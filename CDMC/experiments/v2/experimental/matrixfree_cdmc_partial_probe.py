"""Frozen-CDMC partial Galerkin setup: exact diagonal/coarse, implicit actions.

Stage B compiles per-element coefficient-to-level maps once per mesh. Online it
forms only the level diagonals and the terminal small matrix; fine/middle
matrix-vector products remain implicit. Projected element tensors are built in
vectorized chunks, avoiding a full per-level matrix map.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from adapters.darcy import DarcySplit
from adapters.darcy_system import make_mesh_plan, make_system
from experimental.matrixfree_cdmc_vcycle_probe import (
    FactorizedGalerkin, ImplicitGalerkin, implicit_correct,
)
from experimental.sparse_projection_maps import (
    compile_sparse_affine_maps, compile_terminal_gradients,
)
from run import describe, load_proposal, synchronized_clock
from solver.correctors import CDMCSetup
from solver.inference import FixedMeshProposal
from solver.numerics import (
    TorchLevel, TorchSystem, jacobi_spectral_bound, multiply, sparse_tensor,
)


def compile_diag_terminal_maps(mesh, geometry, free, transfers, device, chunk=128):
    """Exact affine maps for intermediate diagonals and terminal full matrix."""
    area, gradient, _ = geometry
    local = area[:, None, None] * np.einsum(
        "tic,tjc->tij", gradient, gradient, optimize=True
    )
    cells = np.asarray(mesh.cells, dtype=np.int64)
    triangle_count = len(cells)
    basis = sp.eye(len(free), format="csr", dtype=np.float64)
    maps = []
    for level, (prolongation, _) in enumerate(transfers):
        basis = (basis @ prolongation).tocsr()
        width = basis.shape[1]
        full_basis = np.zeros((len(mesh.points), width), dtype=np.float64)
        full_basis[free] = basis.toarray()
        terminal = level == len(transfers) - 1
        rows, cols, values = [], [], []
        for start in range(0, triangle_count, chunk):
            stop = min(start + chunk, triangle_count)
            element_basis = full_basis[cells[start:stop]]
            projected_basis = np.einsum(
                "tij,tjk->tik", local[start:stop], element_basis,
                optimize=True,
            )
            if terminal:
                projected = np.einsum(
                    "tik,til->tkl", element_basis, projected_basis,
                    optimize=True,
                )
                element, i, j = np.nonzero(np.abs(projected) > 1e-13)
                rows.append(i * width + j)
                values.append(projected[element, i, j])
            else:
                diagonal = np.einsum(
                    "tik,tik->tk", element_basis, projected_basis,
                    optimize=True,
                )
                element, i = np.nonzero(np.abs(diagonal) > 1e-13)
                rows.append(i)
                values.append(diagonal[element, i])
            cols.append(element + start)
        row_count = width * width if terminal else width
        mapping = sp.coo_matrix(
            (np.concatenate(values),
             (np.concatenate(rows), np.concatenate(cols))),
            shape=(row_count, triangle_count), dtype=np.float64,
        ).tocsr()
        mapping.sum_duplicates()
        mapping.eliminate_zeros()
        maps.append(sparse_tensor(mapping, device))
    return maps


def gates_from_implicit(setup, system, implicit, element_a, initial):
    residual = system.rhs - implicit.action(0, initial, element_a)
    normalized = residual * system.levels[0].diagonal_inverse.sqrt()
    normalized = normalized / normalized.square().mean().sqrt().clamp_min(1e-7)
    hidden = F.gelu(setup.lift(torch.cat((system.feature, normalized[:, None]), dim=1)))
    for local, neighbor in zip(setup.self_layers, setup.neighbor_layers):
        hidden = hidden + F.gelu(
            local(hidden) + neighbor(torch.sparse.mm(system.graph, hidden))
        )
    logits = setup.output(hidden)
    return 0.5 + torch.sigmoid(logits[:, 0]), 0.5 + torch.sigmoid(logits[:, 1])


class TerminalAction:
    def __init__(self, implicit, terminal_matrix):
        self.implicit = implicit
        self.terminal_matrix = terminal_matrix
        self.terminal_level = len(implicit.level_sizes) - 1

    def action(self, level, vector, element_a):
        if level == self.terminal_level and self.terminal_matrix is not None:
            return self.terminal_matrix @ vector
        return self.implicit.action(level, vector, element_a)


class PartialGalerkin:
    def __init__(self, mesh, device, coarse_size, sparse_plan=False,
                 approx_diagonal=False, approx_terminal=False,
                 implicit_terminal=False):
        if approx_diagonal and not sparse_plan:
            raise ValueError("approx_diagonal requires sparse_plan")
        if approx_terminal and not approx_diagonal:
            raise ValueError("approx_terminal requires approx_diagonal")
        if implicit_terminal and not approx_diagonal:
            raise ValueError("implicit_terminal requires approx_diagonal")
        if implicit_terminal and approx_terminal:
            raise ValueError("select one terminal mode")
        self.approx_diagonal = approx_diagonal
        self.approx_terminal = approx_terminal
        self.implicit_terminal = implicit_terminal
        started = synchronized_clock(device)
        self.plan = make_mesh_plan(
            mesh, device, coarse_size=coarse_size,
            retain_reference_levels=approx_diagonal,
        )
        self.implicit = (FactorizedGalerkin(mesh, self.plan, device)
                         if sparse_plan else ImplicitGalerkin(mesh, self.plan, device))
        map_started = synchronized_clock(device)
        self.terminal_gradients = None
        if approx_terminal or implicit_terminal:
            self.maps = []
        elif approx_diagonal:
            self.maps = []
            gx, gy, area = compile_terminal_gradients(
                mesh, self.plan.geometry, self.plan.free, self.plan.transfers,
            )
            self.terminal_gradients = (
                torch.as_tensor(gx.toarray(), dtype=torch.float32, device=device),
                torch.as_tensor(gy.toarray(), dtype=torch.float32, device=device),
                torch.as_tensor(area, dtype=torch.float32, device=device),
            )
        elif sparse_plan:
            maps, _ = compile_sparse_affine_maps(
                mesh, self.plan.geometry, self.plan.free, self.plan.transfers,
                terminal_mode="gradient",
            )
            self.maps = [sparse_tensor(mapping, device) for mapping in maps[:-1]]
            gx, gy, area = maps[-1]
            self.terminal_gradients = (
                torch.as_tensor(gx.toarray(), dtype=torch.float32, device=device),
                torch.as_tensor(gy.toarray(), dtype=torch.float32, device=device),
                torch.as_tensor(area, dtype=torch.float32, device=device),
            )
        else:
            self.maps = compile_diag_terminal_maps(
                mesh, self.plan.geometry, self.plan.free, self.plan.transfers, device
            )
        self.map_compile_ms_once = 1000 * (synchronized_clock(device) - map_started)
        if self.plan.reference_levels is not None:
            reference_levels = self.plan.reference_levels
            self.reference_omega = [
                min(0.67, 1.6 / jacobi_spectral_bound(matrix))
                for matrix in reference_levels[:-1]
            ]
            terminal_reference = reference_levels[-1].toarray()
            self.reference_inverse = torch.as_tensor(
                np.linalg.inv(terminal_reference), dtype=torch.float32, device=device
            )
            self.reference_terminal_matrix = (
                torch.as_tensor(terminal_reference, dtype=torch.float32, device=device)
                if approx_terminal else None
            )
            self.reference_diagonals = [
                torch.as_tensor(matrix.diagonal(), dtype=torch.float32, device=device)
                for matrix in reference_levels
            ]
        else:
            reference = make_system(
                mesh, np.ones(len(mesh.points), dtype=np.float32),
                np.zeros(len(mesh.points), dtype=np.float32), device, self.plan,
                backend="cpu",
            )
            self.reference_omega = [level.omega for level in reference.levels[:-1]]
            self.reference_inverse = reference.coarse_inverse.float()
            self.reference_terminal_matrix = (
                reference.levels[-1].matrix.to_dense().float()
                if approx_terminal else None
            )
            self.reference_diagonals = [
                level.diagonal_inverse.reciprocal().float() for level in reference.levels
            ]
        self.squared_restrictions = []
        self.squared_reference_denominators = []
        if approx_diagonal:
            squared_transfers = (
                self.plan.transfers if approx_terminal else self.plan.transfers[:-1]
            )
            for level, (p, _) in enumerate(squared_transfers):
                restriction = sparse_tensor(p.multiply(p).T.tocsr(), device)
                self.squared_restrictions.append(restriction)
                self.squared_reference_denominators.append(
                    multiply(restriction, self.reference_diagonals[level]).clamp_min(1e-12)
                )
        self.points_free = torch.as_tensor(self.plan.points_free,
                                            dtype=torch.float32, device=device)
        self.transfers = []
        for (p, _), (p_gpu, r_gpu, gate_gpu), denominator in zip(
            self.plan.transfers, self.plan.gpu_transfers,
            self.plan.gate_denominators,
        ):
            if p.shape[0] <= 512:
                p_gpu, r_gpu = p_gpu.to_dense(), r_gpu.to_dense()
            self.transfers.append((p_gpu, r_gpu, gate_gpu, denominator))
        self.mesh_plan_ms_once = 1000 * (synchronized_clock(device) - started)
        self.device = device

    def build(self, coefficient, lift, coarse_steps=0, timed=True):
        if self.implicit_terminal and coarse_steps < 1:
            raise ValueError("implicit terminal requires coarse_steps >= 1")
        t0 = synchronized_clock(self.device) if timed else None
        element_a, fine_diagonal = self.implicit.fine.prepare(coefficient[None])
        element_a = element_a[0]
        fine_diagonal = fine_diagonal[0]
        rhs = self.implicit.fine.residual(
            torch.zeros_like(fine_diagonal)[None], lift[None], element_a[None]
        )[0]
        fine_ms = 1000 * (synchronized_clock(self.device) - t0) if timed else 0.0
        t0 = synchronized_clock(self.device) if timed else None
        if self.approx_diagonal:
            intermediate_diagonals = []
            diagonal = fine_diagonal
            for level, (restriction, denominator) in enumerate(zip(
                self.squared_restrictions, self.squared_reference_denominators
            )):
                diagonal = (
                    self.reference_diagonals[level + 1]
                    * multiply(restriction, diagonal) / denominator
                )
                if level < len(self.plan.transfers) - 1:
                    intermediate_diagonals.append(diagonal)
        else:
            intermediate_maps = (self.maps if self.terminal_gradients is not None
                                 else self.maps[:-1])
            intermediate_diagonals = [multiply(mapping, element_a)
                                      for mapping in intermediate_maps]
        terminal_width = self.implicit.level_sizes[-1]
        if self.implicit_terminal:
            terminal_matrix = None
        elif self.approx_terminal:
            scaling = (
                diagonal / self.reference_diagonals[-1].clamp_min(1e-12)
            ).clamp_min(1e-12).sqrt()
            terminal_matrix = (
                scaling[:, None] * self.reference_terminal_matrix
                * scaling[None, :]
            )
        elif self.terminal_gradients is None:
            terminal_matrix = multiply(self.maps[-1], element_a).reshape(
                terminal_width, terminal_width
            )
        else:
            gx, gy, area = self.terminal_gradients
            weighted = (area * element_a)[:, None]
            terminal_matrix = gx.T @ (weighted * gx) + gy.T @ (weighted * gy)
        maps_ms = 1000 * (synchronized_clock(self.device) - t0) if timed else 0.0
        t0 = synchronized_clock(self.device) if timed else None
        levels = []
        terminal_diagonal = (
            self.reference_diagonals[-1] if terminal_matrix is None
            else terminal_matrix.diagonal()
        )
        diagonals = [fine_diagonal] + intermediate_diagonals + [terminal_diagonal]
        for level, (p_gpu, r_gpu, gate_gpu, denominator) in enumerate(self.transfers):
            levels.append(TorchLevel(
                matrix=torch.empty(0, device=self.device),
                diagonal_inverse=diagonals[level].reciprocal(),
                prolongation=p_gpu, restriction=r_gpu,
                gate_restriction=gate_gpu, gate_denominator=denominator,
                omega=self.reference_omega[level],
            ))
        levels.append(TorchLevel(
            matrix=torch.empty(0, device=self.device),
            diagonal_inverse=diagonals[-1].reciprocal(),
        ))
        inverse = (None if coarse_steps else
                   torch.linalg.inv(terminal_matrix.double()))
        feature = torch.stack((
            coefficient[self.implicit.fine.free].log() / np.log(12),
            (fine_diagonal / fine_diagonal.quantile(0.5)).log(),
            self.points_free[:, 0],
            self.points_free[:, 1],
            torch.log(self.plan.mass / self.plan.mass.median()),
        ), dim=-1).float()
        system = TorchSystem(
            levels, rhs, self.plan.free, self.plan.graph, feature,
            self.plan.mass, inverse,
        )
        coarse_ms = 1000 * (synchronized_clock(self.device) - t0) if timed else 0.0
        return system, element_a[None], TerminalAction(self.implicit, terminal_matrix), {
            "fine_ms": fine_ms, "maps_ms": maps_ms, "coarse_and_feature_ms": coarse_ms,
            "setup_ms": fine_ms + maps_ms + coarse_ms,
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--proposal-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--count", type=int, default=16)
    parser.add_argument("--coarse-steps", type=int, default=0)
    args = parser.parse_args()
    torch.set_num_threads(4)
    device = torch.device("cuda")
    split = DarcySplit(args.data, "test")
    model, scale = load_proposal(args, device)
    checkpoint = torch.load(args.checkpoint_dir / "cdmc.pt", map_location=device,
                            weights_only=False)
    setup = CDMCSetup(checkpoint["width"]).to(device)
    setup.load_state_dict(checkpoint["state"])
    setup.eval()
    partial = PartialGalerkin(split.mesh, device, int(checkpoint["coarse_size"]))
    gold_plan = make_mesh_plan(split.mesh, device, int(checkpoint["coarse_size"]),
                               gpu_galerkin=True, compiled_assembly=True,
                               hybrid_dense_limit=512)
    proposal = FixedMeshProposal(model, split, scale, device,
                                 capture_graph=True, use_prepared=True)
    records = []
    with torch.no_grad():
        for i in range(min(args.count, len(split)) + 1):
            index = 0 if i == 0 else i - 1
            coeff = torch.as_tensor(split.coefficient[index], device=device)
            lift = torch.as_tensor(split.lift[index], device=device)
            system, element_a, action, timing = partial.build(
                coeff, lift, args.coarse_steps
            )
            t0 = synchronized_clock(device)
            initial = proposal.predict(split, index)[system.free]
            no_ms = 1000 * (synchronized_clock(device) - t0)
            t0 = synchronized_clock(device)
            smooth, coarse = gates_from_implicit(
                setup, system, action, element_a, initial
            )
            gate_ms = 1000 * (synchronized_clock(device) - t0)
            t0 = synchronized_clock(device)
            candidate = implicit_correct(
                system, action, element_a, initial, smooth, coarse,
                reference_inverse=partial.reference_inverse,
                coarse_steps=args.coarse_steps,
            )
            correct_ms = 1000 * (synchronized_clock(device) - t0)
            if i:
                gold = make_system(split.mesh, split.coefficient[index],
                                   split.lift[index], device, gold_plan,
                                   backend="gpu_hybrid")
                from solver.numerics import correct
                old_smooth, old_coarse = setup(gold, initial)
                old = correct(gold, initial, 2, old_smooth, old_coarse)
                truth = torch.as_tensor(split.target[index], device=device)
                mass = torch.as_tensor(split.mesh.mass, device=device)
                denominator = (mass * truth.square()).sum().clamp_min(1e-12)
                def error(x):
                    return float(((mass[system.free] * (x - truth[system.free]).square()
                                   ).sum() / denominator).sqrt()) * 100
                records.append({**timing, "NO_ms": no_ms, "gate_ms": gate_ms,
                                "correct_uncaptured_ms": correct_ms,
                                "total_uncaptured_ms": timing["setup_ms"] + no_ms
                                + gate_ms + correct_ms,
                                "field_error_percent": error(candidate),
                                "gold_error_percent": error(old),
                                "solution_max_abs": float((candidate - old).abs().max()),
                                "gate_max_abs": max(float((smooth - old_smooth).abs().max()),
                                                    float((coarse - old_coarse).abs().max()))})
    summary = {key: describe([r[key] for r in records]) for key in records[0]}
    summary["mesh_plan_ms_once"] = partial.mesh_plan_ms_once
    summary["map_compile_ms_once"] = partial.map_compile_ms_once
    summary["peak_allocated_mib"] = torch.cuda.max_memory_allocated(device) / 1024**2
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"summary": summary, "samples": records}, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
