"""Frozen-CDMC equivalence test for exact implicit Galerkin level actions.

Stage A deliberately keeps the old hierarchy setup for diagonals, damping,
and terminal inverse.  Only the V-cycle's level matrix-vector products change.
This separates mathematical correctness from the subsequent setup optimization.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from adapters.darcy import DarcySplit
from adapters.darcy_system import make_mesh_plan, make_system
from experimental.matrixfree_two_step import MatrixFreeDarcy
from experimental.sparse_projection_maps import cumulative_bases
from run import describe, load_proposal, synchronized_clock
from solver.correctors import CDMCSetup
from solver.inference import FixedMeshProposal
from solver.numerics import coarsen_gates, correct, multiply, sparse_tensor


class ImplicitGalerkin:
    def __init__(self, mesh, plan, device):
        self.fine = MatrixFreeDarcy(mesh, device)
        basis = sp.eye(len(plan.free), format="csr", dtype=np.float32)
        self.bases = [torch.eye(len(plan.free), device=device)]
        for prolongation, _ in plan.transfers:
            basis = (basis @ prolongation).tocsr()
            self.bases.append(torch.as_tensor(basis.toarray(),
                                               dtype=torch.float32, device=device))
        self.level_sizes = [basis.shape[1] for basis in self.bases]

    def action(self, level, vector, element_a):
        basis = self.bases[level]
        fine_vector = vector if level == 0 else basis @ vector
        fine_result = self.fine.homogeneous_action(fine_vector[None], element_a)[0]
        return fine_result if level == 0 else basis.T @ fine_result


class SparseImplicitGalerkin:
    """The same Q_l^T A_0 Q_l action using CSR Q_l, with no dense identity."""

    def __init__(self, mesh, plan, device):
        self.fine = MatrixFreeDarcy(mesh, device)
        self.bases = cumulative_bases(plan.free, plan.transfers)
        self.forward = [None] + [sparse_tensor(basis, device)
                                 for basis in self.bases[1:]]
        self.backward = [None] + [sparse_tensor(basis.T, device)
                                  for basis in self.bases[1:]]
        self.level_sizes = [basis.shape[1] for basis in self.bases]

    def action(self, level, vector, element_a):
        fine_vector = vector if level == 0 else multiply(self.forward[level], vector)
        fine_result = self.fine.homogeneous_action(fine_vector[None], element_a)[0]
        return fine_result if level == 0 else multiply(self.backward[level], fine_result)


class FactorizedGalerkin:
    """Exact Q_l^T A_0 Q_l action through the stored sparse P/R chain."""

    def __init__(self, mesh, plan, device):
        self.fine = MatrixFreeDarcy(mesh, device)
        self.prolongations = [p for p, _, _ in plan.gpu_transfers]
        self.restrictions = [r for _, r, _ in plan.gpu_transfers]
        self.level_sizes = [len(plan.free)] + [p.shape[1] for p, _ in plan.transfers]

    def action(self, level, vector, element_a):
        fine_vector = vector
        for depth in reversed(range(level)):
            fine_vector = multiply(self.prolongations[depth], fine_vector)
        fine_result = self.fine.homogeneous_action(fine_vector[None], element_a)[0]
        for depth in range(level):
            fine_result = multiply(self.restrictions[depth], fine_result)
        return fine_result


def implicit_v_cycle(system, implicit, element_a, rhs, smooth_gates, coarse_gates,
                     level=0, smoothing_steps=2, omega_override=None,
                     reference_inverse=None, coarse_steps=0):
    if level == len(system.levels) - 1:
        if coarse_steps:
            if reference_inverse is None:
                raise ValueError("coarse PCG needs a reference coarse inverse")
            scale = element_a.mean().clamp_min(1e-8)
            x = torch.zeros_like(rhs)
            residual = rhs.clone()
            z = (reference_inverse @ residual) / scale
            direction = z.clone()
            inner = residual @ z
            for _ in range(coarse_steps):
                ad = implicit.action(level, direction, element_a)
                alpha = inner / (direction @ ad).clamp_min(1e-20)
                x = x + alpha * direction
                residual = residual - alpha * ad
                z = (reference_inverse @ residual) / scale
                next_inner = residual @ z
                direction = z + next_inner / inner.clamp_min(1e-20) * direction
                inner = next_inner
            return x
        if system.coarse_factor is not None:
            return torch.cholesky_solve(rhs.double()[:, None],
                                        system.coarse_factor)[:, 0].float()
        return (system.coarse_inverse @ rhs.double()).float()
    item = system.levels[level]
    x = torch.zeros_like(rhs)
    omega = item.omega if omega_override is None else omega_override[level]
    damping = omega * smooth_gates[level] * item.diagonal_inverse
    for _ in range(smoothing_steps):
        x = x + damping * (rhs - implicit.action(level, x, element_a))
    remaining = rhs - implicit.action(level, x, element_a)
    child_rhs = multiply(item.restriction, coarse_gates[level] * remaining)
    child = implicit_v_cycle(system, implicit, element_a, child_rhs,
                             smooth_gates, coarse_gates, level + 1, smoothing_steps,
                             omega_override, reference_inverse, coarse_steps)
    x = x + coarse_gates[level] * multiply(item.prolongation, child)
    for _ in range(smoothing_steps):
        x = x + damping * (rhs - implicit.action(level, x, element_a))
    return x


def implicit_correct(system, implicit, element_a, initial, smooth, coarse, cycles=2,
                     omega_override=None, reference_inverse=None, coarse_steps=0):
    smooth_gates, coarse_gates = coarsen_gates(system, smooth, coarse)
    x = initial
    for _ in range(cycles):
        residual = system.rhs - implicit.action(0, x, element_a)
        direction = implicit_v_cycle(system, implicit, element_a, residual,
                                     smooth_gates, coarse_gates,
                                     omega_override=omega_override,
                                     reference_inverse=reference_inverse,
                                     coarse_steps=coarse_steps)
        ad = implicit.action(0, direction, element_a)
        alpha = (residual @ direction / (direction @ ad).clamp_min(1e-12)).clamp(0, 2)
        x = x + alpha * direction
    return x


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--proposal-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--count", type=int, default=16)
    parser.add_argument("--capture-probe", action="store_true")
    parser.add_argument("--fixed-omega", action="store_true")
    parser.add_argument("--coarse-steps", type=int, default=0)
    args = parser.parse_args()
    torch.set_num_threads(4)
    device = torch.device("cuda")
    split = DarcySplit(args.data, "test")
    proposal_model, scale = load_proposal(args, device)
    checkpoint = torch.load(args.checkpoint_dir / "cdmc.pt", map_location=device,
                            weights_only=False)
    setup = CDMCSetup(checkpoint["width"]).to(device)
    setup.load_state_dict(checkpoint["state"])
    setup.eval()
    plan = make_mesh_plan(split.mesh, device, int(checkpoint["coarse_size"]),
                          gpu_galerkin=True, compiled_assembly=True,
                          hybrid_dense_limit=512)
    implicit = ImplicitGalerkin(split.mesh, plan, device)
    reference_omega = None
    reference_inverse = None
    if args.fixed_omega or args.coarse_steps:
        reference = make_system(
            split.mesh, np.ones(len(split.mesh.points), dtype=np.float32),
            np.zeros(len(split.mesh.points), dtype=np.float32), device, plan,
            backend="gpu_hybrid",
        )
        if args.fixed_omega:
            reference_omega = [level.omega for level in reference.levels[:-1]]
        if args.coarse_steps:
            reference_inverse = reference.coarse_inverse.float()
    proposal = FixedMeshProposal(proposal_model, split, scale, device,
                                 capture_graph=True, use_prepared=True)
    records = []
    graph_metrics = None
    with torch.no_grad():
        for i in range(min(args.count, len(split)) + 1):
            index = 0 if i == 0 else i - 1
            system = make_system(split.mesh, split.coefficient[index],
                                 split.lift[index], device, plan,
                                 backend="gpu_hybrid")
            initial = proposal.predict(split, index)[system.free]
            smooth, coarse = setup(system, initial)
            coefficient = torch.as_tensor(split.coefficient[index:index + 1], device=device)
            element_a, _ = implicit.fine.prepare(coefficient)
            action_diffs = []
            for level, item in enumerate(system.levels):
                test_vector = torch.linspace(-0.5, 0.5, item.matrix.shape[0], device=device)
                old = multiply(item.matrix, test_vector)
                new = implicit.action(level, test_vector, element_a)
                action_diffs.append(float((new - old).abs().max()))
            t0 = synchronized_clock(device)
            old = correct(system, initial, 2, smooth, coarse)
            old_ms = 1000 * (synchronized_clock(device) - t0)
            t0 = synchronized_clock(device)
            new = implicit_correct(system, implicit, element_a, initial,
                                   smooth, coarse, omega_override=reference_omega,
                                   reference_inverse=reference_inverse,
                                   coarse_steps=args.coarse_steps)
            implicit_ms = 1000 * (synchronized_clock(device) - t0)
            if i == 0 and args.capture_probe:
                stream = torch.cuda.Stream(device=device)
                stream.wait_stream(torch.cuda.current_stream(device))
                with torch.cuda.stream(stream):
                    for _ in range(3):
                        implicit_correct(system, implicit, element_a, initial,
                                         smooth, coarse,
                                         omega_override=reference_omega,
                                         reference_inverse=reference_inverse,
                                         coarse_steps=args.coarse_steps)
                torch.cuda.current_stream(device).wait_stream(stream)
                torch.cuda.synchronize(device)
                graph = torch.cuda.CUDAGraph()
                capture_started = time.perf_counter()
                with torch.cuda.graph(graph):
                    graph_result = implicit_correct(system, implicit, element_a,
                                                    initial, smooth, coarse,
                                                    omega_override=reference_omega,
                                                    reference_inverse=reference_inverse,
                                                    coarse_steps=args.coarse_steps)
                graph.replay()
                capture_ms = 1000 * (synchronized_clock(device) - capture_started)
                replay_times = []
                for _ in range(20):
                    t0 = synchronized_clock(device)
                    graph.replay()
                    replay_times.append(1000 * (synchronized_clock(device) - t0))
                graph_metrics = {
                    "replay_ms_single_sample": describe(replay_times),
                    "capture_ms_once": capture_ms,
                    "graph_result_max_abs": float((graph_result - new).abs().max()),
                }
            if i:
                truth = torch.as_tensor(split.target[index], device=device)
                mass = torch.as_tensor(split.mesh.mass[system.free], device=device)
                denominator = (torch.as_tensor(split.mesh.mass, device=device)
                               * truth.square()).sum().clamp_min(1e-12)
                def error(x):
                    return float((mass * (x - truth[system.free]).square()
                                  ).sum().div(denominator).sqrt()) * 100
                records.append({"level_action_max_abs": action_diffs,
                                "solution_max_abs": float((new - old).abs().max()),
                                "old_error_percent": error(old),
                                "implicit_error_percent": error(new),
                                "old_apply_ms": old_ms,
                                "implicit_apply_ms": implicit_ms})
    summary = {key: describe([r[key] for r in records]) for key in
               ("solution_max_abs", "old_error_percent", "implicit_error_percent",
                "old_apply_ms", "implicit_apply_ms")}
    summary["level_action_max_abs"] = [max(r["level_action_max_abs"][j]
                                            for r in records)
                                       for j in range(len(implicit.bases))]
    if graph_metrics is not None:
        summary["graph_probe"] = graph_metrics
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"summary": summary, "samples": records},
                                   indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
