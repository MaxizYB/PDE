"""Matrix-free Darcy residual with two learned multiscale error updates.

Independent v2 experiment.  The online path never constructs a sample-specific
stiffness matrix, Galerkin hierarchy, or coarse inverse.  The fixed mesh plan
contains only element geometry and reference-graph transfers.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ["OPENBLAS_NUM_THREADS"] = "1"

import numpy as np
import scipy.sparse as sp
import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from adapters.darcy import DarcySplit
from adapters.darcy_system import make_mesh_plan, make_system
from solver.inference import FixedMeshProposal
from solver.numerics import multiply, sparse_tensor
from run import describe, load_proposal, predict_numpy, synchronized_clock


def batched_sparse(matrix: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
    """Apply one fixed mesh operator to B x N x C node features."""
    batch, nodes, width = value.shape
    flat = value.permute(1, 0, 2).reshape(nodes, batch * width)
    result = torch.sparse.mm(matrix, flat)
    return result.reshape(matrix.shape[0], batch, width).permute(1, 0, 2)


def normalized_nonnegative(matrix: sp.spmatrix, device: torch.device) -> torch.Tensor:
    positive = abs(matrix).tocsr()
    row_sum = np.asarray(positive.sum(axis=1)).ravel()
    positive = (sp.diags(1.0 / np.maximum(row_sum, 1e-12)) @ positive).tocsr()
    return sparse_tensor(positive, device)


class MatrixFreeDarcy:
    """P1 element action, including nonzero Dirichlet lift, without global A."""

    def __init__(self, mesh, device: torch.device):
        from baselines.data.darcy_v2.generate import element_geometry

        points = np.asarray(mesh.points, dtype=np.float64)
        cells = np.asarray(mesh.cells, dtype=np.int64)
        area, gradient, _ = element_geometry(points, cells)
        local = area[:, None, None] * np.einsum("tic,tjc->tij", gradient, gradient)
        free = np.flatnonzero(~mesh.boundary)
        self.cells = torch.as_tensor(cells, dtype=torch.long, device=device)
        self.flat_cells = self.cells.reshape(-1)
        self.local = torch.as_tensor(local, dtype=torch.float32, device=device)
        self.local_diag = self.local.diagonal(dim1=1, dim2=2)
        self.free = torch.as_tensor(free, dtype=torch.long, device=device)
        self.source = torch.as_tensor(mesh.mass[free], dtype=torch.float32, device=device)
        self.mass = torch.as_tensor(mesh.mass[free], dtype=torch.float32, device=device)
        self.full_mass = torch.as_tensor(mesh.mass, dtype=torch.float32, device=device)
        self.node_count = len(points)
        self.device = device

    def prepare(self, coefficient: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        element_a = coefficient[:, self.cells].mean(dim=-1)
        diagonal = torch.zeros(
            (coefficient.shape[0], self.node_count), device=self.device,
            dtype=coefficient.dtype,
        )
        diagonal.scatter_add_(
            1, self.flat_cells[None].expand(coefficient.shape[0], -1),
            (element_a[:, :, None] * self.local_diag[None]).reshape(coefficient.shape[0], -1),
        )
        return element_a, diagonal[:, self.free]

    def action(
        self, free_value: torch.Tensor, lift: torch.Tensor,
        element_a: torch.Tensor,
    ) -> torch.Tensor:
        full = lift.clone()
        full[:, self.free] = free_value
        element_value = full[:, self.cells]
        element_result = torch.einsum("tij,btj->bti", self.local, element_value)
        element_result = element_a[:, :, None] * element_result
        full_result = torch.zeros(
            (free_value.shape[0], self.node_count), device=self.device,
            dtype=free_value.dtype,
        )
        full_result.scatter_add_(
            1, self.flat_cells[None].expand(free_value.shape[0], -1),
            element_result.reshape(free_value.shape[0], -1),
        )
        return full_result[:, self.free]

    def residual(self, x, lift, element_a):
        return self.source[None] - self.action(x, lift, element_a)

    def homogeneous_action(self, x, element_a):
        return self.action(x, torch.zeros((x.shape[0], self.node_count),
                                          device=self.device, dtype=x.dtype), element_a)


class MeshScales:
    """Fixed graph topology; only the residual and coefficient vary by sample."""

    def __init__(self, mesh, device, coarse_size=128):
        from baselines.data.darcy_v2.generate import assemble, element_geometry

        plan = make_mesh_plan(mesh, device, coarse_size=coarse_size)
        points = np.asarray(mesh.points, dtype=np.float64)
        cells = np.asarray(mesh.cells, dtype=np.int32)
        geometry = element_geometry(points, cells)
        reference, _ = assemble(points, cells, np.ones(len(points)), geometry)
        matrix = reference[plan.free][:, plan.free].tocsr()
        self.graphs = []
        self.down, self.up = [], []
        for p, r in plan.transfers:
            graph = abs(matrix).tocsr()
            graph.setdiag(0)
            graph.eliminate_zeros()
            self.graphs.append(normalized_nonnegative(graph, device))
            self.down.append(normalized_nonnegative(r, device))
            self.up.append(normalized_nonnegative(p, device))
            matrix = (r @ matrix @ p).tocsr()
        graph = abs(matrix).tocsr()
        graph.setdiag(0)
        graph.eliminate_zeros()
        self.graphs.append(normalized_nonnegative(graph, device))
        self.points = torch.as_tensor(points[plan.free], dtype=torch.float32, device=device)
        mass = np.asarray(mesh.mass[plan.free], dtype=np.float32)
        self.log_mass = torch.as_tensor(
            np.log(mass / np.median(mass)), dtype=torch.float32, device=device
        )
        self.free = plan.free


class MultiscaleDirection(nn.Module):
    """Local graph exchange, coarse global exchange, and fine reconstruction."""

    def __init__(self, levels: int, width: int = 32):
        super().__init__()
        self.input = nn.Linear(9, width)
        self.self_layers = nn.ModuleList(nn.Linear(width, width) for _ in range(levels))
        self.neighbor_layers = nn.ModuleList(nn.Linear(width, width) for _ in range(levels))
        self.down_layers = nn.ModuleList(nn.Linear(width, width) for _ in range(levels - 1))
        self.up_layers = nn.ModuleList(nn.Linear(width, width) for _ in range(levels - 1))
        self.attention = nn.MultiheadAttention(width, 4, batch_first=True)
        self.output = nn.Sequential(nn.Linear(width, width), nn.GELU(), nn.Linear(width, 2))
        nn.init.zeros_(self.output[-1].weight)
        nn.init.zeros_(self.output[-1].bias)

    def forward(self, features, meshes: MeshScales):
        hidden = F.gelu(self.input(features))
        skips = []
        for level, graph in enumerate(meshes.graphs):
            hidden = hidden + F.gelu(
                self.self_layers[level](hidden)
                + self.neighbor_layers[level](batched_sparse(graph, hidden))
            )
            if level < len(meshes.down):
                skips.append(hidden)
                hidden = F.gelu(self.down_layers[level](
                    batched_sparse(meshes.down[level], hidden)
                ))
        attended, _ = self.attention(hidden, hidden, hidden, need_weights=False)
        hidden = hidden + attended
        for level in reversed(range(len(meshes.up))):
            hidden = skips[level] + F.gelu(
                self.up_layers[level](batched_sparse(meshes.up[level], hidden))
            )
        return self.output(hidden)


class TwoStepSolver(nn.Module):
    def __init__(self, levels: int, width: int = 32):
        super().__init__()
        self.direction = MultiscaleDirection(levels, width)

    def forward(self, operator: MatrixFreeDarcy, meshes: MeshScales,
                coefficient, lift, x0, element_a=None, diagonal=None):
        if element_a is None or diagonal is None:
            element_a, diagonal = operator.prepare(coefficient)
        u_scale = x0.square().mean(dim=1, keepdim=True).sqrt().clamp_min(0.003)
        x = x0
        states = []
        for step in range(2):
            residual = operator.residual(x, lift, element_a)
            jacobi = residual / diagonal.clamp_min(1e-7)
            norm = jacobi.square().mean(dim=1, keepdim=True).sqrt().clamp_min(1e-8)
            features = torch.stack((
                coefficient[:, operator.free].log() / np.log(12),
                (diagonal / diagonal.median(dim=1, keepdim=True).values).log(),
                meshes.points[None, :, 0].expand_as(jacobi),
                meshes.points[None, :, 1].expand_as(jacobi),
                meshes.log_mass[None].expand_as(jacobi),
                jacobi / norm,
                jacobi / u_scale,
                x / u_scale,
                torch.full_like(jacobi, float(step)),
            ), dim=-1)
            prediction = self.direction(features, meshes)
            direction = (
                0.69314718 * jacobi
                + 0.1 * u_scale * torch.tanh(prediction[..., 0])
                + 0.25 * jacobi * torch.tanh(prediction[..., 1])
            )
            ad = operator.homogeneous_action(direction, element_a)
            alpha = ((residual * direction).sum(dim=1, keepdim=True) /
                     (direction * ad).sum(dim=1, keepdim=True).clamp_min(1e-10)).clamp(0, 2)
            x = x + alpha * direction
            states.append(x)
        return states


def rel_error(x, truth, full_truth, op: MatrixFreeDarcy):
    numerator = ((x - truth[:, op.free]).square() * op.mass).sum(dim=1)
    denominator = (full_truth.square() * op.full_mass).sum(dim=1).clamp_min(1e-12)
    return (numerator / denominator).sqrt()


def tensors(split, proposal, device):
    return (
        torch.as_tensor(split.coefficient, device=device),
        torch.as_tensor(split.lift, device=device),
        torch.as_tensor(split.target, device=device),
        proposal[:, :].detach(),
    )


def train(args, device):
    started = time.perf_counter()
    train_data = DarcySplit(args.data, "train")
    val_data = DarcySplit(args.data, "val")
    proposal_model, scale = load_proposal(args, device)
    proposal_train = predict_numpy(proposal_model, train_data, scale, device)
    proposal_val = predict_numpy(proposal_model, val_data, scale, device)
    operator = MatrixFreeDarcy(train_data.mesh, device)
    meshes = MeshScales(train_data.mesh, device)
    model = TwoStepSolver(len(meshes.graphs), args.width).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    train_tensors = tensors(train_data, proposal_train, device)
    val_tensors = tensors(val_data, proposal_val, device)
    logs, best = [], float("inf")
    for epoch in range(args.epochs):
        model.train()
        order = np.random.permutation(len(train_data))
        batch_losses = []
        for start in range(0, len(order), args.batch_size):
            ids = order[start:start + args.batch_size]
            coeff, lift, truth, x0_full = (array[ids] for array in train_tensors)
            x0 = x0_full[:, operator.free]
            states = model(operator, meshes, coeff, lift, x0)
            loss = (0.15 * rel_error(states[0], truth, truth, operator)
                    + rel_error(states[1], truth, truth, operator)).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            batch_losses.append(float(loss.detach()))
        model.eval()
        with torch.no_grad():
            coeff, lift, truth, x0_full = val_tensors
            states = model(operator, meshes, coeff, lift, x0_full[:, operator.free])
            val_step1 = float(rel_error(states[0], truth, truth, operator).mean()) * 100
            val_step2 = float(rel_error(states[1], truth, truth, operator).mean()) * 100
        logs.append({"epoch": epoch + 1, "train_loss": float(np.mean(batch_losses)),
                     "val_step1_percent": val_step1, "val_step2_percent": val_step2})
        if val_step2 < best:
            best = val_step2
            torch.save({"state": model.state_dict(), "levels": len(meshes.graphs),
                        "width": args.width, "epoch": epoch + 1,
                        "best_val_step2_percent": best}, args.out / "model.pt")
        print(f"epoch {epoch + 1}/{args.epochs}: train={np.mean(batch_losses):.5f} "
              f"val1={val_step1:.4f}% val2={val_step2:.4f}% best={best:.4f}%", flush=True)
    (args.out / "train.json").write_text(json.dumps({
        "training_seconds": time.perf_counter() - started, "epochs": logs}, indent=2) + "\n")


def verify_operator(split, operator, device):
    plan = make_mesh_plan(split.mesh, device, coarse_size=128,
                          gpu_galerkin=True, compiled_assembly=True,
                          hybrid_dense_limit=512)
    checked = []
    with torch.no_grad():
        for i in (0, min(1, len(split) - 1)):
            coefficient = torch.as_tensor(split.coefficient[i:i + 1], device=device)
            lift = torch.as_tensor(split.lift[i:i + 1], device=device)
            system = make_system(split.mesh, split.coefficient[i], split.lift[i],
                                 device, plan, backend="gpu_hybrid")
            element_a, diagonal = operator.prepare(coefficient)
            x = torch.as_tensor(split.target[i:i + 1, plan.free], device=device)
            residual = operator.residual(x, lift, element_a)[0]
            reference = system.rhs - multiply(system.matrix, x[0])
            diagonal_reference = system.levels[0].diagonal_inverse.reciprocal()
            checked.append({"residual_max_abs": float((residual - reference).abs().max()),
                            "diagonal_max_abs": float((diagonal[0] - diagonal_reference).abs().max())})
        # Nonzero boundary exercises free/fixed coupling independently of test samples.
        xy = split.mesh.points
        synthetic_lift = torch.as_tensor(
            (0.01 + 0.02 * xy[:, 0] - 0.01 * xy[:, 1]).astype(np.float32)[None],
            device=device,
        )
        system = make_system(split.mesh, split.coefficient[0],
                             synthetic_lift[0].cpu().numpy(), device, plan, backend="gpu_hybrid")
        coeff = torch.as_tensor(split.coefficient[0:1], device=device)
        element_a, diagonal = operator.prepare(coeff)
        x = torch.as_tensor(split.target[0:1, plan.free], device=device)
        residual = operator.residual(x, synthetic_lift, element_a)[0]
        reference = system.rhs - multiply(system.matrix, x[0])
        checked.append({"residual_max_abs": float((residual - reference).abs().max()),
                        "diagonal_max_abs": float((diagonal[0] -
                                                   system.levels[0].diagonal_inverse.reciprocal()).abs().max())})
    if max(row["residual_max_abs"] for row in checked) > 1e-5:
        raise AssertionError("matrix-free residual differs from assembled FEM")
    return checked


def evaluate(args, device):
    split = DarcySplit(args.eval_data or args.data, "test")
    proposal_model, scale = load_proposal(args, device)
    checkpoint = torch.load(args.out / "model.pt", map_location=device, weights_only=False)
    model = TwoStepSolver(checkpoint["levels"], checkpoint["width"]).to(device)
    model.load_state_dict(checkpoint["state"])
    model.eval()
    t0 = synchronized_clock(device)
    operator = MatrixFreeDarcy(split.mesh, device)
    meshes = MeshScales(split.mesh, device)
    mesh_once = 1000 * (synchronized_clock(device) - t0)
    checks = verify_operator(split, operator, device)
    proposal = FixedMeshProposal(proposal_model, split, scale, device,
                                 capture_graph=True, use_prepared=True)
    records = []
    with torch.no_grad():
        for i in range(min(args.count, len(split)) + 1):
            index = 0 if i == 0 else i - 1
            t0 = synchronized_clock(device)
            coefficient = torch.as_tensor(split.coefficient[index:index + 1], device=device)
            lift = torch.as_tensor(split.lift[index:index + 1], device=device)
            element_a, diagonal = operator.prepare(coefficient)
            prepare_ms = 1000 * (synchronized_clock(device) - t0)
            t0 = synchronized_clock(device)
            x0_full = proposal.predict(split, index)[None]
            no_ms = 1000 * (synchronized_clock(device) - t0)
            t0 = synchronized_clock(device)
            states = model(operator, meshes, coefficient, lift, x0_full[:, operator.free],
                           element_a, diagonal)
            update_ms = 1000 * (synchronized_clock(device) - t0)
            truth = torch.as_tensor(split.target[index:index + 1], device=device)
            values = [float(rel_error(x, truth, truth, operator)[0]) * 100
                      for x in (x0_full[:, operator.free], *states)]
            if i:
                records.append({"index": int(split.sample_index[index]),
                                "NO_error_percent": values[0],
                                "step1_error_percent": values[1],
                                "step2_error_percent": values[2],
                                "prepare_ms": prepare_ms, "NO_ms": no_ms,
                                "update_ms": update_ms,
                                "total_ms": prepare_ms + no_ms + update_ms})
    summary = {
        "NO_error_percent": describe([r["NO_error_percent"] for r in records]),
        "step1_error_percent": describe([r["step1_error_percent"] for r in records]),
        "step2_error_percent": describe([r["step2_error_percent"] for r in records]),
        "prepare_ms": describe([r["prepare_ms"] for r in records]),
        "NO_ms": describe([r["NO_ms"] for r in records]),
        "update_ms": describe([r["update_ms"] for r in records]),
        "total_ms": describe([r["total_ms"] for r in records]),
        "mesh_plan_ms_once": mesh_once,
        "proposal_startup_ms_once": proposal.compile_ms_once,
        "peak_allocated_mib": torch.cuda.max_memory_allocated(device) / 1024**2,
    }
    result = {"status": "independent preview", "samples": len(records),
              "training_epoch": checkpoint["epoch"], "residual_checks": checks,
              "summary": summary, "per_sample": records}
    suffix = "ood" if args.eval_data is not None else "id"
    (args.out / f"matrixfree_{suffix}.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("train", "evaluate", "all"))
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--eval-data", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--proposal-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.0007)
    parser.add_argument("--count", type=int, default=16)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    args.checkpoint_dir = None
    torch.set_num_threads(4)
    torch.manual_seed(20260928)
    np.random.seed(20260928)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("this preview uses the server GPU environment")
    if args.command in ("train", "all"):
        train(args, device)
    if args.command in ("evaluate", "all"):
        evaluate(args, device)


if __name__ == "__main__":
    main()
