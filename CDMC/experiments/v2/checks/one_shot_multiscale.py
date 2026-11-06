"""Experimental one-shot learned multiscale error direction, Darcy v2.

This is deliberately separate from the validated CDMC corrector.  It replaces
both V-cycles with one graph-conditioned multilevel error direction and one
SPD energy step; checkpoints are selected on validation, not test samples.
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
import torch
from torch import nn
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from adapters.darcy import DarcySplit
from adapters.darcy_system import make_mesh_plan, make_system
from run import describe, error_percent, load_proposal, predict_numpy, synchronized_clock
from solver.correctors import CDMCSetup, cdmc, fixed_mg
from solver.inference import FixedMeshProposal
from solver.numerics import TorchSystem, coarsen_gates, multiply


class OneShotMultiscale(nn.Module):
    """NO-defect graph features select one parallel, level-wise correction.

    Level right-hand sides are restrictions of the initial defect, not
    recursively updated V-cycle defects.  The only exact solve is the existing
    small terminal inverse. No numerical smoothing loop is used.
    """

    def __init__(self, levels: int, width: int = 32):
        super().__init__()
        self.lift = nn.Linear(6, width)
        self.self_layers = nn.ModuleList(nn.Linear(width, width) for _ in range(2))
        self.neighbor_layers = nn.ModuleList(nn.Linear(width, width) for _ in range(2))
        self.output = nn.Linear(width, 2)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)
        starting = torch.tensor([0.6] + [0.15] * (levels - 2) + [1.0])
        self.level_weights = nn.Parameter(torch.log(torch.expm1(starting)))

    def forward(self, system: TorchSystem, initial: torch.Tensor) -> torch.Tensor:
        residual = system.rhs - multiply(system.matrix, initial)
        scaled = residual * system.levels[0].diagonal_inverse.sqrt()
        normalized = scaled / scaled.square().mean().sqrt().clamp_min(1e-7)
        hidden = F.gelu(self.lift(torch.cat((system.feature, normalized[:, None]), dim=1)))
        for local, neighbor in zip(self.self_layers, self.neighbor_layers):
            hidden = hidden + F.gelu(local(hidden) + neighbor(torch.sparse.mm(system.graph, hidden)))
        logits = self.output(hidden)
        fine_gate = 0.5 + torch.sigmoid(logits[:, 0])
        coarse_gate = 0.5 + torch.sigmoid(logits[:, 1])
        fine_gates, coarse_gates = coarsen_gates(system, fine_gate, coarse_gate)
        rhs_levels = [residual]
        for item in system.levels[:-1]:
            rhs_levels.append(multiply(item.restriction, rhs_levels[-1]))
        weights = F.softplus(self.level_weights)
        last = len(system.levels) - 1
        direction = (weights[last] * coarse_gates[last] *
                     (system.coarse_inverse @ rhs_levels[last].double()).to(residual.dtype))
        for level in range(last - 1, -1, -1):
            item = system.levels[level]
            direction = (multiply(item.prolongation, direction)
                         + weights[level] * fine_gates[level]
                         * item.diagonal_inverse * rhs_levels[level])
        ad = multiply(system.matrix, direction)
        alpha = (residual @ direction / (direction @ ad).clamp_min(1e-12)).clamp(0, 2)
        return initial + alpha * direction


def prepared(split, model, scale, device):
    plan = make_mesh_plan(split.mesh, device, coarse_size=128)
    proposal = predict_numpy(model, split, scale, device)
    examples = []
    for i in range(len(split)):
        system = make_system(split.mesh, split.coefficient[i], split.lift[i], device, plan)
        target = torch.as_tensor(split.target[i], device=device)
        lift = torch.as_tensor(split.lift[i], device=device)
        mass = torch.as_tensor(split.mesh.mass, device=device)
        examples.append((system, proposal[i, system.free].detach(), target, lift, mass))
    return examples, len(plan.transfers) + 1


def relative_loss(network, example):
    system, initial, target, lift, mass = example
    interior = network(system, initial)
    field = lift.clone()
    field[system.free] = interior
    return ((mass * (field - target).square()).sum() /
            (mass * target.square()).sum().clamp_min(1e-12) + 1e-12).sqrt()


def train(args, device):
    model, scale = load_proposal(args, device)
    train_split = DarcySplit(args.data, "train")
    val_split = DarcySplit(args.data, "val")
    t0 = time.perf_counter()
    training, levels = prepared(train_split, model, scale, device)
    validation, val_levels = prepared(val_split, model, scale, device)
    assert levels == val_levels
    network = OneShotMultiscale(levels).to(device)
    optimizer = torch.optim.AdamW(network.parameters(), lr=args.lr)
    history, best = [], float("inf")
    for epoch in range(args.epochs):
        network.train()
        train_losses = []
        for idx in np.random.permutation(len(training)):
            loss = relative_loss(network, training[int(idx)])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(network.parameters(), 5.0)
            optimizer.step()
            train_losses.append(float(loss.detach()))
        network.eval()
        with torch.no_grad():
            val_loss = float(np.mean([float(relative_loss(network, x)) for x in validation]))
        history.append({"epoch": epoch + 1, "train_percent": 100 * np.mean(train_losses),
                        "val_percent": 100 * val_loss})
        if val_loss < best:
            best = val_loss
            torch.save({"state": network.state_dict(), "levels": levels,
                        "best_val_percent": 100 * best, "epoch": epoch + 1}, args.out / "one_shot.pt")
        print(f"epoch={epoch + 1} train={100*np.mean(train_losses):.4f}% "
              f"val={100*val_loss:.4f}% best={100*best:.4f}%", flush=True)
    (args.out / "train.json").write_text(json.dumps({"history": history,
        "training_seconds": time.perf_counter() - t0}, indent=2) + "\n")


def evaluate(args, device):
    split = DarcySplit(args.eval_data or args.data, "test")
    model, scale = load_proposal(args, device)
    ckpt = torch.load(args.out / "one_shot.pt", map_location=device, weights_only=False)
    network = OneShotMultiscale(ckpt["levels"]).to(device)
    network.load_state_dict(ckpt["state"])
    network.eval()
    setup_ckpt = torch.load(args.cdmc_dir / "cdmc.pt", map_location=device, weights_only=False)
    setup = CDMCSetup(width=setup_ckpt["width"]).to(device)
    setup.load_state_dict(setup_ckpt["state"])
    setup.eval()
    t0 = synchronized_clock(device)
    plan = make_mesh_plan(
        split.mesh, device, coarse_size=128, gpu_galerkin=True,
        compiled_assembly=True, hybrid_dense_limit=512,
        affine_compiler=args.system_backend == "gpu_affine",
    )
    plan_ms_once = 1000 * (synchronized_clock(device) - t0)
    proposal = FixedMeshProposal(model, split, scale, device, capture_graph=True, use_prepared=True)
    records = []

    def run_sample(i, timed):
        t0 = synchronized_clock(device)
        system = make_system(split.mesh, split.coefficient[i], split.lift[i], device, plan,
                             backend=args.system_backend)
        system_ms = 1000 * (synchronized_clock(device) - t0)
        t0 = synchronized_clock(device)
        initial_field = proposal.predict(split, i)
        proposal_ms = 1000 * (synchronized_clock(device) - t0)
        initial = initial_field[system.free]
        result = {"sample_index": i, "system_ms": system_ms, "NO_ms": proposal_ms}
        target = torch.as_tensor(split.target[i], device=device)
        mass = torch.as_tensor(split.mesh.mass, device=device)
        result["NO_error_percent"] = error_percent(initial_field, target, mass)
        lift = torch.as_tensor(split.lift[i], device=device)
        for name, call in (
            ("one_shot", lambda: network(system, initial)),
            ("MG_K1", lambda: fixed_mg(system, initial, 1)),
            ("CDMC_K2", lambda: cdmc(setup, system, initial, 2)),
        ):
            t0 = synchronized_clock(device)
            x = call()
            apply_ms = 1000 * (synchronized_clock(device) - t0)
            field = lift.clone()
            field[system.free] = x
            result[name + "_error_percent"] = error_percent(field, target, mass)
            result[name + "_apply_ms"] = apply_ms
            result[name + "_total_ms"] = system_ms + proposal_ms + apply_ms
        if timed:
            records.append(result)

    with torch.no_grad():
        run_sample(0, timed=False)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        for i in range(min(args.eval_count, len(split))):
            run_sample(i, timed=True)
    names = ("one_shot", "MG_K1", "CDMC_K2")
    summary = {name: {
        "field_error_percent": describe([row[name + "_error_percent"] for row in records]),
        "apply_ms": describe([row[name + "_apply_ms"] for row in records]),
        "total_ms": describe([row[name + "_total_ms"] for row in records]),
    } for name in names}
    summary["NO"] = {
        "field_error_percent": describe([row["NO_error_percent"] for row in records]),
        "total_ms": describe([row["NO_ms"] for row in records]),
    }
    summary["system_ms"] = describe([row["system_ms"] for row in records])
    summary["mesh_plan_ms_once"] = plan_ms_once
    summary["peak_allocated_mib"] = (torch.cuda.max_memory_allocated(device) / 1024**2
                                    if device.type == "cuda" else None)
    result = {"data": str(args.eval_data or args.data), "backend": args.system_backend,
              "trained_epoch": ckpt["epoch"], "best_val_percent": ckpt["best_val_percent"],
              "evaluated_samples": len(records), "summary": summary, "per_sample": records}
    suffix = "ood" if args.eval_data is not None else "id"
    (args.out / f"one_shot_{suffix}_{args.system_backend}.json").write_text(
        json.dumps(result, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("train", "evaluate", "all"))
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--eval-data", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--proposal-dir", type=Path, required=True)
    parser.add_argument("--cdmc-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=16)
    parser.add_argument("--lr", type=float, default=0.0005)
    parser.add_argument("--eval-count", type=int, default=16)
    parser.add_argument("--system-backend", choices=("gpu_hybrid", "gpu_affine"), default="gpu_hybrid")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    args.checkpoint_dir = None
    torch.set_num_threads(4)
    torch.manual_seed(20260928)
    np.random.seed(20260928)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("one-shot preview requires the server GPU environment")
    if args.command in ("train", "all"):
        train(args, device)
    if args.command in ("evaluate", "all"):
        evaluate(args, device)


if __name__ == "__main__":
    main()
