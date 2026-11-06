#!/usr/bin/env python3
"""Train and evaluate CDMC with the same matrix-free, direct-terminal V-cycle.

The frozen NO provides the initial field. A single CDMCSetup is shared across
mesh sizes. The last Galerkin system is assembled exactly and solved by
Cholesky; no terminal or global PCG is used.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from adapters.darcy import DarcySplit
from adapters.darcy_system import solve_sample
from run import load_proposal, predict_numpy
from solver.correctors import CDMCSetup
from solver.matrixfree_cdmc import (
    MeshCompiledCDMC, correct_matrixfree, gates_from_implicit,
)


def clock(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return time.perf_counter()


def field_errors(states, example):
    system, _, _, target, lift, mass = example
    denominator = (mass * target.square()).sum().clamp_min(1e-12)
    errors = []
    for interior in states:
        field = lift.clone()
        field[system.free] = interior
        errors.append(
            ((mass * (field - target).square()).sum() / denominator + 1e-12).sqrt()
        )
    return torch.stack(errors)


def prepare(split, model, scale, device, compiled):
    predictions = predict_numpy(model, split, scale, device)
    examples = []
    with torch.no_grad():
        for index in range(len(split)):
            coefficient = torch.as_tensor(split.coefficient[index], device=device)
            lift = torch.as_tensor(split.lift[index], device=device)
            system, element_a = compiled.build(coefficient, lift)
            initial = predictions[index, system.free].detach()
            target = torch.as_tensor(split.target[index], device=device)
            mass = torch.as_tensor(split.mesh.mass, device=device)
            examples.append((system, element_a, initial, target, lift, mass))
    return examples


def solve(setup, compiled, example, learned=True, coarse_enabled=True):
    system, element_a, initial, _, _, _ = example
    if learned:
        smooth, coarse = gates_from_implicit(
            setup, system, compiled.action, element_a, initial
        )
        if not coarse_enabled:
            coarse = torch.ones_like(coarse)
    else:
        smooth = coarse = torch.ones_like(initial)
    return correct_matrixfree(
        system, compiled.action, element_a, initial, smooth, coarse,
        compiled.reference_inverse, cycles=2, coarse_steps=0,
        return_states=True,
    )


def hierarchy_options(args, checkpoint=None):
    """Resolve explicit setup choices; legacy checkpoints use the exact plan."""
    saved = (checkpoint or {}).get("hierarchy_options", {})
    options = {
        "aggregate_size": saved.get("aggregate_size", 2),
        "interpolation_max_entries": saved.get("interpolation_max_entries", 0),
    }
    for name in options:
        value = getattr(args, name)
        if value is not None:
            if checkpoint is not None and value != options[name] and not args.allow_hierarchy_change:
                raise ValueError(
                    f"{name} differs from checkpoint setup; use --allow-hierarchy-change "
                    "for an explicitly recorded frozen-weight setup-transfer test"
                )
            options[name] = value
    return options


def train(args, device):
    model, scale = load_proposal(args, device)
    options = hierarchy_options(args)
    datasets = []
    for root in args.data:
        train_split = DarcySplit(root, "train")
        val_split = DarcySplit(root, "val")
        if (not np.array_equal(train_split.mesh.points, val_split.mesh.points)
                or not np.array_equal(train_split.mesh.cells, val_split.mesh.cells)):
            raise ValueError(f"train and val mesh differ in {root}")
        compiled = MeshCompiledCDMC(
            train_split.mesh, device, args.coarse_size, terminal_solver="direct",
            **options,
        )
        datasets.append({
            "root": str(root), "compiled": compiled,
            "train": prepare(train_split, model, scale, device, compiled),
            "val": prepare(val_split, model, scale, device, compiled),
            "levels": compiled.action.level_sizes,
        })
        print(f"prepared {root}: levels={compiled.action.level_sizes}", flush=True)
    if args.init_cdmc_dir:
        checkpoint = torch.load(
            args.init_cdmc_dir / "cdmc.pt", map_location=device, weights_only=False
        )
        setup = CDMCSetup(checkpoint["width"]).to(device)
        setup.load_state_dict(checkpoint["state"])
    else:
        setup = CDMCSetup(args.setup_width).to(device)
    optimizer = torch.optim.AdamW(setup.parameters(), lr=args.lr)
    best, history = float("inf"), []
    for epoch in range(args.epochs):
        started = clock(device)
        setup.train()
        order = []
        for dataset_index, dataset in enumerate(datasets):
            count = min(args.train_per_scale, len(dataset["train"]))
            order.extend((dataset_index, int(i)) for i in np.random.permutation(
                len(dataset["train"])
            )[:count])
        random.shuffle(order)
        train_errors = []
        for dataset_index, index in order:
            dataset = datasets[dataset_index]
            example = dataset["train"][index]
            errors = field_errors(solve(
                setup, dataset["compiled"], example,
                coarse_enabled=args.gate_mode == "full",
            ), example)
            loss = errors.mean() if args.loss_mode == "mean" else errors[1]
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(setup.parameters(), 5.0)
            optimizer.step()
            train_errors.append(errors.detach())
        setup.eval()
        val_errors = []
        with torch.no_grad():
            for dataset in datasets:
                count = min(args.val_per_scale, len(dataset["val"]))
                scale_errors = [
                    field_errors(
                        solve(
                            setup, dataset["compiled"], example,
                            coarse_enabled=args.gate_mode == "full",
                        ), example
                    ) for example in dataset["val"][:count]
                ]
                val_errors.append(torch.stack(scale_errors).mean(0))
        train_mean = torch.stack(train_errors).mean(0)
        val_mean = torch.stack(val_errors).mean(0)
        row = {
            "epoch": epoch + 1,
            "train_K1_percent": 100 * float(train_mean[0]),
            "train_K2_percent": 100 * float(train_mean[1]),
            "val_K1_percent": 100 * float(val_mean[0]),
            "val_K2_percent": 100 * float(val_mean[1]),
            "val_by_scale_K2_percent": [100 * float(x[1]) for x in val_errors],
            "seconds": clock(device) - started,
        }
        history.append(row)
        print(json.dumps(row), flush=True)
        if row["val_K2_percent"] < best:
            best = row["val_K2_percent"]
            torch.save({
                "state": setup.state_dict(), "width": setup.lift.out_features,
                "coarse_size": args.coarse_size,
                "training_backend": "matrixfree_direct_terminal",
                "trained_cycles": 2,
                "gate_mode": args.gate_mode,
                "loss_mode": args.loss_mode,
                "training_data": [d["root"] for d in datasets],
                "training_levels": [d["levels"] for d in datasets],
                "best_val_K2_percent": best,
                "seed": args.seed,
                "hierarchy_options": options,
            }, args.out / "cdmc.pt")
    (args.out / "train_log.json").write_text(
        json.dumps(history, indent=2) + "\n", encoding="utf-8"
    )


def evaluate(args, device):
    if len(args.data) != 1:
        raise ValueError("evaluation requires exactly one --data path")
    model, scale = load_proposal(args, device)
    checkpoint = torch.load(
        args.checkpoint_dir / "cdmc.pt", map_location=device, weights_only=False
    )
    if checkpoint.get("training_backend") != "matrixfree_direct_terminal":
        raise ValueError("checkpoint was not trained with the direct-terminal kernel")
    if int(checkpoint["coarse_size"]) != args.coarse_size:
        raise ValueError("coarse-size differs from training")
    setup = CDMCSetup(checkpoint["width"]).to(device)
    setup.load_state_dict(checkpoint["state"])
    setup.eval()
    options = hierarchy_options(args, checkpoint)
    gate_mode = checkpoint.get("gate_mode", "full")
    splits = ("train", "val", "test") if args.split == "all" else (args.split,)
    rows = []
    compiled = None
    with torch.no_grad():
        for split_name in splits:
            split = DarcySplit(args.data[0], split_name)
            if compiled is None:
                compiled = MeshCompiledCDMC(
                    split.mesh, device, args.coarse_size, terminal_solver="direct",
                    **options,
                )
            count = min(args.eval_count or len(split), len(split))
            predictions = predict_numpy(model, split, scale, device)
            for index in range(count):
                coefficient = torch.as_tensor(split.coefficient[index], device=device)
                lift = torch.as_tensor(split.lift[index], device=device)
                target = torch.as_tensor(split.target[index], device=device)
                mass = torch.as_tensor(split.mesh.mass, device=device)
                initial = predictions[index, compiled.plan.free]
                started = clock(device)
                system, element_a = compiled.build(coefficient, lift)
                build_ms = 1000 * (clock(device) - started)
                example = (system, element_a, initial, target, lift, mass)
                started = clock(device)
                learned = field_errors(solve(
                    setup, compiled, example,
                    coarse_enabled=gate_mode == "full",
                ), example)
                cdmc_ms = 1000 * (clock(device) - started)
                smooth_only = (
                    learned if gate_mode == "smooth_only" else field_errors(
                        solve(setup, compiled, example, coarse_enabled=False), example
                    )
                )
                started = clock(device)
                fixed = field_errors(solve(setup, compiled, example, False), example)
                mg_ms = 1000 * (clock(device) - started)
                # Same assembled sample system and V-cycle; only the free-node
                # starting field changes. Boundary values remain in `lift`.
                zero_example = (
                    system, element_a, torch.zeros_like(initial), target, lift, mass
                )
                started = clock(device)
                zero_fixed = field_errors(
                    solve(setup, compiled, zero_example, False), zero_example
                )
                mg_zero_ms = 1000 * (clock(device) - started)
                no_error = 100 * float(field_errors([initial], example)[0])
                row = {
                    "split": split_name, "sample_index": int(split.sample_index[index]),
                    "NO_error_percent": no_error,
                    "CDMC_K1_error_percent": 100 * float(learned[0]),
                    "CDMC_K2_error_percent": 100 * float(learned[1]),
                    "smooth_only_K2_error_percent": 100 * float(smooth_only[1]),
                    "MG_K1_error_percent": 100 * float(fixed[0]),
                    "MG_K2_error_percent": 100 * float(fixed[1]),
                    "MG_zero_K1_error_percent": 100 * float(zero_fixed[0]),
                    "MG_zero_K2_error_percent": 100 * float(zero_fixed[1]),
                    "sample_build_ms": build_ms,
                    "CDMC_K2_eager_ms": cdmc_ms,
                    "MG_K2_eager_ms": mg_ms,
                    "MG_zero_K2_eager_ms": mg_zero_ms,
                }
                if args.with_fem:
                    started = time.perf_counter()
                    fem, _ = solve_sample(
                        split.mesh.points, split.mesh.cells, split.mesh.boundary,
                        split.coefficient[index], split.lift[index],
                        compiled.plan.geometry,
                    )
                    row["FEM_direct_ms"] = 1000 * (time.perf_counter() - started)
                    fem_tensor = torch.as_tensor(fem, device=device, dtype=target.dtype)
                    row["FEM_saved_field_error_percent"] = 100 * float(
                        ((mass * (fem_tensor - target).square()).sum()
                         / (mass * target.square()).sum().clamp_min(1e-12)).sqrt()
                    )
                rows.append(row)
                print(json.dumps(row), flush=True)
    keys = ("NO_error_percent", "CDMC_K1_error_percent",
            "CDMC_K2_error_percent", "smooth_only_K2_error_percent",
            "MG_K1_error_percent", "MG_K2_error_percent",
            "MG_zero_K1_error_percent", "MG_zero_K2_error_percent",
            "sample_build_ms", "CDMC_K2_eager_ms", "MG_K2_eager_ms",
            "MG_zero_K2_eager_ms")
    if args.with_fem:
        keys += ("FEM_direct_ms", "FEM_saved_field_error_percent")
    result = {
        "dataset": str(args.data[0]), "checkpoint_dir": str(args.checkpoint_dir),
        "terminal_solver": "sample-specific direct Cholesky; no PCG",
        "gate_mode": gate_mode,
        "levels": compiled.action.level_sizes,
        "mesh_plan_ms_once": compiled.mesh_plan_ms_once,
        "mesh_plan_stage_ms": compiled.mesh_plan_stage_ms,
        "hierarchy_options": options,
        "checkpoint_hierarchy_options": checkpoint.get("hierarchy_options", {
            "aggregate_size": 2, "interpolation_max_entries": 0,
        }),
        "samples": rows,
        "mean": {key: float(np.mean([row[key] for row in rows])) for key in keys},
        "timing_note": "single-pass eager timing; NO and one-time mesh plan excluded",
    }
    (args.out / "direct_result.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("train", "evaluate"))
    parser.add_argument("--data", type=Path, action="append", required=True)
    parser.add_argument("--proposal-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path)
    parser.add_argument("--init-cdmc-dir", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--coarse-size", type=int, default=128)
    parser.add_argument("--aggregate-size", type=int)
    parser.add_argument("--interpolation-max-entries", type=int)
    parser.add_argument("--allow-hierarchy-change", action="store_true",
                        help="explicit frozen-weight transfer to a different mesh plan")
    parser.add_argument("--setup-width", type=int, default=32)
    parser.add_argument("--train-per-scale", type=int, default=16)
    parser.add_argument("--val-per-scale", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--lr", type=float, default=0.0003)
    parser.add_argument("--gate-mode", choices=("full", "smooth_only"), default="full")
    parser.add_argument("--loss-mode", choices=("mean", "K2"), default="mean")
    parser.add_argument("--split", choices=("train", "val", "test", "all"), default="test")
    parser.add_argument("--eval-count", type=int, default=0)
    parser.add_argument("--with-fem", action="store_true",
                        help="time the mature CPU sparse direct FEM on the same samples")
    parser.add_argument("--gpu-fraction", type=float, default=0.4)
    parser.add_argument("--seed", type=int, default=20260929)
    args = parser.parse_args()
    if args.command == "evaluate" and args.checkpoint_dir is None:
        parser.error("evaluate requires --checkpoint-dir")
    if args.aggregate_size is not None and args.aggregate_size < 2:
        parser.error("aggregate-size must be at least 2")
    if args.interpolation_max_entries is not None and args.interpolation_max_entries < 0:
        parser.error("interpolation-max-entries must be nonnegative (0: exact/unlimited)")
    args.out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    device = torch.device("cuda:0")
    torch.cuda.set_per_process_memory_fraction(args.gpu_fraction, device)
    if args.command == "train":
        train(args, device)
    else:
        evaluate(args, device)


if __name__ == "__main__":
    main()
