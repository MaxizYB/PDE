"""Check the sample-specific direct terminal solve in the matrix-free V-cycle."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from adapters.darcy import DarcySplit
from run import load_proposal, predict_numpy
from solver.correctors import CDMCSetup
from solver.matrixfree_cdmc import (
    MeshCompiledCDMC, correct_matrixfree, gates_from_implicit,
)


def clock(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return time.perf_counter()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    parser.add_argument("--proposal-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--gpu-fraction", type=float, default=0.25)
    args = parser.parse_args()
    torch.set_num_threads(4)
    device = torch.device("cuda:0")
    torch.cuda.set_per_process_memory_fraction(args.gpu_fraction, device)
    split = DarcySplit(args.data, args.split)
    model, scale = load_proposal(args, device)
    checkpoint = torch.load(
        args.checkpoint_dir / "cdmc.pt", map_location=device, weights_only=False
    )
    setup = CDMCSetup(checkpoint["width"]).to(device)
    setup.load_state_dict(checkpoint["state"])
    setup.eval()
    compiled = MeshCompiledCDMC(
        split.mesh, device, int(checkpoint["coarse_size"]), terminal_solver="direct"
    )
    coefficient = torch.as_tensor(split.coefficient[0], device=device)
    lift = torch.as_tensor(split.lift[0], device=device)
    target = torch.as_tensor(split.target[0], device=device)
    mass = torch.as_tensor(split.mesh.mass, device=device)
    with torch.no_grad():
        initial = predict_numpy(model, split, scale, device, batch_size=1)[
            0, compiled.plan.free
        ]
        start = clock(device)
        system, element_a = compiled.build(coefficient, lift)
        build_ms = 1000 * (clock(device) - start)
        width = compiled.action.level_sizes[-1]
        direction = torch.linspace(-0.5, 0.5, width, device=device)
        terminal = system.coarse_factor @ system.coarse_factor.T
        action_result = compiled.action.action(len(system.levels) - 1, direction, element_a)
        terminal_action_error = float(
            ((terminal.float() @ direction - action_result).norm()
             / action_result.norm().clamp_min(1e-12)).item()
        )
        smooth, coarse = gates_from_implicit(
            setup, system, compiled.action, element_a, initial
        )
        start = clock(device)
        cdmc = correct_matrixfree(
            system, compiled.action, element_a, initial, smooth, coarse,
            compiled.reference_inverse, cycles=2, coarse_steps=0,
        )
        cdmc_ms = 1000 * (clock(device) - start)
        ones = torch.ones_like(initial)
        start = clock(device)
        mg = correct_matrixfree(
            system, compiled.action, element_a, initial, ones, ones,
            compiled.reference_inverse, cycles=2, coarse_steps=0,
        )
        mg_ms = 1000 * (clock(device) - start)
        denominator = (mass * target.square()).sum().clamp_min(1e-12)

        def field_error(value):
            full = lift.clone()
            full[system.free] = value
            return 100 * float(((mass * (full - target).square()).sum() / denominator).sqrt())

        result = {
            "data": str(args.data), "split": args.split,
            "nodes": len(split.mesh.points), "levels": compiled.action.level_sizes,
            "terminal_solver": "sample-specific direct Cholesky",
            "terminal_action_relative_error": terminal_action_error,
            "NO_error_percent": field_error(initial),
            "CDMC_K2_error_percent": field_error(cdmc),
            "MG_K2_error_percent": field_error(mg),
            "mesh_plan_ms_once": compiled.mesh_plan_ms_once,
            "sample_build_ms": build_ms,
            "CDMC_K2_eager_ms": cdmc_ms,
            "MG_K2_eager_ms": mg_ms,
            "peak_gpu_allocated_mib": torch.cuda.max_memory_allocated(device) / 1024**2,
        }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
