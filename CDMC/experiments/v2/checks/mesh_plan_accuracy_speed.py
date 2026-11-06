"""Compare cold mesh plans and unchanged two-cycle CDMC on frozen checkpoints.

Only reference setup sparsity/coarsening changes. Current-sample Galerkin
actions, exact terminal Cholesky, field targets and network weights are fixed.
Timings are synchronized batch=1 wall time; no accuracy-tolerance stopping.
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from adapters.darcy import DarcySplit
from direct_multiscale import field_errors, solve
from run import load_proposal, predict_numpy
from solver.correctors import CDMCSetup
from solver.matrixfree_cdmc import MeshCompiledCDMC
from solver.matrixfree_cdmc import correct_matrixfree, gates_from_implicit


VARIANTS = {
    "exact_agg2": dict(aggregate_size=2, galerkin_drop_tol=0.0),
    "drop_1e4_agg2": dict(aggregate_size=2, galerkin_drop_tol=1e-4),
    "drop_1e3_agg2": dict(aggregate_size=2, galerkin_drop_tol=1e-3),
    "drop_1e2_agg2": dict(aggregate_size=2, galerkin_drop_tol=1e-2),
    "exact_agg4": dict(aggregate_size=4, galerkin_drop_tol=0.0),
    "exact_agg3": dict(aggregate_size=3, galerkin_drop_tol=0.0),
    "drop_1e5_agg2": dict(aggregate_size=2, galerkin_drop_tol=1e-5),
    "lump_1e4_agg2": dict(aggregate_size=2, galerkin_drop_tol=1e-4,
                         galerkin_preserve_row_sum=True),
    "lump_1e3_agg2": dict(aggregate_size=2, galerkin_drop_tol=1e-3,
                         galerkin_preserve_row_sum=True),
    "local16_agg2": dict(aggregate_size=2, interpolation_max_entries=16),
    "local8_agg2": dict(aggregate_size=2, interpolation_max_entries=8),
    "local32_agg2": dict(aggregate_size=2, interpolation_max_entries=32),
    "local64_agg2": dict(aggregate_size=2, interpolation_max_entries=64),
}


def clock(device):
    torch.cuda.synchronize(device)
    return time.perf_counter()


def timed(call, device, warmup, repeats):
    for _ in range(warmup):
        call()
    values = []
    for _ in range(repeats):
        start = clock(device)
        value = call()
        values.append(1000 * (clock(device) - start))
    return value, float(np.median(values))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--proposal-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--variants", nargs="+", choices=VARIANTS,
                        default=["exact_agg2", "drop_1e4_agg2", "drop_1e3_agg2", "exact_agg4"])
    parser.add_argument("--split", default="test", choices=("test", "val"))
    parser.add_argument("--count", type=int, default=12)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--coarse-size", type=int, default=128)
    parser.add_argument("--cycles", type=int, default=2)
    parser.add_argument("--time-cycles", type=int, nargs="*", default=[])
    parser.add_argument("--smoothing-steps", type=int, default=2)
    parser.add_argument("--gpu-fraction", type=float, default=0.25)
    args = parser.parse_args()
    if (args.count < 1 or args.warmup < 0 or args.repeats < 1
            or args.cycles < 2 or args.smoothing_steps < 1):
        parser.error("count/repeats must be positive and warmup nonnegative")
    if any(k < 1 or k > args.cycles for k in args.time_cycles):
        parser.error("time-cycles must be within [1, cycles]")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    device = torch.device("cuda:0")
    torch.cuda.set_per_process_memory_fraction(args.gpu_fraction, device)
    split = DarcySplit(args.data, args.split)
    model, scale = load_proposal(args, device)
    checkpoint = torch.load(args.checkpoint_dir / "cdmc.pt", map_location=device,
                            weights_only=False)
    setup = CDMCSetup(checkpoint["width"]).to(device)
    setup.load_state_dict(checkpoint["state"])
    setup.eval()
    gate_mode = checkpoint.get("gate_mode", "full")
    result = {
        "data": str(args.data), "split": args.split,
        "proposal_dir": str(args.proposal_dir),
        "checkpoint_dir": str(args.checkpoint_dir), "gate_mode": gate_mode,
        "torch_version": torch.__version__, "gpu": torch.cuda.get_device_name(device),
        "warmup": args.warmup, "repeats": args.repeats,
        "cycles": args.cycles,
        "smoothing_steps": args.smoothing_steps,
        "timing_note": "same process, frozen weights/initial fields, synchronized batch=1; "
            "median repeats per sample, mean samples; NO/input excluded from online correction",
        "variants": [],
    }
    with torch.no_grad():
        predictions = predict_numpy(model, split, scale, device)
        for name in args.variants:
            print(f"building {name}", flush=True)
            torch.cuda.reset_peak_memory_stats(device)
            compiled = MeshCompiledCDMC(
                split.mesh, device, args.coarse_size, terminal_solver="direct",
                **VARIANTS[name],
            )
            print(json.dumps({"variant": name, "plan_ms": compiled.mesh_plan_ms_once,
                              "stages_ms": compiled.mesh_plan_stage_ms}), flush=True)
            samples = []
            for index in range(min(args.count, len(split))):
                coefficient = torch.as_tensor(split.coefficient[index], device=device)
                lift = torch.as_tensor(split.lift[index], device=device)
                target = torch.as_tensor(split.target[index], device=device)
                mass = torch.as_tensor(split.mesh.mass, device=device)
                initial = predictions[index, compiled.plan.free]
                (system, element_a), build_ms = timed(
                    lambda: compiled.build(coefficient, lift), device,
                    args.warmup, args.repeats,
                )
                example = (system, element_a, initial, target, lift, mass)
                def run_correction(learned=True, cycles=None):
                    if learned:
                        smooth, coarse = gates_from_implicit(
                            setup, system, compiled.action, element_a, initial)
                        if gate_mode == "smooth_only":
                            coarse = torch.ones_like(coarse)
                    else:
                        smooth = coarse = torch.ones_like(initial)
                    return correct_matrixfree(
                        system, compiled.action, element_a, initial, smooth, coarse,
                        compiled.reference_inverse,
                        cycles=args.cycles if cycles is None else cycles, coarse_steps=0,
                        smoothing_steps=args.smoothing_steps,
                        return_states=True)
                states, correction_ms = timed(
                    run_correction,
                    device, args.warmup, args.repeats,
                )
                fixed = run_correction(learned=False)
                errors, fixed_errors = field_errors(states, example), field_errors(fixed, example)
                direction = torch.linspace(-0.5, 0.5, compiled.action.level_sizes[-1],
                                           device=device)
                terminal = system.coarse_factor @ system.coarse_factor.T
                action = compiled.action.action(len(system.levels)-1, direction, element_a)
                terminal_error = float((terminal.float() @ direction - action).norm()
                                       / action.norm().clamp_min(1e-12))
                row = {
                    "sample_index": int(split.sample_index[index]),
                    "NO_error_percent": 100 * float(field_errors([initial], example)[0]),
                    "CDMC_K1_error_percent": 100 * float(errors[0]),
                    "CDMC_K2_error_percent": 100 * float(errors[1]),
                    "CDMC_final_error_percent": 100 * float(errors[-1]),
                    "MG_final_error_percent": 100 * float(fixed_errors[-1]),
                    "MG_K2_error_percent": 100 * float(fixed_errors[1]),
                    "sample_build_ms": build_ms, "correction_ms": correction_ms,
                    "online_ms": build_ms + correction_ms,
                    "terminal_action_relative_error": terminal_error,
                }
                row.update({f"CDMC_K{k+1}_error_percent": 100 * float(error)
                            for k, error in enumerate(errors)})
                for k in args.time_cycles:
                    if k == args.cycles:
                        elapsed = correction_ms
                    else:
                        _, elapsed = timed(
                            lambda: run_correction(cycles=k), device,
                            args.warmup, args.repeats)
                    row[f"CDMC_K{k}_correction_ms"] = elapsed
                    row[f"CDMC_K{k}_online_ms"] = elapsed + build_ms
                samples.append(row)
            matrices = compiled.plan.reference_levels
            entry = {
                "name": name, "options": VARIANTS[name],
                "levels": compiled.action.level_sizes,
                "mesh_plan_ms_once": compiled.mesh_plan_ms_once,
                "mesh_plan_stage_ms": compiled.mesh_plan_stage_ms,
                "reference_nnz": [a.nnz for a in matrices],
                "operator_complexity": sum(a.nnz for a in matrices) / matrices[0].nnz,
                "transfer_nnz": [p.nnz for p, _ in compiled.plan.transfers],
                "samples": samples,
                "mean": {key: float(np.mean([s[key] for s in samples]))
                         for key in samples[0] if key != "sample_index"},
                "P95_CDMC_K2_error_percent": float(np.percentile(
                    [s["CDMC_K2_error_percent"] for s in samples], 95)),
                "peak_gpu_allocated_mib": torch.cuda.max_memory_allocated(device)/1024**2,
            }
            result["variants"].append(entry)
            args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
            print(json.dumps({key: value for key, value in entry.items()
                              if key not in ("samples", "reference_nnz", "transfer_nnz")}), flush=True)
            del compiled, system, example, element_a, matrices
            gc.collect()
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
