#!/usr/bin/env python3
"""Focused same-data audit of hierarchy, smoothing, and coarse consistency."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import replace
from pathlib import Path

os.environ["OPENBLAS_NUM_THREADS"] = "1"

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from adapters.darcy import DarcySplit, relative_field_error
from adapters.darcy_system import make_mesh_plan, make_system
from solver.correctors import CDMCSetup
from solver.inference import FixedMeshProposal
from solver.model import make_model
from solver.numerics import coarsen_gates, correct, multiply, v_cycle


def clock(device: torch.device) -> float:
    torch.cuda.synchronize(device)
    return time.perf_counter()


def summarized(values: list[float]) -> dict[str, float]:
    return {"mean": float(np.mean(values)), "p95": float(np.percentile(values, 95))}


def freeze_coarse(system, constant_system, mean_coefficient: float):
    """Diagnostic only: replace sample Galerkin coarse levels by scaled constant-medium levels."""
    scale = mean_coefficient / 7.5
    coarse = [replace(
        item, matrix=item.matrix * scale,
        diagonal_inverse=item.diagonal_inverse / scale,
    ) for item in constant_system.levels[1:]]
    return replace(
        system, levels=[system.levels[0], *coarse],
        coarse_inverse=constant_system.coarse_inverse / scale,
    )


def correct_without_energy_step(system, initial, smooth, coarse):
    """Diagnostic: use unit update instead of the trained energy-optimal step."""
    smooth_levels, coarse_levels = coarsen_gates(system, smooth, coarse)
    x = initial
    for _ in range(2):
        residual = system.rhs - multiply(system.matrix, x)
        x = x + v_cycle(system, residual, smooth_levels, coarse_levels)
    return x


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--proposal-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--count", type=int, default=16)
    parser.add_argument("--coarse-size", type=int, default=128)
    args = parser.parse_args()
    device = torch.device("cuda")
    torch.set_num_threads(4)
    test = DarcySplit(args.data, "test")
    proposal_checkpoint = torch.load(
        args.proposal_dir / "proposal.pt", map_location=device, weights_only=False
    )
    model = make_model(
        proposal_checkpoint["model"], input_dim=5,
        **proposal_checkpoint["model_kwargs"],
    ).to(device)
    model.load_state_dict(proposal_checkpoint["state"])
    model.eval()
    corrector_checkpoint = torch.load(
        args.checkpoint_dir / "cdmc.pt", map_location=device, weights_only=False
    )
    if corrector_checkpoint.get("coarse_size") != args.coarse_size:
        raise ValueError("CDMC checkpoint coarse size does not match")
    setup = CDMCSetup(corrector_checkpoint["width"]).to(device)
    setup.load_state_dict(corrector_checkpoint["state"])
    setup.eval()
    proposal = FixedMeshProposal(
        model, test, float(proposal_checkpoint["scale"]), device,
        capture_graph=True,
    )
    mass = torch.as_tensor(test.mesh.mass, device=device)
    results = {}
    with torch.no_grad():
        for aggregate_size in (2, 4, 8):
            t0 = clock(device)
            plan = make_mesh_plan(
                test.mesh, device, args.coarse_size,
                gpu_galerkin=True, compiled_assembly=True,
                aggregate_size=aggregate_size,
            )
            compile_ms = 1000 * (clock(device) - t0)
            reference = None
            if aggregate_size == 2:
                reference = make_system(
                    test.mesh, np.full(len(test.mesh.points), 7.5, dtype=np.float32),
                    np.zeros(len(test.mesh.points), dtype=np.float32), device,
                    plan, backend="gpu_compiled",
                )
            warm = make_system(
                test.mesh, test.coefficient[0], test.lift[0], device,
                plan, backend="gpu_compiled",
            )
            warm_initial = proposal.predict(test, 0)[warm.free]
            warm_smooth, warm_coarse = setup(warm, warm_initial)
            correct(warm, warm_initial, 2, warm_smooth, warm_coarse)
            clock(device)
            bucket = {key: {"error_percent": [], "total_ms": []} for key in (
                "cdmc_nu2", "cdmc_nu1", "mg_nu2",
                *(("frozen_coarse",) if reference is not None else ()),
                *(("smooth_gate_only", "coarse_gate_only", "unit_energy_step")
                  if reference is not None else ()),
            )}
            hierarchy = [item.matrix.shape[0] for item in warm.levels]
            for i in range(min(args.count, len(test))):
                t0 = clock(device)
                system = make_system(
                    test.mesh, test.coefficient[i], test.lift[i], device,
                    plan, backend="gpu_compiled",
                )
                system_ms = 1000 * (clock(device) - t0)
                t0 = clock(device)
                initial_full = proposal.predict(test, i)
                proposal_ms = 1000 * (clock(device) - t0)
                initial = initial_full[system.free]
                t0 = clock(device)
                smooth, coarse = setup(system, initial)
                setup_ms = 1000 * (clock(device) - t0)
                target = torch.as_tensor(test.target[i], device=device)
                for key, nu, with_setup in (
                    ("cdmc_nu2", 2, True),
                    ("cdmc_nu1", 1, True),
                    ("mg_nu2", 2, False),
                ):
                    t0 = clock(device)
                    interior = correct(
                        system, initial, 2,
                        smooth if with_setup else None,
                        coarse if with_setup else None,
                        smoothing_steps=nu,
                    )
                    apply_ms = 1000 * (clock(device) - t0)
                    full = torch.as_tensor(test.lift[i], device=device).clone()
                    full[system.free] = interior
                    bucket[key]["error_percent"].append(
                        100 * float(relative_field_error(
                            full[None], target[None], mass[None]
                        ))
                    )
                    bucket[key]["total_ms"].append(
                        system_ms + proposal_ms + apply_ms
                        + (setup_ms if with_setup else 0)
                    )
                if reference is not None:
                    frozen = freeze_coarse(
                        system, reference, float(test.coefficient[i].mean())
                    )
                    interior = correct(frozen, initial, 2, smooth, coarse)
                    full = torch.as_tensor(test.lift[i], device=device).clone()
                    full[frozen.free] = interior
                    bucket["frozen_coarse"]["error_percent"].append(
                        100 * float(relative_field_error(
                            full[None], target[None], mass[None]
                        ))
                    )
                    ones = torch.ones_like(smooth)
                    for key, action in (
                        ("smooth_gate_only", lambda: correct(
                            system, initial, 2, smooth, ones)),
                        ("coarse_gate_only", lambda: correct(
                            system, initial, 2, ones, coarse)),
                        ("unit_energy_step", lambda: correct_without_energy_step(
                            system, initial, smooth, coarse)),
                    ):
                        t0 = clock(device)
                        interior = action()
                        apply_ms = 1000 * (clock(device) - t0)
                        full = torch.as_tensor(test.lift[i], device=device).clone()
                        full[system.free] = interior
                        bucket[key]["error_percent"].append(
                            100 * float(relative_field_error(
                                full[None], target[None], mass[None]
                            ))
                        )
                        bucket[key]["total_ms"].append(
                            system_ms + proposal_ms + setup_ms + apply_ms
                        )
            results[f"aggregate_{aggregate_size}"] = {
                "hierarchy": hierarchy, "plan_compile_ms_once": compile_ms,
                "variants": {
                    key: {metric: summarized(values)
                          for metric, values in metrics.items() if values}
                    for key, metrics in bucket.items()
                },
            }
            print(f"aggregate={aggregate_size} hierarchy={hierarchy} "
                  f"CDMC_K2={results[f'aggregate_{aggregate_size}']['variants']['cdmc_nu2']}",
                  flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
