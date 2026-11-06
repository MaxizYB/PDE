"""Bounded one-sample Darcy scale proxy: CDMC, fixed MG, and sparse direct FEM.

This is a same-discretization comparison, not a claim about real industrial
geometry. The NO/CDMC weights are frozen small-mesh pilot checkpoints.
"""
from __future__ import annotations

import argparse
import json
import resource
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE.parents[4]))

from adapters.darcy import DarcySplit
from adapters.darcy_system import solve_sample
from run import load_proposal, synchronized_clock
from solver.correctors import CDMCSetup
from solver.inference import FixedMeshProposal
from solver.matrixfree_cdmc import (
    MeshCompiledCDMC, correct_matrixfree, gates_from_implicit,
)
from solver.matrixfree_cdmc import _v_cycle
from solver.numerics import coarsen_gates


def describe(values):
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    parser.add_argument("--sample-position", type=int, default=0,
                        help="select one existing sample from the named split")
    parser.add_argument("--proposal-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--gpu-fraction", type=float, default=0.15)
    parser.add_argument("--max-nodes", type=int, default=70000)
    parser.add_argument("--diagnose-gates", action="store_true",
                        help="untimed gate-only controlled comparisons (does not change solver)")
    parser.add_argument("--diagnose-exact-hierarchy", action="store_true",
                        help="CPU sample Galerkin hierarchy for untimed approximation audit")
    args = parser.parse_args()
    if args.repeats < 1 or not 0 < args.gpu_fraction <= 0.25:
        parser.error("require repeats >= 1 and 0 < gpu-fraction <= 0.25")
    if args.diagnose_exact_hierarchy and not args.diagnose_gates:
        parser.error("--diagnose-exact-hierarchy requires --diagnose-gates")
    torch.set_num_threads(4)
    device = torch.device("cuda:0")
    split = DarcySplit(args.data, args.split)
    if not 0 <= args.sample_position < len(split):
        parser.error("sample-position must index an existing sample")
    index = slice(args.sample_position, args.sample_position + 1)
    split.coefficient = split.coefficient[index]
    split.lift = split.lift[index]
    split.target = split.target[index]
    split.sample_index = split.sample_index[index]
    nodes = len(split.mesh.points)
    if nodes > args.max_nodes:
        parser.error(f"{nodes} nodes exceed max-nodes={args.max_nodes}")
    torch.cuda.set_per_process_memory_fraction(args.gpu_fraction, device=device)
    torch.cuda.reset_peak_memory_stats(device)

    # Independent mature CPU sparse-direct FEM: assembly + boundary elimination
    # + solve. Also verifies that the stored target uses the same discretization.
    fem_times = []
    fem_points = np.asarray(split.mesh.points, dtype=np.float64)
    fem_coefficient = np.asarray(split.coefficient[0], dtype=np.float64)
    fem_lift = np.asarray(split.lift[0], dtype=np.float64)
    for _ in range(args.repeats):
        started = time.perf_counter()
        fem, fem_residual = solve_sample(
            fem_points, split.mesh.cells, split.mesh.boundary,
            fem_coefficient, fem_lift,
        )
        fem_times.append(1000 * (time.perf_counter() - started))
    target = split.target[0]
    mass_np = split.mesh.mass.astype(np.float64)
    target_discrepancy = float(np.sqrt(
        np.sum(mass_np * (fem - target) ** 2)
        / max(np.sum(mass_np * target**2), 1e-12)
    )) * 100

    model, scale = load_proposal(args, device)
    checkpoint = torch.load(
        args.checkpoint_dir / "cdmc.pt", map_location=device, weights_only=False,
    )
    setup = CDMCSetup(checkpoint["width"]).to(device)
    setup.load_state_dict(checkpoint["state"])
    setup.eval()
    compiled = MeshCompiledCDMC(split.mesh, device, int(checkpoint["coarse_size"]))
    proposal = FixedMeshProposal(
        model, split, scale, device, capture_graph=True, use_prepared=True,
    )
    coefficient = torch.as_tensor(split.coefficient[0], device=device).clone()
    lift = torch.as_tensor(split.lift[0], device=device).clone()
    initial = proposal.predict(split, 0)[compiled.plan.free].clone()
    truth = torch.as_tensor(target, device=device)
    mass = torch.as_tensor(split.mesh.mass, device=device)
    free = compiled.plan.free
    denominator = (mass * truth.square()).sum().clamp_min(1e-12)

    def error(interior):
        return float(torch.sqrt(
            (mass[free] * (interior - truth[free]).square()).sum() / denominator
        )) * 100

    def residual(interior):
        system, element_a = compiled.build(coefficient, lift)
        remainder = system.rhs - compiled.action.action(0, interior, element_a)
        return float(torch.linalg.vector_norm(remainder)
                     / torch.linalg.vector_norm(system.rhs).clamp_min(1e-12))

    def capture(kind):
        def solve():
            system, element_a = compiled.build(coefficient, lift)
            if kind == "cdmc":
                smooth, coarse = gates_from_implicit(
                    setup, system, compiled.action, element_a, initial,
                )
            else:
                smooth = coarse = torch.ones_like(initial)
            return correct_matrixfree(
                system, compiled.action, element_a, initial, smooth, coarse,
                compiled.reference_inverse, cycles=2, coarse_steps=2,
            )

        stream = torch.cuda.Stream(device=device)
        stream.wait_stream(torch.cuda.current_stream(device))
        with torch.no_grad(), torch.cuda.stream(stream):
            for _ in range(2):
                solve()
        torch.cuda.current_stream(device).wait_stream(stream)
        synchronized_clock(device)
        graph = torch.cuda.CUDAGraph()
        started = time.perf_counter()
        with torch.no_grad(), torch.cuda.graph(graph):
            output = solve()
        graph.replay()
        capture_ms = 1000 * (synchronized_clock(device) - started)
        return graph, output, capture_ms

    def gate_diagnostics():
        system, element_a = compiled.build(coefficient, lift)
        smooth, coarse = gates_from_implicit(
            setup, system, compiled.action, element_a, initial,
        )
        smooth_levels, coarse_levels = coarsen_gates(system, smooth, coarse)
        def stats(x):
            return {
                "mean": float(x.mean()), "std": float(x.std()),
                "min": float(x.min()), "max": float(x.max()),
                "p05": float(torch.quantile(x, 0.05)),
                "p95": float(torch.quantile(x, 0.95)),
            }
        residual_initial = system.rhs - compiled.action.action(0, initial, element_a)
        result = {
            "fine_smooth": stats(smooth), "fine_coarse": stats(coarse),
            "levels_smooth": [stats(x) for x in smooth_levels],
            "levels_coarse": [stats(x) for x in coarse_levels],
            "initial_relative_discrete_residual": float(
                torch.linalg.vector_norm(residual_initial)
                / torch.linalg.vector_norm(system.rhs).clamp_min(1e-12)
            ),
        }
        one = torch.ones_like(smooth)

        def traced_correction(smooth_first, coarse_first, *,
                              neutral_coarse_after=None, refresh_second=False,
                              second_coarse_fixed=False, coarse_steps=2):
            x = initial
            errors, residuals, alphas, energies = [], [], [], []
            fine_x = compiled.action.action(0, x, element_a)
            energies.append(float(0.5 * (x @ fine_x) - system.rhs @ x))
            for step in range(2):
                if step == 1 and refresh_second:
                    smooth_now, coarse_now = gates_from_implicit(
                        setup, system, compiled.action, element_a, x,
                    )
                else:
                    smooth_now, coarse_now = smooth_first, coarse_first
                if step == 1 and second_coarse_fixed:
                    coarse_now = one
                s_levels, c_levels = coarsen_gates(system, smooth_now, coarse_now)
                if neutral_coarse_after is not None:
                    c_levels = [c if depth < neutral_coarse_after else torch.ones_like(c)
                                for depth, c in enumerate(c_levels)]
                r = system.rhs - fine_x
                d = _v_cycle(
                    system, compiled.action, element_a, r, s_levels, c_levels,
                    compiled.reference_inverse, coarse_steps,
                )
                ad = compiled.action.action(0, d, element_a)
                alpha = (r @ d / (d @ ad).clamp_min(1e-12)).clamp(0, 2)
                x = x + alpha * d
                fine_x = compiled.action.action(0, x, element_a)
                errors.append(error(x))
                residuals.append(float(torch.linalg.vector_norm(system.rhs - fine_x)
                                       / torch.linalg.vector_norm(system.rhs).clamp_min(1e-12)))
                alphas.append(float(alpha))
                energies.append(float(0.5 * (x @ fine_x) - system.rhs @ x))
            return {"errors_percent": errors, "relative_residuals": residuals,
                    "energy_alphas": alphas, "energies": energies}

        for label, s, c in (
            ("fixed", one, one),
            ("learned", smooth, coarse),
            ("smooth_only", smooth, one),
            ("coarse_only", one, coarse),
            ("smooth_uniform_coarse", smooth, one * coarse.mean()),
            ("smooth_mean_centered_coarse", smooth, coarse / coarse.mean()),
            ("learned_half_strength", one + 0.5 * (smooth - one),
             one + 0.5 * (coarse - one)),
        ):
            candidate = initial
            errors = []
            for cycle in (1, 2):
                candidate = correct_matrixfree(
                    system, compiled.action, element_a, initial, s, c,
                    compiled.reference_inverse, cycles=cycle, coarse_steps=2,
                )
                errors.append(error(candidate))
            result[label + "_error_percent_K1_K2"] = errors
        for label, s, c, kwargs in (
            ("fixed_trace", one, one, {}),
            ("learned_trace", smooth, coarse, {}),
            ("smooth_only_trace", smooth, one, {}),
            ("coarse_train_depth_only_trace", smooth, coarse,
             {"neutral_coarse_after": 3}),
            ("learned_refresh_second_trace", smooth, coarse,
             {"refresh_second": True}),
            ("learned_fixed_coarse_second_trace", smooth, coarse,
             {"second_coarse_fixed": True}),
        ):
            result[label] = traced_correction(s, c, **kwargs)
        if args.diagnose_exact_hierarchy:
            from baselines.data.darcy_v2.generate import assemble
            from solver.numerics import jacobi_spectral_bound

            started = time.perf_counter()
            full, _ = assemble(fem_points, split.mesh.cells, fem_coefficient,
                               compiled.plan.geometry)
            current = full[free][:, free].tocsr()
            original_levels = [(level.diagonal_inverse, level.omega)
                               for level in system.levels]
            exact_diagonals = []
            sample_omega = []
            for prolongation, restriction in compiled.plan.transfers:
                exact_diagonals.append(torch.as_tensor(
                    current.diagonal().copy(), device=device, dtype=torch.float32
                ))
                sample_omega.append(min(0.67, 1.6 / jacobi_spectral_bound(current)))
                current = (restriction @ current @ prolongation).tocsr()
            terminal = current.toarray()
            exact_diagonals.append(torch.as_tensor(
                np.diag(terminal).copy(), device=device, dtype=torch.float32
            ))
            terminal_inverse = torch.as_tensor(
                np.linalg.inv(terminal), dtype=torch.float64, device=device,
            )
            # This includes a fresh CPU sample-dependent hierarchy: diagnostic only.
            result["exact_hierarchy_compile_ms_once"] = 1000 * (
                synchronized_clock(device) - started
            )
            result["diag_comparison"] = []
            for layer, (old_inverse, _) in zip(system.levels, original_levels):
                level = len(result["diag_comparison"])
                ratio = old_inverse.reciprocal() / exact_diagonals[level]
                result["diag_comparison"].append({
                    "level": level, "size": int(len(ratio)),
                    "mean_approx_over_exact": float(ratio.mean()),
                    "p05": float(torch.quantile(ratio, 0.05)),
                    "p95": float(torch.quantile(ratio, 0.95)),
                    "reference_omega": (float(original_levels[level][1])
                                        if level < len(sample_omega) else None),
                    "sample_omega": (sample_omega[level]
                                     if level < len(sample_omega) else None),
                })
            for label, diag_exact, omega_exact, terminal_exact in (
                ("deployed", False, False, False),
                ("exact_diagonals", True, False, False),
                ("sample_omega", False, True, False),
                ("exact_diag_omega", True, True, False),
                ("exact_diag_omega_terminal", True, True, True),
            ):
                for level, item in enumerate(system.levels):
                    if diag_exact:
                        item.diagonal_inverse = exact_diagonals[level].reciprocal()
                    else:
                        item.diagonal_inverse = original_levels[level][0]
                    if level < len(sample_omega):
                        item.omega = (sample_omega[level] if omega_exact
                                      else original_levels[level][1])
                if terminal_exact:
                    system.coarse_inverse = terminal_inverse
                for name, smooth_now, coarse_now in (
                    ("cdmc", smooth, coarse), ("mg", one, one),
                ):
                    result[label + "_" + name + "_K2"] = traced_correction(
                        smooth_now, coarse_now,
                        coarse_steps=(0 if terminal_exact else 2),
                    )["errors_percent"][1]
            for item, (diagonal_inverse, omega) in zip(system.levels, original_levels):
                item.diagonal_inverse, item.omega = diagonal_inverse, omega
            system.coarse_inverse = None
        return result

    with torch.no_grad():
        gate_stats = gate_diagnostics() if args.diagnose_gates else None
        cdmc_graph, cdmc_output, cdmc_capture_ms = capture("cdmc")
        mg_graph, mg_output, mg_capture_ms = capture("mg")
        input_no_times = []
        cdmc_times = []
        mg_no_times = []
        mg_zero_times = []
        no_prediction = None
        cdmc_error = mg_no_error = mg_zero_error = None
        cdmc_residual = mg_no_residual = mg_zero_residual = None
        for _ in range(args.repeats):
            started = synchronized_clock(device)
            coefficient.copy_(torch.as_tensor(split.coefficient[0], device=device))
            lift.copy_(torch.as_tensor(split.lift[0], device=device))
            no_prediction = proposal.predict(split, 0)
            initial.copy_(no_prediction[free])
            input_no_times.append(1000 * (synchronized_clock(device) - started))
            started = synchronized_clock(device)
            cdmc_graph.replay()
            cdmc_times.append(1000 * (synchronized_clock(device) - started))
            cdmc_error = error(cdmc_output)
            cdmc_residual = residual(cdmc_output)
            started = synchronized_clock(device)
            mg_graph.replay()
            mg_no_times.append(1000 * (synchronized_clock(device) - started))
            mg_no_error = error(mg_output)
            mg_no_residual = residual(mg_output)
            initial.zero_()
            started = synchronized_clock(device)
            mg_graph.replay()
            mg_zero_times.append(1000 * (synchronized_clock(device) - started))
            mg_zero_error = error(mg_output)
            mg_zero_residual = residual(mg_output)
    no_error = float(torch.sqrt(
        (mass * (no_prediction - truth).square()).sum() / denominator
    )) * 100
    result = {
        "dataset": str(args.data),
        "split": args.split,
        "nodes": nodes,
        "triangles": len(split.mesh.cells),
        "sample_index": int(split.sample_index[0]),
        "coarse_size": int(checkpoint["coarse_size"]),
        "cycles": 2,
        "terminal_pcg_steps": 2,
        "gpu_fraction_cap": args.gpu_fraction,
        "fem": {
            "assembly_and_direct_solve_ms": describe(fem_times),
            "relative_discrete_residual": fem_residual,
            "stored_target_difference_percent": target_discrepancy,
        },
        "NO": {"field_error_percent": no_error},
        "CDMC_NO_K2": {
            "field_error_percent": cdmc_error,
            "relative_discrete_residual": cdmc_residual,
            "corrector_graph_ms": describe(cdmc_times),
            "hot_total_ms_mean": float(np.mean(np.asarray(input_no_times)
                                                   + np.asarray(cdmc_times))),
        },
        "MG_NO_K2": {
            "field_error_percent": mg_no_error,
            "relative_discrete_residual": mg_no_residual,
            "corrector_graph_ms": describe(mg_no_times),
            "hot_total_ms_mean": float(np.mean(np.asarray(input_no_times)
                                                   + np.asarray(mg_no_times))),
        },
        "MG_zero_K2": {
            "field_error_percent": mg_zero_error,
            "relative_discrete_residual": mg_zero_residual,
            "corrector_graph_ms": describe(mg_zero_times),
            "hot_total_ms_mean": float(np.mean(mg_zero_times)),
        },
        "shared_input_and_NO_ms": describe(input_no_times),
        "one_time": {
            "mesh_plan_ms": compiled.mesh_plan_ms_once,
            "NO_compile_ms": proposal.compile_ms_once,
            "CDMC_graph_capture_ms": cdmc_capture_ms,
            "MG_graph_capture_ms": mg_capture_ms,
        },
        "gate_diagnostics": gate_stats,
        "peak_pytorch_gpu_allocated_mib": (
            torch.cuda.max_memory_allocated(device) / 1024**2
        ),
        "peak_process_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
        "note": "Synthetic regular Darcy scale proxy. FEM is CPU SciPy sparse direct "
                "on the same P1 system; GPU methods use frozen 33x33 pilot weights. "
                "MG_NO uses the same matrix-free hierarchy without learned gates. "
                "Cold plan/model/graph costs are excluded from hot_total.",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
