"""Full captured CDMC preview: partial setup, learned gates, two V-cycles.

The terminal solve uses two PCG steps with a fixed reference inverse and an
exact sample-specific terminal Galerkin action; no sample inverse is formed.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from adapters.darcy import DarcySplit
from adapters.darcy_system import make_mesh_plan, make_system
from experimental.matrixfree_cdmc_partial_probe import PartialGalerkin, gates_from_implicit
from experimental.matrixfree_cdmc_vcycle_probe import implicit_correct
from run import describe, load_proposal, synchronized_clock
from solver.correctors import CDMCSetup
from solver.inference import FixedMeshProposal
from solver.numerics import correct


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--proposal-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--count", type=int, default=16)
    parser.add_argument("--coarse-steps", type=int, default=2)
    parser.add_argument("--corrector", choices=("cdmc", "mg"), default="cdmc")
    parser.add_argument("--sparse-plan", action="store_true",
                        help="compile per-element Galerkin maps from sparse basis support")
    parser.add_argument("--approx-diagonal", action="store_true",
                        help="use reference-scaled intermediate diagonals; exact terminal A")
    parser.add_argument("--approx-terminal", action="store_true",
                        help="reference terminal A with current diagonal congruence scaling")
    parser.add_argument("--implicit-terminal", action="store_true",
                        help="exact current terminal action via the sparse transfer chain")
    parser.add_argument("--skip-reference", action="store_true",
                        help="measure standalone inference memory without reference solver")
    parser.add_argument("--time-reference", action="store_true",
                        help="also time the old CDMC path on the same samples and GPU")
    args = parser.parse_args()
    if args.coarse_steps < 1:
        raise ValueError("full graph requires coarse_steps >= 1 (no online inverse)")
    if args.approx_diagonal and not args.sparse_plan:
        raise ValueError("--approx-diagonal requires --sparse-plan")
    if args.approx_terminal and not args.approx_diagonal:
        raise ValueError("--approx-terminal requires --approx-diagonal")
    if args.implicit_terminal and not args.approx_diagonal:
        raise ValueError("--implicit-terminal requires --approx-diagonal")
    if args.implicit_terminal and args.approx_terminal:
        raise ValueError("select one terminal mode")
    torch.set_num_threads(4)
    device = torch.device("cuda")
    split = DarcySplit(args.data, "test")
    model, scale = load_proposal(args, device)
    checkpoint = torch.load(args.checkpoint_dir / "cdmc.pt", map_location=device,
                            weights_only=False)
    setup = CDMCSetup(checkpoint["width"]).to(device)
    setup.load_state_dict(checkpoint["state"])
    setup.eval()
    partial = PartialGalerkin(split.mesh, device, int(checkpoint["coarse_size"]),
                              sparse_plan=args.sparse_plan,
                              approx_diagonal=args.approx_diagonal,
                              approx_terminal=args.approx_terminal,
                              implicit_terminal=args.implicit_terminal)
    gold_plan = (None if args.skip_reference else
                 make_mesh_plan(split.mesh, device, int(checkpoint["coarse_size"]),
                                gpu_galerkin=True, compiled_assembly=True,
                                hybrid_dense_limit=512))
    proposal = FixedMeshProposal(model, split, scale, device,
                                 capture_graph=True, use_prepared=True)
    coefficient_static = torch.as_tensor(split.coefficient[0], device=device).clone()
    lift_static = torch.as_tensor(split.lift[0], device=device).clone()
    initial_static = proposal.predict(split, 0)[partial.plan.free].clone()

    def solve():
        system, element_a, action, _ = partial.build(
            coefficient_static, lift_static, args.coarse_steps, timed=False
        )
        if args.corrector == "cdmc":
            smooth, coarse = gates_from_implicit(
                setup, system, action, element_a, initial_static
            )
        else:
            smooth = coarse = torch.ones_like(initial_static)
        return implicit_correct(
            system, action, element_a, initial_static, smooth, coarse,
            reference_inverse=partial.reference_inverse,
            coarse_steps=args.coarse_steps,
        )

    stream = torch.cuda.Stream(device=device)
    stream.wait_stream(torch.cuda.current_stream(device))
    with torch.no_grad(), torch.cuda.stream(stream):
        for _ in range(3):
            solve()
    torch.cuda.current_stream(device).wait_stream(stream)
    torch.cuda.synchronize(device)
    graph = torch.cuda.CUDAGraph()
    started = time.perf_counter()
    with torch.no_grad(), torch.cuda.graph(graph):
        output = solve()
    graph.replay()
    capture_ms = 1000 * (synchronized_clock(device) - started)
    if gold_plan is not None and args.time_reference:
        with torch.no_grad():
            for _ in range(3):
                gold_warm = make_system(split.mesh, split.coefficient[0],
                                        split.lift[0], device, gold_plan,
                                        backend="gpu_hybrid")
                if args.corrector == "cdmc":
                    warm_smooth, warm_coarse = setup(gold_warm, initial_static)
                else:
                    warm_smooth = warm_coarse = torch.ones_like(initial_static)
                correct(gold_warm, initial_static, 2, warm_smooth, warm_coarse)
            torch.cuda.synchronize(device)
    records = []
    with torch.no_grad():
        for i in range(min(args.count, len(split)) + 1):
            index = 0 if i == 0 else i - 1
            started = synchronized_clock(device)
            coefficient_static.copy_(torch.as_tensor(split.coefficient[index], device=device))
            lift_static.copy_(torch.as_tensor(split.lift[index], device=device))
            input_ms = 1000 * (synchronized_clock(device) - started)
            started = synchronized_clock(device)
            initial_static.copy_(proposal.predict(split, index)[partial.plan.free])
            no_ms = 1000 * (synchronized_clock(device) - started)
            started = synchronized_clock(device)
            graph.replay()
            graph_ms = 1000 * (synchronized_clock(device) - started)
            if i:
                truth = torch.as_tensor(split.target[index], device=device)
                mass = torch.as_tensor(split.mesh.mass, device=device)
                denominator = (mass * truth.square()).sum().clamp_min(1e-12)
                free = partial.plan.free
                error = float(((mass[free] * (output - truth[free]).square()
                                ).sum() / denominator).sqrt()) * 100
                record = {
                    "error_percent": error,
                    "input_ms": input_ms, "NO_ms": no_ms, "graph_ms": graph_ms,
                    "total_ms": input_ms + no_ms + graph_ms,
                }
                if gold_plan is not None:
                    gold_started = synchronized_clock(device) if args.time_reference else None
                    gold = make_system(split.mesh, split.coefficient[index],
                                       split.lift[index], device, gold_plan,
                                       backend="gpu_hybrid")
                    gold_system_ms = (1000 * (synchronized_clock(device) - gold_started)
                                      if args.time_reference else None)
                    gold_started = synchronized_clock(device) if args.time_reference else None
                    if args.corrector == "cdmc":
                        gold_smooth, gold_coarse = setup(gold, initial_static)
                    else:
                        gold_smooth = gold_coarse = torch.ones_like(initial_static)
                    gold_gate_ms = (1000 * (synchronized_clock(device) - gold_started)
                                    if args.time_reference else None)
                    gold_started = synchronized_clock(device) if args.time_reference else None
                    gold_output = correct(gold, initial_static, 2,
                                          gold_smooth, gold_coarse)
                    if args.time_reference:
                        gold_correct_ms = 1000 * (synchronized_clock(device) - gold_started)
                        record.update({
                            "gold_system_ms": gold_system_ms,
                            "gold_gate_ms": gold_gate_ms,
                            "gold_correct_ms": gold_correct_ms,
                            "gold_total_ms": no_ms + gold_system_ms + gold_gate_ms
                            + gold_correct_ms,
                        })
                    record["gold_error_percent"] = float(((mass[free] *
                        (gold_output - truth[free]).square()).sum() /
                        denominator).sqrt()) * 100
                    record["max_solution_difference"] = float(
                        (output - gold_output).abs().max()
                    )
                records.append(record)
    summary = {key: describe([row[key] for row in records]) for key in records[0]}
    summary["mesh_plan_ms_once"] = partial.mesh_plan_ms_once
    summary["map_compile_ms_once"] = partial.map_compile_ms_once
    summary["graph_capture_ms_once"] = capture_ms
    summary["NO_compile_ms_once"] = proposal.compile_ms_once
    summary["peak_allocated_mib"] = torch.cuda.max_memory_allocated(device) / 1024**2
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"coarse_steps": args.coarse_steps,
                                    "sparse_plan": args.sparse_plan,
                                    "approx_diagonal": args.approx_diagonal,
                                    "approx_terminal": args.approx_terminal,
                                    "implicit_terminal": args.implicit_terminal,
                                    "corrector": args.corrector,
                                    "summary": summary, "samples": records},
                                   indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
