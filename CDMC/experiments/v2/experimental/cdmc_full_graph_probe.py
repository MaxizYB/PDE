"""Capture frozen CDMC's exact online setup and two V-cycles in one CUDA graph.

This diagnostic retains the original Galerkin mathematics.  It tests whether
GPU launch overhead, rather than the arithmetic itself, dominates its latency.
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
    plan = make_mesh_plan(split.mesh, device, int(checkpoint["coarse_size"]),
                          gpu_galerkin=True, compiled_assembly=True,
                          hybrid_dense_limit=512)
    proposal = FixedMeshProposal(model, split, scale, device,
                                 capture_graph=True, use_prepared=True)
    coefficient_static = torch.as_tensor(split.coefficient[0], dtype=torch.float64,
                                         device=device).clone()
    lift_static = torch.as_tensor(split.lift[0], dtype=torch.float64,
                                  device=device).clone()
    initial_static = proposal.predict(split, 0)[plan.free].clone()

    def solve():
        system = make_system(split.mesh, coefficient_static, lift_static, device,
                             plan, backend="gpu_hybrid")
        smooth, coarse = setup(system, initial_static)
        return correct(system, initial_static, 2, smooth, coarse)

    stream = torch.cuda.Stream(device=device)
    stream.wait_stream(torch.cuda.current_stream(device))
    with torch.no_grad(), torch.cuda.stream(stream):
        for _ in range(3):
            solve()
    torch.cuda.current_stream(device).wait_stream(stream)
    torch.cuda.synchronize(device)
    graph = torch.cuda.CUDAGraph()
    t0 = time.perf_counter()
    with torch.no_grad(), torch.cuda.graph(graph):
        output = solve()
    graph.replay()
    capture_ms = 1000 * (synchronized_clock(device) - t0)
    records = []
    with torch.no_grad():
        for i in range(min(args.count, len(split)) + 1):
            index = 0 if i == 0 else i - 1
            t0 = synchronized_clock(device)
            coefficient_static.copy_(torch.as_tensor(split.coefficient[index],
                                                      dtype=torch.float64, device=device))
            lift_static.copy_(torch.as_tensor(split.lift[index], dtype=torch.float64,
                                              device=device))
            input_ms = 1000 * (synchronized_clock(device) - t0)
            t0 = synchronized_clock(device)
            initial_static.copy_(proposal.predict(split, index)[plan.free])
            no_ms = 1000 * (synchronized_clock(device) - t0)
            t0 = synchronized_clock(device)
            graph.replay()
            graph_ms = 1000 * (synchronized_clock(device) - t0)
            if i:
                reference = solve()
                truth = torch.as_tensor(split.target[index], device=device)
                mass = torch.as_tensor(split.mesh.mass[plan.free], device=device)
                full_mass = torch.as_tensor(split.mesh.mass, device=device)
                denominator = (full_mass * truth.square()).sum().clamp_min(1e-12)
                error = float(((mass * (output - truth[plan.free]).square()
                                ).sum() / denominator).sqrt()) * 100
                records.append({"graph_vs_reference_max_abs": float((output - reference).abs().max()),
                                "error_percent": error, "input_ms": input_ms,
                                "NO_ms": no_ms, "graph_ms": graph_ms,
                                "total_ms": input_ms + no_ms + graph_ms})
    summary = {key: describe([row[key] for row in records]) for key in
               ("graph_vs_reference_max_abs", "error_percent", "input_ms",
                "NO_ms", "graph_ms", "total_ms")}
    summary["capture_ms_once"] = capture_ms
    summary["proposal_compile_ms_once"] = proposal.compile_ms_once
    summary["peak_allocated_mib"] = torch.cuda.max_memory_allocated(device) / 1024**2
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"summary": summary, "samples": records},
                                   indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
