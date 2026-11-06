"""Isolated old v2 CDMC timing, without other methods interleaved."""
from __future__ import annotations

import argparse
import json
import sys
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
    started = synchronized_clock(device)
    plan = make_mesh_plan(split.mesh, device, int(checkpoint["coarse_size"]),
                          gpu_galerkin=True, compiled_assembly=True,
                          hybrid_dense_limit=512)
    plan_ms = 1000 * (synchronized_clock(device) - started)
    proposal = FixedMeshProposal(model, split, scale, device,
                                 capture_graph=True, use_prepared=True)
    with torch.no_grad():
        for _ in range(3):
            system = make_system(split.mesh, split.coefficient[0],
                                 split.lift[0], device, plan,
                                 backend="gpu_hybrid")
            initial = proposal.predict(split, 0)[system.free]
            smooth, coarse = setup(system, initial)
            correct(system, initial, 2, smooth, coarse)
        synchronized_clock(device)
        records = []
        for i in range(min(args.count, len(split))):
            started = synchronized_clock(device)
            system = make_system(split.mesh, split.coefficient[i],
                                 split.lift[i], device, plan,
                                 backend="gpu_hybrid")
            system_ms = 1000 * (synchronized_clock(device) - started)
            started = synchronized_clock(device)
            initial = proposal.predict(split, i)[system.free]
            no_ms = 1000 * (synchronized_clock(device) - started)
            started = synchronized_clock(device)
            smooth, coarse = setup(system, initial)
            gate_ms = 1000 * (synchronized_clock(device) - started)
            started = synchronized_clock(device)
            output = correct(system, initial, 2, smooth, coarse)
            correct_ms = 1000 * (synchronized_clock(device) - started)
            truth = torch.as_tensor(split.target[i], device=device)
            mass = torch.as_tensor(split.mesh.mass, device=device)
            error = float(((mass[system.free] * (output - truth[system.free]).square()
                            ).sum() / (mass * truth.square()).sum().clamp_min(1e-12)
                            ).sqrt()) * 100
            records.append({"error_percent": error, "system_ms": system_ms,
                            "NO_ms": no_ms, "gate_ms": gate_ms,
                            "correct_ms": correct_ms,
                            "total_ms": system_ms + no_ms + gate_ms + correct_ms})
    summary = {key: describe([row[key] for row in records]) for key in records[0]}
    summary["mesh_plan_ms_once"] = plan_ms
    summary["NO_compile_ms_once"] = proposal.compile_ms_once
    summary["peak_allocated_mib"] = torch.cuda.max_memory_allocated(device) / 1024**2
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"summary": summary, "samples": records},
                                   indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
