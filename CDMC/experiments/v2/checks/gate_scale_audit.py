"""Read-only gate scale diagnostic on frozen NO/CDMC checkpoints.

Evaluates one fixed mesh's existing test split under the same numerical
backend. No retraining and no production solver changes are made.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from adapters.darcy import DarcySplit
from run import load_proposal
from solver.correctors import CDMCSetup
from solver.inference import FixedMeshProposal
from solver.matrixfree_cdmc import MeshCompiledCDMC, correct_matrixfree, gates_from_implicit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--proposal-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--count", type=int, default=16)
    parser.add_argument("--gpu-fraction", type=float, default=0.15)
    args = parser.parse_args()
    if not 1 <= args.count <= 100 or not 0 < args.gpu_fraction <= 0.25:
        parser.error("count must be 1..100 and GPU fraction in (0, .25]")
    torch.set_num_threads(4)
    device = torch.device("cuda:0")
    torch.cuda.set_per_process_memory_fraction(args.gpu_fraction, device=device)
    split = DarcySplit(args.data, "test")
    model, scale = load_proposal(args, device)
    checkpoint = torch.load(args.checkpoint_dir / "cdmc.pt", map_location=device,
                            weights_only=False)
    setup = CDMCSetup(checkpoint["width"]).to(device)
    setup.load_state_dict(checkpoint["state"])
    setup.eval()
    compiled = MeshCompiledCDMC(split.mesh, device, int(checkpoint["coarse_size"]))
    proposal = FixedMeshProposal(model, split, scale, device,
                                 capture_graph=True, use_prepared=True)
    mass = torch.as_tensor(split.mesh.mass, device=device)
    rows = []
    with torch.no_grad():
        for i in range(min(len(split), args.count)):
            coefficient = torch.as_tensor(split.coefficient[i], device=device)
            lift = torch.as_tensor(split.lift[i], device=device)
            truth = torch.as_tensor(split.target[i], device=device)
            denominator = (mass * truth.square()).sum().clamp_min(1e-12)
            system, element_a = compiled.build(coefficient, lift)
            initial = proposal.predict(split, i)[compiled.plan.free]
            smooth, coarse = gates_from_implicit(
                setup, system, compiled.action, element_a, initial,
            )
            one = torch.ones_like(smooth)
            row = {"sample_index": int(split.sample_index[i]),
                   "smooth_mean": float(smooth.mean()),
                   "coarse_mean": float(coarse.mean())}
            for label, s, c in (
                ("fixed", one, one),
                ("learned", smooth, coarse),
                ("smooth_only", smooth, one),
                ("coarse_only", one, coarse),
                ("learned_half_strength", one + .5 * (smooth - one),
                 one + .5 * (coarse - one)),
            ):
                for cycles in (1, 2):
                    candidate = correct_matrixfree(
                        system, compiled.action, element_a, initial, s, c,
                        compiled.reference_inverse, cycles=cycles, coarse_steps=2,
                    )
                    row[f"{label}_K{cycles}_error_percent"] = float(torch.sqrt(
                        (mass[system.free] * (candidate - truth[system.free]).square()
                         ).sum() / denominator,
                    )) * 100
            rows.append(row)
    summary = {key: {"mean": float(np.mean([row[key] for row in rows])),
                     "p95": float(np.percentile([row[key] for row in rows], 95))}
               for key in rows[0] if key != "sample_index"}
    result = {"dataset": str(args.data), "nodes": len(split.mesh.points),
              "levels": compiled.action.level_sizes, "count": len(rows),
              "summary": summary, "samples": rows,
              "note": "Frozen weights and same compiled K=1/K=2 kernel; "
                      "only smooth/coarse gate values are varied."}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"nodes": result["nodes"], "levels": result["levels"],
                      "count": result["count"], "summary": summary}, indent=2))


if __name__ == "__main__":
    main()
