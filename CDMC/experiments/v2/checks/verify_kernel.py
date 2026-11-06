#!/usr/bin/env python3
"""Check trained CDMC directions and two-cycle energy descent on test data."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from run import load_proposal  # sets the one-thread BLAS policy before NumPy loads
import torch
from adapters.darcy import DarcySplit, infer_solution
from adapters.darcy_system import make_mesh_plan, make_system
from solver.correctors import CDMCSetup
from solver.numerics import coarsen_gates, correct, multiply, v_cycle


def energy(system, x):
    return float((0.5 * (x @ multiply(system.matrix, x)) - system.rhs @ x).item())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--proposal-dir", type=Path, required=True)
    parser.add_argument("--setup-dir", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, scale = load_proposal(SimpleNamespace(
        proposal_dir=args.proposal_dir, checkpoint_dir=None, out=args.setup_dir,
    ), device)
    checkpoint = torch.load(args.setup_dir / "cdmc.pt", map_location=device, weights_only=False)
    setup = CDMCSetup(checkpoint["width"]).to(device)
    setup.load_state_dict(checkpoint["state"])
    setup.eval()
    data = DarcySplit(args.data, "test")
    plan = make_mesh_plan(data.mesh, device, checkpoint.get("coarse_size", 28))
    directions, nonpositive, rises, disagreement = 0, 0, 0, 0
    min_normalized_direction = float("inf")
    min_alpha = float("inf")
    with torch.no_grad():
        for i in range(len(data)):
            system = make_system(data.mesh, data.coefficient[i], data.lift[i], device, plan)
            proposal = infer_solution(model, data, [i], device, scale)[0]
            initial = proposal[system.free]
            smooth, coarse = setup(system, initial)
            smooth_levels, coarse_levels = coarsen_gates(system, smooth, coarse)
            x = initial
            for _ in range(2):
                before = energy(system, x)
                r = system.rhs - multiply(system.matrix, x)
                d = v_cycle(system, r, smooth_levels, coarse_levels)
                rd = float((r @ d).item())
                ratio = rd / max(float(torch.linalg.vector_norm(r) * torch.linalg.vector_norm(d)), 1e-30)
                min_normalized_direction = min(min_normalized_direction, ratio)
                nonpositive += rd <= 0
                alpha = (r @ d / (d @ multiply(system.matrix, d)).clamp_min(1e-12)).clamp(0, 2)
                min_alpha = min(min_alpha, float(alpha.item()))
                x = x + alpha * d
                rises += energy(system, x) > before + 1e-6 * max(1, abs(before))
                directions += 1
            disagreement += not torch.allclose(x, correct(system, initial, 2, smooth, coarse), atol=1e-6)
    report = {
        "samples": len(data), "directions": directions,
        "nonpositive_directions": nonpositive, "energy_rises": rises,
        "trace_disagreements": disagreement,
        "min_normalized_r_dot_d": min_normalized_direction,
        "min_alpha": min_alpha,
    }
    print(json.dumps(report, indent=2))
    if nonpositive or rises or disagreement:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
