"""Diagnostic fixed-step Jacobi PCG using the same matrix-free P1 action."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from adapters.darcy import DarcySplit
from experimental.matrixfree_two_step import MatrixFreeDarcy, rel_error
from run import describe, load_proposal, synchronized_clock
from solver.inference import FixedMeshProposal


def pcg(operator, x0, lift, element_a, diagonal, steps):
    x = x0.clone()
    residual = operator.residual(x, lift, element_a)
    preconditioned = residual / diagonal
    direction = preconditioned.clone()
    inner = (residual * preconditioned).sum(dim=1, keepdim=True)
    for _ in range(steps):
        ad = operator.homogeneous_action(direction, element_a)
        alpha = inner / (direction * ad).sum(dim=1, keepdim=True).clamp_min(1e-20)
        x = x + alpha * direction
        residual = residual - alpha * ad
        preconditioned = residual / diagonal
        next_inner = (residual * preconditioned).sum(dim=1, keepdim=True)
        direction = preconditioned + next_inner / inner.clamp_min(1e-20) * direction
        inner = next_inner
    return x


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--proposal-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--steps", type=int, nargs="+", default=[4, 8, 16, 32, 64])
    parser.add_argument("--count", type=int, default=16)
    args = parser.parse_args()
    args.checkpoint_dir = None
    torch.set_num_threads(4)
    device = torch.device("cuda")
    split = DarcySplit(args.data, "test")
    model, scale = load_proposal(args, device)
    operator = MatrixFreeDarcy(split.mesh, device)
    proposal = FixedMeshProposal(model, split, scale, device,
                                 capture_graph=True, use_prepared=True)
    measures = {str(k): [] for k in args.steps}
    with torch.no_grad():
        for i in range(min(args.count, len(split)) + 1):
            index = 0 if i == 0 else i - 1
            t0 = synchronized_clock(device)
            coefficient = torch.as_tensor(split.coefficient[index:index + 1], device=device)
            lift = torch.as_tensor(split.lift[index:index + 1], device=device)
            element_a, diagonal = operator.prepare(coefficient)
            prepare_ms = 1000 * (synchronized_clock(device) - t0)
            t0 = synchronized_clock(device)
            initial_full = proposal.predict(split, index)[None]
            no_ms = 1000 * (synchronized_clock(device) - t0)
            initial = initial_full[:, operator.free]
            truth = torch.as_tensor(split.target[index:index + 1], device=device)
            for steps in args.steps:
                t0 = synchronized_clock(device)
                result = pcg(operator, initial, lift, element_a, diagonal, steps)
                update_ms = 1000 * (synchronized_clock(device) - t0)
                if i:
                    error = float(rel_error(result, truth, truth, operator)[0]) * 100
                    measures[str(steps)].append({
                        "error_percent": error, "prepare_ms": prepare_ms,
                        "NO_ms": no_ms, "update_ms": update_ms,
                        "total_ms": prepare_ms + no_ms + update_ms,
                    })
    summary = {key: {
        "error_percent": describe([row["error_percent"] for row in values]),
        "update_ms": describe([row["update_ms"] for row in values]),
        "total_ms": describe([row["total_ms"] for row in values]),
    } for key, values in measures.items()}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"summary": summary, "samples": measures}, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
