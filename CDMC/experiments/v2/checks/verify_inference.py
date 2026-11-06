#!/usr/bin/env python3
"""Compare reference, geometry-prepared, and CUDA-Graph NO predictions."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from run import load_proposal  # sets BLAS policy before NumPy loads
import torch

from adapters.darcy import DarcySplit, infer_solution
from solver.inference import FixedMeshProposal


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--proposal-dir", type=Path, required=True)
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    split = DarcySplit(args.data, "test")
    model, scale = load_proposal(SimpleNamespace(
        proposal_dir=args.proposal_dir, checkpoint_dir=None, out=None,
    ), device)
    runner = FixedMeshProposal(model, split, scale, device)
    resident = runner.stage(split)
    max_host, max_resident = 0.0, 0.0
    with torch.no_grad():
        for i in range(len(split)):
            reference = infer_solution(model, split, [i], device, scale)[0]
            host = runner.predict(split, i).clone()
            staged = runner.predict(split, i, resident).clone()
            max_host = max(max_host, float((reference - host).abs().max()))
            max_resident = max(max_resident, float((reference - staged).abs().max()))
    print({"samples": len(split), "max_abs_host": max_host,
           "max_abs_resident": max_resident,
           "graph_enabled": runner.graph is not None})
    if max(max_host, max_resident) > 1e-7:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
