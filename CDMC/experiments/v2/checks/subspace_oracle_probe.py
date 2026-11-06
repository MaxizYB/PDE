"""Oracle rank test for a one-shot correction on the existing small dataset.

Projects test errors onto the best training-error subspace using *test truth*.
This is an unattainable lower bound, NOT an inference method or a baseline.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ["OPENBLAS_NUM_THREADS"] = "1"
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from adapters.darcy import DarcySplit
from adapters.darcy_system import make_mesh_plan, make_system
from run import describe, load_proposal, predict_numpy
from solver.correctors import fixed_mg


def fields(split, model, scale, device, stage):
    proposal = predict_numpy(model, split, scale, device)
    if stage == "NO":
        return proposal
    plan = make_mesh_plan(split.mesh, device, coarse_size=128)
    corrected = []
    for i in range(len(split)):
        system = make_system(split.mesh, split.coefficient[i], split.lift[i], device, plan)
        interior = fixed_mg(system, proposal[i, system.free], 1)
        result = torch.as_tensor(split.lift[i], device=device).clone()
        result[system.free] = interior
        corrected.append(result)
    return torch.stack(corrected)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--proposal-dir", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    args.checkpoint_dir = None
    torch.set_num_threads(4)
    device = torch.device("cuda")
    model, scale = load_proposal(args, device)
    train = DarcySplit(args.data, "train")
    test = DarcySplit(args.data, "test")
    free = np.flatnonzero(~test.mesh.boundary)
    weight = torch.as_tensor(test.mesh.mass[free], device=device).sqrt()
    output = {}
    with torch.no_grad():
        for stage in ("NO", "MG_K1"):
            train_field = fields(train, model, scale, device, stage)
            test_field = fields(test, model, scale, device, stage)
            train_error = torch.as_tensor(train.target[:, free], device=device) - train_field[:, free]
            test_error = torch.as_tensor(test.target[:, free], device=device) - test_field[:, free]
            left, singular, _ = torch.linalg.svd((train_error * weight).T, full_matrices=False)
            response = (test_error * weight) @ left
            denominator = (torch.as_tensor(test.target, device=device).square() *
                           torch.as_tensor(test.mesh.mass, device=device)).sum(dim=1).sqrt()
            output[stage] = {
                "original_percent": describe((100 * torch.linalg.vector_norm(
                    test_error * weight, dim=1) / denominator).cpu().tolist()),
                "oracle_by_rank": {}, "singular_values": singular.cpu().tolist(),
            }
            for rank in (8, 16, 32, 64):
                rank = min(rank, len(train))
                projected = (response[:, :rank] @ left[:, :rank].T)
                percent = (100 * torch.linalg.vector_norm(test_error * weight - projected, dim=1)
                           / denominator).cpu().tolist()
                output[stage]["oracle_by_rank"][str(rank)] = describe(percent)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps({k: {"original_percent": v["original_percent"],
                           "oracle_by_rank": v["oracle_by_rank"]} for k,v in output.items()}, indent=2))


if __name__ == "__main__":
    main()
