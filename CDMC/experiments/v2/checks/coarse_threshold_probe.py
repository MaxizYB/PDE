"""Check whether one precise coarse solve can replace a second V-cycle.

Fixed trained NO and CDMC setup are reused; different coarsest sizes are a
diagnostic, not a retrained baseline.  Reports both error and full latency.
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
from run import describe, error_percent, load_proposal, synchronized_clock
from solver.correctors import CDMCSetup, cdmc, fixed_mg
from solver.inference import FixedMeshProposal


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--proposal-dir", type=Path, required=True)
    p.add_argument("--cdmc-dir", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--coarse-sizes", type=int, nargs="+", default=[128, 256, 512])
    p.add_argument("--count", type=int, default=16)
    args = p.parse_args()
    args.checkpoint_dir = None
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    device = torch.device("cuda")
    split = DarcySplit(args.data, "test")
    model, scale = load_proposal(args, device)
    runner = FixedMeshProposal(model, split, scale, device, capture_graph=True, use_prepared=True)
    ckpt = torch.load(args.cdmc_dir / "cdmc.pt", map_location=device, weights_only=False)
    setup = CDMCSetup(width=ckpt["width"]).to(device)
    setup.load_state_dict(ckpt["state"])
    setup.eval()
    output = []
    with torch.no_grad():
        for coarse_size in args.coarse_sizes:
            t0 = synchronized_clock(device)
            plan = make_mesh_plan(split.mesh, device, coarse_size=coarse_size,
                                  gpu_galerkin=True, compiled_assembly=True,
                                  hybrid_dense_limit=512)
            plan_once = 1000 * (synchronized_clock(device) - t0)
            measurements = []
            for i in range(min(args.count, len(split)) + 1):
                index = 0 if i == 0 else i - 1
                t0 = synchronized_clock(device)
                system = make_system(split.mesh, split.coefficient[index], split.lift[index],
                                     device, plan, backend="gpu_hybrid")
                system_ms = 1000 * (synchronized_clock(device) - t0)
                t0 = synchronized_clock(device)
                initial_field = runner.predict(split, index)
                no_ms = 1000 * (synchronized_clock(device) - t0)
                initial = initial_field[system.free]
                target = torch.as_tensor(split.target[index], device=device)
                mass = torch.as_tensor(split.mesh.mass, device=device)
                lift = torch.as_tensor(split.lift[index], device=device)
                row = {"system_ms": system_ms, "NO_ms": no_ms}
                for name, fn in (("MG_K1", lambda: fixed_mg(system, initial, 1)),
                                 ("CDMC_K1", lambda: cdmc(setup, system, initial, 1)),
                                 ("CDMC_K2", lambda: cdmc(setup, system, initial, 2))):
                    t0 = synchronized_clock(device)
                    prediction = fn()
                    apply_ms = 1000 * (synchronized_clock(device) - t0)
                    field = lift.clone()
                    field[system.free] = prediction
                    row[name + "_error_percent"] = error_percent(field, target, mass)
                    row[name + "_total_ms"] = system_ms + no_ms + apply_ms
                if i:
                    measurements.append(row)
            summary = {name: {
                "field_error_percent": describe([v[name + "_error_percent"] for v in measurements]),
                "total_ms": describe([v[name + "_total_ms"] for v in measurements]),
            } for name in ("MG_K1", "CDMC_K1", "CDMC_K2")}
            summary["system_ms"] = describe([v["system_ms"] for v in measurements])
            result = {"coarse_size": coarse_size, "level_sizes": [
                item.matrix.shape[0] for item in system.levels],
                "plan_once_ms": plan_once, "summary": summary, "per_sample": measurements}
            output.append(result)
            print(json.dumps({"coarse_size": coarse_size, "level_sizes": result["level_sizes"],
                              "summary": summary}, indent=2), flush=True)
    args.out.write_text(json.dumps(output, indent=2) + "\n")


if __name__ == "__main__":
    main()
