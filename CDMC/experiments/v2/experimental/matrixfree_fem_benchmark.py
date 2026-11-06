"""Same-split CPU sparse direct reference for matrix-free timing probes."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from adapters.darcy import DarcySplit
from baselines.data.darcy_v2.generate import element_geometry, solve_sample
from run import describe


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--count", type=int, default=16)
    args = parser.parse_args()
    split = DarcySplit(args.data, "test")
    mesh = split.mesh
    started = time.perf_counter()
    geometry = element_geometry(mesh.points, mesh.cells)
    plan_ms = 1000 * (time.perf_counter() - started)
    records = []
    for i in range(min(args.count, len(split)) + 1):
        index = 0 if i == 0 else i - 1
        started = time.perf_counter()
        solution, residual = solve_sample(
            mesh.points, mesh.cells, mesh.boundary,
            split.coefficient[index], split.lift[index], geometry,
        )
        elapsed_ms = 1000 * (time.perf_counter() - started)
        if i:
            truth = split.target[index]
            relative = np.sqrt(np.sum(mesh.mass * (solution - truth) ** 2) /
                               max(np.sum(mesh.mass * truth ** 2), 1e-12))
            records.append({"error_percent": float(100 * relative),
                            "residual": float(residual), "total_ms": elapsed_ms})
    summary = {key: describe([record[key] for record in records])
               for key in ("error_percent", "residual", "total_ms")}
    summary["geometry_plan_ms_once"] = plan_ms
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"summary": summary, "samples": records}, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
