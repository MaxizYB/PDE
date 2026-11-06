#!/usr/bin/env python3
"""Read-only numerical diagnostics for the generated v2 Darcy data."""
from __future__ import annotations

import argparse
import cProfile
import io
import pstats
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from adapters.darcy import DarcySplit
from adapters.darcy_system import make_mesh_plan, make_system


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--count", type=int, default=16)
    parser.add_argument("--coarse-size", type=int, default=128)
    parser.add_argument("--backend", choices=("cpu", "gpu", "gpu_compiled"),
                        default="cpu")
    args = parser.parse_args()
    torch.set_num_threads(4)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    split = DarcySplit(args.data, "test")
    plan = make_mesh_plan(
        split.mesh, device, args.coarse_size,
        gpu_galerkin=args.backend != "cpu",
        compiled_assembly=args.backend == "gpu_compiled",
    )
    count = min(args.count, len(split))
    systems = []
    profiler = cProfile.Profile()
    profiler.enable()
    for i in range(count):
        systems.append(make_system(
            split.mesh, split.coefficient[i], split.lift[i], device, plan,
            backend=args.backend,
        ))
    if device.type == "cuda":
        torch.cuda.synchronize()
    profiler.disable()
    stream = io.StringIO()
    pstats.Stats(profiler, stream=stream).sort_stats("cumtime").print_stats(30)
    print(stream.getvalue())
    print("levels", [level.matrix.shape[0] for level in systems[0].levels])
    print("omega", [level.omega for level in systems[0].levels[:-1]])
    print("coarse_condition", np.linalg.cond(systems[0].levels[-1].matrix.to_dense().cpu().numpy()))


if __name__ == "__main__":
    main()
