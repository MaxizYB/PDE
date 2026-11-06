#!/usr/bin/env python3
"""Check compiled and optional affine FEM/Galerkin against CPU construction."""
from __future__ import annotations

import argparse
import json
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
    parser.add_argument("--hybrid-dense-limit", type=int, default=512)
    parser.add_argument("--affine", action="store_true", help="also check exact offline element maps")
    args = parser.parse_args()
    device = torch.device("cuda")
    split = DarcySplit(args.data, "test")
    plan = make_mesh_plan(
        split.mesh, device, args.coarse_size,
        gpu_galerkin=True, compiled_assembly=True,
        hybrid_dense_limit=args.hybrid_dense_limit,
        affine_compiler=args.affine,
    )
    worst = {key: 0.0 for key in (
        "matrix_abs", "matrix_relative_frobenius", "rhs_abs", "feature_abs",
        "coarse_inverse_abs", "omega_abs",
    )}
    hybrid_worst = {key: 0.0 for key in worst}
    affine_worst = {key: 0.0 for key in worst}
    # A nonzero boundary trace tests the eliminated free/fixed block, which
    # zero-Dirichlet generated samples alone would not exercise.
    indices = list(range(min(args.count, len(split)))) + [0]
    with torch.no_grad():
        for position, i in enumerate(indices):
            lift = split.lift[i]
            if position == len(indices) - 1:
                x, y = split.mesh.points.T
                lift = (0.01 + 0.02 * x - 0.01 * y).astype(np.float32)
            reference = make_system(
                split.mesh, split.coefficient[i], lift, device, plan,
                backend="cpu",
            )
            compiled = make_system(
                split.mesh, split.coefficient[i], lift, device, plan,
                backend="gpu_compiled",
            )
            hybrid = make_system(
                split.mesh, split.coefficient[i], lift, device, plan,
                backend="gpu_hybrid",
            )
            comparisons = [(reference, compiled, worst), (compiled, hybrid, hybrid_worst)]
            if args.affine:
                affine = make_system(
                    split.mesh, split.coefficient[i], lift, device, plan,
                    backend="gpu_affine",
                )
                comparisons.append((hybrid, affine, affine_worst))
            for baseline, candidate, bucket in comparisons:
                for a, b in zip(baseline.levels, candidate.levels):
                    left = a.matrix.to_dense() if a.matrix.layout != torch.strided else a.matrix
                    right = b.matrix.to_dense() if b.matrix.layout != torch.strided else b.matrix
                    delta = left - right
                    bucket["matrix_abs"] = max(bucket["matrix_abs"], float(delta.abs().max()))
                    bucket["matrix_relative_frobenius"] = max(
                        bucket["matrix_relative_frobenius"],
                        float(torch.linalg.vector_norm(delta) /
                              torch.linalg.vector_norm(left).clamp_min(1e-12)),
                    )
                    bucket["omega_abs"] = max(
                        bucket["omega_abs"],
                        float(torch.as_tensor(a.omega, device=device).sub(b.omega).abs()),
                    )
                bucket["rhs_abs"] = max(bucket["rhs_abs"], float((baseline.rhs - candidate.rhs).abs().max()))
                bucket["feature_abs"] = max(bucket["feature_abs"], float((baseline.feature - candidate.feature).abs().max()))
                bucket["coarse_inverse_abs"] = max(
                    bucket["coarse_inverse_abs"],
                    float((baseline.coarse_inverse - candidate.coarse_inverse).abs().max()),
                )
    output = {"samples_including_nonzero_boundary": len(indices),
              "cpu_vs_compiled": worst, "compiled_vs_hybrid": hybrid_worst}
    if args.affine:
        output["hybrid_vs_affine"] = affine_worst
    print(json.dumps(output, indent=2))
    buckets = [("compiled", worst), ("hybrid", hybrid_worst)]
    if args.affine:
        buckets.append(("affine", affine_worst))
    for name, values in buckets:
        if values["matrix_relative_frobenius"] > 1e-5 or values["rhs_abs"] > 1e-5:
            raise AssertionError(f"{name} construction differs materially from its reference")


if __name__ == "__main__":
    main()
