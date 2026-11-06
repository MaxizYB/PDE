"""CPU-only check: sparse affine maps reproduce the exact Galerkin matrices."""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import scipy.sparse as sp

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from baselines.data.darcy_v2.generate import assemble, element_geometry, regular_mesh
from CDMC.experiments.v2.experimental.sparse_projection_maps import (
    compile_sparse_affine_maps, compile_terminal_gradients,
)


def transfers_from_reference(matrix: sp.csr_matrix, count: int = 2):
    transfers = []
    current = matrix
    for _ in range(count):
        n = current.shape[0]
        tentative = sp.csr_matrix(
            (np.ones(n), (np.arange(n), np.arange(n) // 2)),
            shape=(n, (n + 1) // 2),
        )
        smoother = sp.eye(n, format="csr") - 0.45 * sp.diags(1 / current.diagonal()) @ current
        prolongation = (smoother @ tentative).tocsr()
        restriction = prolongation.T.tocsr()
        transfers.append((prolongation, restriction))
        current = (restriction @ current @ prolongation).tocsr()
    return transfers


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resolution", type=int, default=33)
    parser.add_argument("--mesh", type=Path,
                        help="optional mesh.npz with points/cells/boundary, for an irregular domain")
    args = parser.parse_args()
    if args.mesh:
        with np.load(args.mesh) as archive:
            points, cells, boundary = (archive[name] for name in ("points", "cells", "boundary"))
    else:
        points, cells, boundary = regular_mesh(args.resolution)
    mesh = SimpleNamespace(points=points, cells=cells, boundary=boundary)
    free = np.flatnonzero(~boundary)
    geometry = element_geometry(points, cells)
    reference = assemble(points, cells, np.ones(len(points)), geometry)[0][free][:, free]
    transfers = transfers_from_reference(reference)
    started = time.perf_counter()
    maps, bases = compile_sparse_affine_maps(mesh, geometry, free, transfers)
    compile_s = time.perf_counter() - started
    gradient_maps, _ = compile_sparse_affine_maps(
        mesh, geometry, free, transfers, terminal_mode="gradient"
    )
    terminal_only = compile_terminal_gradients(mesh, geometry, free, transfers)
    rng = np.random.default_rng(42)
    max_abs = 0.0
    for _ in range(3):
        coefficient = 0.5 + rng.random(len(points)) * 4
        element_coefficient = coefficient[cells].mean(axis=1)
        fine = assemble(points, cells, coefficient, geometry)[0][free][:, free]
        for level, mapping in enumerate(maps):
            galerkin = (bases[level + 1].T @ fine @ bases[level + 1]).tocsr()
            predicted = np.asarray(mapping @ element_coefficient)
            expected = (galerkin.toarray().reshape(-1) if level == len(maps) - 1
                        else galerkin.diagonal())
            difference = float(np.max(np.abs(predicted - expected)))
            max_abs = max(max_abs, difference)
            if difference > 1e-9:
                raise AssertionError(f"level {level + 1} mismatch {difference:.3e}")
            if level == len(maps) - 1:
                gx, gy, area = gradient_maps[-1]
                weights = sp.diags(area * element_coefficient)
                factorized = (gx.T @ weights @ gx + gy.T @ weights @ gy).toarray()
                difference = float(np.max(np.abs(factorized - galerkin.toarray())))
                max_abs = max(max_abs, difference)
                if difference > 1e-9:
                    raise AssertionError(f"terminal gradient mismatch {difference:.3e}")
                tx, ty, ta = terminal_only
                terminal_result = (
                    tx.T @ sp.diags(ta * element_coefficient) @ tx
                    + ty.T @ sp.diags(ta * element_coefficient) @ ty
                ).toarray()
                difference = float(np.max(np.abs(terminal_result - galerkin.toarray())))
                max_abs = max(max_abs, difference)
                if difference > 1e-9:
                    raise AssertionError(f"terminal-only mismatch {difference:.3e}")
    print({"nodes": len(points), "triangles": len(cells),
           "map_build_s": round(compile_s, 4), "map_nnz": [m.nnz for m in maps],
           "basis_nnz": [b.nnz for b in bases], "max_abs_error": max_abs})


if __name__ == "__main__":
    main()
