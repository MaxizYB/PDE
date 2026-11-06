"""Compile exact P1 Galerkin maps using sparse basis support.

For each triangle T and cumulative prolongation Q, the two components of
grad(phi @ Q) are sparse rows Gx[T,:], Gy[T,:]. The triangle contribution is
area[T] * (outer(Gx,Gx) + outer(Gy,Gy)). No dense N-by-N identity or expanded
N-by-coarse array is required. This module deliberately has no torch import.
"""
from __future__ import annotations

import time
import numpy as np
import scipy.sparse as sp


def cumulative_bases(free: np.ndarray, transfers: list, progress=None) -> list[sp.csr_matrix]:
    basis = sp.eye(len(free), format="csr", dtype=np.float64)
    bases = [basis]
    for level, (prolongation, _) in enumerate(transfers, start=1):
        started = time.perf_counter()
        basis = (basis @ prolongation).tocsr()
        basis.sum_duplicates()
        basis.eliminate_zeros()
        bases.append(basis)
        if progress is not None:
            progress({"stage": "cumulative_basis", "level": level,
                      "width": basis.shape[1], "nnz": basis.nnz,
                      "seconds": time.perf_counter() - started})
    return bases


def compile_terminal_gradients(mesh, geometry, free, transfers, progress=None):
    """Compile only the exact terminal Gx/Gy factors, without intermediate maps."""
    area, gradient, _ = geometry
    cells = np.asarray(mesh.cells, dtype=np.int64)
    triangle_count = len(cells)
    basis = sp.eye(len(free), format="csr", dtype=np.float64)
    for level, (prolongation, _) in enumerate(transfers, start=1):
        started = time.perf_counter()
        basis = (basis @ prolongation).tocsr()
        basis.sum_duplicates()
        basis.eliminate_zeros()
        if progress is not None:
            progress({"stage": "terminal_basis", "level": level,
                      "width": basis.shape[1], "nnz": basis.nnz,
                      "seconds": time.perf_counter() - started})
    injection = sp.csr_matrix(
        (np.ones(len(free)), (free, np.arange(len(free)))),
        shape=(len(mesh.points), len(free)), dtype=np.float64,
    )
    full_basis = (injection @ basis).tocsr()
    gradients = []
    for component in range(2):
        projected = sp.csr_matrix((triangle_count, basis.shape[1]), dtype=np.float64)
        for corner in range(3):
            projected = projected + (
                sp.diags(gradient[:, corner, component])
                @ full_basis[cells[:, corner], :]
            )
        projected.sum_duplicates()
        projected.eliminate_zeros()
        gradients.append(projected)
    return gradients[0], gradients[1], area


def compile_sparse_affine_maps(
    mesh, geometry, free, transfers, *, max_terminal_nnz: int = 12_000_000,
    terminal_mode: str = "expanded",
    progress=None,
) -> tuple[list, list[sp.csr_matrix]]:
    """Return intermediate diagonal maps and a terminal representation.

    Expanded terminal mode returns a sparse coefficient-to-A map for independent
    CPU verification. Gradient mode returns (Gx, Gy, area), allowing the exact
    current A_L to be assembled as Gx.T @ diag(area*a_T) @ Gx plus its y term.
    The fine diagonal is computed by the existing element action.
    """
    if terminal_mode not in ("expanded", "gradient"):
        raise ValueError("terminal_mode must be expanded or gradient")
    area, gradient, _ = geometry
    cells = np.asarray(mesh.cells, dtype=np.int64)
    triangle_count = len(cells)
    bases = cumulative_bases(free, transfers, progress=progress)
    injection = sp.csr_matrix(
        (np.ones(len(free)), (free, np.arange(len(free)))),
        shape=(len(mesh.points), len(free)), dtype=np.float64,
    )
    maps = []
    for level, basis in enumerate(bases[1:]):
        started = time.perf_counter()
        width = basis.shape[1]
        full_basis = (injection @ basis).tocsr()
        gradients = []
        for component in range(2):
            projected = sp.csr_matrix((triangle_count, width), dtype=np.float64)
            for corner in range(3):
                values = gradient[:, corner, component]
                projected = projected + sp.diags(values) @ full_basis[cells[:, corner], :]
            projected.sum_duplicates()
            projected.eliminate_zeros()
            gradients.append(projected)
        gx, gy = gradients
        if progress is not None:
            progress({"stage": "gradients", "level": level + 1,
                      "width": width, "nnz": gx.nnz + gy.nnz,
                      "seconds": time.perf_counter() - started})
        terminal = level == len(transfers) - 1
        if terminal and terminal_mode == "gradient":
            mapping = (gx, gy, area)
        elif terminal:
            rows, cols, values = [], [], []
            nnz = 0
            for triangle in range(triangle_count):
                sx, ex = gx.indptr[triangle:triangle + 2]
                sy, ey = gy.indptr[triangle:triangle + 2]
                ix, iy = gx.indices[sx:ex], gy.indices[sy:ey]
                support = np.union1d(ix, iy)
                if not len(support):
                    continue
                dx = np.zeros(len(support), dtype=np.float64)
                dy = np.zeros(len(support), dtype=np.float64)
                dx[np.searchsorted(support, ix)] = gx.data[sx:ex]
                dy[np.searchsorted(support, iy)] = gy.data[sy:ey]
                contribution = area[triangle] * (
                    np.outer(dx, dx) + np.outer(dy, dy)
                )
                ii, jj = np.nonzero(np.abs(contribution) > 1e-13)
                count = len(ii)
                nnz += count
                if nnz > max_terminal_nnz:
                    raise MemoryError(
                        f"terminal map would exceed {max_terminal_nnz:,} entries; "
                        "reduce coarse size or use matrix-free terminal action"
                    )
                if count:
                    rows.append((support[ii] * width + support[jj]).astype(np.int32))
                    cols.append(np.full(count, triangle, dtype=np.int32))
                    values.append(contribution[ii, jj])
            mapping = sp.coo_matrix(
                (np.concatenate(values) if values else np.empty(0),
                 (np.concatenate(rows) if rows else np.empty(0, dtype=np.int32),
                  np.concatenate(cols) if cols else np.empty(0, dtype=np.int32))),
                shape=(width * width, triangle_count), dtype=np.float64,
            ).tocsr()
        else:
            diagonal = gx.multiply(gx) + gy.multiply(gy)
            diagonal = (sp.diags(area) @ diagonal).tocsr()
            diagonal.data[np.abs(diagonal.data) <= 1e-13] = 0
            diagonal.eliminate_zeros()
            mapping = diagonal.T.tocsr()
        maps.append(mapping)
        if progress is not None:
            progress({"stage": "map", "level": level + 1,
                      "seconds": time.perf_counter() - started})
    return maps, bases
