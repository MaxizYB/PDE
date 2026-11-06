"""PDE-independent SPD Galerkin hierarchy and fixed-cycle correction kernel."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
import torch


def sparse_tensor(matrix: sp.spmatrix, device: torch.device) -> torch.Tensor:
    """Transfer canonical CSR directly; COO coalesce was a major online cost."""
    csr = matrix.tocsr()
    csr.sum_duplicates()
    csr.sort_indices()
    return torch.sparse_csr_tensor(
        torch.as_tensor(csr.indptr, dtype=torch.int64, device=device),
        torch.as_tensor(csr.indices, dtype=torch.int64, device=device),
        torch.as_tensor(csr.data.astype(np.float32), device=device),
        size=csr.shape, device=device,
    )


def multiply(matrix: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
    if matrix.layout == torch.strided:
        return matrix @ vector
    return torch.sparse.mm(matrix, vector.unsqueeze(-1)).squeeze(-1)


def jacobi_spectral_bound(matrix: sp.csr_matrix) -> float:
    """Gershgorin upper bound for rho(D^-1/2 A D^-1/2), without eigensolves."""
    diagonal = matrix.diagonal()
    if np.any(diagonal <= 0):
        raise ValueError("Galerkin stiffness must have a positive diagonal")
    row_absolute = np.asarray(abs(matrix).sum(axis=1)).ravel()
    return float(np.max(row_absolute / diagonal))


def aggregate_prolongation(
    matrix: sp.csr_matrix, aggregate_size: int = 2,
    interpolation_max_entries: int = 0,
) -> sp.csr_matrix:
    """Deterministic strong-edge aggregates followed by smoothed aggregation."""
    if aggregate_size < 2:
        raise ValueError("aggregate_size must be at least 2")
    n = matrix.shape[0]
    graph = matrix.tocsr(copy=True)
    graph.setdiag(0)
    graph.eliminate_zeros()
    assigned = np.full(n, -1, dtype=np.int32)
    groups = 0
    for i in range(n):
        if assigned[i] >= 0:
            continue
        assigned[i] = groups
        begin, end = graph.indptr[i:i + 2]
        neighbors = graph.indices[begin:end]
        weights = np.abs(graph.data[begin:end])
        available = assigned[neighbors] < 0
        if np.any(available):
            candidates = neighbors[available]
            strength = weights[available]
            order = np.argsort(-strength, kind="stable")[:aggregate_size - 1]
            assigned[candidates[order]] = groups
        groups += 1
    tentative = sp.csr_matrix(
        (np.ones(n), (np.arange(n), assigned)), shape=(n, groups)
    )
    diagonal = matrix.diagonal()
    omega = min(0.65, 1.4 / jacobi_spectral_bound(matrix))
    smoother = sp.eye(n, format="csr") - omega * sp.diags(1 / diagonal) @ matrix
    # Standard smoothed aggregation does not row-normalize P. On coarse
    # Galerkin levels its entries may legitimately be signed.
    prolongation = (smoother @ tentative).tocsr()
    if interpolation_max_entries:
        prolongation = localize_prolongation(
            prolongation, assigned, interpolation_max_entries
        )
    return prolongation


def localize_prolongation(
    matrix: sp.csr_matrix, aggregate: np.ndarray, max_entries: int,
) -> sp.csr_matrix:
    """Bound interpolation support, preserving each original row sum.

    Retain the aggregate's own column and the strongest remaining entries;
    lump omitted weights into the own-aggregate entry. This preserves P@1,
    not by forcing rows to sum to one (Dirichlet/signed transfers need not).
    All subsequent reference/sample Galerkin actions use this same P.
    """
    if max_entries < 1:
        raise ValueError("interpolation_max_entries must be positive")
    p = matrix.tocsr(copy=True)
    p.sum_duplicates()
    p.eliminate_zeros()
    counts = np.diff(p.indptr)
    if counts.max(initial=0) <= max_entries:
        return p
    rows = np.repeat(np.arange(p.shape[0]), counts)
    own = p.indices == aggregate[rows]
    has_own = np.bincount(rows[own], minlength=p.shape[0]) > 0
    order = np.lexsort((p.indices, -np.abs(p.data), ~own, rows))
    sorted_rows = rows[order]
    starts = np.flatnonzero(np.r_[True, sorted_rows[1:] != sorted_rows[:-1]])
    ranks = np.arange(len(order)) - np.repeat(starts, np.diff(np.r_[starts, len(order)]))
    keep = np.zeros(len(order), dtype=bool)
    # Reserve one slot for compensation even if the own entry was absent.
    row_budget = max_entries - (~has_own).astype(np.int32)
    keep[order[ranks < row_budget[sorted_rows]]] = True
    filtered = sp.coo_matrix(
        (p.data[keep], (rows[keep], p.indices[keep])), shape=p.shape
    ).tocsr()
    omitted = np.asarray((p - filtered).sum(axis=1)).ravel()
    compensation = sp.coo_matrix(
        (omitted, (np.arange(p.shape[0]), aggregate)), shape=p.shape
    ).tocsr()
    result = (filtered + compensation).tocsr()
    result.eliminate_zeros()
    return result


def drop_galerkin_entries(
    matrix: sp.csr_matrix, relative_tol: float = 0.0,
    max_row_nnz: int | None = None,
    preserve_row_sum: bool = False,
) -> sp.csr_matrix:
    """Sparsify a symmetric Galerkin matrix while preserving its diagonal."""
    if relative_tol <= 0 and max_row_nnz is None:
        return matrix.tocsr()
    a = matrix.tocsr(copy=True)
    a.sum_duplicates()
    a = ((a + a.T) * 0.5).tocsr()
    diagonal = a.diagonal()
    if np.any(diagonal <= 0):
        raise ValueError("cannot sparsify a matrix with non-positive diagonal")
    rows = np.repeat(np.arange(a.shape[0]), np.diff(a.indptr))
    cols = a.indices.copy()
    values = a.data.copy()
    keep = rows == cols
    normalized = np.abs(values) / np.sqrt(diagonal[rows] * diagonal[cols])
    if relative_tol > 0:
        keep |= normalized >= relative_tol
    else:
        keep[:] = True
    if max_row_nnz is not None:
        if max_row_nnz < 1:
            raise ValueError("max_row_nnz must be positive")
        for row in range(a.shape[0]):
            begin, end = a.indptr[row], a.indptr[row + 1]
            candidates = np.flatnonzero(keep[begin:end])
            offdiag = candidates[cols[begin:end][candidates] != row]
            budget = max(0, max_row_nnz - 1)
            if len(offdiag) > budget:
                order = np.argsort(-normalized[begin:end][offdiag], kind="stable")
                local_keep = keep[begin:end].copy()
                local_keep[offdiag[order[budget:]]] = False
                keep[begin:end] = local_keep
    pattern = sp.coo_matrix((keep.astype(np.float32), (rows, cols)), shape=a.shape)
    pattern = ((pattern + pattern.T) > 0).tocsr()
    filtered = a.multiply(pattern).tocsr()
    if preserve_row_sum:
        # Preserve the reference operator's action on the constant vector.
        # Uncompensated truncation inserts a spurious reaction into low modes.
        correction = np.asarray((a - filtered).sum(axis=1)).ravel()
        diagonal = diagonal + correction
        if np.any(diagonal <= 0):
            raise ValueError("row-sum-preserving truncation produced a nonpositive diagonal")
    filtered.setdiag(diagonal)
    filtered.eliminate_zeros()
    return filtered


def build_hierarchy(
    matrix: sp.csr_matrix, coarse_size: int = 128, max_levels: int = 12,
    aggregate_size: int = 2, galerkin_drop_tol: float = 0.0,
    galerkin_max_row_nnz: int | None = None,
    galerkin_preserve_row_sum: bool = False,
    interpolation_max_entries: int = 0,
) -> list[sp.csr_matrix | tuple[sp.csr_matrix, sp.csr_matrix, sp.csr_matrix]]:
    """Level entries are (A,P,R), final entry is the coarsest A."""
    current = matrix.tocsr()
    hierarchy = []
    for _ in range(max_levels - 1):
        if current.shape[0] <= coarse_size:
            break
        p = aggregate_prolongation(current, aggregate_size, interpolation_max_entries)
        if p.shape[1] >= current.shape[0] * 0.9:
            break
        r = p.T.tocsr()
        hierarchy.append((current, p, r))
        current = (r @ current @ p).tocsr()
        if galerkin_drop_tol > 0 or galerkin_max_row_nnz is not None:
            current = drop_galerkin_entries(
                current, galerkin_drop_tol, galerkin_max_row_nnz,
                galerkin_preserve_row_sum,
            )
    hierarchy.append(current)
    return hierarchy


@dataclass
class TorchLevel:
    matrix: torch.Tensor
    diagonal_inverse: torch.Tensor
    prolongation: torch.Tensor | None = None
    restriction: torch.Tensor | None = None
    gate_restriction: torch.Tensor | None = None
    gate_denominator: torch.Tensor | None = None
    omega: float | torch.Tensor = 0.67


@dataclass
class TorchSystem:
    levels: list[TorchLevel]
    rhs: torch.Tensor
    free: np.ndarray
    graph: torch.Tensor
    feature: torch.Tensor
    mass: torch.Tensor
    coarse_inverse: torch.Tensor | None
    coarse_factor: torch.Tensor | None = None

    @property
    def matrix(self) -> torch.Tensor:
        return self.levels[0].matrix


def v_cycle(
    system: TorchSystem, rhs: torch.Tensor,
    smooth_gates: list[torch.Tensor], coarse_gates: list[torch.Tensor],
    level: int = 0, smoothing_steps: int = 2,
) -> torch.Tensor:
    """One Galerkin V-cycle; the same function executes fixed MG and CDMC."""
    if level == len(system.levels) - 1:
        if system.coarse_factor is not None:
            return torch.cholesky_solve(
                rhs.double().unsqueeze(-1), system.coarse_factor
            ).squeeze(-1).to(rhs.dtype)
        if system.coarse_inverse is None:
            raise AssertionError("terminal level has no coarse solver")
        return (system.coarse_inverse @ rhs.double()).to(rhs.dtype)
    item = system.levels[level]
    if item.prolongation is None or item.restriction is None:
        raise AssertionError("nonterminal level needs transfer operators")
    x = torch.zeros_like(rhs)
    damping = item.omega * smooth_gates[level] * item.diagonal_inverse
    for _ in range(smoothing_steps):
        x = x + damping * (rhs - multiply(item.matrix, x))
    remaining = rhs - multiply(item.matrix, x)
    child_rhs = multiply(item.restriction, coarse_gates[level] * remaining)
    child = v_cycle(
        system, child_rhs, smooth_gates, coarse_gates,
        level + 1, smoothing_steps,
    )
    x = x + coarse_gates[level] * multiply(item.prolongation, child)
    for _ in range(smoothing_steps):
        x = x + damping * (rhs - multiply(item.matrix, x))
    return x


def coarsen_gates(
    system: TorchSystem, smooth: torch.Tensor, coarse: torch.Tensor
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    smooth_levels, coarse_levels = [smooth], [coarse]
    for item in system.levels[:-1]:
        if item.gate_restriction is None:
            raise AssertionError("missing positive transfer for setup gates")
        if item.gate_denominator is None:
            ones = torch.ones_like(smooth_levels[-1])
            denominator = multiply(item.gate_restriction, ones).clamp_min(1e-8)
        else:
            denominator = item.gate_denominator
        smooth_levels.append(multiply(item.gate_restriction, smooth_levels[-1]) / denominator)
        coarse_levels.append(multiply(item.gate_restriction, coarse_levels[-1]) / denominator)
    return smooth_levels, coarse_levels


def correct(
    system: TorchSystem, initial: torch.Tensor, cycles: int,
    smooth: torch.Tensor | None = None, coarse: torch.Tensor | None = None,
    smoothing_steps: int = 2,
) -> torch.Tensor:
    """Fixed-cycle correction with SPD energy step, no tolerance stopping."""
    x = initial
    one = torch.ones_like(initial)
    smooth_levels, coarse_levels = coarsen_gates(
        system, one if smooth is None else smooth, one if coarse is None else coarse
    )
    for _ in range(cycles):
        residual = system.rhs - multiply(system.matrix, x)
        direction = v_cycle(
            system, residual, smooth_levels, coarse_levels,
            smoothing_steps=smoothing_steps,
        )
        ad = multiply(system.matrix, direction)
        alpha = (residual @ direction / (direction @ ad).clamp_min(1e-12)).clamp(0, 2)
        x = x + alpha * direction
    return x
