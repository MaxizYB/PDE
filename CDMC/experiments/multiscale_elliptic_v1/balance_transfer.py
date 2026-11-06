"""Local strong-coupling elimination and a protected coarse representation.

No solution labels or known equilibrium vectors enter construction.
The prototype partitions the Cartesian graph into interiors and separators.
Its numerical ingredients are static condensation / subspace correction.
"""
from __future__ import annotations

from dataclasses import dataclass
import time

import numpy as np
import scipy.linalg as dense
import scipy.sparse as sp
import scipy.sparse.linalg as sparse


def separator_partition(n, span, indices=None):
    """Return disjoint interiors separated by retained grid lines."""
    side = n-2
    indices = np.arange(side*side) if indices is None else indices
    i, j = indices//side+1, indices % side+1
    retained = ((i % span == 0) | (j % span == 0)).ravel()
    coarse = np.flatnonzero(retained)
    groups = (i//span*((n-1)//span+1)+j//span).ravel()
    blocks = [np.flatnonzero((groups == key) & ~retained)
              for key in np.unique(groups[~retained])]
    assert sum(map(len, blocks))+len(coarse) == len(indices)
    return coarse, blocks


def protect_local_freedoms(H, coarse, blocks, L=None, weak_floor=0.):
    """Retain coordinates for every numerically unconstrained local freedom.

    A rank-revealing PSD factorization selects a positive principal block to
    eliminate. The remaining coordinates join the coarse space; no ridge is
    added to the strong operator.
    """
    retained, interiors = [coarse], []
    protected = 0
    for block in blocks:
        hb = H[block][:, block].toarray()
        hb = .5*(hb+hb.T)
        if weak_floor:
            lb = L[block][:, block].toarray()
            hb -= weak_floor*.5*(lb+lb.T)
        tolerance = np.sqrt(np.finfo(float).eps)*max(
            float(np.max(hb.diagonal())), np.finfo(float).tiny)
        _, piv, rank, _ = dense.lapack.dpstrf(hb, lower=1, tol=tolerance)
        selected = piv[:rank]-1
        extra = piv[rank:]-1
        retained.append(block[extra])
        protected += len(extra)
        if rank:
            interiors.append(np.sort(block[selected]))
    return np.sort(np.concatenate(retained)), interiors, protected


@dataclass
class BalanceTransfer:
    P: sp.csr_matrix
    Hc: sp.csr_matrix
    Lc: sp.csr_matrix
    blocks: list
    Hblocks: list
    Lblocks: list
    setup_ms: dict
    coarse: np.ndarray
    weak_cross: list | None = None

    @classmethod
    def build(cls, H, L, n, span=8, epsilon_for_exact=None, indices=None):
        """Build H-harmonic transfer; optional A-harmonic numerical control."""
        start = time.perf_counter()
        coarse, blocks = separator_partition(n, span, indices)
        coarse, blocks, protected = protect_local_freedoms(H, coarse, blocks)
        coordinate = np.full(H.shape[0], -1, dtype=np.int64)
        coordinate[coarse] = np.arange(len(coarse))
        rows, cols, values = [coarse], [np.arange(len(coarse))], [np.ones(len(coarse))]
        Hblocks, Lblocks = [], []
        base = H if epsilon_for_exact is None else H+epsilon_for_exact*L
        factor_ms = 0.
        for block in blocks:
            hb = H[block][:, block].toarray()
            lb = L[block][:, block].toarray()
            # Retaining separators must isolate all interior components.
            outside = np.setdiff1d(base[block].indices, np.r_[block, coarse],
                                   assume_unique=False)
            assert len(outside) == 0, "Partition does not isolate interiors"
            coupling = base[block][:, coarse].tocsr()
            active = np.unique(coupling.indices)
            local = hb if epsilon_for_exact is None else hb+epsilon_for_exact*lb
            mark = time.perf_counter()
            factor = dense.cho_factor(local, check_finite=False)
            lift = -dense.cho_solve(factor, coupling[:, active].toarray(),
                                    check_finite=False)
            factor_ms += 1000*(time.perf_counter()-mark)
            rows.append(np.repeat(block, len(active)))
            cols.append(np.tile(active, len(block)))
            values.append(lift.ravel())
            Hblocks.append(hb)
            Lblocks.append(lb)
        P = sp.coo_matrix((np.concatenate(values),
                           (np.concatenate(rows), np.concatenate(cols))),
                          shape=(H.shape[0], len(coarse))).tocsr()
        lift_ms = 1000*(time.perf_counter()-start)
        mark = time.perf_counter()
        Hc, Lc = (P.T@H@P).tocsr(), (P.T@L@P).tocsr()
        return cls(P, Hc, Lc, blocks, Hblocks, Lblocks,
                   dict(partition_and_lift_ms=lift_ms,
                        local_lift_factor_solve_ms=factor_ms,
                        protected_local_coordinates=protected,
                        coarse_products_ms=1000*(time.perf_counter()-mark),
                        total_ms=1000*(time.perf_counter()-start)), coarse)

    @classmethod
    def build_local(cls, H, L, n, span=8, indices=None, weak_floor=0.):
        """Assemble both reduced forms from local blocks, without global RAP."""
        start = time.perf_counter()
        coarse, blocks = separator_partition(n, span, indices)
        coarse, blocks, protected = protect_local_freedoms(H, coarse, blocks, L, weak_floor)
        coordinate = np.full(H.shape[0], -1, dtype=np.int64)
        coordinate[coarse] = np.arange(len(coarse))
        position = np.full(H.shape[0], -1, dtype=np.int64)
        owner = np.full(H.shape[0], -1, dtype=np.int64)
        for k, block in enumerate(blocks):
            position[block] = np.arange(len(block))
            owner[block] = k
        prow, pcol, pval = [coarse], [np.arange(len(coarse))], [np.ones(len(coarse))]
        cr, cc, hv, lv = [], [], [], []
        Hblocks, Lblocks, weak_cross = [], [], []
        local_ms = 0.
        for k, block in enumerate(blocks):
            hrow, lrow = H[block].tocoo(), L[block].tocoo()
            hc_mask, lc_mask = coordinate[hrow.col]>=0, coordinate[lrow.col]>=0
            assert np.all(owner[hrow.col[~hc_mask]] == k)
            assert np.all(owner[lrow.col[~lc_mask]] == k)
            active = np.union1d(coordinate[hrow.col[hc_mask]], coordinate[lrow.col[lc_mask]])
            hb, lb = np.zeros((len(block), len(block))), np.zeros((len(block), len(block)))
            hb[hrow.row[~hc_mask], position[hrow.col[~hc_mask]]] = hrow.data[~hc_mask]
            lb[lrow.row[~lc_mask], position[lrow.col[~lc_mask]]] = lrow.data[~lc_mask]
            hc, lc = np.zeros((len(block), len(active))), np.zeros((len(block), len(active)))
            hc[hrow.row[hc_mask], np.searchsorted(active, coordinate[hrow.col[hc_mask]])] = hrow.data[hc_mask]
            lc[lrow.row[lc_mask], np.searchsorted(active, coordinate[lrow.col[lc_mask]])] = lrow.data[lc_mask]
            mark = time.perf_counter()
            factor = dense.cho_factor(.5*(hb+hb.T), check_finite=False)
            W = -dense.cho_solve(factor, hc, check_finite=False)
            h_delta = hc.T@W
            cross = lc.T@W
            l_delta = cross+cross.T+W.T@lb@W
            local_ms += 1000*(time.perf_counter()-mark)
            prow.append(np.repeat(block, len(active)))
            pcol.append(np.tile(active, len(block)))
            pval.append(W.ravel())
            cr.append(np.repeat(active, len(active)))
            cc.append(np.tile(active, len(active)))
            hv.append((.5*(h_delta+h_delta.T)).ravel())
            lv.append((.5*(l_delta+l_delta.T)).ravel())
            Hblocks.append(hb); Lblocks.append(lb)
            weak_cross.append((active, lb@W+lc))
        P = sp.coo_matrix((np.concatenate(pval), (np.concatenate(prow), np.concatenate(pcol))),
                          shape=(H.shape[0], len(coarse))).tocsr()
        reduced_shape = (len(coarse), len(coarse))
        cr, cc = np.concatenate(cr), np.concatenate(cc)
        Hc = (H[coarse][:, coarse]+sp.coo_matrix(
            (np.concatenate(hv), (cr, cc)), shape=reduced_shape)).tocsr()
        Lc = (L[coarse][:, coarse]+sp.coo_matrix(
            (np.concatenate(lv), (cr, cc)), shape=reduced_shape)).tocsr()
        total_ms = 1000*(time.perf_counter()-start)
        return cls(P, Hc, Lc, blocks, Hblocks, Lblocks,
                   dict(total_ms=total_ms, local_algebra_ms=local_ms,
                        protected_local_coordinates=protected,
                        weak_floor=weak_floor,
                        assembly="local_block_forms"), coarse, weak_cross)

    def solver(self, epsilon, symmetric=False):
        mark = time.perf_counter()
        local = [dense.cho_factor(hb+epsilon*lb, check_finite=False)
                 for hb, lb in zip(self.Hblocks, self.Lblocks)]
        local_ms = 1000*(time.perf_counter()-mark)
        mark = time.perf_counter()
        coarse_factor = sparse.splu((self.Hc+epsilon*self.Lc).tocsc())
        coarse_ms = 1000*(time.perf_counter()-mark)
        return ProtectedSolve(self, local, coarse_factor,
                              dict(local_factor_ms=local_ms,
                                   coarse_factor_ms=coarse_ms,
                                   total_ms=local_ms+coarse_ms), epsilon, symmetric)


@dataclass
class ProtectedSolve:
    transfer: BalanceTransfer
    local: list
    coarse_factor: object
    setup_ms: dict
    epsilon: float
    symmetric: bool = False

    def apply(self, rhs):
        t = self.transfer
        if self.symmetric:
            local = np.zeros_like(rhs)
            reduced_rhs = t.P.T@rhs
            for block, factor, (active, cross) in zip(t.blocks, self.local, t.weak_cross):
                local[block] = dense.cho_solve(factor, rhs[block], check_finite=False)
                reduced_rhs[active] -= self.epsilon*(cross.T@local[block])
            coefficient = self.coarse_factor.solve(reduced_rhs)
            result = t.P@coefficient+local
            for block, factor, (active, cross) in zip(t.blocks, self.local, t.weak_cross):
                result[block] -= self.epsilon*dense.cho_solve(
                    factor, cross@coefficient[active], check_finite=False)
            return result
        result = t.P@self.coarse_factor.solve(t.P.T@rhs)
        for block, factor in zip(t.blocks, self.local):
            result[block] += dense.cho_solve(factor, rhs[block], check_finite=False)
        return result

    def aspreconditioner(self):
        size = self.transfer.P.shape[0]
        return sparse.LinearOperator((size, size), matvec=self.apply,
                                     rmatvec=self.apply, dtype=np.float64)


@dataclass
class BalanceHierarchy:
    """Apply the same H-harmonic construction to nested separators."""
    levels: list
    setup_ms: dict
    terminal_H: sp.csr_matrix
    terminal_L: sp.csr_matrix
    initial_size: int

    @classmethod
    def build(cls, H, L, n, span=8, max_coarse=1000, first_level=None, local_assembly=False, weak_floor=0.):
        start = time.perf_counter()
        initial_size = H.shape[0]
        levels, sizes = [], [H.shape[0]]
        indices = np.arange(H.shape[0])
        width = span
        while H.shape[0] > max_coarse:
            builder = BalanceTransfer.build_local if local_assembly else BalanceTransfer.build
            extra = {"weak_floor": weak_floor} if local_assembly else {}
            level = first_level if not levels and first_level is not None else builder(
                H, L, n, width, indices=indices, **extra)
            if level.P.shape[1] == 0:
                raise ValueError("Choose max_coarse large enough to retain the last separator")
            levels.append(level)
            indices = indices[level.coarse]
            H, L = level.Hc, level.Lc
            sizes.append(H.shape[0])
            width *= 2
        initial_ms = first_level.setup_ms["total_ms"] if first_level is not None else 0.
        return cls(levels, dict(total_ms=initial_ms+1000*(time.perf_counter()-start),
                               sizes=sizes,
                               levels=[level.setup_ms for level in levels]),
                   H, L, initial_size)

    def solver(self, epsilon, symmetric=False):
        start = time.perf_counter()
        local = [[dense.cho_factor(hb+epsilon*lb, check_finite=False)
                  for hb, lb in zip(t.Hblocks, t.Lblocks)] for t in self.levels]
        local_ms = 1000*(time.perf_counter()-start)
        mark = time.perf_counter()
        terminal = sparse.splu((self.terminal_H+epsilon*self.terminal_L).tocsc())
        coarse_ms = 1000*(time.perf_counter()-mark)
        return HierarchySolve(self, local, terminal,
                              dict(local_factor_ms=local_ms, coarse_factor_ms=coarse_ms,
                                   total_ms=1000*(time.perf_counter()-start)), epsilon, symmetric)


@dataclass
class HierarchySolve:
    transfer: BalanceHierarchy
    local: list
    terminal: object
    setup_ms: dict
    epsilon: float
    symmetric: bool = False

    def apply(self, rhs):
        saved, coarse_rhs = [], rhs
        for level, factors in zip(self.transfer.levels, self.local):
            if self.symmetric:
                local = np.zeros_like(coarse_rhs)
                reduced_rhs = level.P.T@coarse_rhs
                for block, factor, (active, cross) in zip(level.blocks, factors, level.weak_cross):
                    local[block] = dense.cho_solve(factor, coarse_rhs[block], check_finite=False)
                    reduced_rhs[active] -= self.epsilon*(cross.T@local[block])
                saved.append(local)
                coarse_rhs = reduced_rhs
                continue
            saved.append(coarse_rhs)
            coarse_rhs = level.P.T@coarse_rhs
        result = self.terminal.solve(coarse_rhs)
        for level, factors, value in zip(reversed(self.transfer.levels),
                                          reversed(self.local), reversed(saved)):
            if self.symmetric:
                coefficient = result
                result = level.P@coefficient+value
                for block, factor, (active, cross) in zip(level.blocks, factors, level.weak_cross):
                    result[block] -= self.epsilon*dense.cho_solve(
                        factor, cross@coefficient[active], check_finite=False)
                continue
            result = level.P@result
            for block, factor in zip(level.blocks, factors):
                result[block] += dense.cho_solve(factor, value[block], check_finite=False)
        return result

    def aspreconditioner(self):
        size = self.transfer.initial_size
        return sparse.LinearOperator((size, size), matvec=self.apply,
                                     rmatvec=self.apply, dtype=np.float64)
