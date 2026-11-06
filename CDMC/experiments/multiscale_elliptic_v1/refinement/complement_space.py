"""Preserve the reference solve; learn only its unresolved energy complement.

No full-grid inverse, Krylov loop, or test-dependent fallback is introduced.
This module leaves frame_space.py and its checkpoints unchanged.
"""
from __future__ import annotations

from dataclasses import dataclass
import torch
from torch import nn
from coupled_coarse import CoupledCoarseSetup


@dataclass
class ComplementFactor:
    q: torch.Tensor
    aq: torch.Tensor
    lq: torch.Tensor
    w: torch.Tensor
    aw: torch.Tensor
    lw: torch.Tensor


def solve_columns(factor, rhs):
    return torch.cholesky_solve(rhs, factor)


def reference_factor(matrix, q):
    q = q.double()
    aq = torch.sparse.mm(matrix.double(), q)
    gram = q.T @ aq
    return q, aq, torch.linalg.cholesky((gram + gram.T) / 2)


def reference_update(x, residual, reference):
    q, _, factor = reference
    return x.double() + q @ solve_columns(factor, (q.T @ residual.double())[:, None])[:, 0]


class ComplementGraph(nn.Module):
    """Multilevel context plus fine material/action readout.

The coarse GNN communicates globally. A fine readout sees the actual
coefficient and D^-1 A applied to its context, so output is not restricted
to the piecewise-smooth range of mesh-only prolongation. The model sees
the post-reference iterate and residual, not an error already handled by Q.
"""
    def __init__(self, width=24, rank=8, fine_readout=True):
        super().__init__()
        self.rank = rank
        self.fine_readout = fine_readout
        self.graph = CoupledCoarseSetup(width=width, rank=rank, start_level=1)
        if fine_readout:
            self.readout = nn.Sequential(nn.Linear(7 + 2 * rank, width), nn.GELU(),
                                         nn.Linear(width, rank))

    def forward(self, features, context, system, matrix):
        coarse = self.graph(features, context)
        if not self.fine_readout:
            return torch.tanh(coarse)
        action = torch.sparse.mm(matrix, coarse)
        action = system.levels[0].diagonal_inverse[:, None] * action
        return torch.tanh(self.readout(torch.cat((features[0], coarse, action), -1)))


def prepare_complement(matrix, reference, candidates, ridge=1e-12):
    """Block Galerkin factorization preserving the full reference space.

W=(I-Q(Q^TAQ)^-1 Q^TA)H, hence Q^TAW=0 in exact arithmetic.
The tiny fixed dimensionless ridge regularizes only the new block and
never perturbs the reference solve. Report it explicitly in experiments.
"""
    q, aq, lq = reference
    h = candidates.double()
    w = h - q @ solve_columns(lq, aq.T @ h)
    aw = torch.sparse.mm(matrix.double(), w)
    scale = (w * aw).sum(0).clamp_min(1e-30).sqrt()
    w, aw = w / scale, aw / scale
    gram = w.T @ aw
    gram = (gram + gram.T) / 2
    lw = torch.linalg.cholesky(gram + ridge * torch.eye(gram.shape[0], device=h.device, dtype=h.dtype))
    return ComplementFactor(q, aq, lq, w, aw, lw)


def complement_update(x, residual, prepared):
    r = residual.double()
    cq = solve_columns(prepared.lq, (prepared.q.T @ r)[:, None])[:, 0]
    cw = solve_columns(prepared.lw, (prepared.w.T @ r)[:, None])[:, 0]
    return (x.double() + prepared.q @ cq + prepared.w @ cw).to(x.dtype)


def complement_diagnostics(prepared):
    with torch.no_grad():
        gram = prepared.w.T @ prepared.aw
        ev = torch.linalg.eigvalsh((gram + gram.T) / 2)
        cross = prepared.q.T @ prepared.aw
        return dict(complement_columns=prepared.w.shape[1],
                    normalized_energy_eigenvalues=ev.cpu().tolist(),
                    reference_complement_cross_norm=float(cross.norm()))
