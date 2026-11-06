"""NO-defect-conditioned coupled coarse modes for the existing CDMC hierarchy.

The neural module changes correction directions, not pointwise damping.
Numerical Galerkin calibration retains the current fine PDE and boundary map.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class ModeOnlyCompiled:
    """Fine PDE plus mesh graph hierarchy, without numerical MG coarse solves.

    Graph pooling/prolongation remains part of the neural architecture. Online
    material setup creates only the fine diagonal, RHS, and input features: no
    material Galerkin hierarchy, coarse inverse, or terminal factorization.
    """

    def __init__(self, mesh, device, shared=None):
        import time
        from adapters.darcy_system import make_mesh_plan
        from solver.matrixfree_cdmc import FactorizedGalerkin

        started = time.perf_counter()
        self.device = device
        if shared is None:
            self.plan = make_mesh_plan(mesh, device, coarse_size=128,
                        retain_reference_levels=True, aggregate_size=2,
                        interpolation_max_entries=64)
            self.action = FactorizedGalerkin(mesh, self.plan, device)
            self.points_free = torch.as_tensor(self.plan.points_free, dtype=torch.float32, device=device)
        else:
            self.plan, self.action = shared.plan, shared.action
            self.points_free = shared.points_free
        if device.type=='cuda':
            torch.cuda.synchronize(device)
        self.mesh_plan_ms_once = 1000*(time.perf_counter()-started)
        self.shared_plan = shared is not None

    def build(self, coefficient, lift):
        from solver.numerics import TorchLevel, TorchSystem

        element_a, diagonal = self.action.fine.prepare(coefficient[None])
        diagonal = diagonal[0]
        rhs = self.action.fine.residual(torch.zeros_like(diagonal)[None], lift[None], element_a)[0]
        feature = torch.stack((
            coefficient[self.action.fine.free].log()/np.log(12),
            (diagonal/diagonal.quantile(.5)).log(),
            self.points_free[:,0], self.points_free[:,1],
            torch.log(self.plan.mass/self.plan.mass.median())), -1).float()
        system = TorchSystem([TorchLevel(torch.empty(0, device=self.device), diagonal.reciprocal())],
                    rhs, self.plan.free, self.plan.graph, feature, self.plan.mass, None)
        return system, element_a


def mm(matrix, values):
    return matrix@values if matrix.layout == torch.strided else torch.sparse.mm(matrix, values)


class CoarseContext:
    """Mesh-only graph/averaging maps. No material-specific sparse products."""

    def __init__(self, compiled, start_level=1):
        from solver.numerics import sparse_tensor

        self.compiled = compiled
        self.start_level = start_level
        self.graphs, self.restrict, self.prolong = [], [], []
        for depth, (p, _) in enumerate(compiled.plan.transfers):
            # Positive normalized feature averaging is distinct from signed
            # variational restriction used for the actual coarse correction.
            positive = abs(p).T.tocsr()
            positive = positive.multiply(1/np.maximum(np.asarray(positive.sum(1)).ravel(), 1e-30)[:, None])
            self.restrict.append(sparse_tensor(positive.tocsr(), compiled.device))
            self.prolong.append(compiled.action.prolongations[depth])
        for matrix in compiled.plan.reference_levels:
            graph = abs(matrix).tocsr()
            graph.setdiag(0)
            graph.eliminate_zeros()
            graph = graph.multiply(1/np.maximum(np.asarray(graph.sum(1)).ravel(), 1e-30)[:, None])
            self.graphs.append(sparse_tensor(graph.tocsr(), compiled.device))

    def features(self, system, element_a, initial):
        action = self.compiled.action
        residual = system.rhs-action.action(0, initial, element_a)
        # D^-1 r exposes a potential correction; the separately normalized NO
        # field keeps global initial-solution context. Neither uses test truth.
        defect = residual*system.levels[0].diagonal_inverse
        defect = defect/defect.square().mean().sqrt().clamp_min(1e-12)
        no_field = initial/initial.square().mean().sqrt().clamp_min(1e-12)
        features = torch.cat((system.feature, no_field[:, None], defect[:, None]), 1)
        result = [features]
        for restriction in self.restrict:
            result.append(mm(restriction, result[-1]))
        return result

    def lift(self, coarse):
        fine = coarse
        for p in reversed(self.prolong[:self.start_level]):
            fine = mm(p, fine)
        return fine


class GraphBlock(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.local = nn.Linear(width, width)
        self.neighbor = nn.Linear(width, width, bias=False)
        self.norm = nn.LayerNorm(width)

    def forward(self, values, graph):
        return values+F.gelu(self.norm(self.local(values)+self.neighbor(mm(graph, values))))


class CoupledCoarseSetup(nn.Module):
    """Shared multilevel graph encoder/decoder producing r coupled modes.

    Downward messages give every coarse feature a global material/defect
    context; upward messages return that context to individual regions.
    Parameters are shared over levels, so graph sizes are not model sizes.
    """

    def __init__(self, width=16, rank=8, start_level=1):
        super().__init__()
        self.width, self.rank, self.start_level = width, rank, start_level
        self.lift = nn.Linear(7, width)
        self.down = GraphBlock(width)
        self.up = GraphBlock(width)
        self.merge = nn.Linear(2*width, width)
        self.output = nn.Linear(width, rank)

    def forward(self, features, context):
        saved = []
        hidden = None
        last = len(features)-1
        for level in range(self.start_level, last+1):
            own = F.gelu(self.lift(features[level]))
            hidden = own if hidden is None else own+mm(context.restrict[level-1], hidden)
            hidden = self.down(hidden, context.graphs[level])
            saved.append(hidden)
        for level in reversed(range(self.start_level, last)):
            hidden = mm(context.prolong[level], hidden)
            hidden = F.gelu(self.merge(torch.cat((saved[level-self.start_level], hidden), 1)))
            hidden = self.up(hidden, context.graphs[level])
        return context.lift(self.output(hidden))


def harmonize_modes(modes, context, system, element_a, steps):
    """A fixed amount of current-material energy relaxation, no tolerance loop."""
    fine = context.compiled.action.fine
    diagonal_inverse = system.levels[0].diagonal_inverse[:, None]
    # Exact current-material relaxation shapes the learned modes. Unlike a
    # fixed POD dictionary, their internal material compatibility changes with A.
    for _ in range(steps):
        applied = fine.homogeneous_action(modes.T, element_a).T
        modes = modes-(2/3)*diagonal_inverse*applied
    return modes


def factor_modes(modes, context, element_a):
    """Exact current-A action, energy scaling, and tiny regularized Cholesky."""
    fine = context.compiled.action.fine
    applied = fine.homogeneous_action(modes.T, element_a).T
    scale = (modes*applied).sum(0).clamp_min(1e-20).sqrt()
    modes, applied = modes/scale, applied/scale
    gram = modes.double().T@applied.double()
    gram = .5*(gram+gram.T)
    # A fixed tiny ridge bounds coefficient sensitivity for nearly dependent
    # columns. It is not an error threshold, fallback, or stopping criterion.
    regularized = gram+1e-5*torch.eye(modes.shape[1], dtype=gram.dtype, device=gram.device)
    return modes, applied, torch.linalg.cholesky(regularized), gram


def prepare_modes(model, features, context, system, element_a, harmonic_steps=2):
    """Construct sample-dependent modes and their exact small Galerkin matrix."""
    modes = harmonize_modes(model(features, context), context, system, element_a, harmonic_steps)
    return factor_modes(modes, context, element_a)


def apply_modes(prepared, system, element_a, initial, context):
    modes, _, factor, _ = prepared
    residual = system.rhs-context.compiled.action.action(0, initial, element_a)
    coordinate = torch.cholesky_solve((modes.double().T@residual.double())[:, None], factor)[:, 0]
    return initial+modes@coordinate.to(initial.dtype)
