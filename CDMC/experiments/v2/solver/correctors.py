"""Interchangeable post-solvers over the shared v2 Galerkin numerical kernel."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from solver.numerics import TorchSystem, correct, multiply


class CDMCSetup(nn.Module):
    """NO-defect-conditioned graph setup, called once then reused for K cycles."""

    def __init__(self, width: int = 32):
        super().__init__()
        self.lift = nn.Linear(6, width)
        self.self_layers = nn.ModuleList(nn.Linear(width, width) for _ in range(2))
        self.neighbor_layers = nn.ModuleList(nn.Linear(width, width) for _ in range(2))
        self.output = nn.Linear(width, 2)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, system: TorchSystem, initial: torch.Tensor):
        residual = system.rhs - multiply(system.matrix, initial)
        normalized = residual * system.levels[0].diagonal_inverse.sqrt()
        normalized = normalized / normalized.square().mean().sqrt().clamp_min(1e-7)
        hidden = F.gelu(self.lift(torch.cat((system.feature, normalized[:, None]), dim=1)))
        for local, neighbor in zip(self.self_layers, self.neighbor_layers):
            hidden = hidden + F.gelu(
                local(hidden) + neighbor(torch.sparse.mm(system.graph, hidden))
            )
        logits = self.output(hidden)
        return 0.5 + torch.sigmoid(logits[:, 0]), 0.5 + torch.sigmoid(logits[:, 1])


def fixed_mg(system: TorchSystem, initial: torch.Tensor, cycles: int) -> torch.Tensor:
    return correct(system, initial, cycles)


def cdmc(
    setup: CDMCSetup, system: TorchSystem, initial: torch.Tensor, cycles: int
) -> torch.Tensor:
    smooth, coarse = setup(system, initial)
    return correct(system, initial, cycles, smooth, coarse)


CORRECTOR_REGISTRY = {"mg": fixed_mg, "cdmc": cdmc}
