"""HINTS 2-D DeepONet, matching the pinned official NumPy implementation.

Only device handling, input/output units and interpolation are adapted. The
branch and trunk layer sequence, activations and homogeneous boundary lift
are the official ones, not a new residual-corrector architecture.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class HintsDeepONet(nn.Module):
    def __init__(self, coefficient_mean, coefficient_std, solution_scale):
        super().__init__()
        layers = []
        for ci, co in zip((2, 40, 60, 100), (40, 60, 100, 180)):
            layers += [nn.Conv2d(ci, co, 3, stride=2), nn.ReLU()]
        layers += [nn.Flatten(), nn.Linear(180, 80), nn.ReLU(),
                   nn.Linear(80, 80), nn.ReLU(), nn.Linear(80, 80)]
        self.branch_net = nn.Sequential(*layers)
        self.trunk_net = nn.Sequential(nn.Linear(2, 80), nn.Tanh(),
                                       nn.Linear(80, 80), nn.Tanh(),
                                       nn.Linear(80, 80), nn.Tanh())
        self.register_buffer('coefficient_mean', torch.tensor(float(coefficient_mean)))
        self.register_buffer('coefficient_std', torch.tensor(float(coefficient_std)))
        self.register_buffer('solution_scale', torch.tensor(float(solution_scale)))

    def trunk(self, coordinates):
        x, y = coordinates.unbind(-1)
        return self.trunk_net(coordinates) * (x*(x-1)*y*(y-1))[:, None]

    def forward(self, coefficient_sensors, rhs_sensors, coordinates=None, trunk=None):
        amplitude = rhs_sensors.square().mean((-2, -1), keepdim=True).sqrt().clamp_min(1e-20)
        inputs = torch.stack(((coefficient_sensors-self.coefficient_mean)/self.coefficient_std,
                              rhs_sensors/amplitude), 1)
        branch = self.branch_net(inputs)
        if trunk is None:
            trunk = self.trunk(coordinates)
        return (branch @ trunk.T) * amplitude.flatten()[:, None] * self.solution_scale


def sensors(grid):
    """The official inference uses bilinear interpolation to 31x31 sensors."""
    return F.interpolate(grid[:, None], size=(31, 31), mode='bilinear',
                         align_corners=True)[:, 0]
