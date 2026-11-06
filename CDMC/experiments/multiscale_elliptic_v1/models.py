"""A standard four-block FNO; the CDMC model is reused unmodified from v2."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class SpectralConv(nn.Module):
    def __init__(self, width, modes):
        super().__init__()
        self.modes = modes
        self.top = nn.Parameter(torch.rand(width, width, modes, modes, dtype=torch.cfloat)/width**2)
        self.bottom = nn.Parameter(torch.rand(width, width, modes, modes, dtype=torch.cfloat)/width**2)

    def forward(self, value):
        b, c, h, w = value.shape
        transformed = torch.fft.rfft2(value)
        out = torch.zeros(b, c, h, w//2+1, device=value.device, dtype=torch.cfloat)
        m = min(self.modes, h//2, w//2+1)
        out[:, :, :m, :m] = torch.einsum('bixy,ioxy->boxy', transformed[:, :, :m, :m], self.top[:, :, :m, :m])
        out[:, :, -m:, :m] = torch.einsum('bixy,ioxy->boxy', transformed[:, :, -m:, :m], self.bottom[:, :, :m, :m])
        return torch.fft.irfft2(out, s=(h, w))


class FNO(nn.Module):
    def __init__(self, width=32, modes=16, depth=4, padding=9):
        super().__init__()
        self.padding = padding
        self.lift = nn.Linear(3, width)
        self.spectral = nn.ModuleList(SpectralConv(width, modes) for _ in range(depth))
        self.point = nn.ModuleList(nn.Conv2d(width, width, 1) for _ in range(depth))
        self.project = nn.Sequential(nn.Linear(width, 128), nn.GELU(), nn.Linear(128, 1))

    def forward(self, value, coordinates):
        value = self.lift(torch.cat((value[..., None], coordinates.expand(len(value), -1, -1, -1)), -1))
        value = F.pad(value.permute(0, 3, 1, 2), (0, self.padding, 0, self.padding))
        for i, (spectral, point) in enumerate(zip(self.spectral, self.point)):
            value = spectral(value) + point(value)
            if i + 1 < len(self.point):
                value = F.gelu(value)
        if self.padding:
            value = value[..., :-self.padding, :-self.padding]
        return self.project(value.permute(0, 2, 3, 1)).squeeze(-1)
