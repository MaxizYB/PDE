"""Compact Geo-FNO for nodal fields on regular or irregular 2-D meshes.

The model follows Geo-FNO's geometric Fourier analysis -> FFT latent operator
-> geometric Fourier synthesis.  Unlike the historical official script, the
quadrature-weighted transforms here are normalized so the same model class can
be used on meshes with different numbers of vertices.  This is a new trained
instance, not an official-paper checkpoint or a claim of exact code parity.
"""
from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


class FourierBlock(nn.Module):
    def __init__(self, width: int, modes: int):
        super().__init__()
        self.modes = modes
        scale = 1 / (width * width)
        self.positive = nn.Parameter(scale * torch.randn(width, width, modes, modes, dtype=torch.cfloat))
        self.negative = nn.Parameter(scale * torch.randn(width, width, modes, modes, dtype=torch.cfloat))
        self.local = nn.Conv2d(width, width, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n = x.shape[-1]
        frequency = torch.fft.rfft2(x)
        output = torch.zeros_like(frequency)
        m = min(self.modes, n // 2)
        output[:, :, :m, :m] = torch.einsum(
            "bcxy,cdxy->bdxy", frequency[:, :, :m, :m], self.positive[:, :, :m, :m]
        )
        output[:, :, -m:, :m] = torch.einsum(
            "bcxy,cdxy->bdxy", frequency[:, :, -m:, :m], self.negative[:, :, :m, :m]
        )
        return F.gelu(x + self.local(x) + torch.fft.irfft2(output, s=(n, n)))


class GeoFNO2d(nn.Module):
    """One architecture for regular and irregular Darcy; weights train separately."""

    def __init__(
        self, input_dim: int = 5, width: int = 24, modes: int = 8,
        latent_resolution: int = 32, depth: int = 4,
    ):
        super().__init__()
        if latent_resolution <= 2 * modes:
            raise ValueError("latent grid must resolve the retained Fourier modes")
        self.modes = modes
        self.latent_resolution = latent_resolution
        self.input = nn.Linear(input_dim, width)
        self.deformation = nn.Sequential(
            nn.Linear(2, 32), nn.Tanh(), nn.Linear(32, 32), nn.Tanh(), nn.Linear(32, 2)
        )
        nn.init.zeros_(self.deformation[-1].weight)
        nn.init.zeros_(self.deformation[-1].bias)
        self.blocks = nn.ModuleList(FourierBlock(width, modes) for _ in range(depth))
        self.output = nn.Sequential(nn.Linear(width * 2, 64), nn.GELU(), nn.Linear(64, 1))

        k = torch.arange(-modes, modes + 1)
        wave = torch.stack(torch.meshgrid(k, k, indexing="ij"), dim=-1).reshape(-1, 2)
        self.register_buffer("wave", wave)
        n = latent_resolution
        self.register_buffer("flat_frequency", (wave[:, 0] % n) * n + wave[:, 1] % n)

    def forward(
        self, features: torch.Tensor, points: torch.Tensor, mass: torch.Tensor
    ) -> torch.Tensor:
        """features[B,N,C], points[B,N,2], mass[B,N] -> prediction[B,N]."""
        if points.ndim != 3 or features.shape[:2] != points.shape[:2]:
            raise ValueError("expected batched nodal features and coordinates")
        basis, normalized_mass = self.prepare_geometry(points, mass)
        return self.forward_prepared(features, basis, normalized_mass)

    def prepare_geometry(
        self, points: torch.Tensor, mass: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Geometry Fourier basis; cache only after weights are frozen."""
        warped = points + 0.15 * self.deformation(points)
        phase = 2 * math.pi * torch.einsum("bnd,kd->bnk", warped, self.wave.float())
        basis = torch.polar(torch.ones_like(phase), -phase)
        return basis, mass / mass.sum(dim=1, keepdim=True)

    def forward_prepared(
        self, features: torch.Tensor, basis: torch.Tensor,
        normalized_mass: torch.Tensor,
    ) -> torch.Tensor:
        """Same learned operator with the fixed-mesh geometry already compiled."""
        n = self.latent_resolution
        encoded = self.input(features)
        coefficients = torch.einsum(
            "bnc,bnk->bck",
            (encoded * normalized_mass.unsqueeze(-1)).to(basis.dtype), basis,
        )
        b, _, channels = encoded.shape
        grid_spectrum = torch.zeros(
            b, channels, n * n, dtype=coefficients.dtype, device=features.device
        )
        grid_spectrum.index_copy_(2, self.flat_frequency, coefficients * (n * n))
        latent = torch.fft.ifft2(grid_spectrum.reshape(b, channels, n, n)).real
        for block in self.blocks:
            latent = block(latent)
        output_spectrum = torch.fft.fft2(latent).reshape(b, channels, n * n) / (n * n)
        retained = output_spectrum.index_select(2, self.flat_frequency)
        decoded = torch.einsum("bck,bnk->bnc", retained, basis.conj()).real
        return self.output(torch.cat((decoded, encoded), dim=-1)).squeeze(-1)


MODEL_REGISTRY = {"geofno": GeoFNO2d}


def make_model(name: str, **kwargs) -> nn.Module:
    try:
        return MODEL_REGISTRY[name](**kwargs)
    except KeyError as exc:
        raise ValueError(f"unknown model {name}; available: {tuple(MODEL_REGISTRY)}") from exc
