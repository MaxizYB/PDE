"""Darcy P1-FEM data adapter for both regular and irregular meshes."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch


@dataclass(frozen=True)
class Mesh:
    points: np.ndarray
    cells: np.ndarray
    boundary: np.ndarray
    mass: np.ndarray

    @classmethod
    def read(cls, root: Path) -> "Mesh":
        with np.load(root / "mesh.npz") as z:
            return cls(z["points"], z["cells"], z["boundary"], z["mass"])


@dataclass(frozen=True)
class DarcyProfile:
    NAME: str
    DESCRIPTION: str
    DEFAULT_MODEL: dict

    def validate(self, manifest: dict) -> None:
        if manifest.get("kind") != self.NAME or manifest.get("discretization") != "P1 FEM":
            raise ValueError(f"expected generated {self.NAME} P1-FEM Darcy dataset")


_DEFAULT_MODEL = {"width": 24, "modes": 8, "latent_resolution": 32, "depth": 4}
DATASETS = {
    "regular": DarcyProfile(
        "regular", "Unit square, Cartesian vertices split into P1 triangles", _DEFAULT_MODEL
    ),
    "irregular": DarcyProfile(
        "irregular", "Top-notched domain with conforming unstructured P1 triangles", _DEFAULT_MODEL
    ),
}


class DarcySplit:
    def __init__(self, root: Path, split: str):
        self.mesh = Mesh.read(root)
        with np.load(root / f"{split}.npz") as z:
            self.coefficient = z["coefficient"].astype(np.float32)
            self.lift = z["lift"].astype(np.float32)
            self.target = z["solution"].astype(np.float32)
            self.sample_index = z["sample_index"].copy()
        self._device_mesh = {}
        if self.coefficient.shape != self.target.shape:
            raise ValueError("coefficient and target shape mismatch")

    def __len__(self) -> int:
        return len(self.target)

    def features(self, indices, device: torch.device):
        """Physical nodal input with identical meanings on either geometry."""
        index = np.atleast_1d(indices)
        coefficient = torch.as_tensor(self.coefficient[index], device=device)
        lift = torch.as_tensor(self.lift[index], device=device)
        key = (f"cuda:{device.index if device.index is not None else torch.cuda.current_device()}"
               if device.type == "cuda" else str(device))
        if key not in self._device_mesh:
            self._device_mesh[key] = (
                torch.as_tensor(self.mesh.points, device=device),
                torch.as_tensor(self.mesh.mass, device=device),
                torch.as_tensor(self.mesh.boundary.astype(np.float32), device=device),
            )
        points0, mass0, boundary0 = self._device_mesh[key]
        points = points0.expand(len(index), -1, -1)
        mass = mass0.expand(len(index), -1)
        boundary = boundary0.expand(len(index), -1)
        features = torch.stack(
            ((coefficient - 7.5) / 4.5, lift / 0.05, points[..., 0], points[..., 1], boundary),
            dim=-1,
        )
        return features, points, mass, lift, boundary

    def target_tensor(self, indices, device: torch.device) -> torch.Tensor:
        return torch.as_tensor(self.target[np.atleast_1d(indices)], device=device)


def relative_field_error(prediction: torch.Tensor, target: torch.Tensor, mass: torch.Tensor):
    numerator = ((prediction - target).square() * mass).sum(dim=-1)
    denominator = (target.square() * mass).sum(dim=-1).clamp_min(1e-12)
    return torch.sqrt(numerator / denominator)


def infer_solution(
    model: torch.nn.Module, split: DarcySplit, indices, device: torch.device,
    scale: float,
) -> torch.Tensor:
    features, points, mass, lift, boundary = split.features(indices, device)
    correction = model(features, points, mass)
    return lift + (1 - boundary) * (scale * correction)
