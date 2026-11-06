"""Element-wise P1 Darcy action for the compiled v2 inference path."""
from __future__ import annotations

import numpy as np
import torch


class MatrixFreeDarcy:
    """Apply the sample-specific stiffness operator without assembling it."""

    def __init__(self, mesh, device: torch.device):
        from baselines.data.darcy_v2.generate import element_geometry

        points = np.asarray(mesh.points, dtype=np.float64)
        cells = np.asarray(mesh.cells, dtype=np.int64)
        area, gradient, _ = element_geometry(points, cells)
        local = area[:, None, None] * np.einsum("tic,tjc->tij", gradient, gradient)
        free = np.flatnonzero(~mesh.boundary)
        self.cells = torch.as_tensor(cells, dtype=torch.long, device=device)
        self.flat_cells = self.cells.reshape(-1)
        self.local = torch.as_tensor(local, dtype=torch.float32, device=device)
        self.local_diag = self.local.diagonal(dim1=1, dim2=2)
        self.free = torch.as_tensor(free, dtype=torch.long, device=device)
        self.source = torch.as_tensor(mesh.mass[free], dtype=torch.float32, device=device)
        self.node_count = len(points)
        self.device = device

    def prepare(self, coefficient: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        element_a = coefficient[:, self.cells].mean(dim=-1)
        diagonal = torch.zeros(
            (coefficient.shape[0], self.node_count), device=self.device,
            dtype=coefficient.dtype,
        )
        diagonal.scatter_add_(
            1, self.flat_cells[None].expand(coefficient.shape[0], -1),
            (element_a[:, :, None] * self.local_diag[None]).reshape(coefficient.shape[0], -1),
        )
        return element_a, diagonal[:, self.free]

    def action(self, free_value: torch.Tensor, lift: torch.Tensor,
               element_a: torch.Tensor) -> torch.Tensor:
        full = lift.clone()
        full[:, self.free] = free_value
        element_value = full[:, self.cells]
        element_result = torch.einsum("tij,btj->bti", self.local, element_value)
        element_result = element_a[:, :, None] * element_result
        full_result = torch.zeros(
            (free_value.shape[0], self.node_count), device=self.device,
            dtype=free_value.dtype,
        )
        full_result.scatter_add_(
            1, self.flat_cells[None].expand(free_value.shape[0], -1),
            element_result.reshape(free_value.shape[0], -1),
        )
        return full_result[:, self.free]

    def residual(self, x, lift, element_a):
        return self.source[None] - self.action(x, lift, element_a)

    def homogeneous_action(self, x, element_a):
        return self.action(
            x, torch.zeros((x.shape[0], self.node_count), device=self.device,
                           dtype=x.dtype), element_a,
        )
