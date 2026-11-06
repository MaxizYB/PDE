"""Fixed-mesh Geo-FNO inference with unchanged weights and prediction values.

The geometry Fourier basis is constant after training. CUDA Graph captures the
same forward pass for one mesh and batch size, eliminating per-layer Python
launch overhead. Its static output is valid until the next replay.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import torch

from adapters.darcy import DarcySplit, infer_solution


@dataclass
class ResidentInputs:
    features: torch.Tensor
    lift: torch.Tensor
    boundary: torch.Tensor


class FixedMeshProposal:
    def __init__(
        self, model: torch.nn.Module, split: DarcySplit, scale: float,
        device: torch.device, capture_graph: bool = True,
        use_prepared: bool = True,
    ) -> None:
        self.model = model.eval()
        self.scale = scale
        self.device = device
        self.graph = None
        self.compile_ms_once = 0.0
        self.prepared = (use_prepared and hasattr(model, "prepare_geometry")
                         and hasattr(model, "forward_prepared"))
        if not self.prepared:
            return
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        started = time.perf_counter()
        with torch.no_grad():
            first, points, mass, lift, boundary = split.features([0], device)
            self.basis, self.normalized_mass = model.prepare_geometry(points, mass)
            self.static_features = first.clone()
            self.static_lift = lift.clone()
            self.static_boundary = boundary.clone()
            if capture_graph and device.type == "cuda":
                for _ in range(5):
                    self._forward(
                        self.static_features, self.static_lift, self.static_boundary
                    )
                torch.cuda.synchronize(device)
                self.graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(self.graph):
                    self.static_output = self._forward(
                        self.static_features, self.static_lift, self.static_boundary
                    )
                self.graph.replay()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        self.compile_ms_once = 1000 * (time.perf_counter() - started)

    def _forward(
        self, features: torch.Tensor, lift: torch.Tensor, boundary: torch.Tensor
    ) -> torch.Tensor:
        correction = self.model.forward_prepared(
            features, self.basis, self.normalized_mass
        )
        return lift + (1 - boundary) * (self.scale * correction)

    def stage(self, split: DarcySplit) -> ResidentInputs:
        """Stage an existing dataset on GPU for a throughput-only timing mode."""
        features, _, _, lift, boundary = split.features(
            list(range(len(split))), self.device
        )
        return ResidentInputs(features, lift, boundary)

    @torch.no_grad()
    def predict(
        self, split: DarcySplit, index: int,
        resident: ResidentInputs | None = None,
    ) -> torch.Tensor:
        if not self.prepared:
            return infer_solution(
                self.model, split, [index], self.device, self.scale
            )[0]
        if resident is None:
            features, _, _, lift, boundary = split.features([index], self.device)
        else:
            features = resident.features[index:index + 1]
            lift = resident.lift[index:index + 1]
            boundary = resident.boundary[index:index + 1]
        if self.graph is None:
            return self._forward(features, lift, boundary)[0]
        self.static_features.copy_(features)
        self.static_lift.copy_(lift)
        self.static_boundary.copy_(boundary)
        self.graph.replay()
        return self.static_output[0]
