"""Compiled v2 CDMC: factorized Galerkin actions and a lightweight mesh plan.

The trained CDMCSetup is unchanged. Only its numerical execution backend is
replaced: sample-dependent fine operators remain element-wise, while coarse
actions are factored through the fixed sparse transfer chain.
"""
from __future__ import annotations

import time

import numpy as np
import torch
from torch.nn import functional as F

from adapters.darcy_matrixfree import MatrixFreeDarcy
from adapters.darcy_system import make_mesh_plan
from solver.numerics import (
    TorchLevel, TorchSystem, coarsen_gates, jacobi_spectral_bound, multiply,
    sparse_tensor,
)


def _clock(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return time.perf_counter()


class FactorizedGalerkin:
    """Exact A_l = R_0...R_(l-1) A_0 P_(l-1)...P_0 action."""

    def __init__(self, mesh, plan, device):
        self.fine = MatrixFreeDarcy(mesh, device)
        self.prolongations = [p for p, _, _ in plan.gpu_transfers]
        self.restrictions = [r for _, r, _ in plan.gpu_transfers]
        self.level_sizes = [len(plan.free)] + [p.shape[1] for p, _ in plan.transfers]

    def action(self, level, vector, element_a):
        fine_vector = vector
        for depth in reversed(range(level)):
            fine_vector = multiply(self.prolongations[depth], fine_vector)
        fine_result = self.fine.homogeneous_action(fine_vector[None], element_a)[0]
        for depth in range(level):
            fine_result = multiply(self.restrictions[depth], fine_result)
        return fine_result


def gates_from_implicit(setup, system, action, element_a, initial):
    """Run the trained CDMC graph setup with an exact matrix-free residual."""
    residual = system.rhs - action.action(0, initial, element_a)
    normalized = residual * system.levels[0].diagonal_inverse.sqrt()
    normalized = normalized / normalized.square().mean().sqrt().clamp_min(1e-7)
    hidden = F.gelu(setup.lift(torch.cat((system.feature, normalized[:, None]), dim=1)))
    for local, neighbor in zip(setup.self_layers, setup.neighbor_layers):
        hidden = hidden + F.gelu(
            local(hidden) + neighbor(torch.sparse.mm(system.graph, hidden))
        )
    logits = setup.output(hidden)
    return 0.5 + torch.sigmoid(logits[:, 0]), 0.5 + torch.sigmoid(logits[:, 1])


class MeshCompiledCDMC:
    """One-time geometry plan; online build avoids assembled level matrices."""

    def __init__(self, mesh, device, coarse_size, terminal_solver="pcg",
                 galerkin_drop_tol=0.0, galerkin_max_row_nnz=None,
                 aggregate_size=2, galerkin_preserve_row_sum=False,
                 interpolation_max_entries=0):
        if terminal_solver not in ("pcg", "direct"):
            raise ValueError("terminal_solver must be pcg or direct")
        self.terminal_solver = terminal_solver
        self.hierarchy_options = {
            "aggregate_size": aggregate_size,
            "reference_drop_tol": galerkin_drop_tol,
            "reference_max_row_nnz": galerkin_max_row_nnz,
            "reference_preserve_row_sum": galerkin_preserve_row_sum,
            "interpolation_max_entries": interpolation_max_entries,
        }
        started = _clock(device)
        self.plan = make_mesh_plan(
            mesh, device, coarse_size=coarse_size, retain_reference_levels=True,
            galerkin_drop_tol=galerkin_drop_tol,
            galerkin_max_row_nnz=galerkin_max_row_nnz,
            aggregate_size=aggregate_size,
            galerkin_preserve_row_sum=galerkin_preserve_row_sum,
            interpolation_max_entries=interpolation_max_entries,
        )
        plan_at = _clock(device)
        self.action = FactorizedGalerkin(mesh, self.plan, device)
        reference = self.plan.reference_levels
        if reference is None:
            raise RuntimeError("reference hierarchy was not retained")
        self.reference_omega = [
            min(0.67, 1.6 / jacobi_spectral_bound(matrix))
            for matrix in reference[:-1]
        ]
        self.reference_inverse = (
            torch.as_tensor(
                np.linalg.inv(reference[-1].toarray()), dtype=torch.float32, device=device
            ) if terminal_solver == "pcg" else None
        )
        self.reference_diagonals = [
            torch.as_tensor(matrix.diagonal(), dtype=torch.float32, device=device)
            for matrix in reference
        ]
        self.squared_restrictions = []
        self.squared_reference_denominators = []
        for level, (prolongation, _) in enumerate(self.plan.transfers[:-1]):
            restriction = sparse_tensor(
                prolongation.multiply(prolongation).T.tocsr(), device
            )
            self.squared_restrictions.append(restriction)
            self.squared_reference_denominators.append(
                multiply(restriction, self.reference_diagonals[level]).clamp_min(1e-12)
            )
        self.points_free = torch.as_tensor(
            self.plan.points_free, dtype=torch.float32, device=device
        )
        reference_at = _clock(device)
        if terminal_solver == "direct":
            from baselines.data.darcy_v2.generate import element_geometry

            width = self.action.level_sizes[-1]
            basis = torch.eye(width, dtype=torch.float32, device=device)
            for prolongation in reversed(self.action.prolongations):
                basis = multiply(prolongation, basis) if prolongation.layout == torch.strided else torch.sparse.mm(prolongation, basis)
            full_basis = torch.zeros(
                (len(mesh.points), width), dtype=torch.float32, device=device
            )
            full_basis[self.action.fine.free] = basis
            area, gradient, _ = element_geometry(mesh.points, mesh.cells)
            gradient = torch.as_tensor(gradient, dtype=torch.float32, device=device)
            cell_basis = full_basis[self.action.fine.cells]
            self.terminal_gx = torch.einsum("ti,tij->tj", gradient[:, :, 0], cell_basis)
            self.terminal_gy = torch.einsum("ti,tij->tj", gradient[:, :, 1], cell_basis)
            self.terminal_area = torch.as_tensor(
                area, dtype=torch.float32, device=device
            )
        terminal_at = _clock(device)
        self.transfers = []
        for (prolongation, _), (p_gpu, r_gpu, gate_gpu), denominator in zip(
            self.plan.transfers, self.plan.gpu_transfers, self.plan.gate_denominators,
        ):
            if prolongation.shape[0] <= 512:
                p_gpu, r_gpu = p_gpu.to_dense(), r_gpu.to_dense()
            self.transfers.append((p_gpu, r_gpu, gate_gpu, denominator))
        self.mesh_plan_ms_once = 1000 * (_clock(device) - started)
        self.mesh_plan_stage_ms = dict(self.plan.stage_ms)
        self.mesh_plan_stage_ms.update({
            "reference_auxiliary": 1000 * (reference_at - plan_at),
            "terminal_geometry": 1000 * (terminal_at - reference_at),
            "remaining": self.mesh_plan_ms_once - sum(self.plan.stage_ms.values())
                - 1000 * (terminal_at - plan_at),
        })
        self.device = device

    def build(self, coefficient, lift):
        element_a, fine_diagonal = self.action.fine.prepare(coefficient[None])
        element_a, fine_diagonal = element_a[0], fine_diagonal[0]
        rhs = self.action.fine.residual(
            torch.zeros_like(fine_diagonal)[None], lift[None], element_a[None]
        )[0]
        intermediate = []
        diagonal = fine_diagonal
        for level, (restriction, denominator) in enumerate(zip(
            self.squared_restrictions, self.squared_reference_denominators
        )):
            diagonal = (
                self.reference_diagonals[level + 1]
                * multiply(restriction, diagonal) / denominator
            )
            if level < len(self.plan.transfers) - 1:
                intermediate.append(diagonal)
        diagonals = [fine_diagonal] + intermediate + [self.reference_diagonals[-1]]
        levels = []
        for level, (p_gpu, r_gpu, gate_gpu, denominator) in enumerate(self.transfers):
            levels.append(TorchLevel(
                matrix=torch.empty(0, device=self.device),
                diagonal_inverse=diagonals[level].reciprocal(),
                prolongation=p_gpu, restriction=r_gpu,
                gate_restriction=gate_gpu, gate_denominator=denominator,
                omega=self.reference_omega[level],
            ))
        levels.append(TorchLevel(
            matrix=torch.empty(0, device=self.device),
            diagonal_inverse=diagonals[-1].reciprocal(),
        ))
        feature = torch.stack((
            coefficient[self.action.fine.free].log() / np.log(12),
            (fine_diagonal / fine_diagonal.quantile(0.5)).log(),
            self.points_free[:, 0],
            self.points_free[:, 1],
            torch.log(self.plan.mass / self.plan.mass.median()),
        ), dim=-1).float()
        system = TorchSystem(
            levels, rhs, self.plan.free, self.plan.graph, feature,
            self.plan.mass, None,
        )
        if self.terminal_solver == "direct":
            weight = self.terminal_area * element_a
            terminal = (
                self.terminal_gx.T @ (weight[:, None] * self.terminal_gx)
                + self.terminal_gy.T @ (weight[:, None] * self.terminal_gy)
            )
            terminal = 0.5 * (terminal + terminal.T)
            system.coarse_factor = torch.linalg.cholesky(terminal.double())
        return system, element_a[None]


def _v_cycle(system, action, element_a, rhs, smooth_gates, coarse_gates,
             reference_inverse, coarse_steps, smoothing_steps=2, level=0):
    if level == len(system.levels) - 1:
        if system.coarse_factor is not None:
            return torch.cholesky_solve(
                rhs.double().unsqueeze(-1), system.coarse_factor
            ).squeeze(-1).to(rhs.dtype)
        scale = element_a.mean().clamp_min(1e-8)
        x = torch.zeros_like(rhs)
        residual = rhs.clone()
        z = (reference_inverse @ residual) / scale
        direction = z.clone()
        inner = residual @ z
        for _ in range(coarse_steps):
            ad = action.action(level, direction, element_a)
            alpha = inner / (direction @ ad).clamp_min(1e-20)
            x = x + alpha * direction
            residual = residual - alpha * ad
            z = (reference_inverse @ residual) / scale
            next_inner = residual @ z
            direction = z + next_inner / inner.clamp_min(1e-20) * direction
            inner = next_inner
        return x
    item = system.levels[level]
    x = torch.zeros_like(rhs)
    damping = item.omega * smooth_gates[level] * item.diagonal_inverse
    for _ in range(smoothing_steps):
        x = x + damping * (rhs - action.action(level, x, element_a))
    remaining = rhs - action.action(level, x, element_a)
    child_rhs = multiply(item.restriction, coarse_gates[level] * remaining)
    child = _v_cycle(
        system, action, element_a, child_rhs, smooth_gates, coarse_gates,
        reference_inverse, coarse_steps, smoothing_steps, level + 1,
    )
    x = x + coarse_gates[level] * multiply(item.prolongation, child)
    for _ in range(smoothing_steps):
        x = x + damping * (rhs - action.action(level, x, element_a))
    return x


def correct_matrixfree(system, action, element_a, initial, smooth, coarse,
                       reference_inverse, cycles=2, coarse_steps=2,
                       smoothing_steps=2, return_states=False):
    if system.coarse_factor is None and coarse_steps < 1:
        raise ValueError("matrix-free terminal solve requires coarse_steps >= 1")
    smooth_gates, coarse_gates = coarsen_gates(system, smooth, coarse)
    x = initial
    states = []
    for _ in range(cycles):
        residual = system.rhs - action.action(0, x, element_a)
        direction = _v_cycle(
            system, action, element_a, residual, smooth_gates, coarse_gates,
            reference_inverse, coarse_steps, smoothing_steps,
        )
        ad = action.action(0, direction, element_a)
        alpha = (residual @ direction / (direction @ ad).clamp_min(1e-12)).clamp(0, 2)
        x = x + alpha * direction
        if return_states:
            states.append(x)
    return states if return_states else x
