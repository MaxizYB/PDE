"""Darcy P1-FEM assembly and per-sample multilevel system adapter."""
from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch

WORKSPACE = Path(__file__).resolve().parents[4]
if str(WORKSPACE) not in sys.path:
    sys.path.insert(0, str(WORKSPACE))
from baselines.data.darcy_v2.generate import assemble, element_geometry, solve_sample  # noqa: E402

from adapters.darcy import Mesh
from solver.numerics import (
    TorchLevel, TorchSystem, build_hierarchy, jacobi_spectral_bound,
    multiply, sparse_tensor,
)


@dataclass
class GpuAssemblyPlan:
    """Fixed P1 element-to-CSR scatter and boundary-elimination map."""
    cells: torch.Tensor
    element_index: torch.Tensor
    edge_index: torch.Tensor
    edge_weight: torch.Tensor
    boundary_element: torch.Tensor
    boundary_row: torch.Tensor
    boundary_node: torch.Tensor
    boundary_weight: torch.Tensor
    fine_crow: torch.Tensor
    fine_col: torch.Tensor
    load_free: torch.Tensor
    free_index: torch.Tensor
    points_free: torch.Tensor
    log_mass_free: torch.Tensor
    fine_diagonal_position: torch.Tensor


@dataclass
class MeshPlan:
    """One-time geometry and transfer plan shared by all coefficient samples."""
    geometry: tuple[np.ndarray, np.ndarray, np.ndarray]
    free: np.ndarray
    fixed: np.ndarray
    transfers: list[tuple[sp.csr_matrix, sp.csr_matrix]]
    gpu_transfers: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]
    dense_transfers: list[tuple[torch.Tensor, torch.Tensor] | None]
    hybrid_dense_limit: int | None
    gate_denominators: list[torch.Tensor]
    gpu_patterns: list[tuple[torch.Tensor, torch.Tensor]]
    gpu_assembly: GpuAssemblyPlan | None
    affine_levels: list[torch.Tensor] | None
    graph: torch.Tensor
    mass: torch.Tensor
    points_free: np.ndarray
    node_mass_free: np.ndarray
    reference_levels: list[sp.csr_matrix] | None = None
    stage_ms: dict[str, float] = field(default_factory=dict)


def compile_gpu_assembly(
    mesh: Mesh, geometry: tuple[np.ndarray, np.ndarray, np.ndarray],
    free: np.ndarray, reference: sp.csr_matrix, device: torch.device,
) -> GpuAssemblyPlan:
    """Compile the exact P1 element and Dirichlet maps once per mesh."""
    cells = np.asarray(mesh.cells, dtype=np.int64)
    area, gradients, mass = geometry
    local = np.einsum("tic,tjc->tij", gradients, gradients)
    local *= area[:, None, None]
    free_position = np.full(len(mesh.points), -1, dtype=np.int64)
    free_position[free] = np.arange(len(free))
    rows = np.repeat(np.arange(len(free)), np.diff(reference.indptr))
    cols = reference.indices
    lookup = {(int(i), int(j)): k for k, (i, j) in enumerate(zip(rows, cols))}
    element_index, edge_index, edge_weight = [], [], []
    boundary_element, boundary_row, boundary_node, boundary_weight = [], [], [], []
    for element, triangle in enumerate(cells):
        for i, node_i in enumerate(triangle):
            reduced_i = free_position[node_i]
            if reduced_i < 0:
                continue
            for j, node_j in enumerate(triangle):
                weight = local[element, i, j]
                reduced_j = free_position[node_j]
                if reduced_j >= 0:
                    element_index.append(element)
                    edge_index.append(lookup[(int(reduced_i), int(reduced_j))])
                    edge_weight.append(weight)
                else:
                    boundary_element.append(element)
                    boundary_row.append(reduced_i)
                    boundary_node.append(node_j)
                    boundary_weight.append(weight)
    diagonal_position = np.flatnonzero(rows == cols)
    if len(diagonal_position) != len(free):
        raise ValueError("fine FEM matrix has a missing diagonal")
    to_index = lambda values: torch.as_tensor(values, dtype=torch.int64, device=device)
    to_value = lambda values: torch.as_tensor(values, dtype=torch.float64, device=device)
    node_mass_free = np.asarray(mesh.mass[free], dtype=np.float32)
    return GpuAssemblyPlan(
        cells=to_index(cells),
        element_index=to_index(element_index),
        edge_index=to_index(edge_index),
        edge_weight=to_value(edge_weight),
        boundary_element=to_index(boundary_element),
        boundary_row=to_index(boundary_row),
        boundary_node=to_index(boundary_node),
        boundary_weight=to_value(boundary_weight),
        fine_crow=to_index(reference.indptr),
        fine_col=to_index(reference.indices),
        load_free=to_value(mass[free]),
        free_index=to_index(free),
        points_free=torch.as_tensor(mesh.points[free], dtype=torch.float32, device=device),
        log_mass_free=torch.as_tensor(
            np.log(node_mass_free / np.median(node_mass_free)),
            dtype=torch.float32, device=device,
        ),
        fine_diagonal_position=to_index(diagonal_position),
    )


def compiled_gpu_assembly(
    plan: GpuAssemblyPlan, coefficient: np.ndarray, lift: np.ndarray,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    coefficient_gpu = torch.as_tensor(coefficient, dtype=torch.float64, device=device)
    lift_gpu = torch.as_tensor(lift, dtype=torch.float64, device=device)
    element_coefficient = coefficient_gpu[plan.cells].mean(dim=1)
    values = torch.zeros(plan.fine_col.numel(), dtype=torch.float64, device=device)
    values.index_add_(
        0, plan.edge_index,
        element_coefficient[plan.element_index] * plan.edge_weight,
    )
    rhs = plan.load_free.clone()
    rhs.index_add_(
        0, plan.boundary_row,
        -element_coefficient[plan.boundary_element]
        * plan.boundary_weight * lift_gpu[plan.boundary_node],
    )
    diagonal = values[plan.fine_diagonal_position]
    feature = torch.stack((
        (coefficient_gpu[plan.free_index].log() / np.log(12)).float(),
        (diagonal / diagonal.quantile(0.5)).log().float(),
        plan.points_free[:, 0], plan.points_free[:, 1],
        plan.log_mass_free,
    ), dim=-1)
    matrix = torch.sparse_csr_tensor(
        plan.fine_crow, plan.fine_col, values.float(),
        size=(len(plan.free_index), len(plan.free_index)), device=device,
    )
    return matrix, rhs.float(), feature, element_coefficient.float()


def compile_affine_levels(
    mesh: Mesh, geometry: tuple[np.ndarray, np.ndarray, np.ndarray],
    free: np.ndarray, transfers: list[tuple[sp.csr_matrix, sp.csr_matrix]],
    device: torch.device,
) -> list[torch.Tensor]:
    """Offline exact element-to-Galerkin operators for a fixed P1 mesh.

    Each sparse map sends per-element coefficients to the *full* flattened
    coarse matrix.  The same element template works for arbitrary sampled
    coefficients; no reference-medium approximation is used online.
    """
    area, gradients, _ = geometry
    cells = np.asarray(mesh.cells)
    free_position = np.full(len(mesh.points), -1, dtype=np.int64)
    free_position[free] = np.arange(len(free))
    basis = sp.eye(len(free), format="csr", dtype=np.float64)
    levels = []
    for prolongation, _ in transfers:
        basis = (basis @ prolongation).tocsr()
        basis.sum_duplicates()
        n = basis.shape[1]
        rows, cols, weights = [], [], []
        for element, triangle in enumerate(cells):
            indices = free_position[triangle]
            valid = np.flatnonzero(indices >= 0)
            if len(valid) == 0:
                continue
            subbasis = basis[indices[valid]]
            support = np.unique(subbasis.indices)
            local_basis = subbasis[:, support].toarray()
            local_k = area[element] * (gradients[element] @ gradients[element].T)
            projected = local_basis.T @ local_k[np.ix_(valid, valid)] @ local_basis
            ii, jj = np.nonzero(np.abs(projected) > 1e-13)
            if len(ii):
                rows.extend((support[ii] * n + support[jj]).tolist())
                cols.extend([element] * len(ii))
                weights.extend(projected[ii, jj].tolist())
        mapping = sp.coo_matrix(
            (weights, (rows, cols)), shape=(n * n, len(cells)), dtype=np.float64,
        ).tocsr()
        mapping.sum_duplicates()
        mapping.eliminate_zeros()
        levels.append(sparse_tensor(mapping, device))
    return levels


def make_system(
    mesh: Mesh, coefficient: np.ndarray, lift: np.ndarray, device: torch.device,
    plan: MeshPlan | None = None,
    backend: str = "cpu",
    coarse_solver: str = "inverse",
) -> TorchSystem:
    if backend not in ("cpu", "gpu", "gpu_compiled", "gpu_hybrid", "gpu_affine"):
        raise ValueError(f"unknown system backend: {backend}")
    if coarse_solver not in ("inverse", "cholesky"):
        raise ValueError(f"unknown coarse solver: {coarse_solver}")
    if backend != "cpu" and device.type != "cuda":
        raise ValueError("GPU Galerkin backend requires a CUDA device")
    if plan is None:
        plan = make_mesh_plan(
            mesh, device, gpu_galerkin=backend != "cpu",
            compiled_assembly=backend in ("gpu_compiled", "gpu_hybrid", "gpu_affine"),
            hybrid_dense_limit=512 if backend in ("gpu_hybrid", "gpu_affine") else None,
            affine_compiler=backend == "gpu_affine",
        )
    if backend != "cpu" and len(plan.gpu_patterns) != len(plan.transfers):
        raise ValueError("mesh plan was not compiled for GPU Galerkin")
    if backend in ("gpu_hybrid", "gpu_affine") and plan.hybrid_dense_limit is None:
        raise ValueError("mesh plan was not compiled for hybrid coarse levels")
    if backend == "gpu_affine" and plan.affine_levels is None:
        raise ValueError("mesh plan was not compiled for affine Galerkin levels")
    free, fixed = plan.free, plan.fixed
    element_coefficient = None
    if backend in ("gpu_compiled", "gpu_hybrid", "gpu_affine"):
        if plan.gpu_assembly is None:
            raise ValueError("mesh plan was not compiled for GPU assembly")
        current, rhs_tensor, feature, element_coefficient = compiled_gpu_assembly(
            plan.gpu_assembly, coefficient, lift, device
        )
    else:
        points = np.asarray(mesh.points, dtype=np.float64)
        cells = np.asarray(mesh.cells, dtype=np.int32)
        matrix, load = assemble(
            points, cells, coefficient.astype(np.float64), plan.geometry
        )
        reduced = matrix[free][:, free].tocsr()
        rhs = load[free] - matrix[free][:, fixed] @ lift[fixed]
        current = reduced if backend == "cpu" else sparse_tensor(reduced, device)
    levels = []
    for level_index, ((p, r), (p_gpu, r_gpu, gate_gpu), gate_denominator) in enumerate(zip(
        plan.transfers, plan.gpu_transfers, plan.gate_denominators
    )):
        if backend != "cpu":
            dense_transfer = plan.dense_transfers[level_index] if backend in ("gpu_hybrid", "gpu_affine") else None
            if dense_transfer is not None:
                # Preserve the exact Galerkin product; only its storage and
                # execution change once the coarse matrix is small enough.
                if current.layout != torch.strided:
                    current = current.to_dense()
                p_dense, r_dense = dense_transfer
                diagonal = current.diagonal()
                row_absolute = current.abs().sum(dim=1)
                omega = (1.6 / (row_absolute / diagonal).max()).clamp(max=0.67)
                levels.append(TorchLevel(
                    matrix=current,
                    diagonal_inverse=diagonal.reciprocal(),
                    prolongation=p_dense,
                    restriction=r_dense,
                    gate_restriction=gate_gpu,
                    gate_denominator=gate_denominator,
                    omega=omega,
                ))
                current = (
                    multiply(plan.affine_levels[level_index], element_coefficient)
                    .reshape(p_dense.shape[1], p_dense.shape[1])
                    if backend == "gpu_affine" else r_dense @ current @ p_dense
                )
                continue
            # The transfer graph is mesh-dependent and already resident on GPU.
            # Keeping the sample-dependent Galerkin products there avoids six
            # SciPy sparse products and repeated CSR transfers per sample.
            row_index, diagonal_position = plan.gpu_patterns[level_index]
            values = current.values()
            diagonal = values[diagonal_position]
            row_absolute = torch.zeros_like(diagonal).index_add_(
                0, row_index, values.abs()
            )
            bound = (row_absolute / diagonal).max()
            omega = (1.6 / bound).clamp(max=0.67)
            levels.append(TorchLevel(
                matrix=current,
                diagonal_inverse=diagonal.reciprocal(),
                prolongation=p_gpu,
                restriction=r_gpu,
                gate_restriction=gate_gpu,
                gate_denominator=gate_denominator,
                omega=omega,
            ))
            if backend == "gpu_affine":
                current = multiply(
                    plan.affine_levels[level_index], element_coefficient
                ).reshape(p_gpu.shape[1], p_gpu.shape[1])
            else:
                current = torch.sparse.mm(
                    torch.sparse.mm(r_gpu, current), p_gpu
                )
            continue
        levels.append(TorchLevel(
            matrix=sparse_tensor(current, device),
            diagonal_inverse=torch.as_tensor(
                1 / current.diagonal(), dtype=torch.float32, device=device
            ),
            prolongation=p_gpu,
            restriction=r_gpu,
            gate_restriction=gate_gpu,
            gate_denominator=gate_denominator,
            omega=min(0.67, 1.6 / jacobi_spectral_bound(current)),
        ))
        current = (r @ current @ p).tocsr()
    if backend != "cpu":
        terminal_dense = current if current.layout == torch.strided else current.to_dense()
        levels.append(TorchLevel(
            matrix=current,
            diagonal_inverse=terminal_dense.diagonal().clone().reciprocal(),
        ))
        coarse_inverse = (
            torch.linalg.inv(terminal_dense.double())
            if coarse_solver == "inverse" else None
        )
        coarse_factor = (
            torch.linalg.cholesky(terminal_dense.double())
            if coarse_solver == "cholesky" else None
        )
    else:
        levels.append(TorchLevel(
            matrix=sparse_tensor(current, device),
            diagonal_inverse=torch.as_tensor(
                1 / current.diagonal(), dtype=torch.float32, device=device
            ),
        ))
        coarse_inverse = torch.as_tensor(
            np.linalg.inv(current.toarray()), dtype=torch.float64, device=device
        ) if coarse_solver == "inverse" else None
        coarse_factor = (
            torch.linalg.cholesky(torch.as_tensor(current.toarray(), dtype=torch.float64, device=device))
            if coarse_solver == "cholesky" else None
        )
    if backend not in ("gpu_compiled", "gpu_hybrid", "gpu_affine"):
        diag = reduced.diagonal()
        a_free = coefficient[free]
        node_mass = plan.node_mass_free
        feature = np.stack((
            np.log(a_free) / np.log(12),
            np.log(diag / np.median(diag)),
            plan.points_free[:, 0], plan.points_free[:, 1],
            np.log(node_mass / np.median(node_mass)),
        ), axis=-1).astype(np.float32)
        rhs_tensor = torch.as_tensor(rhs, dtype=torch.float32, device=device)
        feature = torch.as_tensor(feature, device=device)
    return TorchSystem(
        levels, rhs_tensor, free,
        plan.graph, feature,
        plan.mass, coarse_inverse, coarse_factor,
    )


def make_mesh_plan(
    mesh: Mesh, device: torch.device, coarse_size: int = 128,
    gpu_galerkin: bool = False,
    compiled_assembly: bool = False,
    aggregate_size: int = 2,
    hybrid_dense_limit: int | None = None,
    affine_compiler: bool = False,
    retain_reference_levels: bool = False,
    galerkin_drop_tol: float = 0.0,
    galerkin_max_row_nnz: int | None = None,
    galerkin_preserve_row_sum: bool = False,
    interpolation_max_entries: int = 0,
) -> MeshPlan:
    """Compile only mesh-dependent work once, with a constant reference medium."""
    started = time.perf_counter()
    points = np.asarray(mesh.points, dtype=np.float64)
    cells = np.asarray(mesh.cells, dtype=np.int32)
    geometry = element_geometry(points, cells)
    free = np.flatnonzero(~mesh.boundary)
    fixed = np.flatnonzero(mesh.boundary)
    reference, _ = assemble(
        points, cells, np.ones(len(points), dtype=np.float64), geometry
    )
    reference = reference[free][:, free].tocsr()
    assembled_at = time.perf_counter()
    if compiled_assembly and not gpu_galerkin:
        raise ValueError("compiled assembly requires GPU Galerkin")
    if hybrid_dense_limit is not None and (hybrid_dense_limit < 1 or not gpu_galerkin):
        raise ValueError("hybrid dense limit requires GPU Galerkin and must be positive")
    if affine_compiler and (not compiled_assembly or hybrid_dense_limit is None):
        raise ValueError("affine compiler requires compiled GPU assembly and hybrid transfers")
    hierarchy = build_hierarchy(
        reference, coarse_size=coarse_size, aggregate_size=aggregate_size,
        galerkin_drop_tol=galerkin_drop_tol,
        galerkin_max_row_nnz=galerkin_max_row_nnz,
        galerkin_preserve_row_sum=galerkin_preserve_row_sum,
        interpolation_max_entries=interpolation_max_entries,
    )
    hierarchy_at = time.perf_counter()
    reference_levels = (
        [entry[0] if isinstance(entry, tuple) else entry for entry in hierarchy]
        if retain_reference_levels else None
    )
    transfers = [(p, r) for a, p, r in hierarchy[:-1]]
    gpu_transfers = [
        (sparse_tensor(p, device), sparse_tensor(r, device),
         sparse_tensor(abs(r), device))
        for p, r in transfers
    ]
    dense_transfers = [
        (p_gpu.to_dense(), r_gpu.to_dense())
        if hybrid_dense_limit is not None and p.shape[0] <= hybrid_dense_limit else None
        for (p, _), (p_gpu, r_gpu, _) in zip(transfers, gpu_transfers)
    ]
    gate_denominators = [
        multiply(gate_gpu, torch.ones(p.shape[0], dtype=torch.float32, device=device))
        .clamp_min(1e-8)
        for (p, _), (_, _, gate_gpu) in zip(transfers, gpu_transfers)
    ]
    gpu_patterns = []
    if gpu_galerkin:
        current_gpu = sparse_tensor(reference, device)
        for p_gpu, r_gpu, _ in gpu_transfers:
            crow = current_gpu.crow_indices().cpu().numpy()
            cols = current_gpu.col_indices().cpu().numpy()
            rows = np.repeat(np.arange(current_gpu.shape[0]), np.diff(crow))
            diagonal_position = np.flatnonzero(rows == cols)
            if len(diagonal_position) != current_gpu.shape[0]:
                raise ValueError("Galerkin level has a missing diagonal")
            gpu_patterns.append((
                torch.as_tensor(rows, dtype=torch.int64, device=device),
                torch.as_tensor(diagonal_position, dtype=torch.int64, device=device),
            ))
            current_gpu = torch.sparse.mm(torch.sparse.mm(r_gpu, current_gpu), p_gpu)
    gpu_assembly = (
        compile_gpu_assembly(mesh, geometry, free, reference, device)
        if compiled_assembly else None
    )
    affine_levels = (
        compile_affine_levels(mesh, geometry, free, transfers, device)
        if affine_compiler else None
    )
    graph = abs(reference).tocsr()
    graph.setdiag(0)
    graph.eliminate_zeros()
    row_sum = np.asarray(graph.sum(axis=1)).ravel()
    graph = (sp.diags(1 / np.maximum(row_sum, 1e-12)) @ graph).tocsr()
    plan = MeshPlan(
        geometry, free, fixed, transfers, gpu_transfers, dense_transfers,
        hybrid_dense_limit, gate_denominators,
        gpu_patterns, gpu_assembly, affine_levels,
        sparse_tensor(graph, device),
        torch.as_tensor(mesh.mass[free], dtype=torch.float32, device=device),
        points[free].astype(np.float32), mesh.mass[free].astype(np.float32),
        reference_levels,
    )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    finished_at = time.perf_counter()
    plan.stage_ms = {
        "reference_geometry_assembly": 1000 * (assembled_at - started),
        "reference_hierarchy": 1000 * (hierarchy_at - assembled_at),
        "transfers_graph_device": 1000 * (finished_at - hierarchy_at),
    }
    return plan
