"""Generate paired, physics-complete regular/irregular Darcy benchmarks.

Both geometries use the *same* P1 finite-element assembly and the same random
field sampler.  A sample index and seed identify the same parent coefficient
field in either geometry; only its restriction to the mesh changes.

Requires NumPy, SciPy, and triangle.  No ML package is needed.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from scipy.fft import dctn, idctn
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator
from scipy.ndimage import map_coordinates
from scipy.sparse.linalg import spsolve


def regular_mesh(n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if n < 5:
        raise ValueError("regular resolution must be at least 5")
    axis = np.linspace(0.0, 1.0, n)
    xx, yy = np.meshgrid(axis, axis, indexing="ij")
    points = np.stack((xx.ravel(), yy.ravel()), axis=-1)
    cells = []
    for i in range(n - 1):
        for j in range(n - 1):
            a = i * n + j
            cells.extend(((a, a + n, a + n + 1), (a, a + n + 1, a + 1)))
    boundary = np.isclose(points[:, 0], 0) | np.isclose(points[:, 0], 1)
    boundary |= np.isclose(points[:, 1], 0) | np.isclose(points[:, 1], 1)
    return points, np.asarray(cells, dtype=np.int32), boundary


def notched_mesh(
    max_area: float, notch_width: float = 0.16, notch_depth: float = 0.68
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Conforming triangle mesh of a square with a top-open rectangular notch."""
    import triangle  # kept here so regular meshes need no mesher dependency

    if not (0 < notch_width < 0.5 and 0 < notch_depth < 0.9):
        raise ValueError("invalid notch dimensions")
    left, right = 0.5 - notch_width / 2, 0.5 + notch_width / 2
    floor = 1 - notch_depth
    polygon = np.asarray(
        [(0, 0), (1, 0), (1, 1), (right, 1), (right, floor),
         (left, floor), (left, 1), (0, 1)], dtype=np.float64
    )
    segments = np.stack((np.arange(8), np.roll(np.arange(8), -1)), axis=1)
    result = triangle.triangulate(
        {"vertices": polygon, "segments": segments}, f"pq25a{max_area:.10f}Q"
    )
    if "triangles" not in result or "segments" not in result:
        raise RuntimeError("triangle failed to produce a conforming mesh")
    points = np.asarray(result["vertices"], dtype=np.float64)
    cells = np.asarray(result["triangles"], dtype=np.int32)
    boundary = np.zeros(len(points), dtype=bool)
    boundary[np.asarray(result["segments"], dtype=np.int32).ravel()] = True
    return points, cells, boundary


def element_geometry(points: np.ndarray, cells: np.ndarray):
    p = points[cells]
    det = (p[:, 1, 0] - p[:, 0, 0]) * (p[:, 2, 1] - p[:, 0, 1])
    det -= (p[:, 2, 0] - p[:, 0, 0]) * (p[:, 1, 1] - p[:, 0, 1])
    if np.any(np.abs(det) < 1e-14):
        raise ValueError("degenerate triangle")
    grads = np.empty((len(cells), 3, 2), dtype=np.float64)
    grads[:, 0, 0] = (p[:, 1, 1] - p[:, 2, 1]) / det
    grads[:, 0, 1] = (p[:, 2, 0] - p[:, 1, 0]) / det
    grads[:, 1, 0] = (p[:, 2, 1] - p[:, 0, 1]) / det
    grads[:, 1, 1] = (p[:, 0, 0] - p[:, 2, 0]) / det
    grads[:, 2, 0] = (p[:, 0, 1] - p[:, 1, 1]) / det
    grads[:, 2, 1] = (p[:, 1, 0] - p[:, 0, 0]) / det
    area = np.abs(det) / 2
    mass = np.zeros(len(points), dtype=np.float64)
    np.add.at(mass, cells.ravel(), np.repeat(area / 3, 3))
    return area, grads, mass


def assemble(
    points: np.ndarray, cells: np.ndarray, coefficient: np.ndarray,
    geometry: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
) -> tuple[sp.csr_matrix, np.ndarray]:
    """P1-FEM stiffness and unit-source load; cell coefficient is nodal mean."""
    area, grads, _ = geometry or element_geometry(points, cells)
    local = np.einsum("tic,tjc->tij", grads, grads)
    local *= (area * coefficient[cells].mean(axis=1))[:, None, None]
    rows = np.broadcast_to(cells[:, :, None], local.shape).ravel()
    cols = np.broadcast_to(cells[:, None, :], local.shape).ravel()
    matrix = sp.coo_matrix((local.ravel(), (rows, cols)), shape=(len(points),) * 2).tocsr()
    load = np.zeros(len(points), dtype=np.float64)
    np.add.at(load, cells.ravel(), np.repeat(area / 3, 3))
    return matrix, load


def parent_random_field(rng: np.random.Generator, n: int, tau: float) -> np.ndarray:
    """Neumann-basis GRF with covariance proportional to (-Delta+tau I)^-2."""
    white = rng.standard_normal((n, n))
    k = np.arange(n, dtype=np.float64) * np.pi
    scale = tau + k[:, None] ** 2 + k[None, :] ** 2
    field = idctn(dctn(white, norm="ortho") / scale, norm="ortho")
    return (field - field.mean()) / max(field.std(), 1e-12)


def sample_field(field: np.ndarray, points: np.ndarray) -> np.ndarray:
    loc = np.stack((points[:, 0], points[:, 1])) * (len(field) - 1)
    return map_coordinates(field, loc, order=1, mode="nearest")


def boundary_lift(points: np.ndarray, parameters: np.ndarray) -> np.ndarray:
    x, y = points.T
    return parameters[0] + parameters[1] * x + parameters[2] * y


def solve_sample(
    points: np.ndarray, cells: np.ndarray, boundary: np.ndarray,
    coefficient: np.ndarray, lift: np.ndarray,
    geometry: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
) -> tuple[np.ndarray, float]:
    matrix, load = assemble(points, cells, coefficient, geometry)
    free = np.flatnonzero(~boundary)
    fixed = np.flatnonzero(boundary)
    rhs = load[free] - matrix[free][:, fixed] @ lift[fixed]
    solution = lift.copy()
    solution[free] = spsolve(matrix[free][:, free].tocsc(), rhs)
    residual = matrix[free] @ solution - load[free]
    relative_residual = np.linalg.norm(residual) / max(np.linalg.norm(load[free]), 1e-12)
    return solution, float(relative_residual)


def mesh_for(args: argparse.Namespace, refinement: int = 1):
    if args.kind == "regular":
        return regular_mesh((args.resolution - 1) * refinement + 1)
    return notched_mesh(
        args.max_area / refinement**2, args.notch_width, args.notch_depth
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=("regular", "irregular"), required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--train", type=int, default=1000)
    parser.add_argument("--val", type=int, default=100)
    parser.add_argument("--test", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20260927)
    parser.add_argument("--resolution", type=int, default=85)
    parser.add_argument("--max-area", type=float, default=0.00022)
    parser.add_argument("--notch-width", type=float, default=0.16)
    parser.add_argument("--notch-depth", type=float, default=0.68)
    parser.add_argument("--field-grid", type=int, default=129)
    parser.add_argument("--tau", type=float, default=9.0)
    parser.add_argument("--a-low", type=float, default=3.0)
    parser.add_argument("--a-high", type=float, default=12.0)
    parser.add_argument("--boundary", choices=("zero", "smooth"), default="zero")
    parser.add_argument("--refine-factor", type=int, default=1)
    args = parser.parse_args()
    if min(args.train, args.val, args.test) < 1:
        parser.error("each split must contain at least one sample")
    if args.a_low <= 0 or args.a_high <= args.a_low:
        parser.error("require 0 < a-low < a-high")
    if args.refine_factor < 1:
        parser.error("refine-factor must be at least 1")
    args.out.mkdir(parents=True, exist_ok=True)

    points, cells, boundary = mesh_for(args)
    geometry = element_geometry(points, cells)
    np.savez_compressed(
        args.out / "mesh.npz", points=points.astype(np.float32), cells=cells,
        boundary=boundary, mass=geometry[2].astype(np.float32),
    )
    fine = mesh_for(args, args.refine_factor) if args.refine_factor > 1 else None
    fine_geometry = element_geometry(fine[0], fine[1]) if fine else None
    counts = {"train": args.train, "val": args.val, "test": args.test}
    max_residual = 0.0
    start = 0
    for split, count in counts.items():
        coefficients, lifts, solutions, references, residuals = [], [], [], [], []
        for index in range(start, start + count):
            rng = np.random.default_rng(args.seed + 1000003 * index)
            field = parent_random_field(rng, args.field_grid, args.tau)
            nodal = np.where(sample_field(field, points) >= 0, args.a_high, args.a_low)
            bc_parameters = (
                np.zeros(3) if args.boundary == "zero"
                else np.asarray([0.0, rng.uniform(-0.03, 0.03), rng.uniform(-0.03, 0.03)])
            )
            lift = boundary_lift(points, bc_parameters)
            solution, residual = solve_sample(
                points, cells, boundary, nodal, lift, geometry
            )
            coefficients.append(nodal.astype(np.float32))
            lifts.append(lift.astype(np.float32))
            solutions.append(solution.astype(np.float32))
            residuals.append(residual)
            if fine is not None:
                fpoints, fcells, fboundary = fine
                fnodal = np.where(
                    sample_field(field, fpoints) >= 0, args.a_high, args.a_low
                )
                fsolution, _ = solve_sample(
                    fpoints, fcells, fboundary, fnodal,
                    boundary_lift(fpoints, bc_parameters), fine_geometry,
                )
                interpolated = LinearNDInterpolator(fpoints, fsolution)(points)
                missing = np.isnan(interpolated)
                if np.any(missing):
                    interpolated[missing] = NearestNDInterpolator(
                        fpoints, fsolution
                    )(points[missing])
                references.append(interpolated.astype(np.float32))
        payload = {
            "coefficient": np.asarray(coefficients), "lift": np.asarray(lifts),
            "solution": np.asarray(solutions),
            "relative_residual": np.asarray(residuals),
            "sample_index": np.arange(start, start + count, dtype=np.int32),
        }
        if references:
            payload["fine_reference_on_mesh"] = np.asarray(references)
        np.savez_compressed(args.out / f"{split}.npz", **payload)
        max_residual = max(max_residual, float(np.max(residuals)))
        print(f"{split}: {count} samples, max relative FEM residual={max(residuals):.3e}")
        start += count
    manifest = {
        "version": 1, "equation": "-div(a grad u)=1", "discretization": "P1 FEM",
        "boundary_type": "Dirichlet", "kind": args.kind,
        "nodes": len(points), "triangles": len(cells), "boundary_nodes": int(boundary.sum()),
        "splits": counts, "seed": args.seed, "coefficient_sampling": {
            "parent_grid": args.field_grid, "tau": args.tau,
            "covariance": "proportional to (-Delta_N+tau I)^-2",
            "threshold": 0.0, "a_low": args.a_low, "a_high": args.a_high,
        }, "boundary": args.boundary, "resolution": args.resolution,
        "max_area": args.max_area, "notch_width": args.notch_width,
        "notch_depth": args.notch_depth, "refine_factor": args.refine_factor,
        "max_relative_discrete_residual": max_residual,
        "primary_target": "solution; exactly matches saved mesh and P1-FEM operator",
        "secondary_target": "fine_reference_on_mesh, only when refine-factor > 1",
    }
    (args.out / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"saved {args.out} ({len(points)} vertices, {len(cells)} triangles)")


if __name__ == "__main__":
    main()
