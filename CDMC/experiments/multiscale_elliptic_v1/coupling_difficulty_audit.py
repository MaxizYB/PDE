"""Locate material-coupling difficulties using mature numerical solvers.

No neural model or training. P1 FEM, tensor diffusion, zero Dirichlet data.
Controls separate material contrast, orientation, topology and hierarchy reuse.
Instrumented accuracy traces and clean timing replays are kept separate.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import scipy
import scipy.sparse as sp
import scipy.sparse.linalg as sla

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from baselines.data.darcy_v2.generate import regular_mesh, element_geometry


CASES = (
    "isotropic", "aligned_1e4", "rotated_1e2", "rotated_1e4",
    "curved_1e4", "channels_1e4", "islands_1e4",
)


def material(name, centers):
    count = len(centers)
    tensor = np.broadcast_to(np.eye(2), (count, 2, 2)).copy()
    if name in ("channels_1e4", "islands_1e4"):
        x, y = centers.T
        c = np.array([0.2, 0.4, 0.6, 0.8])
        dx = np.min(abs(x[:, None]-c), axis=1)
        dy = np.min(abs(y[:, None]-c), axis=1)
        distance = dx if name.startswith("channels") else dx*dx+dy*dy
        # Identical high/low histogram; only spatial arrangement changes.
        selected = np.argsort(distance, kind="stable")[:int(0.30*count)]
        scalar = np.ones(count)
        scalar[selected] = 1e4
        return tensor*scalar[:, None, None]
    if name == "isotropic":
        return tensor
    epsilon = 1e-2 if name == "rotated_1e2" else 1e-4
    theta = np.zeros(count) if name.startswith("aligned") else np.full(count, np.pi/8)
    if name.startswith("curved"):
        x, y = centers.T
        theta = np.pi/4+0.55*np.sin(2*np.pi*x)*np.sin(2*np.pi*y)
    cs, sn = np.cos(theta), np.sin(theta)
    tensor[:, 0, 0] = cs*cs+epsilon*sn*sn
    tensor[:, 1, 1] = sn*sn+epsilon*cs*cs
    tensor[:, 0, 1] = tensor[:, 1, 0] = (1-epsilon)*cs*sn
    return tensor


def assemble_tensor(points, cells, tensor, geometry):
    area, grads, mass = geometry
    local = np.einsum("tid,tdk,tjk->tij", grads, tensor, grads)*area[:, None, None]
    rows = np.broadcast_to(cells[:, :, None], local.shape).ravel()
    cols = np.broadcast_to(cells[:, None, :], local.shape).ravel()
    matrix = sp.coo_matrix((local.ravel(), (rows, cols)),
                          shape=(len(points), len(points))).tocsr()
    matrix.eliminate_zeros()
    return matrix, mass.copy()


def physical_errors(x, truth, free, cells, geometry, tensor):
    delta = np.zeros(len(geometry[2]))
    target = np.zeros_like(delta)
    delta[free], target[free] = x-truth, truth
    area, grads, mass = geometry
    ge = np.einsum("tid,ti->td", grads, delta[cells])
    gt = np.einsum("tid,ti->td", grads, target[cells])
    qe = np.einsum("tdk,tk->td", tensor, ge)
    qt = np.einsum("tdk,tk->td", tensor, gt)
    ratio = lambda a, b: float(np.sqrt(a/max(b, 1e-300)))
    return dict(field=ratio(mass@delta**2, mass@target**2),
                gradient=ratio(np.sum(area[:, None]*ge**2), np.sum(area[:, None]*gt**2)),
                energy=ratio(np.sum(area*np.einsum("td,td->t", ge, qe)),
                             np.sum(area*np.einsum("td,td->t", gt, qt))),
                flux=ratio(np.sum(area[:, None]*qe**2), np.sum(area[:, None]*qt**2)))


def reused_hierarchy(A, template):
    from pyamg.multilevel import MultilevelSolver
    from pyamg.relaxation.smoothing import change_smoothers
    levels = []
    current = A
    for old in template.levels[:-1]:
        level = MultilevelSolver.Level()
        level.A, level.P = current, old.P.copy()
        level.R = level.P.T.tocsr()
        levels.append(level)
        current = (level.R@current@level.P).tocsr()
    last = MultilevelSolver.Level()
    last.A = current
    levels.append(last)
    solver = MultilevelSolver(levels, coarse_solver="splu")
    change_smoothers(solver, ("gauss_seidel", {"sweep": "symmetric"}),
                    ("gauss_seidel", {"sweep": "symmetric"}))
    return solver


def build_solver(name, A, template):
    import pyamg
    options = dict(max_coarse=16, max_levels=10, coarse_solver="splu")
    if name == "reused_isotropic_P":
        return reused_hierarchy(A, template)
    if name == "SA":
        return pyamg.smoothed_aggregation_solver(A, **options)
    if name == "RS":
        return pyamg.ruge_stuben_solver(A, **options)
    if name == "rootnode_evolution":
        return pyamg.rootnode_solver(
            A, strength=("evolution", {"epsilon": 4.0, "k": 2}),
            smooth=("energy", {"krylov": "cg", "degree": 2}), **options)
    if name == "adaptive_SA":
        from pyamg.aggregation import adaptive_sa_solver
        return adaptive_sa_solver(A, num_candidates=3, candidate_iters=8,
                                  improvement_iters=1, **options)[0]
    raise ValueError(name)


def coarse_diagnosis(A, solver, probes):
    diagonal = A.diagonal()
    inv_sqrt = diagonal**-0.5
    normalized = sp.diags(inv_sqrt)@A@sp.diags(inv_sqrt)
    bound = float(np.asarray(abs(normalized).sum(axis=1)).max())
    omega = 1.5/bound
    smoothed = probes.copy()
    initial_energy = np.sum(probes*(A@probes), axis=0)
    for _ in range(32):
        smoothed -= omega*(A@smoothed)/diagonal[:, None]
    smooth_energy = np.sum(smoothed*(A@smoothed), axis=0)
    P = solver.levels[0].P
    Ac = (P.T@A@P).tocsc()
    ac_factor = sla.splu(Ac)
    projected = smoothed-P@ac_factor.solve(P.T@(A@smoothed))
    missing_energy = np.sum(projected*(A@projected), axis=0)
    orthogonality = np.linalg.norm(P.T@(A@projected))/max(
        np.linalg.norm(P.T@(A@smoothed)), 1e-30)
    return dict(
        omega=omega, smoothing_energy_amplitude_ratio=np.sqrt(smooth_energy/initial_energy).tolist(),
        coarse_miss_energy_amplitude_ratio=np.sqrt(np.maximum(missing_energy, 0)/smooth_energy).tolist(),
        exact_coarse_orthogonality=float(orthogonality),
        first_coarse_dofs=P.shape[1], first_coarse_nnz=int(Ac.nnz))


def run_case(n, name, args, template):
    points, cells, boundary = regular_mesh(n)
    geometry = element_geometry(points, cells)
    K = material(name, points[cells].mean(axis=1))
    start = time.perf_counter()
    full, load = assemble_tensor(points, cells, K, geometry)
    free = np.flatnonzero(~boundary)
    A = full[free][:, free].tocsr()
    assembly_ms = 1000*(time.perf_counter()-start)
    symmetry = float(sla.norm(A-A.T)/sla.norm(A))
    assert symmetry < 1e-12
    # Both sources are physical loads, using the same mass-lumped source rule.
    x, y = points.T
    sources = {"unit": load[free],
               "smooth": (load*(1+0.5*np.sin(2*np.pi*x)*np.cos(3*np.pi*y)))[free]}
    direct_start = time.perf_counter()
    direct = sla.splu(A.tocsc())
    direct_factor_ms = 1000*(time.perf_counter()-direct_start)
    truths = {key: direct.solve(value) for key, value in sources.items()}
    reference_residual = max(float(np.linalg.norm(A@truths[k]-v)/np.linalg.norm(v))
                             for k, v in sources.items())
    assert reference_residual < 1e-6, reference_residual
    rng = np.random.default_rng(args.seed+n)
    probes = np.column_stack([truths["unit"], rng.normal(size=(len(free), 4))])
    methods = ["SA", "RS", "rootnode_evolution", "reused_isotropic_P"]
    if args.adaptive and name in ("rotated_1e4", "curved_1e4"):
        methods.append("adaptive_SA")
    rows, details = [], {}
    for method in methods:
        np.random.seed(args.seed)
        start = time.perf_counter()
        ml = build_solver(method, A, template)
        ml.coarse_solver(ml.levels[-1].A, np.ones(ml.levels[-1].A.shape[0]))
        setup_ms = 1000*(time.perf_counter()-start)
        preconditioner = ml.aspreconditioner(cycle="V")
        diagnostic = coarse_diagnosis(A, ml, probes)
        method_info = dict(setup_ms=setup_ms, levels=[level.A.shape[0] for level in ml.levels],
                           operator_complexity=float(ml.operator_complexity()),
                           grid_complexity=float(ml.grid_complexity()), coarse_diagnosis=diagnostic,
                           rhs={})
        for rhs_name, b in sources.items():
            truth, history = truths[rhs_name], []

            def record(current):
                errors = physical_errors(current, truth, free, cells, geometry, K)
                errors["relative_residual"] = float(np.linalg.norm(b-A@current)/np.linalg.norm(b))
                errors["iteration"] = len(history)+1
                history.append(errors)

            result, info = sla.cg(A, b, x0=np.zeros_like(b), M=preconditioner,
                                  rtol=1e-8, atol=0, maxiter=args.maxiter, callback=record)
            reached = next((h["iteration"] for h in history
                            if h["field"] <= 1e-3 and h["flux"] <= 5e-3), None)
            # Clean replay: no physical metric calculations in the timer.
            count = reached if reached is not None else len(history)
            start = time.perf_counter()
            replay, _ = sla.cg(A, b, x0=np.zeros_like(b), M=preconditioner,
                               rtol=0, atol=0, maxiter=count)
            solve_ms = 1000*(time.perf_counter()-start)
            replay_errors = physical_errors(replay, truth, free, cells, geometry, K)
            check = history[count-1]
            assert abs(replay_errors["field"]-check["field"]) < 1e-8
            assert abs(replay_errors["flux"]-check["flux"]) < 1e-8
            item = dict(n=n, case=name, method=method, rhs=rhs_name, dofs=len(free),
                        assembly_ms=assembly_ms, setup_ms=setup_ms, solve_ms=solve_ms,
                        setup_solve_ms=setup_ms+solve_ms, iterations_physical=reached,
                        iterations_residual=len(history), residual_converged=(info == 0),
                        operator_complexity=float(ml.operator_complexity()),
                        field_percent=100*replay_errors["field"],
                        flux_percent=100*replay_errors["flux"])
            rows.append(item)
            method_info["rhs"][rhs_name] = dict(final_errors=physical_errors(
                result, truth, free, cells, geometry, K), history=history, clean_replay=item)
            print(json.dumps(item), flush=True)
        details[method] = method_info
    return rows, dict(case=name, n=n, assembly_ms=assembly_ms,
                      direct_factor_ms=direct_factor_ms, reference_residual=reference_residual,
                      symmetry_error=symmetry, tensor_min_eigenvalue=float(np.linalg.eigvalsh(K).min()),
                      methods=details)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", type=int, nargs="+", default=[65, 129])
    parser.add_argument("--cases", nargs="+", choices=CASES, default=list(CASES))
    parser.add_argument("--maxiter", type=int, default=160)
    parser.add_argument("--seed", type=int, default=20261024)
    parser.add_argument("--adaptive", action="store_true")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    import pyamg
    if (args.out/"audit.json").exists():
        parser.error("Output already exists; use a new directory.")
    args.out.mkdir(parents=True, exist_ok=True)
    records, cases, reference_ms = [], [], {}
    for n in args.sizes:
        points, cells, boundary = regular_mesh(n)
        geometry = element_geometry(points, cells)
        full, _ = assemble_tensor(points, cells, material("isotropic", points[cells].mean(1)), geometry)
        free = np.flatnonzero(~boundary)
        np.random.seed(args.seed)
        start = time.perf_counter()
        template = pyamg.smoothed_aggregation_solver(
            full[free][:, free].tocsr(), max_coarse=16, coarse_solver="splu")
        reference_ms[str(n)] = 1000*(time.perf_counter()-start)
        for name in args.cases:
            rows, detail = run_case(n, name, args, template)
            records.extend(rows)
            cases.append(detail)
            result = dict(
                versions=dict(numpy=np.__version__, scipy=scipy.__version__, pyamg=pyamg.__version__),
                cpu=os.uname().nodename, settings=vars(args)|{"out": str(args.out)},
                thread_environment={k: os.environ.get(k) for k in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS")},
                meaning="Pure numerical diagnosis; no NO, no neural correction. All initial guesses zero.",
                physical_target=dict(field_relative=1e-3, flux_relative=5e-3),
                timing="Single clean replay after instrumented trace; setup includes coarse factorization. No GPU timings.",
                reusable_isotropic_plan_ms=reference_ms, rows=records, cases=cases)
            (args.out/"audit.json").write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
            with (args.out/"summary.csv").open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(records[0]))
                writer.writeheader()
                writer.writerows(records)


if __name__ == "__main__":
    main()
