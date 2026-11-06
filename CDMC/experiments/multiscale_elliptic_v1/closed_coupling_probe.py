"""Probe closed directional coupling, preserving a known equilibrium mode.

This is a structure diagnostic, not a new learned solver. A = eps L + G.T G,
where G samples the directional Q1 gradient at Cartesian cell centers and L
is the standard Dirichlet five-point energy matrix. It is distinct from the
P1 tensor discretization in coupling_difficulty_audit.py.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import time

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as sla

from coupling_difficulty_audit import (
    regular_mesh, element_geometry, material, assemble_tensor,
    build_solver, coarse_diagnosis,
)


def matrices(n):
    points, cells, boundary = regular_mesh(n)
    geometry = element_geometry(points, cells)
    free = np.flatnonzero(~boundary)
    full, _ = assemble_tensor(points, cells, material("isotropic", points[cells].mean(1)), geometry)
    L = full[free][:, free].tocsr()
    h = 1/(n-1)
    i, j = np.meshgrid(np.arange(n-1), np.arange(n-1), indexing="ij")
    a = (i*n+j).ravel()
    vertices = np.column_stack((a, a+n, a+n+1, a+1))
    row = np.repeat(np.arange(len(vertices)), 4)
    dx = np.tile(np.array([-0.5, 0.5, 0.5, -0.5])/h, len(vertices))
    dy = np.tile(np.array([-0.5, -0.5, 0.5, 0.5])/h, len(vertices))
    Gx = sp.csr_matrix((dx, (row, vertices.ravel())), shape=(len(vertices), len(points)))[:, free]
    Gy = sp.csr_matrix((dy, (row, vertices.ravel())), shape=(len(vertices), len(points)))[:, free]
    cx, cy = points[vertices].mean(1).T
    ux = np.pi*np.cos(np.pi*cx)*np.sin(np.pi*cy)
    uy = np.pi*np.sin(np.pi*cx)*np.cos(np.pi*cy)
    norm = np.hypot(ux, uy)
    bx, by = uy/norm, -ux/norm
    G = (h*(sp.diags(bx)@Gx+sp.diags(by)@Gy)).tocsr()
    u = np.sin(np.pi*points[free, 0])*np.sin(np.pi*points[free, 1])
    assert np.linalg.norm(G@u)/np.linalg.norm(u)<1e-12
    return L, G, Gx, Gy, bx, by, u, geometry[2][free]


def metrics(x, target, A, Gx, Gy, bx, by, epsilon, mass):
    e = x-target
    def flux(value):
        gx, gy = Gx@value, Gy@value
        parallel = bx*gx+by*gy
        return epsilon*gx+bx*parallel, epsilon*gy+by*parallel
    ex, ey = flux(e)
    tx, ty = flux(target)
    return dict(
        field=float(np.sqrt((mass@e**2)/(mass@target**2))),
        energy=float(np.sqrt(max(e@(A@e), 0)/(target@(A@target)))),
        flux=float(np.sqrt((ex@ex+ey@ey)/(tx@tx+ty@ty))))


def primal_check(args):
    """Check physical accuracy of the ordinary P1 representation separately."""
    rows = []
    for n in args.sizes:
        points, cells, boundary = regular_mesh(n)
        geometry = element_geometry(points, cells)
        free = np.flatnonzero(~boundary)
        centers = points[cells].mean(1)
        cx, cy = centers.T
        gx = np.pi*np.cos(np.pi*cx)*np.sin(np.pi*cy)
        gy = np.pi*np.sin(np.pi*cx)*np.cos(np.pi*cy)
        norm = np.hypot(gx, gy)
        direction = np.column_stack((gy/norm, -gx/norm))
        base = np.sin(np.pi*points[:, 0])*np.sin(np.pi*points[:, 1])
        load = geometry[2]*2*np.pi**2*base
        for eps in args.epsilons:
            K = eps*np.eye(2)[None]+np.einsum("ti,tj->tij", direction, direction)
            full, _ = assemble_tensor(points, cells, K, geometry)
            A = full[free][:, free].tocsc()
            start = time.perf_counter()
            factor = sla.splu(A)
            factor_ms = 1000*(time.perf_counter()-start)
            start = time.perf_counter()
            pred = np.zeros(len(points))
            pred[free] = factor.solve(load[free])
            solve_ms = 1000*(time.perf_counter()-start)
            exact = base/eps
            mass = geometry[2]
            field = float(np.sqrt((mass@(pred-exact)**2)/(mass@exact**2)))
            ge = np.einsum("tid,ti->td", geometry[1], exact[cells])
            leakage = np.sum(geometry[0]*np.einsum("td,td->t", direction, ge)**2)
            isotropic = eps*np.sum(geometry[0][:, None]*ge**2)
            row = dict(n=n, epsilon=eps, field_percent=100*field,
                       direct_factor_ms=factor_ms, direct_solve_ms=solve_ms,
                       direct_relative_residual=float(np.linalg.norm(A@pred[free]-load[free])/np.linalg.norm(load[free])),
                       interpolated_exact_parallel_to_perpendicular_energy=float(leakage/isotropic))
            rows.append(row)
            print(json.dumps(row), flush=True)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out/"primal_check.json").write_text(json.dumps(dict(
        meaning="Exact continuum u=sin(pi*x)sin(pi*y)/eps; no iterative or learned solver. Cellwise P1 material approximation included.",
        rows=rows), indent=2, allow_nan=False)+"\n")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sizes", nargs="+", type=int, default=[65, 129])
    p.add_argument("--epsilons", nargs="+", type=float, default=[1e-2, 1e-4, 1e-6])
    p.add_argument("--maxiter", type=int, default=200)
    p.add_argument("--primal-check", action="store_true")
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    if args.primal_check:
        if (args.out/"primal_check.json").exists():
            p.error("Select a new output directory.")
        primal_check(args)
        return
    import pyamg
    if (args.out/"audit.json").exists():
        p.error("Select a new output directory.")
    args.out.mkdir(parents=True, exist_ok=True)
    rows, details = [], []
    for n in args.sizes:
        L, G, Gx, Gy, bx, by, known, mass = matrices(n)
        np.random.seed(20261024)
        template = pyamg.smoothed_aggregation_solver(L, max_coarse=16, coarse_solver="splu")
        for eps in args.epsilons:
            A = (eps*L+G.T@G).tocsr()
            # The same forcing in all eps cases; exact balance gives u/eps.
            sources = {"balanced": L@known, "unit": mass}
            direct_start = time.perf_counter()
            factor = sla.splu(A.tocsc())
            factor_ms = 1000*(time.perf_counter()-direct_start)
            truths = {key: factor.solve(b) for key, b in sources.items()}
            ref_residual = max(float(np.linalg.norm(A@truths[k]-b)/np.linalg.norm(b))
                               for k, b in sources.items())
            analytic_discrepancy = metrics(truths["balanced"], known/eps,
                                          A, Gx, Gy, bx, by, eps, mass)
            rng = np.random.default_rng(20261024+n)
            probes = np.column_stack((truths["balanced"], rng.normal(size=(len(known), 4))))
            info_case = dict(n=n, epsilon=eps, dofs=len(known), reference_residual=ref_residual,
                             analytic_discrepancy=analytic_discrepancy,
                             direct_factor_ms=factor_ms, methods={})
            for method in ("SA", "RS", "rootnode_evolution", "reused_isotropic_P", "adaptive_SA"):
                np.random.seed(20261024)
                start = time.perf_counter()
                ml = build_solver(method, A, template)
                ml.coarse_solver(ml.levels[-1].A, np.ones(ml.levels[-1].A.shape[0]))
                setup_ms = 1000*(time.perf_counter()-start)
                diagnostic = coarse_diagnosis(A, ml, probes)
                preconditioner = ml.aspreconditioner(cycle="V")
                method_detail = dict(setup_ms=setup_ms, coarse_diagnosis=diagnostic, rhs={})
                for rhs_name, b in sources.items():
                    truth = truths[rhs_name]
                    def measure(current):
                        m = metrics(current, truth, A, Gx, Gy, bx, by, eps, mass)
                        m["relative_residual"] = float(np.linalg.norm(b-A@current)/np.linalg.norm(b))
                        return m
                    history = []
                    def record(x):
                        history.append(measure(x))
                    result, status = sla.cg(A, b, M=preconditioner, maxiter=args.maxiter,
                                           rtol=1e-8, atol=0, callback=record)
                    reached = next((i+1 for i, r in enumerate(history)
                                    if r["field"]<=1e-3 and r["flux"]<=5e-3), None)
                    count = reached if reached is not None else len(history)
                    start = time.perf_counter()
                    replay, _ = sla.cg(A, b, M=preconditioner, maxiter=count, rtol=0, atol=0)
                    solve_ms = 1000*(time.perf_counter()-start)
                    errors = measure(replay)
                    assert abs(errors["field"]-history[count-1]["field"])<1e-7
                    assert abs(errors["flux"]-history[count-1]["flux"])<1e-7
                    v_states = []
                    # Four direct V cycles: a diagnostic control, no PCG.
                    ml.solve(b, x0=np.zeros_like(b), maxiter=4, tol=0, cycle="V",
                             callback=lambda x: v_states.append(measure(x)))
                    row = dict(n=n, epsilon=eps, method=method, rhs=rhs_name,
                               iterations_physical=reached, iterations_residual=len(history),
                               residual_converged=(status==0), setup_ms=setup_ms,
                               solve_ms=solve_ms, setup_solve_ms=setup_ms+solve_ms,
                               operator_complexity=float(ml.operator_complexity()),
                               field_percent=100*errors["field"], flux_percent=100*errors["flux"],
                               V2_field_percent=100*v_states[1]["field"],
                               V2_flux_percent=100*v_states[1]["flux"],
                               V4_field_percent=100*v_states[3]["field"],
                               V4_flux_percent=100*v_states[3]["flux"])
                    rows.append(row)
                    method_detail["rhs"][rhs_name] = dict(history=history, v_cycles=v_states,
                                                          clean_replay=row)
                    print(json.dumps(row), flush=True)
                info_case["methods"][method] = method_detail
            details.append(info_case)
            (args.out/"audit.json").write_text(json.dumps(dict(
                settings=vars(args)|{"out": str(args.out)}, pyamg_version=pyamg.__version__,
                meaning="Numerical closed-coupling diagnostic, no neural model, all starts zero.",
                discretization="A=eps*L+G.T*G, cell-centered Q1 directional derivative and 5-point L.",
                physical_target=dict(field_relative=1e-3, flux_relative=5e-3),
                cases=details, rows=rows), indent=2, allow_nan=False)+"\n")
            with (args.out/"summary.csv").open("w", newline="") as stream:
                writer=csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)


if __name__ == "__main__":
    main()
