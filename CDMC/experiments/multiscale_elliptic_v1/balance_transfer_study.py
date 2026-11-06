"""Evaluate protected transfer against AMG and exact static condensation."""
from __future__ import annotations

import argparse
import csv
import json
import itertools
from pathlib import Path
import time

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as sla

from balance_transfer import BalanceTransfer, BalanceHierarchy
from closed_coupling_probe import matrices, metrics
from coupling_difficulty_audit import build_solver, regular_mesh, element_geometry


def geometry_matrices(n, name):
    L, G, Gx, Gy, bx, by, known, mass = matrices(n)
    if name == "base":
        return L, G, Gx, Gy, bx, by, known, mass
    x, y = np.meshgrid(np.arange(1, n-1)/(n-1),
                       np.arange(1, n-1)/(n-1), indexing="ij")
    if name == "warped":
        known = known*(1+.3*np.sin(2*np.pi*x.ravel())*np.sin(2*np.pi*y.ravel()))
    elif name == "two_cells":
        known = np.sin(2*np.pi*x.ravel())*np.sin(np.pi*y.ravel())
    elif name == "triangle":
        points, cells, boundary = regular_mesh(n)
        area, gradients, _ = element_geometry(points, cells)
        free = np.flatnonzero(~boundary)
        row = np.repeat(np.arange(len(cells)), 3)
        Gx = sp.csr_matrix((gradients[:, :, 0].ravel(), (row, cells.ravel())),
                           shape=(len(cells), len(points)))[:, free]
        Gy = sp.csr_matrix((gradients[:, :, 1].ravel(), (row, cells.ravel())),
                           shape=(len(cells), len(points)))[:, free]
    else:
        raise ValueError(name)
    dx, dy = Gx@known, Gy@known
    scale = np.hypot(dx, dy)
    regular = scale>1e-14
    bx = np.divide(dy, scale, out=np.ones_like(dy), where=regular)
    by = np.divide(-dx, scale, out=np.zeros_like(dx), where=regular)
    weight = np.sqrt(area) if name == "triangle" else np.full(len(bx), 1/(n-1))
    G = sp.diags(weight)@(sp.diags(bx)@Gx+sp.diags(by)@Gy)
    assert np.linalg.norm(G@known)/np.linalg.norm(known)<1e-12
    return L, G.tocsr(), Gx, Gy, bx, by, known, mass


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sizes", nargs="+", type=int, default=[65, 129])
    p.add_argument("--epsilons", nargs="+", type=float, default=[1e-2, 1e-4])
    p.add_argument("--span", type=int, default=8)
    p.add_argument("--maxiter", type=int, default=200)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--controls", action="store_true")
    p.add_argument("--geometries", nargs="+", choices=["base", "warped", "two_cells", "triangle"], default=["base"])
    p.add_argument("--methods", nargs="+", choices=["protected", "local", "symmetric", "multilevel", "multilevel_symmetric", "SA",
        "rootnode_evolution", "reused_isotropic_P", "frozen_shift_transfer", "exact_condensation", "direct"])
    p.add_argument("--max-coarse", type=int, default=1000)
    p.add_argument("--weak-floor", type=float, default=1e-4)
    args = p.parse_args()
    if (args.out/"audit.json").exists():
        p.error("Choose a new result directory")
    args.out.mkdir(parents=True, exist_ok=True)
    import pyamg
    rows, cases = [], []
    for n, geometry in itertools.product(args.sizes, args.geometries):
        L, G, Gx, Gy, bx, by, known, mass = geometry_matrices(n, geometry)
        H = (G.T@G).tocsr()
        methods = args.methods or (["protected", "SA", "rootnode_evolution", "reused_isotropic_P",
            "frozen_shift_transfer", "exact_condensation", "direct"] if args.controls else ["protected"])
        fast_needed = bool(set(methods)&{"local", "symmetric", "multilevel", "multilevel_symmetric"})
        transfer = (BalanceTransfer.build_local(H, L, n, span=args.span, weak_floor=args.weak_floor)
                    if fast_needed else BalanceTransfer.build(H, L, n, span=args.span))
        reference_transfer = BalanceTransfer.build(H, L, n, span=args.span) if (
            fast_needed and "protected" in methods) else transfer
        frozen = BalanceTransfer.build(H, L, n, span=args.span, epsilon_for_exact=1e-2) if "frozen_shift_transfer" in methods else None
        hierarchy = BalanceHierarchy.build(H, L, n, span=args.span, max_coarse=args.max_coarse,
            first_level=transfer, local_assembly=True, weak_floor=args.weak_floor
            ) if set(methods)&{"multilevel", "multilevel_symmetric"} else None
        template = pyamg.smoothed_aggregation_solver(L, max_coarse=16, coarse_solver="splu")
        # This diagnostic uses known only AFTER construction, never as input.
        reproduced = transfer.P@known[transfer.coarse]
        reproduction = float(np.linalg.norm(reproduced-known)/np.linalg.norm(known))
        for eps in args.epsilons:
            A = (H+eps*L).tocsr()
            start = time.perf_counter()
            direct = sla.splu(A.tocsc())
            direct_factor_ms = 1000*(time.perf_counter()-start)
            sources = {"balanced": L@known, "unit": mass,
                       "smooth": mass*(1+.4*np.sin(np.linspace(0, 5, len(mass))))}
            truths = {key: direct.solve(b) for key, b in sources.items()}
            case = dict(n=n, geometry=geometry, epsilon=eps, span=args.span,
                        reproduction_relative=reproduction,
                        transfer_setup_ms=transfer.setup_ms,
                        dofs=A.shape[0], coarse_dofs=transfer.P.shape[1],
                        transfer_nnz=int(transfer.P.nnz), methods={})
            for name in methods:
                start = time.perf_counter()
                if name in ("protected", "local", "symmetric", "multilevel", "multilevel_symmetric",
                            "exact_condensation", "frozen_shift_transfer"):
                    current = hierarchy if name.startswith("multilevel") else (
                        reference_transfer if name == "protected" else (transfer if name in (
                        "local", "symmetric") else (
                        frozen if name == "frozen_shift_transfer" else BalanceTransfer.build(
                            H, L, n, span=args.span, epsilon_for_exact=eps))))
                    solver = current.solver(eps, symmetric="symmetric" in name)
                    pre = solver.aspreconditioner()
                    extra = dict(transfer_ms=current.setup_ms, solve_setup_ms=solver.setup_ms)
                    # Standalone timing counts full transfer construction once.
                    setup_ms = current.setup_ms["total_ms"]+solver.setup_ms["total_ms"]
                    apply = solver.apply
                elif name == "direct":
                    setup_ms = direct_factor_ms
                    pre = None
                    apply = direct.solve
                    extra = {}
                else:
                    np.random.seed(20261024)
                    ml = build_solver(name, A, template)
                    ml.coarse_solver(ml.levels[-1].A, np.ones(ml.levels[-1].A.shape[0]))
                    pre = ml.aspreconditioner(cycle="V")
                    setup_ms = 1000*(time.perf_counter()-start)
                    apply = pre.matvec
                    extra = dict(operator_complexity=float(ml.operator_complexity()))
                detail = dict(setup_ms=setup_ms, **extra, rhs={})
                for rhs_name, b in sources.items():
                    truth = truths[rhs_name]
                    measure = lambda x: metrics(x, truth, A, Gx, Gy, bx, by, eps, mass)
                    history = []
                    if name == "direct":
                        start = time.perf_counter()
                        replay = direct.solve(b)
                        cg_ms = 1000*(time.perf_counter()-start)
                        reached, count = 1, 1
                        history = [measure(replay)]
                    else:
                        sla.cg(A, b, M=pre, rtol=1e-10, atol=0, maxiter=args.maxiter,
                               callback=lambda x: history.append(measure(x)))
                        reached = next((i+1 for i, h in enumerate(history)
                                        if h["field"]<=1e-3 and h["flux"]<=5e-3), None)
                        count = reached or len(history)
                        start = time.perf_counter()
                        replay, _ = sla.cg(A, b, M=pre, rtol=0, atol=0, maxiter=count)
                        cg_ms = 1000*(time.perf_counter()-start)
                    err = measure(replay)
                    assert abs(err["field"]-history[count-1]["field"])<1e-7
                    fixed, fixed_times = [], {}
                    for rounds in (1, 2, 4):
                        x = np.zeros_like(b)
                        start = time.perf_counter()
                        for _ in range(rounds):
                            x += apply(b-A@x)
                        fixed_times[rounds] = 1000*(time.perf_counter()-start)
                        fixed.append(measure(x))
                    row = dict(n=n, geometry=geometry, epsilon=eps, span=args.span, method=name, rhs=rhs_name,
                               iterations_physical=reached, iterations_run=count,
                               setup_ms=setup_ms, cg_ms=cg_ms, total_ms=setup_ms+cg_ms,
                               field_percent=100*err["field"], flux_percent=100*err["flux"],
                               K1_field_percent=100*fixed[0]["field"],
                               K1_flux_percent=100*fixed[0]["flux"],
                               K2_field_percent=100*fixed[1]["field"],
                               K2_flux_percent=100*fixed[1]["flux"],
                               K4_field_percent=100*fixed[2]["field"],
                               K4_flux_percent=100*fixed[2]["flux"],
                               K1_ms=fixed_times[1], K2_ms=fixed_times[2], K4_ms=fixed_times[4])
                    rows.append(row)
                    detail["rhs"][rhs_name] = dict(history=history, fixed=fixed, row=row)
                    print(json.dumps(row), flush=True)
                case["methods"][name] = detail
            cases.append(case)
            (args.out/"audit.json").write_text(json.dumps(dict(
                settings=vars(args)|{"out": str(args.out)},
                meaning="Zero initial guesses. Transfer constructed only from H,L and graph partition. Reference never used in construction.",
                cases=cases, rows=rows), indent=2, allow_nan=False)+"\n")
            with (args.out/"summary.csv").open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader(); writer.writerows(rows)


if __name__ == "__main__":
    main()
