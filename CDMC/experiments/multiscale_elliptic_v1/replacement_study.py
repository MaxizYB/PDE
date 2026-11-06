"""Matched diagnostics for replacing work, not adding a head after full MG.

Reuses the frozen FNO, data, transfers and P1 operator. Existing artifacts are
never overwritten. CPU Galerkin setup is timed explicitly, not hidden in solve.
"""
from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import scipy.sparse.linalg as sla
import torch

from run import (DarcySplit, compiled_plan, make_example, field_metrics,
                 save_json, summary, clock, measure, assemble, element_geometry)
from solver.numerics import sparse_tensor, multiply
from solver.matrixfree_cdmc import correct_matrixfree


class StoredAction:
    """Exact sample Galerkin matrices; never lifts coarse matvecs to the fine grid."""

    def __init__(self, compiled, coefficient):
        start = clock(compiled.device)
        mesh = compiled.action.fine
        self.fine = mesh
        self.level_sizes = compiled.action.level_sizes
        self.prolongations = compiled.action.prolongations
        self.restrictions = compiled.action.restrictions
        full, _ = assemble(compiled.mesh.points, compiled.mesh.cells, coefficient,
                           compiled.plan.geometry)
        current = full[compiled.plan.free][:, compiled.plan.free].tocsr()
        self.cpu, self.matrices = [], []
        for level in range(len(self.level_sizes)):
            current.sum_duplicates()
            current.sort_indices()
            if current.nnz > 8_000_000:
                raise RuntimeError('Sparse diagnosis safety limit exceeded')
            self.cpu.append(current)
            self.matrices.append(torch.as_tensor(current.toarray(), dtype=torch.float32,
                device=compiled.device) if current.shape[0] <= 512 else
                sparse_tensor(current, compiled.device))
            if level < len(compiled.plan.transfers):
                p, r = compiled.plan.transfers[level]
                current = (r @ current @ p).tocsr()
        self.setup_ms = 1000*(clock(compiled.device)-start)

    def action(self, level, vector, element_a):
        return multiply(self.matrices[level], vector)


def correction(system, action, ae, initial, cycles=1, nu=2, cut=None):
    """Fixed-count symmetric correction; optional exact intermediate solve is diagnostic."""
    factors = {}
    if cut is not None:
        factors[cut] = sla.splu(action.cpu[cut].tocsc(), permc_spec='MMD_AT_PLUS_A')

    def solve(rhs, level):
        if level == cut:
            result = factors[level].solve(rhs.detach().cpu().numpy().astype(np.float64))
            return torch.as_tensor(result, dtype=rhs.dtype, device=rhs.device)
        if level == len(system.levels)-1:
            return torch.cholesky_solve(rhs.double()[:, None], system.coarse_factor)[:, 0].float()
        item = system.levels[level]
        x = item.omega*item.diagonal_inverse*rhs
        for _ in range(nu-1):
            x = x+item.omega*item.diagonal_inverse*(rhs-action.action(level, x, ae))
        child = solve(multiply(item.restriction, rhs-action.action(level, x, ae)), level+1)
        x = x+multiply(item.prolongation, child)
        for _ in range(nu):
            x = x+item.omega*item.diagonal_inverse*(rhs-action.action(level, x, ae))
        return x

    x = initial
    for _ in range(cycles):
        residual = system.rhs-action.action(0, x, ae)
        direction = solve(residual, 0)
        ad = action.action(0, direction, ae)
        alpha = (residual@direction/(direction@ad).clamp_min(1e-20)).clamp(0, 2)
        x = x+alpha*direction
    return x


def additive(system, action, ae, initial, cycles=1):
    """Non-learned all-level additive control: same P, no recursive coarse matvecs."""
    x = initial
    for _ in range(cycles):
        rhs = system.rhs-action.action(0, x, ae)
        restricted = [rhs]
        for item in system.levels[:-1]:
            restricted.append(multiply(item.restriction, restricted[-1]))
        direction = torch.cholesky_solve(restricted[-1].double()[:, None], system.coarse_factor)[:, 0].float()
        for level in reversed(range(len(system.levels)-1)):
            item = system.levels[level]
            direction = multiply(item.prolongation, direction)+item.diagonal_inverse*restricted[level]
        ad = action.action(0, direction, ae)
        alpha = (rhs@direction/(direction@ad).clamp_min(1e-20)).clamp(0, 2)
        x = x+alpha*direction
    return x


def aggregate(rows):
    return {name: {key: summary([row['methods'][name][key] for row in rows])
                   for key in rows[0]['methods'][name]}
            for name in rows[0]['methods']}


def write_summary(out, result):
    save_json(out/'diagnosis.json', result)
    fields = ['method', 'L2_percent', 'gradient_percent', 'flux_L2_percent', 'energy_percent', 'apply_ms']
    with (out/'summary.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for name, values in result['summary'].items():
            writer.writerow({'method':name, **{k: values[k]['mean'] for k in fields[1:]}})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('data', 'no-dir', 'out'):
        p.add_argument('--'+name, required=True, type=Path)
    p.add_argument('--count', type=int, default=8)
    p.add_argument('--device', default='cuda')
    p.add_argument('--repeats', type=int, default=3)
    p.add_argument('--cuts', type=int, nargs='*', default=[3, 6])
    args = p.parse_args()
    if args.out.exists():
        p.error('Choose a new output directory; previous runs are immutable.')
    args.out.mkdir(parents=True)
    torch.set_num_threads(2)
    device = torch.device(args.device)
    split = DarcySplit(args.data, 'test')
    compiled = compiled_plan(split, device, SimpleNamespace(coarse_size=128))
    compiled.mesh = split.mesh
    with np.load(args.no_dir/'predictions.npz') as z:
        predictions = z['test'].copy()
    rows = []
    with torch.no_grad():
        for index in range(min(args.count, len(split))):
            system, ae, initial, target = make_example(split, predictions, index, compiled, device)
            stored = StoredAction(compiled, split.coefficient[index])
            probe = torch.sin(torch.arange(len(initial), device=device).float())
            fine_relative_difference = float((stored.action(0, probe, ae)-compiled.action.action(0, probe, ae)).norm()/compiled.action.action(0, probe, ae).norm())
            if fine_relative_difference > 2e-5:
                raise AssertionError('Stored and matrix-free fine operators disagree')
            methods = {}
            def record(name, output, elapsed):
                full = torch.zeros_like(target)
                full[system.free] = output
                metrics = field_metrics(full.cpu().numpy(), split.target[index], split.coefficient[index], split.mesh)
                metrics['apply_ms'] = elapsed
                methods[name] = metrics
            record('FNO_cached', initial, 0.)
            for cycles in (1, 2, 4):
                value, elapsed = measure(lambda: correct_matrixfree(system, compiled.action, ae, initial,
                    torch.ones_like(initial), torch.ones_like(initial), None, cycles=cycles, coarse_steps=0), device, args.repeats)
                record(f'factorized_MG_K{cycles}', value, elapsed)
                value, elapsed = measure(lambda: correction(system, stored, ae, initial, cycles), device, args.repeats)
                record(f'stored_MG_K{cycles}', value, elapsed)
            for nu in (1, 3):
                value, elapsed = measure(lambda: correction(system, stored, ae, initial, 2, nu=nu), device, args.repeats)
                record(f'stored_MG_nu{nu}_K2', value, elapsed)
            for cycles in (1, 2, 4):
                value, elapsed = measure(lambda: additive(system, stored, ae, initial, cycles), device, args.repeats)
                record(f'additive_K{cycles}', value, elapsed)
            for cut in args.cuts:
                if cut >= len(system.levels)-1:
                    continue
                started = clock(device)
                value = correction(system, stored, ae, initial, 1, cut=cut)
                record(f'exact_cut{cut}_diagnostic', value, 1000*(clock(device)-started))
            rows.append(dict(index=index, stored_setup_ms=stored.setup_ms,
                fine_operator_relative_difference=fine_relative_difference, methods=methods))
            result = dict(summary=aggregate(rows), samples=rows,
                level_sizes=compiled.action.level_sizes, level_nnz=[m.nnz for m in stored.cpu],
                mesh_plan_ms_once=compiled.mesh_plan_ms_once,
                device=torch.cuda.get_device_name(device), seed=20261027,
                timing='Cached FNO initial; apply_ms excludes per-material setup, which is reported separately. Exact-cut timing includes CPU factorization and copies and is diagnostic only.',
                objective='Determine local/coarse work and separate recursive implementation overhead from mathematical correction difficulty; no retraining.')
            write_summary(args.out, result)
            print({'index':index, 'stored_setup_ms':stored.setup_ms,
                   'methods':{k: {q:v[q] for q in ('L2_percent','flux_L2_percent','apply_ms')} for k,v in methods.items()}}, flush=True)


if __name__ == '__main__':
    main()
