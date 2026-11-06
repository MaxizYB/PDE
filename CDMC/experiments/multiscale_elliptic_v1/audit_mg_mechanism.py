"""Audit existing MG/mode correction mechanisms; no training or solver changes.

Reports fine-level V-cycle stages, fixed learned-space oracle projections and
actual operator-call counts. Oracle projections use test truth only offline.
This instrumented audit does not supply inference timings.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager
import csv
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from run import (DarcySplit, compiled_plan, make_example, field_metrics,
                 save_json, summary)
from solver.matrixfree_cdmc import _v_cycle, correct_matrixfree
from solver.numerics import coarsen_gates, multiply
from coupled_coarse import (CoarseContext, CoupledCoarseSetup, ModeOnlyCompiled,
                            prepare_modes, apply_modes)


@contextmanager
def count_actions(action):
    """Count level actions and fine RHS-equivalent work without changing values."""
    level_counts, fine_counts = Counter(), Counter()
    original_level = action.action
    original_fine = action.fine.homogeneous_action

    def level(level, vector, element_a):
        level_counts[str(level)] += 1
        return original_level(level, vector, element_a)

    def fine(vectors, element_a):
        fine_counts['kernel_calls'] += 1
        fine_counts['vector_equivalents'] += len(vectors)
        return original_fine(vectors, element_a)

    action.action, action.fine.homogeneous_action = level, fine
    try:
        yield level_counts, fine_counts
    finally:
        action.action, action.fine.homogeneous_action = original_level, original_fine


def cycle_stages(system, action, ae, initial):
    """Expose the finest pre/coarse/post stages of the existing fixed V-cycle."""
    gates = torch.ones_like(initial)
    smooth, coarse = coarsen_gates(system, gates, gates)
    rhs = system.rhs-action.action(0, initial, ae)
    item = system.levels[0]
    damping = item.omega*smooth[0]*item.diagonal_inverse
    correction = torch.zeros_like(initial)
    for _ in range(2):
        correction = correction+damping*(rhs-action.action(0, correction, ae))
    stages = {'fine_pre2': initial+correction}
    child_rhs = multiply(item.restriction, coarse[0]*(rhs-action.action(0, correction, ae)))
    child = _v_cycle(system, action, ae, child_rhs, smooth, coarse, None, 0, level=1)
    correction = correction+coarse[0]*multiply(item.prolongation, child)
    stages['after_recursive_coarse'] = initial+correction
    for _ in range(2):
        correction = correction+damping*(rhs-action.action(0, correction, ae))
    stages['after_fine_post2'] = initial+correction
    ad = action.action(0, correction, ae)
    alpha = (rhs@correction/(correction@ad).clamp_min(1e-12)).clamp(0, 2)
    stages['after_outer_energy_step'] = initial+alpha*correction
    return stages, float(alpha)


def projection_oracles(prepared, initial, target, mass, fine, ae):
    """Best field/energy error in this *fixed* learned span, using test truth."""
    z, _, _, gram_a = prepared
    z64, e64 = z.double(), (target-initial).double()
    weighted_z = mass.double()[:, None]*z64
    gram_m = z64.T@weighted_z
    rhs_m = weighted_z.T@e64
    # Tiny r x r eigendecomposition avoids an arbitrary ridge in the oracle.
    values, vectors = torch.linalg.eigh(0.5*(gram_m+gram_m.T))
    keep = values > values.max()*1e-12
    coords = vectors[:, keep]@((vectors[:, keep].T@rhs_m)/values[keep])
    field_oracle = initial+(z64@coords).float()
    applied_error = fine.homogeneous_action((target-initial)[None], ae)[0]
    values_a, vectors_a = torch.linalg.eigh(gram_a)
    keep_a = values_a > values_a.max()*1e-12
    rhs_a = z64.T@applied_error.double()
    ca = vectors_a[:, keep_a]@((vectors_a[:, keep_a].T@rhs_a)/values_a[keep_a])
    energy_oracle = initial+(z64@ca).float()
    return field_oracle, energy_oracle, int(keep.sum())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('data', 'no-dir', 'checkpoint', 'no-mg-checkpoint', 'out'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--count', type=int, default=8)
    p.add_argument('--device', default='cuda')
    args = p.parse_args()
    if (args.out/'audit.json').exists():
        p.error('Audit exists; select a new --out to preserve results.')
    torch.set_num_threads(2)
    device = torch.device(args.device)
    split = DarcySplit(args.data, 'test')
    compiled = compiled_plan(split, device, SimpleNamespace(coarse_size=128))
    light = ModeOnlyCompiled(split.mesh, device, shared=compiled)
    loaded = [torch.load(path, map_location=device, weights_only=False)
              for path in (args.checkpoint, args.no_mg_checkpoint)]
    models = []
    for ck in loaded:
        model = CoupledCoarseSetup(**ck['options']).to(device).eval()
        model.load_state_dict(ck['state'])
        models.append(model)
    context = CoarseContext(compiled, loaded[0]['options']['start_level'])
    with np.load(args.no_dir/'predictions.npz') as archive:
        predictions = archive['test'].copy()
    rows, action_counts = [], {}
    with torch.no_grad():
        for i in range(min(args.count, len(split))):
            system, ae, initial, target_full = make_example(split, predictions, i, compiled, device)
            target = target_full[system.free]
            metrics = {}

            def record(name, x):
                full = torch.zeros_like(target_full)
                full[system.free] = x
                metrics[name] = field_metrics(full.cpu().numpy(), split.target[i],
                                             split.coefficient[i], split.mesh)

            record('FNO', initial)
            x = initial
            for k in (1, 2):
                stages, alpha = cycle_stages(system, compiled.action, ae, x)
                for name, state in stages.items():
                    record(f'MG{k}_{name}', state)
                x = stages['after_outer_energy_step']
                metrics[f'MG{k}_after_outer_energy_step']['outer_alpha'] = alpha
            # Pure fine Jacobi control: same actual fine smoother as MG.
            fine_only = initial
            for sweep in range(1, 33):
                residual = system.rhs-compiled.action.action(0, fine_only, ae)
                fine_only = fine_only+system.levels[0].omega*system.levels[0].diagonal_inverse*residual
                if sweep in (2, 4, 8, 16, 32):
                    record(f'fine_Jacobi_{sweep}', fine_only)
            for label, start, model, ck in (
                    ('no_MG', initial, models[1], loaded[1]),
                    ('after_MG2', x, models[0], loaded[0])):
                features = context.features(system, ae, start)
                prepared = prepare_modes(model, features, context, system, ae, ck['harmonic_steps'])
                output = apply_modes(prepared, system, ae, start, context)
                record(label+'_actual', output)
                field_oracle, energy_oracle, rank = projection_oracles(
                    prepared, start, target, compiled.plan.mass, compiled.action.fine, ae)
                record(label+'_field_oracle', field_oracle)
                record(label+'_energy_oracle', energy_oracle)
                metrics[label+'_field_oracle']['mass_gram_rank'] = rank
            if i==0:
                for k in (1, 2):
                    with count_actions(compiled.action) as (levels, fine):
                        correct_matrixfree(system, compiled.action, ae, initial,
                              torch.ones_like(initial), torch.ones_like(initial), None,
                              cycles=k, coarse_steps=0)
                    action_counts[f'MG_K{k}'] = dict(level_actions=dict(levels), fine_actions=dict(fine))
                with count_actions(compiled.action) as (levels, fine):
                    features = context.features(system, ae, initial)
                    prepared = prepare_modes(models[1], features, context, system, ae, loaded[1]['harmonic_steps'])
                    apply_modes(prepared, system, ae, initial, context)
                action_counts['no_MG_modes'] = dict(level_actions=dict(levels), fine_actions=dict(fine))
                # Fine PDE equality for the light preparation is an integration invariant.
                ns, na = light.build(torch.as_tensor(split.coefficient[i], device=device),
                                    torch.as_tensor(split.lift[i], device=device))
                assert torch.allclose(ns.rhs, system.rhs, rtol=1e-5, atol=1e-7)
                assert torch.allclose(na, ae, rtol=1e-5, atol=1e-7)
            rows.append(dict(index=i, methods=metrics))
            print(f'audited {i+1}', flush=True)
    names = list(rows[0]['methods'])
    aggregated = {name: {key: summary([r['methods'][name][key] for r in rows])
                       for key in rows[0]['methods'][name]} for name in names}
    args.out.mkdir(parents=True, exist_ok=True)
    save_json(args.out/'audit.json', dict(
        summary=aggregated, samples=rows, action_counts=action_counts,
        level_sizes=compiled.action.level_sizes,
        reference_nnz=[a.nnz for a in compiled.plan.reference_levels],
        prolongation_nnz=[a.nnz for a, _ in compiled.plan.transfers],
        checkpoints=[str(args.checkpoint), str(args.no_mg_checkpoint)],
        device=torch.cuda.get_device_name(device), test_count=len(rows),
        purpose='Offline mechanism diagnosis of unchanged models; no training and no inference timing.',
        oracle='Test truth used only for best coefficients in each frozen predicted span, never for online prediction.'))
    fields = ['method', 'L2_percent', 'gradient_percent', 'energy_percent', 'flux_L2_percent']
    with (args.out/'summary.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for name in names:
            writer.writerow(dict(method=name, **{k:aggregated[name][k]['mean'] for k in fields[1:]}))


if __name__=='__main__':
    main()
