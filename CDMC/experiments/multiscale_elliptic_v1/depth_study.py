"""Depth study of the established filtered response-space solver.

No new backbone or PDE-specific learned head. Train the existing response
space through its actual two-cycle use; diagnose approximation, calibration,
conditioning, and reuse on the exact same discrete equations.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from polynomial_joint import (SparseCompiled, P1Norms, chebyshev, shape_response,
                              LiftedSparseAction)
from coupled_coarse import CoarseContext, CoupledCoarseSetup, apply_modes, mm
from run import (DarcySplit, clock, save_json, measure, load_no, predict, summary,
                 tensors, field_metrics, compiled_plan)
from solver.matrixfree_cdmc import correct_matrixfree


def geometric(context, rank=8):
    points = context.compiled.points_free
    pairs = sorted(((i, j) for i in range(1, 9) for j in range(1, 9)),
                   key=lambda t: (t[0]**2+t[1]**2, t))[:rank]
    values = torch.stack([torch.sin(np.pi*i*points[:, 0]) *
                          torch.sin(np.pi*j*points[:, 1]) for i, j in pairs], 1)
    return context.lift(mm(context.restrict[0], values))


def load_model(path, device):
    saved = torch.load(path, map_location=device, weights_only=False)
    model = CoupledCoarseSetup(**saved['model_options']).to(device)
    model.load_state_dict(saved['state'])
    return model, saved


def config(saved):
    return SimpleNamespace(**saved['options'])


def condition(features, mode):
    if mode == 'all':
        return features
    if mode != 'material':
        raise ValueError(mode)
    return [torch.cat((f[:, :5], torch.zeros_like(f[:, 5:])), 1) for f in features]


def cycle(x, prepared, system, ae, context, opts, already_filtered=False):
    if not already_filtered:
        x = chebyshev(system, context.compiled.action, ae, x, opts.pre, opts.lower)
    x = apply_modes(prepared, system, ae, x, context)
    return chebyshev(system, context.compiled.action, ae, x, opts.post, opts.post_lower)


def cache_data(args, split_name, count, compiled, context, opts):
    split = DarcySplit(args.data, split_name)
    with np.load(args.no_dir/'predictions.npz') as data:
        predictions = data[split_name]
    rows = []
    with torch.no_grad():
        for i in range(min(count, len(split))):
            system, ae, matrix = compiled.build(split.coefficient[i], split.lift[i])
            target = torch.as_tensor(split.target[i], device=compiled.device)
            initial = torch.as_tensor(predictions[i], device=compiled.device)[system.free]
            filtered = chebyshev(system, compiled.action, ae, initial, opts.pre, opts.lower)
            rows.append(dict(index=i, system=system, ae=ae, matrix=matrix,
                             initial=initial, filtered=filtered, target=target,
                             features=context.features(system, ae, filtered)))
    return split, rows


def train(args, device):
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    model, saved = load_model(args.checkpoint, device)
    opts = config(saved)
    split = DarcySplit(args.data, 'val')
    compiled = SparseCompiled(split.mesh, device)
    context, norms = CoarseContext(compiled, 1), P1Norms(split.mesh, device)
    started = clock(device)
    _, tr = cache_data(args, 'train', args.train_count, compiled, context, opts)
    _, va = cache_data(args, 'val', args.val_count, compiled, context, opts)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, eta_min=1e-5)
    history, best = [], float('inf')

    def loss_for(item):
        compiled.action.matrix = item['matrix']
        system, ae = item['system'], item['ae']
        prepared = shape_response(model(condition(item['features'], args.conditioning), context),
                                  context, system, ae, opts)
        x = item['filtered']
        values = []
        for k in range(args.cycles):
            x = cycle(x, prepared, system, ae, context, opts, already_filtered=(k == 0))
            values.append(norms.ratios(x, item['target'], system, ae))
        # Equal loss on all deployed cycle outputs, normalized physical metrics.
        return torch.stack(values)

    for epoch in range(args.epochs):
        tick = clock(device)
        model.train()
        train_values = []
        for index in rng.permutation(len(tr)):
            values = loss_for(tr[index])
            loss = values.square().mean().sqrt()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.)
            optimizer.step()
            train_values.append(values.detach())
        schedule.step()
        model.eval()
        with torch.no_grad():
            val = torch.stack([loss_for(item) for item in va])
        score = float(val.square().mean().sqrt())
        if not np.isfinite(score):
            raise RuntimeError('Non-finite objective; preserve failure.')
        row = dict(epoch=epoch+1,
                   train_percent=(100*torch.stack(train_values).mean(0)).tolist(),
                   val_percent=(100*val.mean(0)).tolist(), seconds=clock(device)-tick)
        history.append(row)
        if score < best:
            best = score
            torch.save(dict(state=model.state_dict(), options=saved['options'],
                            model_options=saved['model_options'], epoch=epoch+1,
                            training_cycles=args.cycles, validation=row,
                            conditioning=args.conditioning,
                            initialize=str(args.checkpoint), seed=args.seed), args.out/'model.pt')
        save_json(args.out/'training.json', dict(history=history, best=best,
                  cycles=args.cycles, train_count=len(tr), val_count=len(va),
                  conditioning=args.conditioning,
                  initialize=str(args.checkpoint), seed=args.seed,
                  total_seconds=clock(device)-started,
                  objective='RMS over K deployed cycles and field/gradient/flux ratios; same frozen NO; same 72-dimensional space; no test selection.'))
        print(json.dumps(row), flush=True)


def metrics(x, item, split, compiled):
    full = torch.zeros_like(item['target'])
    full[item['system'].free] = x
    return field_metrics(full.cpu().numpy(), split.target[item['index']],
                         split.coefficient[item['index']], split.mesh)


def oracle(prepared, filtered, target, system, context, norms, ae, mode):
    """Offline metric-best projection before postfilter. Never a deployable method."""
    z = prepared[0].double()
    error = (target[system.free]-filtered).double()
    if mode == 'field':
        weights = system.mass.double()
        gram, rhs = z.T@(weights[:, None]*z), z.T@(weights*error)
    else:
        full = torch.zeros((len(target), z.shape[1]), dtype=torch.float64, device=z.device)
        full[system.free] = z
        gradients = torch.einsum('tic,tik->tck', norms.gradient.double(), full[norms.cells])
        err_full = torch.zeros_like(target, dtype=torch.float64)
        err_full[system.free] = error
        err_grad = torch.einsum('tic,ti->tc', norms.gradient.double(), err_full[norms.cells])
        weight = norms.area.double()
        if mode == 'flux':
            weight = weight*ae.flatten().double().square()
        gram = torch.einsum('tck,tcl,t->kl', gradients, gradients, weight)
        rhs = torch.einsum('tck,tc,t->k', gradients, err_grad, weight)
    eigen, vectors = torch.linalg.eigh(.5*(gram+gram.T))
    keep = eigen > eigen[-1]*1e-10
    coefficient = vectors[:, keep]@((vectors[:, keep].T@rhs)/eigen[keep])
    return filtered+(z@coefficient).float(), int(keep.sum())


def diagnose(args, device):
    model, saved = load_model(args.checkpoint, device)
    model.eval()
    opts = config(saved)
    split0 = DarcySplit(args.data, args.split)
    compiled = SparseCompiled(split0.mesh, device)
    context, norms = CoarseContext(compiled, 1), P1Norms(split0.mesh, device)
    split, data = cache_data(args, args.split, args.count, compiled, context, opts)
    fixed = geometric(context)
    rows = []
    with torch.no_grad():
        for number, item in enumerate(data):
            compiled.action.matrix = item['matrix']
            system, ae = item['system'], item['ae']
            features = item['features']
            wrong = data[(number+1) % len(data)]['features']
            no_defect = [torch.cat((f[:, :5], torch.zeros_like(f[:, 5:])), 1) for f in features]
            bases = {'learned': model(features, context), 'fixed_sine': fixed,
                     'other_material_space': model(wrong, context),
                     'material_only_space': model(no_defect, context)}
            methods = {'FNO': metrics(item['initial'], item, split, compiled),
                       'prefilter': metrics(item['filtered'], item, split, compiled)}
            details = {}
            for name, basis in bases.items():
                prepared = shape_response(basis, context, system, ae, opts)
                projected = apply_modes(prepared, system, ae, item['filtered'], context)
                error = item['target'][system.free]-item['filtered']
                remain = item['target'][system.free]-projected
                energy_before = error@mm(item['matrix'], error[:, None])[:, 0]
                energy_after = remain@mm(item['matrix'], remain[:, None])[:, 0]
                details[name] = dict(dimension=prepared[0].shape[1],
                    captured_A_energy_fraction=float(1-energy_after/energy_before),
                    gram_eigenvalues=torch.linalg.eigvalsh(prepared[3]).cpu().tolist())
                methods[name+'_projection'] = metrics(projected, item, split, compiled)
                x = item['filtered']
                for k in range(1, 5):
                    x = cycle(x, prepared, system, ae, context, opts, already_filtered=(k == 1))
                    methods[name+f'_K{k}'] = metrics(x, item, split, compiled)
                if name in ('learned', 'fixed_sine'):
                    for objective in ('field', 'gradient', 'flux'):
                        best, rank = oracle(prepared, item['filtered'], item['target'],
                                            system, context, norms, ae, objective)
                        methods[name+'_oracle_'+objective] = metrics(best, item, split, compiled)
                        details[name]['rank_'+objective] = rank
            coef = split.coefficient[item['index']]
            rows.append(dict(index=item['index'], sample_index=int(split.sample_index[item['index']]),
                             contrast=float(coef.max()/coef.min()), methods=methods, details=details))
            save_json(args.out/'diagnosis.json', dict(samples=rows, split=args.split,
                checkpoint=str(args.checkpoint),
                explanation='Shuffled features are a sensitivity intervention, not a trained baseline; all bases reshaped/recalibrated with the correct current A. Oracles use truth offline before postfilter.'))
            print(json.dumps(dict(index=item['index'], captured={k:v['captured_A_energy_fraction'] for k,v in details.items()})), flush=True)


def evaluate(args, device):
    model, saved = load_model(args.checkpoint, device)
    old, old_saved = load_model(args.original, device)
    model.eval(); old.eval()
    material = None
    if args.material_checkpoint:
        material, material_saved = load_model(args.material_checkpoint, device)
        material.eval()
    opts = config(saved)
    split = DarcySplit(args.data, args.split)
    base = compiled_plan(split, device, SimpleNamespace(coarse_size=128))
    compiled = SparseCompiled(split.mesh, device, shared=base)
    context = CoarseContext(compiled, 1)
    no, stats, _ = load_no(args.no_dir, device)
    a, y, xy, _, mask = tensors(split, device)
    fixed, fixed16 = geometric(context), geometric(context, 16)
    rows = []
    with torch.no_grad():
        for i in range(min(args.count, len(split))):
            names = ['FNO', 'MG_K2', 'MG_K4']+[f'{n}_K{k}' for n in ('original', args.candidate_name, 'fixed_sine') for k in (1, 2, 4)]
            names += [f'MG_zero_K{k}' for k in (2, 4, 8)]
            names += [f'{n}_K{k}' for n in ('union144', 'fixed144') for k in (2, 4)]
            if material is not None:
                names += [f'material_only_K{k}' for k in (1, 2, 4)]
            if args.methods:
                if set(args.methods)-set(names):
                    raise ValueError('Unavailable method requested')
                names = args.methods
            def online(name):
                full = (torch.zeros_like(y[i]).flatten() if name.startswith('MG_zero_') else
                        predict(no, a[i:i+1], xy, mask, stats).flatten())
                if name == 'FNO':
                    return full
                system, ae, _ = compiled.build(a[i].flatten(), torch.zeros_like(full))
                if name.startswith('MG_'):
                    mg_system, mg_ae = base.build(a[i].flatten(), torch.zeros_like(full))
                    action = LiftedSparseAction(base.action, compiled.action.matrix)
                    x = correct_matrixfree(mg_system, action, mg_ae, full[mg_system.free],
                        torch.ones_like(full[mg_system.free]), torch.ones_like(full[mg_system.free]),
                        None, cycles=int(name[-1]), coarse_steps=0)
                    result = torch.zeros_like(full)
                    result[mg_system.free] = x
                    return result
                filtered = chebyshev(system, compiled.action, ae, full[system.free], opts.pre, opts.lower)
                if name.startswith('fixed144'):
                    basis = fixed16
                elif name.startswith('fixed_sine'):
                    basis = fixed
                else:
                    if name.startswith('original'):
                        network, metadata = old, old_saved
                    elif name.startswith('material_only'):
                        network, metadata = material, material_saved
                    else:
                        network, metadata = model, saved
                    features = condition(context.features(system, ae, filtered),
                                         metadata.get('conditioning', 'all'))
                    basis = network(features, context)
                    if name.startswith('union144'):
                        basis = torch.cat((fixed, basis), 1)
                prepared = shape_response(basis, context, system, ae, opts)
                x = filtered
                for k in range(int(name[-1])):
                    x = cycle(x, prepared, system, ae, context, opts, already_filtered=(k == 0))
                result = torch.zeros_like(full)
                result[system.free] = x
                return result
            methods = {}
            for name in names:
                output, elapsed = measure(lambda: online(name), device, args.repeats)
                methods[name] = dict(field_metrics(output.cpu().numpy(), split.target[i],
                                      split.coefficient[i], split.mesh), total_ms=elapsed)
            rows.append(dict(index=i, sample_index=int(split.sample_index[i]), methods=methods))
            aggregated = {n:{k:summary([r['methods'][n][k] for r in rows]) for k in methods[n]} for n in names}
            save_json(args.out/'evaluation.json', dict(samples=rows, summary=aggregated,
                split=args.split, checkpoint=str(args.checkpoint), original=str(args.original),
                device=torch.cuda.get_device_name(device), seed=args.seed,
                timing='Batch1; fresh frozen FNO (omitted for MG_zero), material fine matrix, space and factor per call; one warmup then synchronized wall-clock median; mesh plan separate.',
                repeats=args.repeats, mesh_plan_ms_once=base.mesh_plan_ms_once,
                shared_plan_attachment_ms=compiled.mesh_plan_ms_once,
                assembly_plan_ms_once=compiled.assembly_plan_ms_once,
                options=saved['options']))
            keys = ['L2_percent', 'gradient_percent', 'flux_L2_percent', 'energy_percent', 'total_ms']
            with (args.out/'summary.csv').open('w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=['method', 'n']+keys)
                writer.writeheader()
                for name in names:
                    writer.writerow(dict(method=name, n=len(rows), **{k:aggregated[name][k]['mean'] for k in keys}))
            print(json.dumps(dict(index=i, methods={n:{k:methods[n][k] for k in keys} for n in names})), flush=True)


def transfer(args, device):
    """Frozen-weight contrast transfer, followed by a same-material multi-RHS audit.

    Each altered coefficient is quantized BEFORE its own aligned reference solve.
    For unseen RHS, all corrections start from zero; the constant-source NO is
    used only once to construct the material's reusable space, not as a purported
    general-source neural predictor. This is a reuse test, not a new NO benchmark.
    """
    from scipy.sparse.linalg import splu
    from baselines.data.darcy_v2.generate import assemble, element_geometry
    model, saved = load_model(args.checkpoint, device)
    model.eval()
    opts = config(saved)
    split = DarcySplit(args.data, args.split)
    compiled = SparseCompiled(split.mesh, device)
    context, norms = CoarseContext(compiled, 1), P1Norms(split.mesh, device)
    no, stats, _ = load_no(args.no_dir, device)
    _, _, xy, _, mask = tensors(split, device)
    fixed, fixed16 = geometric(context), geometric(context, 16)
    geometry = element_geometry(split.mesh.points, split.mesh.cells)
    free = ~split.mesh.boundary
    pts = split.mesh.points
    free_idx = np.flatnonzero(free)
    sources = {'constant': np.ones(len(pts)),
               'left_source': 1+.8*np.cos(2*np.pi*pts[:, 0]),
               'signed_source': np.sin(2*np.pi*pts[:, 0])*np.sin(np.pi*pts[:, 1])}
    rows = []
    for i in range(min(args.count, len(split))):
        for power in (.75, 1., 1.25):
            coefficient = (split.coefficient[i].astype('float64')**power).astype('float32')
            matrix_cpu, rhs_cpu = assemble(split.mesh.points, split.mesh.cells,
                                           coefficient.astype('float64'), geometry)
            af = matrix_cpu[free][:, free].tocsc()
            factor = splu(af)
            coef_tensor = torch.as_tensor(coefficient, device=device)
            with torch.no_grad():
                full = predict(no, coef_tensor.reshape(1, *mask.shape), xy, mask, stats).flatten()
                system, ae, _ = compiled.build(coef_tensor, torch.zeros_like(full))
                base_rhs = system.rhs
                filtered = chebyshev(system, compiled.action, ae, full[system.free], opts.pre, opts.lower)
                learned_basis = model(condition(context.features(system, ae, filtered),
                                                saved.get('conditioning', 'all')), context)
                prepared = shape_response(learned_basis, context, system, ae, opts)
                sine = shape_response(fixed, context, system, ae, opts)
                union = shape_response(torch.cat((fixed, learned_basis), 1), context, system, ae, opts)
                sine144 = shape_response(fixed16, context, system, ae, opts)
                for source, values in sources.items():
                    # Consistent nodal-source P1 load: element local M*f, not lumping.
                    cells = split.mesh.cells
                    local = geometry[0][:, None]/12*(values[cells].sum(1)[:, None]+values[cells])
                    rhs = np.bincount(cells.ravel(), weights=local.ravel(), minlength=len(pts))[free]
                    if source == 'constant' and not np.allclose(rhs, rhs_cpu[free], rtol=2e-6, atol=1e-12):
                        raise AssertionError('Source quadrature differs from original P1 assembly')
                    truth = factor.solve(rhs)
                    residual = float(np.linalg.norm(af@truth-rhs)/np.linalg.norm(rhs))
                    target = torch.zeros_like(full)
                    target[system.free] = torch.as_tensor(truth, device=device, dtype=torch.float32)
                    rhs_t = torch.as_tensor(rhs, device=device, dtype=torch.float32)
                    system.rhs = rhs_t
                    initial = full[system.free] if source == 'constant' else torch.zeros_like(rhs_t)
                    methods = {}
                    candidates = {'learned_reused': prepared, 'fixed_sine': sine,
                                  'union144_reused': union, 'fixed144': sine144, 'filter_only': None}
                    for name, space in candidates.items():
                        def solve():
                            x = initial
                            for _ in range(2):
                                x = chebyshev(system, compiled.action, ae, x, opts.pre, opts.lower)
                                if space is not None:
                                    x = apply_modes(space, system, ae, x, context)
                                x = chebyshev(system, compiled.action, ae, x, opts.post, opts.post_lower)
                            return x
                        output, apply_ms = measure(solve, device, args.repeats)
                        ratios = (100*norms.ratios(output, target, system, ae)).cpu().tolist()
                        methods[name] = dict(zip(('L2_percent','gradient_percent','flux_L2_percent'), ratios),
                                             apply_ms=apply_ms)
                    baseline = (100*norms.ratios(initial, target, system, ae)).cpu().tolist()
                    rows.append(dict(index=i, sample_index=int(split.sample_index[i]), power=power,
                        source=source, contrast=float(coefficient.max()/coefficient.min()),
                        reference_relative_residual=residual, initial_percent=baseline, methods=methods))
                    system.rhs = base_rhs
            save_json(args.out/'transfer.json', dict(samples=rows, split=args.split,
                checkpoint=str(args.checkpoint),
                protocol='No retraining. Constant-source reference regenerated from stored transformed coefficient. Space made once from this coefficient and the constant-source FNO, then reused. New loads start from zero; timings are apply-only with material/space/LU-reference setup excluded, not full online times.'))
            print(json.dumps(dict(index=i, power=power, results=rows[-3:])), flush=True)


def spectral(args, device):
    """Offline exact-current-material slow eigenspace, never used in training."""
    import scipy.sparse as sp
    from scipy.sparse.linalg import eigsh
    from baselines.data.darcy_v2.generate import assemble, element_geometry
    model, saved = load_model(args.checkpoint, device)
    model.eval()
    opts = config(saved)
    split0 = DarcySplit(args.data, args.split)
    compiled = SparseCompiled(split0.mesh, device)
    context = CoarseContext(compiled, 1)
    split, data = cache_data(args, args.split, args.count, compiled, context, opts)
    fixed = geometric(context)
    geometry = element_geometry(split.mesh.points, split.mesh.cells)
    free = ~split.mesh.boundary
    rows = []
    with torch.no_grad():
        for item in data:
            index = item['index']
            cpu, _ = assemble(split.mesh.points, split.mesh.cells,
                              split.coefficient[index].astype('float64'), geometry)
            cpu = cpu[free][:, free].tocsc()
            diag = cpu.diagonal()
            eigen, vectors = eigsh(cpu, k=16, M=sp.diags(diag), sigma=0., which='LM',
                                    tol=1e-9, v0=np.ones(len(diag)))
            eigen_residual = np.linalg.norm(cpu@vectors-diag[:, None]*vectors*eigen, axis=0)
            eigen_residual /= np.linalg.norm(cpu@vectors, axis=0).clip(1e-30)
            compiled.action.matrix = item['matrix']
            system, ae = item['system'], item['ae']
            error0 = (item['target'][system.free]-item['initial']).double().cpu().numpy()
            error1 = (item['target'][system.free]-item['filtered']).double().cpu().numpy()
            energy0, energy1 = error0@(cpu@error0), error1@(cpu@error1)
            coordinates0, coordinates1 = vectors.T@(diag*error0), vectors.T@(diag*error1)
            captures, actual_captures = {}, {}
            v = torch.as_tensor(vectors, device=device, dtype=torch.float64)
            av = torch.as_tensor(cpu@vectors, device=device, dtype=torch.float64)
            for name, basis in [('learned', model(item['features'], context)), ('fixed_sine', fixed)]:
                prepared = shape_response(basis, context, system, ae, opts)
                z, factor = prepared[0].double(), prepared[2]
                c = torch.cholesky_solve(z.T@av, factor)
                error = v-z@c
                # CPU A used for accurate eigenvector energy; GPU A only calibrates.
                remain = error.cpu().numpy()
                ratio = np.einsum('ik,ik->k', remain, cpu@remain)/eigen
                captures[name] = (1-ratio).tolist()
                actual_rhs = z.T@torch.as_tensor(cpu@error1, device=device, dtype=torch.float64)
                actual_c = torch.cholesky_solve(actual_rhs[:, None], factor)[:, 0]
                actual_remainder = error1-(z@actual_c).cpu().numpy()
                actual_captures[name] = float(1-actual_remainder@(cpu@actual_remainder)/energy1)
            rows.append(dict(index=index, eigenvalues=eigen.tolist(),
                max_eigen_relative_residual=float(eigen_residual.max()),
                slow16_energy_fraction_initial=float(np.sum(eigen*coordinates0**2)/energy0),
                slow16_energy_fraction_filtered=float(np.sum(eigen*coordinates1**2)/energy1),
                total_prefilter_energy_ratio=float(energy1/energy0),
                actual_filtered_error_energy_captured=actual_captures,
                fraction_of_each_slow_eigenmode_captured=captures))
            save_json(args.out/'spectral.json', dict(samples=rows, split=args.split,
                checkpoint=str(args.checkpoint),
                definition='A v=lambda D v, v^T D v=1. Sparse shift-invert eigenanalysis is offline diagnosis only. Learned task-space need not approximate every slow eigenmode.'))
            print(json.dumps(rows[-1]), flush=True)


def profile(args, device):
    model, saved = load_model(args.checkpoint, device)
    model.eval()
    opts = config(saved)
    split = DarcySplit(args.data, args.split)
    compiled = SparseCompiled(split.mesh, device)
    context = CoarseContext(compiled, 1)
    no, stats, _ = load_no(args.no_dir, device)
    a, _, xy, _, mask = tensors(split, device)
    fixed = geometric(context)
    rows = []
    with torch.no_grad():
        for i in range(min(args.count, len(split))):
            full, no_ms = measure(lambda: predict(no, a[i:i+1], xy, mask, stats).flatten(), device, args.repeats)
            (system, ae, _), build_ms = measure(lambda: compiled.build(a[i].flatten(), torch.zeros_like(full)), device, args.repeats)
            filtered, pre_ms = measure(lambda: chebyshev(system, compiled.action, ae, full[system.free], opts.pre, opts.lower), device, args.repeats)
            features, features_ms = measure(lambda: context.features(system, ae, filtered), device, args.repeats)
            learned, graph_ms = measure(lambda: model(features, context), device, args.repeats)
            spaces = {}
            for name, basis in [('learned72', learned), ('union144', torch.cat((fixed, learned), 1))]:
                prepared, setup_ms = measure(lambda: shape_response(basis, context, system, ae, opts), device, args.repeats)
                x, apply_ms = measure(lambda: apply_modes(prepared, system, ae, filtered, context), device, args.repeats)
                _, post_ms = measure(lambda: chebyshev(system, compiled.action, ae, x, opts.post, opts.post_lower), device, args.repeats)
                spaces[name] = dict(expand_shape_factor_ms=setup_ms, projection_ms=apply_ms,
                                    postfilter_ms=post_ms)
            rows.append(dict(index=i, FNO_ms=no_ms, fine_build_ms=build_ms,
                             prefilter_ms=pre_ms, features_ms=features_ms,
                             graph_ms=graph_ms, spaces=spaces))
            save_json(args.out/'profile.json', dict(samples=rows,
                device=torch.cuda.get_device_name(device),
                definition='Independently synchronized stage medians; not an additive end-to-end latency. Fixed mesh geometric hats/sines cached for both methods. No truth used.'))
            print(json.dumps(rows[-1]), flush=True)


def example(args, device):
    """Export the fixed first test material for a spatial mechanism figure."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    from baselines.data.darcy_v2.generate import element_geometry
    model, saved = load_model(args.checkpoint, device)
    model.eval()
    opts = config(saved)
    split = DarcySplit(args.data, 'test')
    base = compiled_plan(split, device, SimpleNamespace(coarse_size=128))
    compiled = SparseCompiled(split.mesh, device, shared=base)
    context = CoarseContext(compiled, 1)
    no, stats, _ = load_no(args.no_dir, device)
    a, y, xy, _, mask = tensors(split, device)
    with torch.no_grad():
        full = predict(no, a[:1], xy, mask, stats).flatten()
        system, ae, matrix = compiled.build(a[0].flatten(), torch.zeros_like(full))
        filtered = chebyshev(system, compiled.action, ae, full[system.free], opts.pre, opts.lower)
        learned = model(context.features(system, ae, filtered), context)
        prepared = shape_response(torch.cat((geometric(context), learned), 1), context, system, ae, opts)
        x = cycle(filtered, prepared, system, ae, context, opts, already_filtered=True)
        x = cycle(x, prepared, system, ae, context, opts)
        joint = torch.zeros_like(full); joint[system.free] = x
        mg_system, mg_ae = base.build(a[0].flatten(), torch.zeros_like(full))
        mg_x0 = full[mg_system.free]
        mg_x = correct_matrixfree(mg_system, LiftedSparseAction(base.action, matrix), mg_ae,
            mg_x0, torch.ones_like(mg_x0),
            torch.ones_like(mg_x0), None, cycles=2, coarse_steps=0)
        mg = torch.zeros_like(full); mg[mg_system.free] = mg_x
    coefficient, target = split.coefficient[0], split.target[0]
    area, gradient, mass = element_geometry(split.mesh.points, split.mesh.cells)
    cells, pts = split.mesh.cells, split.mesh.points
    ae_np = coefficient[cells].mean(1)
    grad_target = np.einsum('tic,ti->tc', gradient, target[cells])
    qtarget = -ae_np[:, None]*grad_target
    qscale = np.sqrt(np.sum(area*np.sum(qtarget**2, 1))/area.sum())
    uscale = np.sqrt(np.sum(mass*target**2)/mass.sum())
    fields = {'FNO':full.cpu().numpy(), 'Numerical MG K2':mg.cpu().numpy(),
              'Joint response K2':joint.cpu().numpy()}
    measured = {n:field_metrics(u, target, coefficient, split.mesh) for n,u in fields.items()}
    fig, axes = plt.subplots(3,3,figsize=(11,9),constrained_layout=True)
    first = [(coefficient, LogNorm(), 'Material coefficient a'),
             (target, None, 'Reference field'),
             (np.linalg.norm(qtarget,axis=1), LogNorm(), 'Reference flux magnitude')]
    for ax,(values,norm,label) in zip(axes[0], first):
        plot = ax.tripcolor(pts[:,0],pts[:,1],cells,values,shading='flat',norm=norm,rasterized=True)
        ax.set_title(label);fig.colorbar(plot,ax=ax,shrink=.75)
    for column,(name,u) in enumerate(fields.items()):
        grad = np.einsum('tic,ti->tc',gradient,(u-target)[cells])
        qe = 100*ae_np*np.linalg.norm(grad,axis=1)/qscale
        ue = 100*np.abs(u-target)/uscale
        for row,values,title in [(1,ue,'Field'),(2,qe,'Flux')]:
            plot=axes[row,column].tripcolor(pts[:,0],pts[:,1],cells,np.maximum(values,1e-5),
                shading='flat',norm=LogNorm(vmin=1e-3,vmax=1e3),cmap='magma',rasterized=True)
            metric='L2_percent' if row==1 else 'flux_L2_percent'
            axes[row,column].set_title(name+'\n'+title+' error '+str(round(measured[name][metric],5))+'%')
            fig.colorbar(plot,ax=axes[row,column],shrink=.75)
    for ax in axes.flat:
        ax.set_aspect('equal');ax.set_xticks([]);ax.set_yticks([])
    fig.savefig(args.out/'MATERIAL_AND_ERRORS.svg',dpi=160)
    fig.savefig(args.out/'MATERIAL_AND_ERRORS.png',dpi=140)
    plt.close(fig)
    save_json(args.out/'example.json', dict(test_index=0,sample_index=int(split.sample_index[0]),
        methods=measured,source='Fixed first held-out material, no example selection.',
        display='Pointwise errors divided by reference global RMS; common logarithmic error color scales. Metrics use the exact P1 definitions.'))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage', choices=['train', 'diagnose', 'evaluate', 'transfer', 'spectral', 'profile', 'example'])
    for name in ('data', 'no-dir', 'checkpoint', 'out'):
        p.add_argument('--'+name, required=True, type=Path)
    p.add_argument('--original', type=Path)
    p.add_argument('--material-checkpoint', type=Path)
    p.add_argument('--methods', nargs='+')
    p.add_argument('--conditioning', choices=['all', 'material'], default='all')
    p.add_argument('--candidate-name', default='cycle_trained')
    p.add_argument('--cycles', type=int, default=2)
    p.add_argument('--epochs', type=int, default=20)
    p.add_argument('--train-count', type=int, default=48)
    p.add_argument('--val-count', type=int, default=12)
    p.add_argument('--count', type=int, default=32)
    p.add_argument('--split', choices=['val', 'test'], default='test')
    p.add_argument('--repeats', type=int, default=5)
    p.add_argument('--seed', type=int, default=20261103)
    p.add_argument('--lr', type=float, default=.0003)
    p.add_argument('--device', default='cuda')
    args = p.parse_args()
    if args.out.exists():
        p.error('Choose a new output directory; completed runs are immutable.')
    if args.stage == 'evaluate' and args.original is None:
        p.error('Evaluation requires --original for paired comparison.')
    args.out.mkdir(parents=True)
    torch.set_num_threads(2)
    save_json(args.out/'provenance.json', dict(arguments={k:str(v) if isinstance(v, Path) else v for k,v in vars(args).items()},
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
    {'train':train, 'diagnose':diagnose, 'evaluate':evaluate,
     'transfer':transfer, 'spectral':spectral, 'profile':profile,
     'example':example}[args.stage](args, torch.device(args.device))


if __name__ == '__main__':
    main()
