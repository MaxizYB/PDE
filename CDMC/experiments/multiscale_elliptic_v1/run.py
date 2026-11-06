"""One entry point for the multiscale-elliptic CDMC mechanism experiment.

Stages: train-no, train-cdmc, evaluate, numerics, report. Existing CDMC v2
models and the no-PCG direct-terminal V-cycle are imported, not copied.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
V2 = ROOT / 'CDMC/experiments/v2'
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(V2))
from adapters.darcy import DarcySplit
from solver.correctors import CDMCSetup
from solver.matrixfree_cdmc import MeshCompiledCDMC, gates_from_implicit, correct_matrixfree
from solver.numerics import multiply
from baselines.data.darcy_v2.generate import assemble, element_geometry
from models import FNO


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def clock(device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    return time.perf_counter()


def summary(values):
    a = np.asarray(values, dtype=float)
    return {'mean': float(a.mean()), 'p95': float(np.percentile(a, 95)),
            'median': float(np.median(a))}


def tensors(split, device):
    n = int(round(np.sqrt(split.target.shape[1])))
    assert n*n == split.target.shape[1]
    a = torch.as_tensor(split.coefficient.reshape(-1, n, n), device=device)
    y = torch.as_tensor(split.target.reshape(-1, n, n), device=device)
    xy = torch.as_tensor(split.mesh.points.reshape(1, n, n, 2), device=device)
    mass = torch.as_tensor(split.mesh.mass.reshape(n, n), device=device)
    mask = torch.as_tensor((~split.mesh.boundary).reshape(n, n), device=device)
    return a, y, xy, mass, mask


def relative(pred, target, mass):
    dims = tuple(range(1, pred.ndim))
    return ((mass*(pred-target).square()).sum(dims)
            / (mass*target.square()).sum(dims).clamp_min(1e-20)).sqrt()


def gradient_error(pred, target):
    ex = (pred-target)[:, 1:, :] - (pred-target)[:, :-1, :]
    ey = (pred-target)[:, :, 1:] - (pred-target)[:, :, :-1]
    tx = target[:, 1:, :] - target[:, :-1, :]
    ty = target[:, :, 1:] - target[:, :, :-1]
    return ((ex.square().sum((1, 2)) + ey.square().sum((1, 2)))
            / (tx.square().sum((1, 2)) + ty.square().sum((1, 2))).clamp_min(1e-20)).sqrt()


def predict(model, coefficient, xy, mask, stats):
    value = (coefficient.log() - stats['a_mean']) / stats['a_std']
    return (model(value, xy)*stats['y_std'] + stats['y_mean']) * mask


def load_no(path, device):
    saved = torch.load(Path(path)/'no.pt', map_location=device, weights_only=False)
    model = FNO(**saved['model_options']).to(device)
    model.load_state_dict(saved['state'])
    model.eval()
    return model, saved['stats'], saved


def train_no(args, device):
    train = DarcySplit(args.data, 'train')
    val = DarcySplit(args.data, 'val')
    a, y, xy, mass, mask = tensors(train, device)
    va, vy, _, _, _ = tensors(val, device)
    stats = {'a_mean': a.log().mean(), 'a_std': a.log().std().clamp_min(1e-6),
             'y_mean': y.mean(0), 'y_std': y.std(0).clamp_min(0.05*y.std())}
    options = {'width': args.width, 'modes': args.modes, 'depth': 4, 'padding': 9}
    model = FNO(**options).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, eta_min=1e-5)
    history, best = [], float('inf')
    started = clock(device)
    for epoch in range(args.epochs):
        epoch_start = clock(device)
        model.train()
        losses = []
        for index in torch.randperm(len(train), device=device).split(args.batch_size):
            output = predict(model, a[index], xy, mask, stats)
            loss = relative(output, y[index], mass).mean()
            if args.no_loss == 'h1':
                loss = loss + 0.1*gradient_error(output, y[index]).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(loss.detach())
        scheduler.step()
        model.eval()
        with torch.no_grad():
            errors, gradients = [], []
            for index in torch.arange(len(val), device=device).split(args.batch_size):
                pred = predict(model, va[index], xy, mask, stats)
                errors.append(relative(pred, vy[index], mass))
                gradients.append(gradient_error(pred, vy[index]))
            error = float(torch.cat(errors).mean())
            grad = float(torch.cat(gradients).mean())
        row = {'epoch': epoch+1, 'train_loss': float(torch.stack(losses).mean()),
               'val_L2_percent': 100*error, 'val_gradient_percent': 100*grad,
               'seconds': clock(device)-epoch_start}
        history.append(row)
        score = error + (0.1*grad if args.no_loss == 'h1' else 0.)
        if score < best:
            best = score
            torch.save({'state': model.state_dict(), 'stats': stats,
                        'model_options': options, 'no_loss': args.no_loss,
                        'seed': args.seed, 'best_epoch': epoch+1,
                        'best_validation': row, 'data': str(args.data),
                        'architecture': 'standard FNO: four spectral+pointwise blocks; no groupnorm'},
                       args.out/'no.pt')
        if epoch < 3 or (epoch+1) % 10 == 0:
            print(json.dumps(row), flush=True)
        if not np.isfinite(score):
            raise RuntimeError('NO training became nonfinite')
    save_json(args.out/'no_train.json', {'history': history,
               'total_seconds': clock(device)-started, 'best_score': best,
               'parameters': sum(p.numel() for p in model.parameters())})


def get_predictions(args, device):
    model, stats, saved = load_no(args.no_dir, device)
    destination = args.no_dir/'predictions.npz'
    if destination.exists():
        with np.load(destination) as z:
            return {name: z[name].copy() for name in ('train', 'val', 'test')}
    result = {}
    with torch.no_grad():
        for name in ('train', 'val', 'test'):
            split = DarcySplit(args.data, name)
            a, _, xy, _, mask = tensors(split, device)
            preds = [predict(model, a[i:i+args.batch_size], xy, mask, stats).flatten(1).cpu().numpy()
                     for i in range(0, len(split), args.batch_size)]
            result[name] = np.concatenate(preds)
    np.savez_compressed(destination, **result)
    return result


def compiled_plan(split, device, args):
    return MeshCompiledCDMC(split.mesh, device, args.coarse_size,
                            terminal_solver='direct', aggregate_size=2,
                            interpolation_max_entries=64)


def make_example(split, prediction, index, compiled, device):
    a = torch.as_tensor(split.coefficient[index], device=device)
    lift = torch.as_tensor(split.lift[index], device=device)
    system, element_a = compiled.build(a, lift)
    target = torch.as_tensor(split.target[index], device=device)
    initial = torch.as_tensor(prediction[index], device=device)[system.free]
    return system, element_a, initial, target


def corrected(setup, compiled, example, cycles, learned=True, initial=None):
    system, element_a, x0, _ = example
    x0 = x0 if initial is None else initial
    if learned:
        smooth, _ = gates_from_implicit(setup, system, compiled.action, element_a, x0)
    else:
        smooth = torch.ones_like(x0)
    return correct_matrixfree(system, compiled.action, element_a, x0,
                              smooth, torch.ones_like(x0), None,
                              cycles=cycles, coarse_steps=0, return_states=True)


def generic_initial(example, compiled, rng):
    """Equal-amplitude generic probes; no test labels or NO data in directions."""
    system, _, initial, target = example
    n = int(round(np.sqrt(len(target))))
    axis = torch.linspace(0, 1, n, device=initial.device)
    xx, yy = torch.meshgrid(axis, axis, indexing='ij')
    probe = torch.zeros_like(xx)
    for lo, hi in ((1, 5), (5, 17), (17, min(65, n//2))):
        part = torch.zeros_like(xx)
        for _ in range(4):
            kx, ky = rng.integers(lo, max(lo+1, hi), 2)
            part = part + float(rng.normal())*torch.sin(np.pi*kx*xx)*torch.sin(np.pi*ky*yy)
        interior = part.flatten()[system.free]
        part = part / (compiled.plan.mass*interior.square()).sum().sqrt().clamp_min(1e-12)
        probe = probe + float(rng.uniform(0.5, 1.5))*part
    error = probe.flatten()[system.free]
    amplitude = (compiled.plan.mass*(target[system.free]-initial).square()).sum().sqrt()
    error = error * amplitude / (compiled.plan.mass*error.square()).sum().sqrt().clamp_min(1e-12)
    return target[system.free] - error


def train_cdmc(args, device):
    predictions = get_predictions(args, device)
    train, val = DarcySplit(args.data, 'train'), DarcySplit(args.data, 'val')
    compiled = compiled_plan(train, device, args)
    tr = [make_example(train, predictions['train'], i, compiled, device)
          for i in range(min(args.corrector_samples, len(train)))]
    vr = [make_example(val, predictions['val'], i, compiled, device)
          for i in range(min(args.corrector_val, len(val)))]
    rng = np.random.default_rng(args.seed)
    if args.task == 'generic':
        # Fresh fixed generic directions on train and validation, same amplitude
        # per sample as the actual NO error. Validation selects on generic probes.
        tr = [(s, a, generic_initial(e, compiled, rng), y) for e in tr for s,a,_,y in [e]]
        vr = [(s, a, generic_initial(e, compiled, rng), y) for e in vr for s,a,_,y in [e]]
    # Identical sample order in the task-matched and generic training controls.
    rng = np.random.default_rng(args.seed + 1)
    setup = CDMCSetup(args.setup_width).to(device)
    optimizer = torch.optim.AdamW(setup.parameters(), lr=args.corrector_lr)
    mass = torch.as_tensor(train.mesh.mass, device=device)
    history, best = [], float('inf')
    started = clock(device)
    for epoch in range(args.corrector_epochs):
        epoch_start = clock(device)
        setup.train()
        train_rows = []
        for idx in rng.permutation(len(tr)):
            example = tr[idx]
            states = corrected(setup, compiled, example, 2)
            system, _, _, target = example
            denominator = (mass*target.square()).sum().clamp_min(1e-20)
            errors = torch.stack([((compiled.plan.mass*(x-target[system.free]).square()).sum()
                                    / denominator+1e-20).sqrt() for x in states])
            loss = errors.mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(setup.parameters(), 5.)
            optimizer.step()
            train_rows.append(errors.detach())
        setup.eval()
        with torch.no_grad():
            errors = []
            for example in vr:
                system, _, _, target = example
                denom = (mass*target.square()).sum().clamp_min(1e-20)
                states = corrected(setup, compiled, example, 2)
                errors.append(torch.stack([((compiled.plan.mass*(x-target[system.free]).square()).sum()
                                             / denom+1e-20).sqrt() for x in states]))
            train_mean = 100*torch.stack(train_rows).mean(0)
            val_mean = 100*torch.stack(errors).mean(0)
        row = {'epoch': epoch+1, 'train_K1_percent': float(train_mean[0]),
               'train_K2_percent': float(train_mean[1]), 'val_K1_percent': float(val_mean[0]),
               'val_K2_percent': float(val_mean[1]), 'seconds': clock(device)-epoch_start}
        history.append(row)
        print(json.dumps(row), flush=True)
        if not np.isfinite(row['val_K2_percent']):
            raise RuntimeError('CDMC nonfinite validation')
        if row['val_K2_percent'] < best:
            best = row['val_K2_percent']
            torch.save({'state': setup.state_dict(), 'width': args.setup_width,
                        'task': args.task, 'no_dir': str(args.no_dir),
                        'coarse_size': args.coarse_size, 'trained_cycles': 2,
                        'gate_mode': 'smooth_only', 'best_epoch': epoch+1,
                        'training_samples': len(tr), 'seed': args.seed}, args.out/'cdmc.pt')
    save_json(args.out/'cdmc_train.json', {'history': history,
                  'seconds': clock(device)-started, 'mesh_plan_ms_once': compiled.mesh_plan_ms_once})


def field_metrics(x, target, coefficient, mesh):
    """Exact P1 field-gradient metrics; no spectral differentiation at interfaces."""
    area, grads, mass = element_geometry(mesh.points, mesh.cells)
    delta = np.asarray(x, dtype=float)-np.asarray(target, dtype=float)
    true = np.asarray(target, dtype=float)
    dg = np.einsum('tic,ti->tc', grads, delta[mesh.cells])
    tg = np.einsum('tic,ti->tc', grads, true[mesh.cells])
    ae = coefficient[mesh.cells].mean(1)
    ratio = lambda a, b: 100*float(np.sqrt(a/max(b, 1e-30)))
    from scipy.fft import dstn
    n = int(round(np.sqrt(len(true))))
    spectrum = dstn(delta.reshape(n,n)[1:-1,1:-1], type=1, norm='ortho')**2
    wave = np.arange(1, n-1)
    radius = np.sqrt(wave[:,None]**2 + wave[None,:]**2)
    spectral_total = max(float(spectrum.sum()), 1e-30)
    result = {'L2_percent': ratio(np.sum(mass*delta**2), np.sum(mass*true**2)),
            'gradient_percent': ratio(np.sum(area[:,None]*dg**2), np.sum(area[:,None]*tg**2)),
            'energy_percent': ratio(np.sum((area*ae)[:,None]*dg**2), np.sum((area*ae)[:,None]*tg**2)),
            'flux_L2_percent': ratio(np.sum((area*ae**2)[:,None]*dg**2), np.sum((area*ae**2)[:,None]*tg**2))}
    for name, selected in [('low', radius <= 8), ('mid', (radius > 8)&(radius <= 32)),
                           ('high', radius > 32)]:
        result[f'error_sine_{name}_fraction_percent'] = 100*float(spectrum[selected].sum())/spectral_total
    return result


def measure(call, device, repeats):
    # Warmup and timing never reuse a previous iterate; each call starts at x0.
    with torch.no_grad():
        output = call()
        times = []
        for _ in range(repeats):
            started = clock(device)
            output = call()
            times.append(1000*(clock(device)-started))
    return output, float(np.median(times))


def evaluate(args, device):
    model, stats, saved = load_no(args.no_dir, device)
    split = DarcySplit(args.data, 'test')
    a, y, xy, mass, mask = tensors(split, device)
    compiled = compiled_plan(split, device, args)
    setups, provenance = {}, {}
    for path in args.corrector_dirs:
        ckpt = torch.load(path/'cdmc.pt', map_location=device, weights_only=False)
        setup = CDMCSetup(ckpt['width']).to(device)
        setup.load_state_dict(ckpt['state'])
        setup.eval()
        setups[path.name] = setup
        provenance[path.name] = {key: ckpt.get(key) for key in
                                ('task', 'no_dir', 'best_epoch', 'training_samples', 'seed')}
    rows, figures, probe_rows = [], {}, []
    count = min(args.count, len(split))
    with torch.no_grad():
        for index in range(count):
            pred, no_ms = measure(lambda: predict(model, a[index:index+1], xy, mask, stats), device, args.repeats)
            pred_full = pred.flatten()
            # Explicit online sample build with the current test coefficient.
            coef = a[index].flatten()
            lift = torch.zeros_like(coef)
            built, build_ms = measure(lambda: compiled.build(coef, lift), device, args.repeats)
            system, element_a = built
            example = (system, element_a, pred_full[system.free], y[index].flatten())
            true = split.target[index]
            coef_np = split.coefficient[index]
            no_full = pred_full.cpu().numpy()
            row = {'index': index, 'NO_ms': no_ms, 'build_ms': build_ms, 'methods': {}}
            no_metrics = field_metrics(no_full, true, coef_np, split.mesh)
            no_metrics['total_ms'] = no_ms
            row['methods']['NO'] = no_metrics
            states_figure = {'GT': true, 'NO': no_full, 'coefficient': coef_np}
            for name, setup in [('zero_MG', None), ('fixed_MG', None), *setups.items()]:
                for k in args.cycles:
                    def online():
                        sys0, ae0 = compiled.build(coef, lift)
                        if name == 'zero_MG':
                            initial = torch.zeros_like(sys0.rhs)
                        else:
                            initial = predict(model, a[index:index+1], xy, mask, stats).flatten()[sys0.free]
                        ex = (sys0, ae0, initial, y[index].flatten())
                        return corrected(setup, compiled, ex, k, learned=setup is not None)
                    states, total_ms = measure(online, device, args.repeats)
                    # Component timing excludes metrics/CPU transfer and reuses the
                    # current system; total above performs NO+build+setup+cycles.
                    apply_initial = torch.zeros_like(system.rhs) if name == 'zero_MG' else None
                    _, apply_ms = measure(lambda: corrected(setup, compiled, example, k,
                                                   learned=setup is not None, initial=apply_initial), device, args.repeats)
                    x = np.zeros_like(true)
                    x[compiled.plan.free] = states[-1].cpu().numpy()
                    metrics = field_metrics(x, true, coef_np, split.mesh)
                    metrics.update(total_ms=total_ms, correction_ms=apply_ms)
                    residual = system.rhs-compiled.action.action(0, states[-1], element_a)
                    metrics['relative_residual'] = float(residual.norm()/system.rhs.norm())
                    key = f'{name}_K{k}'
                    row['methods'][key] = metrics
                    if k == 2:
                        states_figure[key] = x
            rows.append(row)
            # Offline counterfactual only: same operator and initial error norm,
            # different error directions. Never used as a deployment predictor.
            if args.diagnose_probes:
                probe_initial = generic_initial(example, compiled, np.random.default_rng(args.seed+index+5000))
                probe_row = {'index': index, 'methods': {}}
                for name, setup in [('fixed_MG', None), *setups.items()]:
                    state = corrected(setup, compiled, example, 2, learned=setup is not None,
                                      initial=probe_initial)[-1]
                    x = np.zeros_like(true)
                    x[compiled.plan.free] = state.cpu().numpy()
                    probe_row['methods'][name] = field_metrics(x, true, coef_np, split.mesh)
                probe_rows.append(probe_row)
            if index < 4:
                for key, value in states_figure.items():
                    figures[f'sample{index}_{key}'] = value
            print(json.dumps({'test': index+1, 'NO': no_metrics['L2_percent'],
                   **{k: v['L2_percent'] for k,v in row['methods'].items() if k.endswith('K2')}}), flush=True)
    aggregated = {name: {metric: summary([r['methods'][name][metric] for r in rows])
                        for metric in rows[0]['methods'][name]}
                  for name in rows[0]['methods']}
    save_json(args.out/'evaluation.json', {'data': str(args.data), 'no_dir': str(args.no_dir),
                'device': torch.cuda.get_device_name(device) if device.type == 'cuda' else 'CPU',
                'test_count': count, 'timing': 'batch1 synchronized wallclock, device-resident inputs, 1 warmup, median repeats; online total reruns NO/build/setup/cycles; zero_MG skips NO',
                'repeats': args.repeats, 'mesh_plan_ms_once': compiled.mesh_plan_ms_once,
                'corrector_provenance': provenance,
                'level_sizes': compiled.action.level_sizes, 'summary': aggregated, 'samples': rows,
                'offline_matched_amplitude_generic_probes': probe_rows})
    np.savez_compressed(args.out/'fields.npz', **figures)


def numerics(args, device):
    """Independent mature CPU SA-AMG baseline, zero and identical NO starts."""
    import pyamg
    split = DarcySplit(args.data, 'test')
    predictions = get_predictions(args, device)['test']
    free = np.flatnonzero(~split.mesh.boundary)
    geometry = element_geometry(split.mesh.points, split.mesh.cells)
    mass = split.mesh.mass[free].astype(float)
    rows = []
    for index in range(min(args.count, len(split))):
        coefficient = split.coefficient[index].astype(float)
        started = time.perf_counter()
        full, load = assemble(split.mesh.points, split.mesh.cells, coefficient, geometry)
        matrix, rhs = full[free][:,free].tocsr(), load[free]
        assembly_ms = 1000*(time.perf_counter()-started)
        started = time.perf_counter()
        if args.amg_method == 'asa':
            from pyamg.aggregation import adaptive_sa_solver
            ml, _ = adaptive_sa_solver(matrix, symmetry='symmetric',
                        num_candidates=2, candidate_iters=5, improvement_iters=1,
                        strength=('classical', {'theta': 0.25}),
                        max_coarse=args.coarse_size)
        elif args.amg_method == 'rs':
            ml = pyamg.ruge_stuben_solver(matrix, max_coarse=args.coarse_size)
        else:
            ml = pyamg.smoothed_aggregation_solver(matrix, symmetry='symmetric',
                      strength=('symmetric', {'theta': 0.}), max_coarse=args.coarse_size)
        # PyAMG may defer its terminal factorization until the first solve.
        # Charge this coefficient-dependent preparation to setup, not warmup.
        ml.coarse_solver(ml.levels[-1].A, np.ones(ml.levels[-1].A.shape[0]))
        setup_ms = 1000*(time.perf_counter()-started)
        truth = split.target[index].astype(float)
        outputs = {}
        for initial_name, x0 in [('zero', np.zeros(len(free))), ('NO', predictions[index,free].astype(float))]:
            for steps in args.cycles:
                def apply():
                    return ml.solve(rhs, x0=x0.copy(), tol=0., maxiter=steps, cycle='V')
                apply()
                times = []
                for _ in range(args.repeats):
                    started = time.perf_counter()
                    x = apply()
                    times.append(1000*(time.perf_counter()-started))
                field = np.zeros_like(truth)
                field[free] = x
                metrics = field_metrics(field, truth, coefficient, split.mesh)
                metrics.update(solve_ms=float(np.median(times)), assembly_ms=assembly_ms,
                               amg_setup_ms=setup_ms, total_cpu_ms=assembly_ms+setup_ms+float(np.median(times)))
                outputs[f'{initial_name}_AMG_K{steps}'] = metrics
            # Precision endpoint; this is a numerical baseline, never CDMC-PCG.
            started = time.perf_counter()
            res = []
            x = ml.solve(rhs, x0=x0.copy(), tol=1e-10, maxiter=200, accel='cg', residuals=res)
            solve_ms = 1000*(time.perf_counter()-started)
            field = np.zeros_like(truth)
            field[free] = x
            metrics = field_metrics(field, truth, coefficient, split.mesh)
            metrics.update(solve_ms=solve_ms, iterations=len(res)-1,
                           relative_residual=float(np.linalg.norm(rhs-matrix@x)/np.linalg.norm(rhs)),
                           total_cpu_ms=assembly_ms+setup_ms+solve_ms)
            outputs[f'{initial_name}_AMG_CG_tol1e-10'] = metrics
        rows.append({'index': index, 'assembly_ms': assembly_ms, 'amg_setup_ms': setup_ms,
                     'operator_complexity': float(ml.operator_complexity()), 'methods': outputs})
        print(json.dumps({'numerics': index+1, 'setup_ms': setup_ms,
                          'zero_K2': outputs.get('zero_AMG_K2', {}).get('L2_percent')}), flush=True)
    aggregated = {name: {metric: summary([r['methods'][name][metric] for r in rows])
                        for metric in rows[0]['methods'][name]}
                  for name in rows[0]['methods']}
    filename = 'numerics.json' if args.amg_method == 'sa' else f'numerics_{args.amg_method}.json'
    backend = {'rs': 'classical Ruge-Stuben, default strength theta=0.25 and symmetric Gauss-Seidel',
               'sa': 'smoothed aggregation; symmetric strength theta=0; default symmetric smoothing',
               'asa': 'adaptive smoothed aggregation; 2 candidates, 5 candidate iterations, 1 improvement; classical strength theta=.25'}[args.amg_method]
    save_json(args.out/filename, {'backend': f'CPU PyAMG {pyamg.__version__}, {backend}',
              'cpu_threads': torch.get_num_threads(), 'no_dir': str(args.no_dir),
              'summary': aggregated, 'samples': rows,
              'timing_note': 'CPU baseline, not a controlled same-hardware GPU speedup comparison; full coefficient-dependent AMG setup charged separately'})


def report(args, device):
    source = json.loads((args.out/'evaluation.json').read_text())
    lines = ['# Multiscale elliptic: paired mechanism experiment', '',
             'HANO coefficient law; regenerated same-grid P1-FEM targets. Not the original downsampled-label benchmark.', '',
             '| Method | L2 % | Gradient % | Energy % | Flux % | Online ms | L2 reduction vs NO % |',
             '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    records = []
    for name, data in source['summary'].items():
        row = {'method': name, **{k:v['mean'] for k,v in data.items()}}
        no_error = source['summary']['NO']['L2_percent']['mean']
        row['L2_reduction_vs_NO_percent'] = 100*(1-row['L2_percent']/no_error)
        if '_K' in name and not name.startswith(('fixed_MG', 'zero_MG')):
            cycle = name.rsplit('_K', 1)[1]
            fixed = source['summary']['fixed_MG_K'+cycle]['L2_percent']['mean']
            row['L2_reduction_vs_fixed_MG_percent'] = 100*(1-row['L2_percent']/fixed)
        records.append(row)
        lines.append(f"| {name} | {row['L2_percent']:.6g} | {row['gradient_percent']:.6g} | {row['energy_percent']:.6g} | {row['flux_L2_percent']:.6g} | {row['total_ms']:.4f} | {row['L2_reduction_vs_NO_percent']:.2f} |")
    lines += ['', f"GPU: {source['device']}; n={source['test_count']}. {source['timing']}.",
              f"One-time mesh plan: {source['mesh_plan_ms_once']/1000:.3f} s."]
    paired = []
    for name in source['summary']:
        if name.startswith(('actual_K', 'h1_actual_K')):
            cycle = name.rsplit('_K', 1)[1]
            for baseline in ('fixed_MG_K'+cycle, 'generic_K'+cycle):
                if baseline not in source['summary']:
                    continue
                wins = sum(r['methods'][name]['L2_percent'] < r['methods'][baseline]['L2_percent']
                           for r in source['samples'])
                a = source['summary'][name]['L2_percent']['mean']
                b = source['summary'][baseline]['L2_percent']['mean']
                paired.append(f'| {name} vs {baseline} | {100*(1-a/b):.2f} | {wins}/{source["test_count"]} |')
    if paired:
        lines += ['', '## Paired task-matching evidence', '',
                  '| Comparison | Mean L2 reduction % | Per-sample wins |',
                  '| --- | ---: | ---: |', *paired]
    for numerical in sorted(args.out.glob('numerics*.json')):
        data = json.loads(numerical.read_text())
        lines += ['', '## Mature numerical baseline (CPU)', '', data['backend'], '',
                  '| Method | L2 % | Gradient % | Solve ms | Full CPU ms |',
                  '| --- | ---: | ---: | ---: | ---: |']
        for name, metrics in data['summary'].items():
            d = {k:v['mean'] for k,v in metrics.items()}
            lines.append(f"| {name} | {d['L2_percent']:.6g} | {d['gradient_percent']:.6g} | {d['solve_ms']:.4f} | {d['total_cpu_ms']:.4f} |")
        lines += ['', data['timing_note']]
    probes = source.get('offline_matched_amplitude_generic_probes', [])
    if probes:
        lines += ['', '## Counterfactual error-direction diagnostic', '',
                  'Same held-out operators; synthetic initial errors matched to NO error amplitude. This is an offline diagnostic, not a predictor.', '',
                  '| Corrector | Actual NO start, K2 L2 % | Generic-probe start, K2 L2 % |',
                  '| --- | ---: | ---: |']
        for name in probes[0]['methods']:
            value = np.mean([r['methods'][name]['L2_percent'] for r in probes])
            actual = source['summary'][name+'_K2']['L2_percent']['mean']
            lines.append(f'| {name} | {actual:.6g} | {value:.6g} |')
    audit_path = args.out.parent/'stability_audit.json'
    matched_actual = source.get('corrector_provenance', {}).get('actual', {})
    if audit_path.exists() and matched_actual.get('no_dir') == source['no_dir']:
        audit = json.loads(audit_path.read_text())
        control = audit.get('constant_control')
        if control:
            lines += ['', '## Train-calibrated scalar relaxation control', '',
                      f"The global gate {control['gate_value']:.6f} is the mean network output over {control['calibration_samples']} training samples. No test tuning.", '',
                      '| Method | L2 % | Gradient % |', '| --- | ---: | ---: |']
            for name, value in control['summary'].items():
                lines.append(f"| {name} | {value['L2_percent']['mean']:.6g} | {value['gradient_percent']['mean']:.6g} |")
            lines += ['', 'This control is more accurate than the graph setup at K2 in this run: the Actual-vs-Generic gain alone does not establish a spatial-setup advantage.']
        lines += ['', '## Smoother stability diagnosis', '',
                  '| Test index (zero-based) | Max fixed Rayleigh | Max learned Rayleigh | Zero-step cycles |',
                  '| --- | ---: | ---: | --- |']
        for row in audit['samples']:
            fixed = max(level['fixed_lambda_rayleigh'] for level in row['levels'])
            learned = max(level['learned_lambda_rayleigh'] for level in row['levels'])
            stalled = [cycle['cycle'] for cycle in row['cycles'] if cycle['alpha'] == 0]
            lines.append(f"| {row['index']} | {fixed:.4f} | {learned:.4f} | {stalled} |")
        lines += ['', audit['interpretation']]
    manifest = json.loads((args.data/'manifest.json').read_text())
    lines += ['', '## Discretization audit', '',
              'Coarse truth versus restricted 513-grid truth (first held-out parameter vectors): '+
              ', '.join(f"{row['coarse_vs_restricted_fine_L2_percent']:.4f}%" for row in manifest.get('refinement_audit', []))+'.',
              'The table measures algebraic accuracy on the stored P1 system; continuum refinement error is separate.']
    (args.out/'summary.md').write_text('\n'.join(lines)+'\n')
    keys = sorted({k for row in records for k in row})
    with (args.out/'summary.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(records)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    colors = {'fixed_MG': 'C0', 'actual': 'C3', 'generic': 'C2', 'h1_actual': 'C3'}
    for name in colors:
        points = [(source['summary']['NO']['total_ms']['mean'],
                   source['summary']['NO']['L2_percent']['mean'],
                   source['summary']['NO']['gradient_percent']['mean'])]
        for k in args.cycles:
            key = f'{name}_K{k}'
            if key in source['summary']:
                d = source['summary'][key]
                points.append((d['total_ms']['mean'], d['L2_percent']['mean'], d['gradient_percent']['mean']))
        if len(points) > 1:
            t, l2, grad = np.asarray(points).T
            axes[0].semilogy(t, l2, 'o-', color=colors[name], label=name)
            axes[1].semilogy(t, grad, 'o-', color=colors[name], label=name)
    for ax, ylabel in zip(axes, ['Field relative L2 (%)', 'P1 gradient relative L2 (%)']):
        ax.set(xlabel='Online NO + build + correction (ms)', ylabel=ylabel)
        ax.grid(alpha=.25)
        ax.legend()
    fig.tight_layout()
    fig.savefig(args.out/'accuracy_time.png', dpi=160)
    plt.close(fig)
    with np.load(args.out/'fields.npz') as data:
        n = int(round(np.sqrt(len(data['sample0_GT']))))
        methods = ['NO', 'fixed_MG_K2', 'generic_K2', 'actual_K2', 'h1_actual_K2']
        methods = [name for name in methods if f'sample0_{name}' in data]
        truth = data['sample0_GT'].reshape(n,n)
        errors = [np.abs(data[f'sample0_{name}'].reshape(n,n)-truth) for name in methods]
        error_max = max(float(error.max()) for error in errors)
        fig, axes = plt.subplots(2, len(methods)+1, figsize=(3*(len(methods)+1), 6))
        axes[0,0].imshow(np.log(data['sample0_coefficient'].reshape(n,n)).T,
                         origin='lower', extent=(0,1,0,1), cmap='viridis')
        axes[0,0].set_title('log conductivity')
        axes[1,0].imshow(truth.T, origin='lower', extent=(0,1,0,1), cmap='viridis')
        axes[1,0].set_title('Reference')
        for j,name in enumerate(methods, 1):
            field = data[f'sample0_{name}'].reshape(n,n)
            axes[0,j].imshow(field.T, origin='lower', extent=(0,1,0,1),
                             cmap='viridis', vmin=truth.min(), vmax=truth.max())
            axes[0,j].set_title(name)
            image = axes[1,j].imshow(errors[j-1].T, origin='lower', extent=(0,1,0,1), cmap='magma',
                                     vmin=0, vmax=error_max)
            fig.colorbar(image, ax=axes[1,j], shrink=.65)
            axes[1,j].set_title('Absolute error')
        for ax in axes.flat:
            ax.set_xticks([])
            ax.set_yticks([])
        fig.tight_layout()
        fig.savefig(args.out/'fields_example.png', dpi=160)
        plt.close(fig)
        # Derivatives, not just the visually smooth solution, expose the NO defect.
        with np.load(args.data/'mesh.npz') as mesh:
            points, cells = mesh['points'], mesh['cells']
        _, gradients, _ = element_geometry(points, cells)
        grad = lambda value: np.einsum('tic,ti->tc', gradients, value.flatten()[cells])
        true_gradient = grad(truth)
        learned_name = next((name for name in ('actual_K2','h1_actual_K2') if name in methods), None)
        selected = [name for name in ('NO','fixed_MG_K2',learned_name) if name is not None]
        derivative_errors = [np.linalg.norm(grad(data[f'sample0_{name}'])-true_gradient, axis=1)
                             for name in selected]
        from matplotlib.colors import LogNorm
        upper = max(float(value.max()) for value in derivative_errors)
        scale = LogNorm(vmin=max(upper*1e-5, 1e-20), vmax=upper)
        fig, axes = plt.subplots(2, 3, figsize=(12, 7))
        axes[0,0].imshow(np.log(data['sample0_coefficient'].reshape(n,n)).T,
                         origin='lower', extent=(0,1,0,1), cmap='viridis')
        axes[0,0].set_title('Conductivity (log scale)')
        upper_grad = float(np.linalg.norm(true_gradient, axis=1).max())
        for ax, name, value in [(axes[0,1], 'Reference gradient', true_gradient),
                                (axes[0,2], 'FNO gradient', grad(data['sample0_NO']))]:
            image = ax.tripcolor(points[:,0], points[:,1], cells,
                                 facecolors=np.linalg.norm(value, axis=1),
                                 vmin=0, vmax=upper_grad, cmap='viridis', rasterized=True)
            ax.set_title(name)
            fig.colorbar(image, ax=ax, shrink=.65)
        for ax, name, error in zip(axes[1], selected, derivative_errors):
            image = ax.tripcolor(points[:,0], points[:,1], cells, facecolors=error,
                                 norm=scale, cmap='magma', rasterized=True)
            ax.set_title(name+' gradient error')
            fig.colorbar(image, ax=ax, shrink=.65)
        for ax in axes.flat:
            ax.set_aspect('equal')
            ax.set_xticks([])
            ax.set_yticks([])
        fig.tight_layout()
        fig.savefig(args.out/'derivative_example.png', dpi=160)
        plt.close(fig)
    if probes:
        names = ['fixed_MG', 'generic', 'actual']
        names = [name for name in names if name in probes[0]['methods']]
        fig, ax = plt.subplots(figsize=(7,4))
        positions = np.arange(len(names))
        actual = [source['summary'][name+'_K2']['L2_percent']['mean'] for name in names]
        synthetic = [np.mean([r['methods'][name]['L2_percent'] for r in probes]) for name in names]
        ax.bar(positions-.18, actual, .36, label='Actual NO error')
        ax.bar(positions+.18, synthetic, .36, label='Generic error, matched amplitude')
        ax.set(xticks=positions, xticklabels=names, ylabel='Field relative L2 after K2 (%)')
        ax.legend()
        fig.tight_layout()
        fig.savefig(args.out/'task_direction_control.png', dpi=160)
        plt.close(fig)
    print('\n'.join(lines), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage', choices=['train-no', 'train-cdmc', 'evaluate', 'numerics', 'report'])
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--no-dir', type=Path)
    p.add_argument('--corrector-dirs', type=Path, nargs='*', default=[])
    p.add_argument('--device', default='cuda')
    p.add_argument('--seed', type=int, default=20261023)
    p.add_argument('--epochs', type=int, default=150)
    p.add_argument('--width', type=int, default=32)
    p.add_argument('--modes', type=int, default=16)
    p.add_argument('--no-loss', choices=['l2','h1'], default='l2')
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--task', choices=['actual','generic'], default='actual')
    p.add_argument('--setup-width', type=int, default=32)
    p.add_argument('--coarse-size', type=int, default=128)
    p.add_argument('--corrector-samples', type=int, default=48)
    p.add_argument('--corrector-val', type=int, default=12)
    p.add_argument('--corrector-epochs', type=int, default=20)
    p.add_argument('--corrector-lr', type=float, default=1e-3)
    p.add_argument('--cycles', nargs='+', type=int, default=[1,2,4])
    p.add_argument('--count', type=int, default=32)
    p.add_argument('--repeats', type=int, default=3)
    p.add_argument('--diagnose-probes', action='store_true')
    p.add_argument('--amg-method', choices=['sa','rs','asa'], default='sa')
    args = p.parse_args()
    if min(args.cycles) < 1:
        p.error('cycle counts must be positive')
    if args.stage != 'train-no' and args.no_dir is None:
        p.error('--no-dir is required for this stage')
    args.out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device)
    stages = {'train-no': train_no, 'train-cdmc': train_cdmc,
              'evaluate': evaluate, 'numerics': numerics, 'report': report}
    stages[args.stage](args, device)


if __name__ == '__main__':
    main()
