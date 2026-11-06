"""Train/evaluate coupled coarse CDMC without changing existing checkpoints."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from run import (DarcySplit, compiled_plan, make_example, load_no, tensors, predict,
                 measure, clock, summary, save_json, field_metrics,
                 corrected, CDMCSetup, generic_initial)
from solver.matrixfree_cdmc import correct_matrixfree
from coupled_coarse import (CoarseContext, CoupledCoarseSetup, prepare_modes, apply_modes,
                            harmonize_modes, factor_modes, mm, ModeOnlyCompiled)


def prefix(compiled, example, cycles, gate=1.):
    system, element_a, initial, _ = example
    if cycles==0:
        return initial
    return correct_matrixfree(system, compiled.action, element_a, initial,
                              torch.full_like(initial, gate), torch.ones_like(initial),
                              None, cycles=cycles, coarse_steps=0)


def load_context(args, device, split, mode_only=False):
    compiled = (ModeOnlyCompiled(split.mesh, device) if mode_only else
                compiled_plan(split, device, SimpleNamespace(coarse_size=128)))
    started = clock(device)
    context = CoarseContext(compiled, args.start_level)
    context.build_ms = 1000*(clock(device)-started)
    return compiled, context


def train(args, device):
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed+1)
    tr, va = DarcySplit(args.data, 'train'), DarcySplit(args.data, 'val')
    compiled, context = load_context(args, device, tr, mode_only=args.prefix_cycles==[0])
    with np.load(args.no_dir/'predictions.npz') as archive:
        predictions = {s: archive[s].copy() for s in ('train', 'val')}
    def cache(split, label, count):
        result = []
        probe_rng = np.random.default_rng(args.seed+(0 if label=='train' else 10000))
        with torch.no_grad():
            for i in range(min(count, len(split))):
                example = make_example(split, predictions[label], i, compiled, device)
                if args.task=='generic':
                    s, ae, _, y = example
                    example = (s, ae, generic_initial(example, compiled, probe_rng), y)
                system, ae, _, target = example
                states = [(k, prefix(compiled, example, k, args.prefix_gate)) for k in args.prefix_cycles]
                result.append((system, ae, target[system.free],
                               [(k,x,context.features(system, ae, x)) for k,x in states]))
        return result
    started = clock(device)
    train_cache = cache(tr, 'train', args.train_count)
    val_cache = cache(va, 'val', args.val_count)
    print(json.dumps({'event': 'cached', 'seconds': clock(device)-started,
                      'train': len(train_cache), 'val': len(val_cache)}), flush=True)
    model = CoupledCoarseSetup(args.width, args.rank, args.start_level).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, eta_min=1e-5)
    history, best = [], float('inf')
    options = dict(width=args.width, rank=args.rank, start_level=args.start_level)
    def errors(example):
        system, ae, target, states = example
        denom = (compiled.plan.mass*target.square()).sum().clamp_min(1e-20)
        values = []
        for _, x, features in states:
            prepared = prepare_modes(model, features, context, system, ae, args.harmonic_steps)
            output = apply_modes(prepared, system, ae, x, context)
            values.append(((compiled.plan.mass*(output-target).square()).sum()/denom+1e-20).sqrt())
        return torch.stack(values)
    for epoch in range(args.epochs):
        epoch_started = clock(device)
        model.train()
        training = []
        for i in rng.permutation(len(train_cache)):
            values = errors(train_cache[i])
            loss = values.mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.)
            optimizer.step()
            training.append(values.detach())
        scheduler.step()
        model.eval()
        with torch.no_grad():
            validation = torch.stack([errors(e) for e in val_cache]).mean(0)*100
        row = {'epoch': epoch+1, 'train_percent': (100*torch.stack(training).mean(0)).tolist(),
               'val_percent': validation.tolist(), 'seconds': clock(device)-epoch_started}
        history.append(row)
        print(json.dumps(row), flush=True)
        score = float(validation.mean())
        if not np.isfinite(score):
            raise RuntimeError('Nonfinite coupled coarse training')
        if score < best:
            best = score
            torch.save({'state': model.state_dict(), 'options': options,
                        'best_epoch': epoch+1, 'val': row, 'prefix_gate': args.prefix_gate,
                        'harmonic_steps': args.harmonic_steps, 'seed': args.seed,
                        'train_count': len(train_cache), 'no_dir': str(args.no_dir),
                        'task': args.task, 'prefix_cycles': args.prefix_cycles}, args.out/'coupled.pt')
        save_json(args.out/'training.json', {'history': history, 'parameters': sum(p.numel() for p in model.parameters()),
                    'seconds': clock(device)-started, 'best': best,
                    'task': args.task,
                    'prefix_cycles': args.prefix_cycles,
                    'mesh_plan_ms_once': compiled.mesh_plan_ms_once,
                    'graph_context_ms_once': context.build_ms,
                    'training': 'Frozen FNO; mean final field loss at explicitly recorded fixed prefix cycles; generic probes use training/validation truth offline, inference never uses truth.'})


def evaluate(args, device):
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    args.start_level = checkpoint['options']['start_level']
    split = DarcySplit(args.data, args.split)
    compiled, context = load_context(args, device, split)
    model = CoupledCoarseSetup(**checkpoint['options']).to(device)
    model.load_state_dict(checkpoint['state'])
    model.eval()
    torch.manual_seed(checkpoint['seed'])
    untrained = CoupledCoarseSetup(**checkpoint['options']).to(device).eval()
    generic = None
    if args.generic_checkpoint:
        gc = torch.load(args.generic_checkpoint, map_location=device, weights_only=False)
        if gc['options'] != checkpoint['options'] or gc['harmonic_steps'] != checkpoint['harmonic_steps']:
            raise ValueError('Generic control must use the same architecture and shaping')
        generic = CoupledCoarseSetup(**gc['options']).to(device).eval()
        generic.load_state_dict(gc['state'])
    no, stats, _ = load_no(args.no_dir, device)
    a, y, xy, _, mask = tensors(split, device)
    old = torch.load(args.old_corrector/'cdmc.pt', map_location=device, weights_only=False)
    old_setup = CDMCSetup(old['width']).to(device)
    old_setup.load_state_dict(old['state'])
    old_setup.eval()
    no_mg_model = None
    mode_compiled = ModeOnlyCompiled(split.mesh, device, shared=compiled)
    # Same mesh and fine action: reuse neural graph maps, not numerical solves.
    mode_context = context
    if args.no_mg_checkpoint:
        nc = torch.load(args.no_mg_checkpoint, map_location=device, weights_only=False)
        if nc['options'] != checkpoint['options'] or nc['harmonic_steps'] != checkpoint['harmonic_steps']:
            raise ValueError('No-MG control must use the same architecture and mode shaping')
        if nc.get('prefix_cycles') != [0]:
            raise ValueError('No-MG checkpoint must have been trained with --prefix-cycles 0')
        no_mg_model = CoupledCoarseSetup(**nc['options']).to(device).eval()
        no_mg_model.load_state_dict(nc['state'])
    # Fair timing: same-process, same input residency and fresh NO/build per call.
    names = ['NO', 'MG_K1', 'MG_K2', 'MG_K4', 'scalar_K1', 'scalar_K2', 'old_CDMC_K2',
             'coupled_K1', 'coupled_K2', 'untrained_K2', 'geometric_K1', 'geometric_K2']
    if generic is not None:
        names.extend(['generic_K1','generic_K2'])
    if no_mg_model is not None:
        names.extend(['frozen_no_MG','trained_no_MG','geometric_K0'])
    available = names+['coupled_interleaved_K2','geometric_interleaved_K2']
    if generic is not None:
        available.append('generic_interleaved_K2')
    if args.methods:
        if set(args.methods)-set(available):
            raise ValueError(f'Unknown methods: {set(args.methods)-set(available)}')
        names = args.methods
    # Matched-rank numerical enrichment, not a learned module: low sine modes
    # pass through the same first coarse space and current-A shaping/calibration.
    point = compiled.points_free
    pairs = sorted(((i,j) for i in range(1,9) for j in range(1,9)), key=lambda ij: (ij[0]**2+ij[1]**2,ij))
    geometric = torch.stack([torch.sin(np.pi*i*point[:,0])*torch.sin(np.pi*j*point[:,1])
                             for i,j in pairs[:checkpoint['options']['rank']]], 1)
    for restriction in context.restrict[:args.start_level]:
        geometric = mm(restriction, geometric)
    geometric = context.lift(geometric)
    rows, profiles = [], []
    with torch.no_grad():
        for i in range(min(args.count, len(split))):
            coefficient = a[i].flatten()
            lift = torch.zeros_like(coefficient)
            def online(name):
                initial_full = predict(no, a[i:i+1], xy, mask, stats).flatten()
                if name == 'NO':
                    return initial_full
                if name in ('frozen_no_MG','trained_no_MG','geometric_K0'):
                    sys0, ae0 = mode_compiled.build(coefficient, lift)
                    initial = initial_full[sys0.free]
                    if name=='geometric_K0':
                        shaped = harmonize_modes(geometric, mode_context, sys0, ae0, checkpoint['harmonic_steps'])
                        prepared = factor_modes(shaped, mode_context, ae0)
                    else:
                        features = mode_context.features(sys0, ae0, initial)
                        selected = model if name=='frozen_no_MG' else no_mg_model
                        prepared = prepare_modes(selected, features, mode_context, sys0, ae0, checkpoint['harmonic_steps'])
                    output = apply_modes(prepared, sys0, ae0, initial, mode_context)
                    full = torch.zeros_like(initial_full)
                    full[sys0.free] = output
                    return full
                system, ae = compiled.build(coefficient, lift)
                example = (system, ae, initial_full[system.free], y[i].flatten())
                if name == 'old_CDMC_K2':
                    output = corrected(old_setup, compiled, example, 2)[-1]
                elif name.startswith('scalar'):
                    output = prefix(compiled, example, int(name[-1]), 1.217079848)
                else:
                    k = 1 if '_interleaved_' in name else int(name[-1])
                    output = prefix(compiled, example, k,
                                    checkpoint['prefix_gate'] if name.startswith(('coupled','untrained','generic','geometric')) else 1.)
                    if name.startswith(('coupled', 'untrained', 'generic', 'geometric')):
                        if name.startswith('geometric'):
                            shaped = harmonize_modes(geometric, context, system, ae, checkpoint['harmonic_steps'])
                            prepared = factor_modes(shaped, context, ae)
                        else:
                            features = context.features(system, ae, output)
                            selected = generic if name.startswith('generic') else untrained if name.startswith('untrained') else model
                            prepared = prepare_modes(selected,
                                                     features, context, system, ae, checkpoint['harmonic_steps'])
                        output = apply_modes(prepared, system, ae, output, context)
                        if '_interleaved_' in name:
                            output = prefix(compiled, (system,ae,output,y[i].flatten()), 1,
                                            checkpoint['prefix_gate'])
                full = torch.zeros_like(initial_full)
                full[system.free] = output
                return full
            row = {'index': i, 'methods': {}}
            # Independent of the predictor: metrics use the same fine PDE and
            # are outside timed calls. This light setup contains no MG solve.
            metric_system, metric_a = mode_compiled.build(coefficient, lift)
            for name in names:
                output, total_ms = measure(lambda: online(name), device, args.repeats)
                metrics = field_metrics(output.cpu().numpy(), split.target[i], split.coefficient[i], split.mesh)
                metrics['total_ms'] = total_ms
                residual = metric_system.rhs-mode_compiled.action.action(0, output[metric_system.free], metric_a)
                metrics['relative_residual'] = float(residual.norm()/metric_system.rhs.norm())
                row['methods'][name] = metrics
            if i < args.profile_count:
                full, t_no = measure(lambda: predict(no, a[i:i+1], xy, mask, stats).flatten(), device, args.repeats)
                (sys0, ae0), t_build = measure(lambda: compiled.build(coefficient, lift), device, args.repeats)
                ex = (sys0, ae0, full[sys0.free], y[i].flatten())
                for k in (1,2):
                    xp, t_v = measure(lambda: prefix(compiled, ex, k, checkpoint['prefix_gate']), device, args.repeats)
                    features, t_f = measure(lambda: context.features(sys0, ae0, xp), device, args.repeats)
                    modes, t_n = measure(lambda: model(features, context), device, args.repeats)
                    shaped, t_h = measure(lambda: harmonize_modes(modes, context, sys0, ae0, checkpoint['harmonic_steps']), device, args.repeats)
                    prepared, t_g = measure(lambda: factor_modes(shaped, context, ae0), device, args.repeats)
                    _, t_a = measure(lambda: apply_modes(prepared, sys0, ae0, xp, context), device, args.repeats)
                    eig = torch.linalg.eigvalsh(prepared[-1])
                    profiles.append(dict(index=i, cycles=k, NO_ms=t_no, system_ms=t_build,
                            V_cycle_ms=t_v, features_ms=t_f, neural_modes_ms=t_n,
                            material_shaping_ms=t_h, Galerkin_factor_ms=t_g, correction_ms=t_a,
                            gram_min_eigenvalue=float(eig[0]), gram_max_eigenvalue=float(eig[-1])))
                if no_mg_model is not None:
                    (ns, na), t_b = measure(lambda: mode_compiled.build(coefficient, lift), device, args.repeats)
                    nf, t_f = measure(lambda: mode_context.features(ns, na, full[ns.free]), device, args.repeats)
                    nz, t_n = measure(lambda: no_mg_model(nf, mode_context), device, args.repeats)
                    shaped, t_h = measure(lambda: harmonize_modes(nz, mode_context, ns, na, checkpoint['harmonic_steps']), device, args.repeats)
                    prepared, t_g = measure(lambda: factor_modes(shaped, mode_context, na), device, args.repeats)
                    _, t_a = measure(lambda: apply_modes(prepared, ns, na, full[ns.free], mode_context), device, args.repeats)
                    eig = torch.linalg.eigvalsh(prepared[-1])
                    profiles.append(dict(index=i, cycles=0, NO_ms=t_no, system_ms=t_b,
                            V_cycle_ms=0., features_ms=t_f, neural_modes_ms=t_n,
                            material_shaping_ms=t_h, Galerkin_factor_ms=t_g, correction_ms=t_a,
                            gram_min_eigenvalue=float(eig[0]), gram_max_eigenvalue=float(eig[-1])))
            rows.append(row)
            print(json.dumps({'test': i+1, **{n: round(m['L2_percent'],6) for n,m in row['methods'].items()}}), flush=True)
    aggregated = {n: {metric: summary([r['methods'][n][metric] for r in rows])
                       for metric in rows[0]['methods'][n]} for n in names}
    save_json(args.out/'evaluation.json', {'summary': aggregated, 'samples': rows,
                'device': torch.cuda.get_device_name(device), 'checkpoint': str(args.checkpoint),
                'best_epoch': checkpoint['best_epoch'], 'test_count': len(rows), 'split': args.split,
                'timing': 'Same process, batch1 device-resident, 1 warmup / median repeats, fresh NO+system build+fixed V cycles+neural basis+exact Galerkin calibration; no timing from old report reused.',
                'repeats': args.repeats, 'mesh_plan_ms_once': compiled.mesh_plan_ms_once,
                'graph_context_ms_once': context.build_ms,
                'stage_profiles': profiles,
                'no_mg_checkpoint': str(args.no_mg_checkpoint) if args.no_mg_checkpoint else None,
                'no_mg_semantics': 'No numerical V-cycle, coarse solve or PCG. Neural graph hierarchy, two fixed mode-shaping steps and 8x8 Galerkin calibration remain. Mode-only material build skips intermediate numerical levels and terminal Cholesky.',
                'method': 'Existing CDMC multilevel backbone plus learned material/NO-tail-conditioned coarse modes; no PCG; no test labels in prediction.'})
    fields = ['method','L2_percent','L2_p95_percent','gradient_percent','energy_percent','flux_L2_percent','relative_residual','total_ms']
    with (args.out/'summary.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for n in names:
            s = aggregated[n]
            writer.writerow(dict(method=n, L2_percent=s['L2_percent']['mean'],
                                 L2_p95_percent=s['L2_percent']['p95'],
                                 **{m:s[m]['mean'] for m in fields[3:]}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['train','evaluate'])
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--no-dir', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--generic-checkpoint', type=Path)
    parser.add_argument('--no-mg-checkpoint', type=Path)
    parser.add_argument('--task', choices=['actual','generic'], default='actual')
    parser.add_argument('--old-corrector', type=Path)
    parser.add_argument('--train-count', type=int, default=48)
    parser.add_argument('--val-count', type=int, default=12)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--width', type=int, default=16)
    parser.add_argument('--rank', type=int, default=8)
    parser.add_argument('--start-level', type=int, default=1)
    parser.add_argument('--harmonic-steps', type=int, default=2)
    parser.add_argument('--prefix-gate', type=float, default=1.)
    parser.add_argument('--prefix-cycles', type=int, nargs='+', default=[1,2])
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--count', type=int, default=32)
    parser.add_argument('--split', choices=['val','test'], default='test')
    parser.add_argument('--methods', nargs='+')
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--profile-count', type=int, default=8)
    parser.add_argument('--seed', type=int, default=20261023)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    if any(k<0 for k in args.prefix_cycles):
        parser.error('--prefix-cycles must be nonnegative')
    if args.stage=='train' and (args.out/'coupled.pt').exists():
        parser.error('Training checkpoint exists; choose a fresh --out to preserve completed experiments.')
    args.out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    train(args, device) if args.stage=='train' else evaluate(args, device)


if __name__ == '__main__':
    main()
