"""Train and evaluate retained-reference neural complements on aligned Darcy.

Reuses existing FNO/trunk checkpoints, graph hierarchy, filters, and P1 metrics.
Prediction never consumes test labels. Outputs are separate from older runs.
"""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from frame_space import setup
from stable_space import (torch, DarcySplit, tensors, predict, smooth, clock,
    measure, field_metrics, summary, save_json, exact_prepare, exact_cycles)
from polynomial_joint import P1Norms
from complement_space import (ComplementGraph, reference_factor, reference_update,
    prepare_complement, complement_update, complement_diagnostics)


def settings(args, base):
    opts = SimpleNamespace(**vars(base))
    opts.harmonic_steps = 0
    if args.pre is not None:
        opts.pre = args.pre
    if args.cutoff is not None:
        opts.lower = args.cutoff
    return opts


def build_item(compiled, context, no, stats, bank, opts, split, a, xy, mask, i):
    full = predict(no, a[i:i+1], xy, mask, stats).flatten()
    system, ae, matrix = compiled.build(a[i].flatten(), torch.zeros_like(full))
    initial = smooth(system, compiled.action, ae, full[system.free], opts.pre, opts.lower, opts)
    reference = reference_factor(matrix, bank[0])
    residual = system.rhs - compiled.action.action(0, initial, ae)
    after = reference_update(initial, residual, reference).to(initial.dtype)
    features = context.features(system, ae, after)
    return dict(system=system, ae=ae, matrix=matrix, initial=initial, reference=reference,
                features=features, target=torch.as_tensor(split.target[i], device=compiled.device))


def iterate(item, prepared, context, opts, count, learned=True):
    action = context.compiled.action
    action.matrix = item['matrix']
    x = item['initial']
    values = []
    for k in range(count):
        if k:
            x = smooth(item['system'], action, item['ae'], x, opts.pre, opts.lower, opts)
        residual = item['system'].rhs - action.action(0, x, item['ae'])
        if learned:
            x = complement_update(x, residual, prepared)
        else:
            x = reference_update(x, residual, item['reference']).to(x.dtype)
        x = smooth(item['system'], action, item['ae'], x, opts.post, opts.post_lower, opts)
        values.append(x)
    return values


def train(args):
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device, compiled, context, no, stats, base, bank, info, cache_ms = setup(args)
    opts = settings(args, base)
    model_options = dict(width=args.width, rank=args.rank, fine_readout=not args.no_fine_readout)
    model = ComplementGraph(**model_options).to(device)
    norms = P1Norms(DarcySplit(args.data, 'val').mesh, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, eta_min=args.lr / 20)
    def cache(name, count):
        split = DarcySplit(args.data, name)
        a, _, xy, _, mask = tensors(split, device)
        with torch.no_grad():
            return [build_item(compiled, context, no, stats, bank, opts, split, a, xy, mask, i)
                    for i in range(min(count, len(split)))]
    started = clock(device)
    training, validation = cache('train', args.train_count), cache('val', args.val_count)
    def objective(item):
        compiled.action.matrix = item['matrix']
        h = model(item['features'], context, item['system'], item['matrix'])
        prepared = prepare_complement(item['matrix'], item['reference'], h, args.ridge)
        states = iterate(item, prepared, context, opts, 2)
        return torch.stack([norms.ratios(x, item['target'], item['system'], item['ae']) for x in states])
    history, best = [], float('inf')
    for epoch in range(args.epochs + 1):
        tick = clock(device)
        train_values = []
        if epoch:
            model.train()
            for idx in rng.permutation(len(training)):
                value = objective(training[idx])
                selected = value if args.target == 'K1K2' else value[-1:]
                loss = selected.clamp_min(1e-9).log().mean()
                if not bool(torch.isfinite(loss)):
                    raise RuntimeError('Nonfinite training loss')
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                optimizer.step()
                train_values.append(value.detach())
            scheduler.step()
        model.eval()
        with torch.no_grad():
            values = torch.stack([objective(v) for v in validation])
            selected = values if args.target == 'K1K2' else values[:, -1:]
            score = float(selected.clamp_min(1e-9).log().mean())
        if not np.isfinite(score):
            raise RuntimeError('Nonfinite validation score')
        row = dict(epoch=epoch, score=score, val_percent=(100 * values.mean(0)).tolist(),
                   seconds=clock(device)-tick)
        if train_values:
            row['train_percent'] = (100 * torch.stack(train_values).mean(0)).tolist()
        history.append(row)
        if score < best:
            best = score
            torch.save(dict(state=model.state_dict(), model_options=model_options, options=vars(opts),
                epoch=epoch, validation=row, ridge=args.ridge, trunk=info,
                arguments={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}), args.out / 'model.pt')
        save_json(args.out / 'training.json', dict(history=history, seconds=clock(device)-started,
            train_count=len(training), val_count=len(validation), cache_ms_once=cache_ms,
            parameters=sum(p.numel() for p in model.parameters()), seed=args.seed))
        print(json.dumps(row), flush=True)


def evaluate(args):
    device, compiled, context, no, stats, base, bank, info, cache_ms = setup(args)
    saved = torch.load(args.checkpoint, map_location=device, weights_only=False)
    opts = SimpleNamespace(**saved['options'])
    model = ComplementGraph(**saved['model_options']).to(device)
    model.load_state_dict(saved['state'])
    model.eval()
    split = DarcySplit(args.data, args.split)
    if args.count > len(split):
        raise ValueError('Requested test count exceeds split')
    a, _, xy, _, mask = tensors(split, device)
    methods = [(name, k) for name in args.variants for k in args.cycles]
    rows, stages, diagnostics = [], [], []
    with torch.no_grad():
        for i in range(args.count):
            def solve(variant, k, profile=False):
                parts = {}
                last = clock(device) if profile else 0.
                def tick(label):
                    nonlocal last
                    if profile:
                        now = clock(device)
                        parts[label] = 1000 * (now-last)
                        last = now
                full = predict(no, a[i:i+1], xy, mask, stats).flatten()
                tick('FNO_ms')
                system, ae, matrix = compiled.build(a[i].flatten(), torch.zeros_like(full))
                tick('assembly_ms')
                x = smooth(system, compiled.action, ae, full[system.free], opts.pre, opts.lower, opts)
                tick('prefilter_ms')
                reference = reference_factor(matrix, bank[0])
                tick('reference_factor_ms')
                item = dict(system=system, ae=ae, matrix=matrix, initial=x, reference=reference)
                if variant == 'complement':
                    residual = system.rhs - compiled.action.action(0, x, ae)
                    after = reference_update(x, residual, reference).to(x.dtype)
                    h = model(context.features(system, ae, after), context, system, matrix)
                    tick('neural_space_ms')
                    prepared = prepare_complement(matrix, reference, h, saved['ridge'])
                    tick('complement_factor_ms')
                    x = iterate(item, prepared, context, opts, k)[-1]
                else:
                    x = iterate(item, None, context, opts, k, learned=False)[-1]
                tick('correction_ms')
                out = torch.zeros_like(full)
                out[system.free] = x
                return (out, parts) if profile else out
            values = {}
            for variant, k in methods:
                name = variant + '_K' + str(k)
                out, ms = measure(lambda: solve(variant, k), device, args.repeats)
                if not bool(torch.isfinite(out).all()):
                    raise RuntimeError('Nonfinite prediction')
                values[name] = dict(field_metrics(out.cpu().numpy(), split.target[i],
                    split.coefficient[i], split.mesh), total_ms=ms)
                if i < 4:
                    _, parts = solve(variant, k, profile=True)
                    stages.append(dict(index=i, method=name, **parts))
            rows.append(dict(index=i, sample_index=int(split.sample_index[i]), methods=values))
            if i < 4:
                item = build_item(compiled, context, no, stats, bank, opts, split, a, xy, mask, i)
                h = model(item['features'], context, item['system'], item['matrix'])
                prepared = prepare_complement(item['matrix'], item['reference'], h, saved['ridge'])
                diagnostics.append(dict(index=i, **complement_diagnostics(prepared)))
            aggregate = {name: {key: summary([r['methods'][name][key] for r in rows])
                         for key in values[name]} for name in values}
            save_json(args.out / 'evaluation.json', dict(count=len(rows), split=args.split,
                summary=aggregate, samples=rows, stages=stages, diagnostics=diagnostics,
                options=vars(opts), checkpoint=str(args.checkpoint), selected_epoch=saved['epoch'],
                trunk=info, device=torch.cuda.get_device_name(device), cache_ms_once=cache_ms,
                mesh_plan_ms_once=compiled.mesh_plan_ms_once,
                assembly_plan_ms_once=compiled.assembly_plan_ms_once,
                timing='Batch1; warmup then synchronized wall median; FNO+assembly+space+all correction included; geometry-only cache separate',
                train_test='Identical architecture/filter/projection; validation-only checkpoint selection',
                reference_columns=int(bank[0].shape[1]), extra_columns=saved['model_options']['rank']))
            print(json.dumps(dict(index=i, L2={k: v['L2_percent'] for k, v in values.items()})), flush=True)
    keys = ['L2_percent', 'gradient_percent', 'flux_L2_percent', 'total_ms']
    with (args.out / 'summary.csv').open('w') as f:
        w = csv.writer(f)
        w.writerow(['method', *keys])
        for name, values in aggregate.items():
            w.writerow([name, *[values[key]['mean'] for key in keys]])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=['train', 'evaluate'])
    for name in ('data', 'no-dir', 'dictionary', 'trunk-model', 'out'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--checkpoint', type=Path)
    p.add_argument('--width', type=int, default=24)
    p.add_argument('--rank', type=int, default=8)
    p.add_argument('--no-fine-readout', action='store_true')
    p.add_argument('--pre', type=int)
    p.add_argument('--cutoff', type=float)
    p.add_argument('--power', type=int, choices=[0], default=0)
    p.add_argument('--ridge', type=float, default=1e-12)
    p.add_argument('--epochs', type=int, default=20)
    p.add_argument('--train-count', type=int, default=256)
    p.add_argument('--val-count', type=int, default=16)
    p.add_argument('--lr', type=float, default=.0003)
    p.add_argument('--seed', type=int, default=20261106)
    p.add_argument('--target', choices=['K2', 'K1K2'], default='K2')
    p.add_argument('--split', choices=['val', 'test'], default='test')
    p.add_argument('--count', type=int, default=32)
    p.add_argument('--cycles', type=int, nargs='+', default=[1, 2, 4])
    p.add_argument('--repeats', type=int, default=3)
    p.add_argument('--variants', choices=['complement', 'TB'], nargs='+', default=['complement', 'TB'])
    args = p.parse_args()
    if args.mode == 'train' and args.out.exists():
        p.error('Choose a new immutable training directory')
    if args.mode == 'evaluate' and not args.checkpoint:
        p.error('Evaluation requires --checkpoint')
    if min(args.rank, args.train_count, args.val_count, args.count, *args.cycles) < 1:
        p.error('Positive rank, counts, cycles required')
    args.out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    {'train': train, 'evaluate': evaluate}[args.mode](args)


if __name__ == '__main__':
    main()
