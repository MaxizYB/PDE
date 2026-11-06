"""Controlled simplification of the existing fine-equation response corrector.

Reuse the validated assembly, graph hierarchy, filters, norms and calibration.
The new dictionary uses a nonlinear readout instead of Cartesian window products
or a concatenated sine space. Train/validation choose models; tests are held out.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

PARENT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PARENT))
from run import (DarcySplit, clock, save_json, summary, tensors, predict,
                 measure, field_metrics)
from predictors import load_predictor as load_no
from core import fixed_filter, calibrate_dictionary, calibrated_update
from polynomial_joint import (SparseCompiled, P1Norms, chebyshev,
                              shape_response)
from coupled_coarse import CoupledCoarseSetup, CoarseContext, apply_modes, mm
from depth_study import geometric
import numpy as np
import torch


class ResponseDictionary(CoupledCoarseSetup):
    """One shared multilevel graph trunk and a nonlinear spatial dictionary.

    A linear r-channel readout of w features has spatial rank at most w+1.
    The elementwise nonlinearity lifts that restriction before calibration.
    The same model class is used for every coefficient family.
    """
    def __init__(self, width=16, rank=64, start_level=1, nonlinear=True):
        super().__init__(width, rank, start_level)
        self.nonlinear = nonlinear

    def forward(self, features, context):
        modes = super().forward(features, context)
        return torch.tanh(modes) if self.nonlinear else modes


def load_dictionary(path, device):
    saved = torch.load(path, map_location=device, weights_only=False)
    model = ResponseDictionary(**saved['model_options']).to(device)
    model.load_state_dict(saved['state'])
    return model, saved


def options(args):
    return SimpleNamespace(pre=args.pre, post=args.post, lower=args.cutoff,
                           post_lower=args.post_cutoff,
                           harmonic_steps=args.harmonic_steps, partition_grid=1,
                           matrix_bound=args.matrix_bound,
                           calibration_sweeps=args.calibration_sweeps or 1)


def smooth(system, action, ae, initial, steps, cutoff, opts):
    function = fixed_filter if getattr(opts,'matrix_bound',False) else chebyshev
    return function(system,action,ae,initial,steps,cutoff)


def prepare(modes,context,system,ae,opts):
    if getattr(opts,'matrix_bound',False):
        return calibrate_dictionary(modes,context,system,ae,opts)
    return shape_response(modes,context,system,ae,opts)


def cycles(initial, prepared, system, ae, context, opts, count, filtered=False):
    x = initial
    for k in range(count):
        if not (filtered and k == 0):
            x = smooth(system, context.compiled.action, ae, x, opts.pre, opts.lower,opts)
        x = calibrated_update(prepared, system, ae, x, context,
                              getattr(opts,'calibration_sweeps',1))
        x = smooth(system, context.compiled.action, ae, x, opts.post, opts.post_lower,opts)
    return x


def cache(args, split_name, count, compiled, context):
    split = DarcySplit(args.data, split_name)
    with np.load(args.no_dir/'predictions.npz') as z:
        predictions = z[split_name]
    rows = []
    with torch.no_grad():
        for i in range(min(count, len(split))):
            system, ae, matrix = compiled.build(split.coefficient[i], split.lift[i])
            initial = torch.as_tensor(predictions[i], device=compiled.device)[system.free]
            filtered = smooth(system, compiled.action, ae, initial, args.pre, args.cutoff,options(args))
            rows.append(dict(system=system, ae=ae, matrix=matrix, initial=initial,
                             filtered=filtered, features=context.features(system, ae, filtered),
                             target=torch.as_tensor(split.target[i], device=compiled.device)))
    return split, rows


def train(args, device):
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    split = DarcySplit(args.data, 'train')
    compiled = SparseCompiled(split.mesh, device)
    context, norms = CoarseContext(compiled, 1), P1Norms(split.mesh, device)
    opts = options(args)
    model_options = dict(width=args.width, rank=args.rank, start_level=1,
                         nonlinear=not args.linear)
    model = ResponseDictionary(**model_options).to(device)
    if args.initialize:
        previous = torch.load(args.initialize, map_location=device, weights_only=False)['state']
        matched = {k:v for k,v in previous.items()
                   if k in model.state_dict() and v.shape == model.state_dict()[k].shape}
        model.load_state_dict(matched, strict=False)
        # Nested initialization preserves every old response when rank grows.
        # Remaining channels keep their declared random initialization.
        if ('output.weight' in previous and previous['output.weight'].shape[1] == args.width
                and previous['output.weight'].shape[0] < args.rank):
            old_rank = previous['output.weight'].shape[0]
            with torch.no_grad():
                model.output.weight[:old_rank].copy_(previous['output.weight'])
                model.output.bias[:old_rank].copy_(previous['output.bias'])
    started = clock(device)
    _, tr = cache(args, 'train', args.train_count, compiled, context)
    _, va = cache(args, 'val', args.val_count, compiled, context)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs, eta_min=1e-5)
    history, best = [], float('inf')

    def objective(item):
        compiled.action.matrix = item['matrix']
        prepared = prepare(model(item['features'], context), context,
                                  item['system'], item['ae'], opts)
        x, values = item['filtered'], []
        for k in range(args.cycles):
            x = cycles(x, prepared, item['system'], item['ae'], context, opts, 1,
                       filtered=(k == 0))
            values.append(norms.ratios(x, item['target'], item['system'], item['ae']))
        return torch.stack(values)

    for epoch in range(args.epochs):
        tick = clock(device)
        model.train()
        train_values = []
        for i in rng.permutation(len(tr)):
            value = objective(tr[i])
            loss = (value.clamp_min(1e-6).log().mean() if args.objective == 'log'
                    else value.square().mean().sqrt())
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.)
            opt.step()
            train_values.append(value.detach())
        sched.step()
        model.eval()
        with torch.no_grad():
            val = torch.stack([objective(item) for item in va])
        score = float(val.clamp_min(1e-6).log().mean() if args.objective == 'log'
                      else val.square().mean().sqrt())
        if not np.isfinite(score):
            raise RuntimeError('Nonfinite validation; keep this run for diagnosis.')
        row = dict(epoch=epoch+1, train_percent=(100*torch.stack(train_values).mean(0)).tolist(),
                   val_percent=(100*val.mean(0)).tolist(), seconds=clock(device)-tick)
        history.append(row)
        if score < best:
            best = score
            torch.save(dict(state=model.state_dict(), model_options=model_options,
                            options=vars(opts), validation=row, epoch=epoch+1,
                            seed=args.seed, training_cycles=args.cycles,
                            objective=args.objective), args.out/'model.pt')
        save_json(args.out/'training.json', dict(history=history, seconds=clock(device)-started,
                  parameters=sum(p.numel() for p in model.parameters()), best_score=best,
                  train_count=len(tr), val_count=len(va)))
        print(json.dumps(row), flush=True)


def evaluate(args, device):
    model, saved = load_dictionary(args.checkpoint, device)
    model.eval()
    opts = SimpleNamespace(**saved['options'])
    if args.matrix_bound:
        opts.matrix_bound = True
    if args.calibration_sweeps is not None:
        opts.calibration_sweeps = args.calibration_sweeps
    if args.compare_calibration:
        opts.calibration_sweeps = 1
    split = DarcySplit(args.data, args.split)
    compiled = SparseCompiled(split.mesh, device)
    context = CoarseContext(compiled, 1)
    no, stats, no_saved = load_no(args.no_dir, device)
    predictor_name = no_saved.get('predictor_type', 'FNO')
    a, y, xy, _, mask = tensors(split, device)
    norms = P1Norms(split.mesh, device)
    old = None
    if args.reference:
        old_saved = torch.load(args.reference, map_location=device, weights_only=False)
        old = CoupledCoarseSetup(**old_saved['model_options']).to(device)
        old.load_state_dict(old_saved['state'])
        old.eval()
        old_opts = SimpleNamespace(**old_saved['options'])
    names = [predictor_name, 'filter_only_K2', 'dictionary_K1', 'dictionary_K2', 'dictionary_K4']
    if args.compare_calibration:
        names += ['dictionary_D2_K2', 'dictionary_D4_K2']
    if old is not None:
        names += ['old_learned72_K2', 'old_union144_K2', 'old_union144_K4']
    rows = []
    stages = []
    with torch.no_grad():
        for i in range(args.offset, min(args.offset+args.count, len(split))):
            def solve(name, profile=False):
                times, last = {}, clock(device) if profile else 0.
                def tick(label):
                    nonlocal last
                    if profile:
                        now = clock(device)
                        times[label] = 1000*(now-last)
                        last = now
                full = predict(no, a[i:i+1], xy, mask, stats).flatten()
                tick('NO_ms')
                if name == predictor_name:
                    return (full, times) if profile else full
                system, ae, _ = compiled.build(a[i].flatten(), torch.zeros_like(full))
                tick('fine_assembly_ms')
                use_old = name.startswith('old_')
                cfg = SimpleNamespace(**vars(old_opts if use_old else opts))
                if name.startswith('dictionary_D'):
                    cfg.calibration_sweeps = int(name.split('_')[1][1:])
                x = smooth(system, compiled.action, ae, full[system.free], cfg.pre, cfg.lower,cfg)
                tick('prefilter_ms')
                if name.startswith('filter_only'):
                    x = smooth(system, compiled.action, ae, x, cfg.pre+2*cfg.post, cfg.lower,cfg)
                else:
                    features = context.features(system, ae, x)
                    tick('features_ms')
                    basis = (old if use_old else model)(features, context)
                    if name.startswith('old_union'):
                        basis = torch.cat((geometric(context), basis), 1)
                    tick('graph_dictionary_ms')
                    prepared = prepare(basis, context, system, ae, cfg)
                    tick('space_and_factor_ms')
                    x = cycles(x, prepared, system, ae, context, cfg, int(name[-1]), filtered=True)
                tick('calibration_and_filters_ms')
                out = torch.zeros_like(full)
                out[system.free] = x
                return (out, times) if profile else out
            methods = {}
            for name in names:
                out, elapsed = measure(lambda:solve(name), device, args.repeats)
                metrics = field_metrics(out.cpu().numpy(), split.target[i], split.coefficient[i], split.mesh)
                methods[name] = dict(metrics, total_ms=elapsed)
            if i < args.offset+4:
                _, profile = solve('dictionary_K2', profile=True)
                stages.append(dict(index=i, **profile))
            rows.append(dict(index=i, sample_index=int(split.sample_index[i]), methods=methods))
            aggregated = {name:{key:summary([r['methods'][name][key] for r in rows])
                               for key in methods[name]} for name in names}
            save_json(args.out/'evaluation.json', dict(summary=aggregated, samples=rows, stages=stages,
                      device=torch.cuda.get_device_name(device), checkpoint=str(args.checkpoint),
                      model_options=saved['model_options'], options=vars(opts),
                      split=args.split, predictor=predictor_name, matrix_bound=getattr(opts,'matrix_bound',False), mesh_plan_ms_once=compiled.mesh_plan_ms_once,
                      timing='Batch1; inputs on GPU; fresh NO, assembly, dictionary, factor and updates per call; one warmup; synchronized wall median. Mesh plan separate.'))
            print(json.dumps(dict(index=i, values={k:round(v['L2_percent'],6) for k,v in methods.items()})), flush=True)
    keys = ['L2_percent','gradient_percent','flux_L2_percent','total_ms']
    with (args.out/'summary.csv').open('w') as stream:
        writer = csv.writer(stream)
        writer.writerow(['method', *keys])
        for name in names:
            writer.writerow([name, *[aggregated[name][key]['mean'] for key in keys]])


def cache_no(args, device):
    from run import get_predictions
    args.batch_size = 8
    result = get_predictions(args, device)
    save_json(args.out/'predictions.json', dict(counts={k:len(v) for k,v in result.items()},
              checkpoint=str(args.no_dir/'no.pt')))


def diagnose(args,device):
    model,saved=load_dictionary(args.checkpoint,device)
    model.eval()
    opts=SimpleNamespace(**saved['options'])
    for name in ('pre','post','harmonic_steps','matrix_bound'):
        if hasattr(opts,name):
            setattr(args,name,getattr(opts,name))
    args.cutoff,args.post_cutoff=opts.lower,opts.post_lower
    split=DarcySplit(args.data,'val')
    compiled=SparseCompiled(split.mesh,device)
    context=CoarseContext(compiled,1)
    _,items=cache(args,'val',args.count,compiled,context)
    rows=[]
    with torch.no_grad():
        for i,item in enumerate(items):
            compiled.action.matrix=item['matrix']
            raw=CoupledCoarseSetup.forward(model,item['features'],context)
            nonlin=model(item['features'],context)
            ix=torch.linspace(0,len(raw)-1,min(512,len(raw)),device=device).long()
            ranks={}
            for name,modes in [('linear_readout',raw),('nonlinear_dictionary',nonlin)]:
                values=modes[ix].double()
                values=values/values.norm(dim=0).clamp_min(1e-20)
                sv=torch.linalg.svdvals(values)
                ranks[name]=dict(rank_at_relative_1e_6=int((sv>1e-6*sv[0]).sum()),
                                  singular_values=sv.cpu().tolist())
            prepared=prepare(nonlin,context,item['system'],item['ae'],opts)
            before=item['target'][item['system'].free]-item['filtered']
            after=item['target'][item['system'].free]-apply_modes(prepared,item['system'],
                item['ae'],item['filtered'],context)
            def energy(v):
                return (v.double()*mm(item['matrix'],v[:,None])[:,0].double()).sum()
            rows.append(dict(index=i,rank=ranks,captured_energy_fraction=float(1-energy(after)/energy(before))))
    save_json(args.out/'diagnosis.json',dict(split='val',samples=rows,
              theoretical_linear_rank_bound=model.width+1,
              note='512 deterministic evenly spaced nodes for numerical rank; actual current-A error capture on the complete mesh; ground truth used only for offline diagnosis.'))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage', choices=['train','evaluate','cache-no','diagnose'])
    for name in ('data','no-dir','out'):
        p.add_argument('--'+name, type=Path, required=True)
    for name in ('initialize','checkpoint','reference'):
        p.add_argument('--'+name, type=Path)
    p.add_argument('--rank', type=int, default=64)
    p.add_argument('--width', type=int, default=16)
    p.add_argument('--linear', action='store_true')
    p.add_argument('--pre', type=int, default=64)
    p.add_argument('--post', type=int, default=8)
    p.add_argument('--cutoff', type=float, default=.001)
    p.add_argument('--post-cutoff', type=float, default=.05)
    p.add_argument('--harmonic-steps', type=int, default=2)
    p.add_argument('--matrix-bound',action='store_true',help='Use a certified matrix-derived bound, including for dictionary shaping.')
    p.add_argument('--calibration-sweeps',type=int,default=None,
                   help='Fixed small-system debiasing count; default is the checkpoint policy or one.')
    p.add_argument('--compare-calibration',action='store_true',
                   help='Compare one/two/four small-system sweeps at K2, overriding checkpoint calibration count.')
    p.add_argument('--cycles', type=int, default=2)
    p.add_argument('--epochs', type=int, default=30)
    p.add_argument('--objective',choices=['rms','log'],default='rms')
    p.add_argument('--lr', type=float, default=.0003)
    p.add_argument('--train-count', type=int, default=48)
    p.add_argument('--val-count', type=int, default=12)
    p.add_argument('--split', choices=['val','test'], default='test')
    p.add_argument('--count', type=int, default=16)
    p.add_argument('--offset', type=int, default=0)
    p.add_argument('--repeats', type=int, default=3)
    p.add_argument('--seed', type=int, default=20261103)
    p.add_argument('--device', default='cuda')
    args = p.parse_args()
    if args.calibration_sweeps is not None and args.calibration_sweeps < 1:
        p.error('--calibration-sweeps must be positive.')
    if args.out.exists():
        p.error('Choose a new output directory; runs are immutable.')
    args.out.mkdir(parents=True)
    torch.set_num_threads(2)
    save_json(args.out/'provenance.json', dict(arguments={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
    {'train':train, 'evaluate':evaluate, 'cache-no':cache_no,
     'diagnose':diagnose}[args.stage](args, torch.device(args.device))


if __name__ == '__main__':
    main()
