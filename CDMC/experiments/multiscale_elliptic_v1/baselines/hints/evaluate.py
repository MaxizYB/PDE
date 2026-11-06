"""Same-matrix Darcy comparison: HINTS-Jacobi, FNO and learned dictionary.

One HINTS block is 14 damped-Jacobi steps and one DeepONet residual
correction, matching the paper's 2-D ratio 1:14. FEM residual coefficients
are divided by lumped nodal mass before interpolation to source sensors.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from model import HintsDeepONet, sensors

EXP = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(EXP/'refinement'))
sys.path.insert(0, str(EXP))
from run import DarcySplit, tensors, predict, load_no, clock, measure, field_metrics, summary, save_json
from polynomial_joint import SparseCompiled
from coupled_coarse import CoarseContext
from study import load_dictionary, smooth, prepare, cycles


def evaluate(args):
    torch.set_num_threads(2)
    device = torch.device('cuda:0')
    split = DarcySplit(args.data, args.split)
    n = int(round(np.sqrt(split.target.shape[1])))
    a, _, xy, mass, mask = tensors(split, device)
    saved = torch.load(args.model, map_location=device, weights_only=False)
    hints = HintsDeepONet(**saved['model_options']).to(device).eval()
    hints.load_state_dict(saved['state'])
    fno, stats, _ = load_no(args.no_dir, device)
    compiled = SparseCompiled(split.mesh, device)
    coords = xy.reshape(-1, 2)
    dictionary, dictionary_saved, context = None, None, None
    if args.dictionary:
        dictionary, dictionary_saved = load_dictionary(args.dictionary, device)
        dictionary.eval()
        context = CoarseContext(compiled, 1)
    methods = ['FNO', 'DeepONet']
    for init in ('zero', 'FNO'):
        for b in args.blocks:
            methods.extend([f'{init}_HINTS_B{b}', f'{init}_Jacobi_J{15*b}'])
    if dictionary is not None:
        methods.extend(['dictionary_K1', 'dictionary_K2', 'dictionary_K4'])
    stages, rows = [], []
    # A geometry-only trunk is cached for every problem on this fixed mesh.
    trunk_start = clock(device)
    with torch.no_grad():
        trunk = hints.trunk(coords)
    trunk_ms = 1000*(clock(device)-trunk_start)
    with torch.no_grad():
        for i in range(args.count):
            def solve(name, profile=False):
                time_parts, last = {}, clock(device) if profile else 0.
                def tick(label):
                    nonlocal last
                    if profile:
                        now = clock(device)
                        time_parts[label] = 1000*(now-last)
                        last = now
                if name == 'FNO':
                    return predict(fno, a[i:i+1], xy, mask, stats).flatten()
                if name == 'DeepONet':
                    ks = sensors(a[i:i+1])
                    return hints(ks, torch.ones_like(ks), trunk=trunk)[0]
                initial = (predict(fno, a[i:i+1], xy, mask, stats).flatten()
                           if name.startswith('FNO_') or name.startswith('dictionary')
                           else torch.zeros(n*n,device=device))
                tick('NO_ms')
                system, ae, _ = compiled.build(a[i].flatten(), torch.zeros_like(initial))
                tick('fine_assembly_ms')
                x = initial[system.free]
                if name.startswith('dictionary'):
                    cfg = SimpleNamespace(**dictionary_saved['options'])
                    x = smooth(system, compiled.action, ae, x, cfg.pre, cfg.lower, cfg)
                    modes = dictionary(context.features(system,ae,x),context)
                    prepared = prepare(modes,context,system,ae,cfg)
                    x = cycles(x,prepared,system,ae,context,cfg,int(name[-1]),filtered=True)
                    tick('dictionary_and_updates_ms')
                else:
                    ks = sensors(a[i:i+1])
                    tick('coefficient_sensors_ms')
                    diagonal_inverse = system.levels[0].diagonal_inverse
                    def residual(value):
                        return system.rhs-compiled.action.action(0,value,ae)
                    count = int(name.rsplit('_',1)[1][1:])
                    for _ in range(count):
                        for _ in range(14 if '_HINTS_' in name else 1):
                            x = x+0.8*diagonal_inverse*residual(x)
                        if '_HINTS_' in name:
                            source = torch.zeros(n*n,device=device)
                            source[system.free] = residual(x)/mass.flatten()[system.free]
                            fs = sensors(source.reshape(1,n,n))
                            x = x+hints(ks,fs,trunk=trunk)[0,system.free]
                    tick('relaxation_and_neural_updates_ms')
                full = torch.zeros(n*n,device=device)
                full[system.free] = x
                return (full,time_parts) if profile else full
            results = {}
            for name in methods:
                full, ms = measure(lambda:solve(name),device,args.repeats)
                if not bool(torch.isfinite(full).all()):
                    raise RuntimeError(f'Nonfinite HINTS result: sample={i}, method={name}')
                metrics = field_metrics(full.cpu().numpy(),split.target[i],split.coefficient[i],split.mesh)
                if name != 'FNO' and name != 'DeepONet':
                    system,ae,_ = compiled.build(a[i].flatten(),torch.zeros_like(full))
                    r = system.rhs-compiled.action.action(0,full[system.free],ae)
                    metrics['residual_relative'] = float(r.norm()/system.rhs.norm())
                results[name] = dict(metrics,total_ms=ms)
            if i < 4:
                _,parts = solve('FNO_HINTS_B2',True)
                stages.append(dict(index=i,**parts))
            rows.append(dict(index=i,sample_index=int(split.sample_index[i]),methods=results))
            aggregate = {name:{key:summary([row['methods'][name][key] for row in rows])
                                for key in results[name]} for name in methods}
            save_json(args.out/'evaluation.json',dict(summary=aggregate,samples=rows,stages=stages,
                data=str(args.data),split=args.split,no_dir=str(args.no_dir),model=str(args.model),
                dictionary=str(args.dictionary),device=torch.cuda.get_device_name(device),
                count=len(rows),blocks=args.blocks,repeats=args.repeats,
                hints=dict(jacobi_omega=0.8,numerical_steps_per_block=14,
                           neural_steps_per_block=1,source_conversion='r / lumped P1 mass'),
                training=saved['args'],mesh_plan_ms_once=compiled.mesh_plan_ms_once,
                assembly_plan_ms_once=compiled.assembly_plan_ms_once,trunk_cache_ms_once=trunk_ms,
                timing='Batch1 synchronized wall median; one warmup; fresh initial prediction (FNO branches), material assembly, sensors and all updates per call. Mesh plan and fixed-geometry DeepONet trunk are cached and listed separately.'))
            print(json.dumps(dict(index=i,L2_percent={k:round(v['L2_percent'],6) for k,v in results.items()})),flush=True)
    keys = ['L2_percent','gradient_percent','energy_percent','flux_L2_percent','total_ms']
    with (args.out/'summary.csv').open('w') as stream:
        writer = csv.writer(stream)
        writer.writerow(['method',*keys])
        for name in methods:
            writer.writerow([name,*[aggregate[name][key]['mean'] for key in keys]])


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for arg in ('data','model','no-dir','out'):
        p.add_argument('--'+arg,type=Path,required=True)
    p.add_argument('--dictionary',type=Path)
    p.add_argument('--split',choices=('val','test'),default='test')
    p.add_argument('--count',type=int,default=32)
    p.add_argument('--blocks',type=int,nargs='+',default=[1,2,4,8,16,32])
    p.add_argument('--repeats',type=int,default=3)
    evaluate(p.parse_args())
