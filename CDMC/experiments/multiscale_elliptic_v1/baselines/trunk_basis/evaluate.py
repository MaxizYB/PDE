"""DeepONet trunk-space replacement in the deployed CDMC solver.

TB follows Kopanicakova & Karniadakis, arXiv:2401.02016, section 4.2:
coordinate trunk evaluation, homogeneous boundary lifting, QR, current-A
Galerkin coarse solve. This is a matched correction-space experiment, not
the paper's outer Krylov benchmark. No network is retrained here.
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

EXP = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(EXP/'baselines/hints'))
from model import HintsDeepONet
sys.path.insert(0, str(EXP/'refinement'))
sys.path.insert(0, str(EXP))
from run import DarcySplit, tensors, load_no, predict, clock, measure, field_metrics, summary, save_json
from polynomial_joint import SparseCompiled
from coupled_coarse import CoarseContext
from study import load_dictionary, smooth, prepare, cycles
from core import calibrated_update


def trunk_spaces(path, coordinates, free, device):
    saved = torch.load(path, map_location=device, weights_only=False)
    model = HintsDeepONet(**saved['model_options']).to(device)
    model.load_state_dict(saved['state'])
    # The paper computes its bases in double precision. Casting pretrained
    # weights preserves the learned functions while avoiding FP32 QR noise.
    model = model.double().eval()
    with torch.no_grad():
        raw = model.trunk(coordinates.double())[free]
        q, r = torch.linalg.qr(raw, mode='reduced')
        diagonal = r.diagonal().abs()
        keep = diagonal > 1e-10*diagonal.max()
        spaces = {'TB80':q.float(), 'TBqr':q[:,keep].float()}
        sv = torch.linalg.svdvals(raw)
    info = dict(checkpoint=str(path),raw_columns=raw.shape[1],
                full_qr_columns=q.shape[1],stable_qr_columns=int(keep.sum()),
                qr_relative_threshold=1e-10,qr_diagonal=diagonal.cpu().tolist(),
                singular_values=sv.cpu().tolist(),
                orthogonality_error=float((q.T@q-torch.eye(q.shape[1],device=device)).norm()),
                boundary_lift='x(x-1)y(y-1)',evaluation_precision='float64',
                training=saved['args'])
    return spaces, info


def evaluate(args):
    torch.set_num_threads(2)
    device = torch.device('cuda:0')
    split = DarcySplit(args.data,args.split)
    if args.count > len(split):
        raise ValueError('Requested more samples than the split contains')
    a, _, xy, _, mask = tensors(split,device)
    compiled = SparseCompiled(split.mesh,device)
    context = CoarseContext(compiled,1)
    fno, stats, _ = load_no(args.no_dir,device)
    dictionary, saved = load_dictionary(args.dictionary,device)
    dictionary.eval()
    opts = SimpleNamespace(**saved['options'])
    compact = None
    if args.compact_dictionary:
        compact, compact_saved = load_dictionary(args.compact_dictionary,device)
        compact.eval()
    started = clock(device)
    bases, tb_info = trunk_spaces(args.trunk_model,xy.reshape(-1,2),compiled.plan.free,device)
    tb_info['cached_trunk_and_QR_ms_once'] = 1000*(clock(device)-started)
    methods = ['FNO','filter_only_K2']
    for space in ('TB80_native','TB80_shaped','TBqr_native'):
        methods += [f'{space}_K{k}' for k in args.cycles]
    for space in (['CDMC128','CDMC64'] if compact is not None else ['CDMC128']):
        methods += [f'{space}_K{k}' for k in args.cycles]
    rows, stages, diagnostics = [], [], []
    with torch.no_grad():
        for i in range(args.count):
            def solve(name, profile=False):
                parts, last = {}, clock(device) if profile else 0.
                def tick(label):
                    nonlocal last
                    if profile:
                        now=clock(device)
                        parts[label]=1000*(now-last)
                        last=now
                full = predict(fno,a[i:i+1],xy,mask,stats).flatten()
                tick('FNO_ms')
                if name=='FNO':
                    return full
                system, ae, _ = compiled.build(a[i].flatten(),torch.zeros_like(full))
                tick('fine_assembly_ms')
                cfg = SimpleNamespace(**vars(opts))
                x = smooth(system,compiled.action,ae,full[system.free],cfg.pre,cfg.lower,cfg)
                tick('prefilter_ms')
                if name=='filter_only_K2':
                    x = smooth(system,compiled.action,ae,x,cfg.pre+2*cfg.post,cfg.lower,cfg)
                else:
                    if name.startswith('TB'):
                        modes = bases['TBqr' if name.startswith('TBqr') else 'TB80']
                        if '_native_' in name:
                            cfg.harmonic_steps=0
                    else:
                        net = compact if name.startswith('CDMC64') else dictionary
                        modes = net(context.features(system,ae,x),context)
                    tick('space_generation_ms')
                    prepared = prepare(modes,context,system,ae,cfg)
                    tick('space_factor_ms')
                    x = cycles(x,prepared,system,ae,context,cfg,int(name.rsplit('K',1)[1]),filtered=True)
                tick('correction_and_postfilter_ms')
                out = torch.zeros_like(full)
                out[system.free]=x
                return (out,parts) if profile else out
            values = {}
            for name in methods:
                out, ms = measure(lambda:solve(name),device,args.repeats)
                if not bool(torch.isfinite(out).all()):
                    raise RuntimeError(f'Nonfinite result {name}, sample {i}')
                metrics = field_metrics(out.cpu().numpy(),split.target[i],split.coefficient[i],split.mesh)
                values[name]=dict(metrics,total_ms=ms)
            if i < 4:
                for name in ('TB80_native_K2','TB80_shaped_K2','CDMC128_K2'):
                    _, parts=solve(name,profile=True)
                    stages.append(dict(index=i,method=name,**parts))
            # Ground truth is used below only for offline explanation, outside
            # prediction and timing. All spaces have already been evaluated.
            initial=predict(fno,a[i:i+1],xy,mask,stats).flatten()
            system,ae,_=compiled.build(a[i].flatten(),torch.zeros_like(initial))
            filtered=smooth(system,compiled.action,ae,initial[system.free],opts.pre,opts.lower,opts)
            target=torch.as_tensor(split.target[i],device=device)[system.free]
            error=target-filtered
            before=torch.dot(error,compiled.action.action(0,error,ae))
            spaces=dict(TB80_native=bases['TB80'],TB80_shaped=bases['TB80'],TBqr_native=bases['TBqr'],
                        CDMC128=dictionary(context.features(system,ae,filtered),context))
            if compact is not None:
                spaces['CDMC64']=compact(context.features(system,ae,filtered),context)
            diagnostic={}
            for name,modes in spaces.items():
                cfg=SimpleNamespace(**vars(opts))
                if name.endswith('native'):
                    cfg.harmonic_steps=0
                prepared=prepare(modes,context,system,ae,cfg)
                updated=calibrated_update(prepared,system,ae,filtered,context,getattr(cfg,'calibration_sweeps',1))
                error=target-updated
                after=torch.dot(error,compiled.action.action(0,error,ae))
                diagnostic[name]=dict(captured_actual_error_energy_percent=100*float(1-after/before),
                                      columns=modes.shape[1])
            diagnostics.append(dict(index=i,spaces=diagnostic))
            rows.append(dict(index=i,sample_index=int(split.sample_index[i]),methods=values))
            aggregate={name:{key:summary([r['methods'][name][key] for r in rows])
                             for key in values[name]} for name in methods}
            save_json(args.out/'evaluation.json',dict(summary=aggregate,samples=rows,stages=stages,
                diagnostics=diagnostics,trunk=tb_info,data=str(args.data),split=args.split,
                dictionary=str(args.dictionary),compact_dictionary=str(args.compact_dictionary),
                no_dir=str(args.no_dir),options=vars(opts),cycles=args.cycles,count=len(rows),
                device=torch.cuda.get_device_name(device),repeats=args.repeats,
                mesh_plan_ms_once=compiled.mesh_plan_ms_once,
                assembly_plan_ms_once=compiled.assembly_plan_ms_once,
                timing='Batch1; fresh FNO, fine assembly, material-space preparation, factor and all corrections. Mesh-only plans and fixed pretrained trunk/QR cached separately. One warmup then synchronized wall median.'))
            print(json.dumps(dict(index=i,L2_percent={k:round(v['L2_percent'],7) for k,v in values.items()})),flush=True)
    keys=['L2_percent','gradient_percent','energy_percent','flux_L2_percent','total_ms']
    with (args.out/'summary.csv').open('w') as stream:
        writer=csv.writer(stream)
        writer.writerow(['method',*keys])
        for name in methods:
            writer.writerow([name,*[aggregate[name][k]['mean'] for k in keys]])


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('data','no-dir','trunk-model','dictionary','out'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--compact-dictionary',type=Path)
    p.add_argument('--split',choices=('val','test'),default='test')
    p.add_argument('--count',type=int,default=32)
    p.add_argument('--cycles',type=int,nargs='+',default=[1,2,4])
    p.add_argument('--repeats',type=int,default=3)
    evaluate(p.parse_args())
