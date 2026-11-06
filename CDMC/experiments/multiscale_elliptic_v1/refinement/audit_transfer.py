"""Separate predictor amplitude, physical filtering and response-space coverage.

Reference fields enter offline diagnostics only, never the deployed corrector.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from run import DarcySplit, tensors, predict, save_json
from study import load_dictionary, smooth, prepare
from predictors import load_predictor
from polynomial_joint import SparseCompiled, P1Norms
from coupled_coarse import CoarseContext, mm
from core import calibrated_update


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, nargs='+', required=True)
    p.add_argument('--no-dir', type=Path, required=True)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--count', type=int, default=4)
    args=p.parse_args()
    if args.out.exists():
        raise FileExistsError('Keep completed diagnostics immutable.')
    torch.set_num_threads(2)
    device=torch.device('cuda')
    model,saved=load_dictionary(args.checkpoint,device)
    model.eval()
    no,stats,_=load_predictor(args.no_dir,device)
    opts=SimpleNamespace(**saved['options'])
    opts.matrix_bound=True
    cases=[]
    for data in args.data:
        split=DarcySplit(data,'test')
        compiled=SparseCompiled(split.mesh,device)
        context=CoarseContext(compiled,1)
        norms=P1Norms(split.mesh,device)
        a,y,xy,mass,mask=tensors(split,device)
        rows=[]
        with torch.no_grad():
            for i in range(min(args.count,len(split))):
                full=predict(no,a[i:i+1],xy,mask,stats).flatten()
                system,ae,matrix=compiled.build(a[i].flatten(),torch.zeros_like(full))
                x=full[system.free]
                target=y[i].flatten()
                alpha=(mass.flatten()*full*target).sum()/(mass.flatten()*full.square()).sum().clamp_min(1e-20)
                raw=norms.ratios(x,target,system,ae)
                oracle=norms.ratios(alpha*x,target,system,ae)
                filtered=smooth(system,compiled.action,ae,x,opts.pre,opts.lower,opts)
                filtered_error=norms.ratios(filtered,target,system,ae)
                prepared=prepare(model(context.features(system,ae,filtered),context),context,system,ae,opts)
                def energy(state):
                    e=target[system.free]-state
                    return (e.double()*mm(matrix,e[:,None])[:,0].double()).sum()
                initial_energy=energy(filtered)
                captures={}
                for count in (1,2,4):
                    out=calibrated_update(prepared,system,ae,filtered,context,count)
                    captures[str(count)]=float(1-energy(out)/initial_energy)
                rows.append(dict(index=i,NO_percent=(100*raw).tolist(),
                    oracle_amplitude=float(alpha),oracle_amplitude_percent=(100*oracle).tolist(),
                    prefilter_percent=(100*filtered_error).tolist(),captured_energy_fraction=captures))
        cases.append(dict(data=str(data),samples=rows))
    save_json(args.out,dict(cases=cases,
        interpretation='Amplitude fit uses test truth only as an offline diagnostic; it is not a deployable method. Energy capture is measured on the actual postfilter error of each current equation.'))
    for case in cases:
        rows=case['samples']
        print(case['data'],dict(NO=np.mean([r['NO_percent'] for r in rows],0).tolist(),
            amplitude_fit=np.mean([r['oracle_amplitude_percent'] for r in rows],0).tolist(),
            filtered=np.mean([r['prefilter_percent'] for r in rows],0).tolist(),
            captured={k:float(np.mean([r['captured_energy_fraction'][k] for r in rows])) for k in ('1','2','4')}),flush=True)


if __name__=='__main__':
    main()
