"""Train the official HINTS 2-D DeepONet architecture on current Darcy data."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from data import make_pairs
from model import HintsDeepONet


def train(args):
    torch.set_num_threads(2)
    torch.manual_seed(args.seed)
    device = torch.device('cuda:0')
    args.out.mkdir(parents=True,exist_ok=True)
    pairs = args.out/'pairs'
    if not (pairs/'manifest.json').exists():
        make_pairs(args.data,pairs,args.rhs_per_material,args.seed)
    raw = {s:dict(np.load(pairs/f'{s}.npz')) for s in ('train','val')}
    kmean, kstd = float(raw['train']['coefficient'].mean()), float(raw['train']['coefficient'].std())
    uscale = float(np.sqrt(np.mean(raw['train']['solution']**2)))
    model = HintsDeepONet(kmean,kstd,uscale).to(device)
    datasets = {s:{k:torch.as_tensor(v,device=device) for k,v in r.items()} for s,r in raw.items()}
    opt = torch.optim.Adam(model.parameters(),lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt,args.epochs,eta_min=args.lr/10)
    best,history = float('inf'),[]
    start = time.perf_counter()
    def loss(pred,target):
        # Official L2_relative, expressed in normalized solution units.
        return (((pred-target)/uscale).square()/((target/uscale).square()+1e-2)).mean()
    def validation():
        d = datasets['val']
        sums = []
        with torch.no_grad():
            trunk = model.trunk(d['coordinates'])
            for ids in torch.arange(len(d['rhs']),device=device).split(args.batch):
                p = model(d['coefficient'][d['material'][ids]],d['rhs'][ids],trunk=trunk)
                t = d['solution'][ids]
                sums.extend(((p-t).square().mean(1)/t.square().mean(1).clamp_min(1e-30)).sqrt().cpu().tolist())
        return float(np.mean(sums))
    for epoch in range(args.epochs):
        model.train()
        d = datasets['train']
        losses = []
        for ids in torch.randperm(len(d['rhs']),device=device).split(args.batch):
            pred = model(d['coefficient'][d['material'][ids]],d['rhs'][ids],d['coordinates'])
            objective = loss(pred,d['solution'][ids])
            opt.zero_grad(set_to_none=True)
            objective.backward()
            opt.step()
            losses.append(float(objective.detach()))
        sched.step()
        if epoch == 0 or (epoch+1)%10 == 0 or epoch+1 == args.epochs:
            model.eval()
            val = validation()
            record = dict(epoch=epoch+1,train_loss=float(np.mean(losses)),
                          val_relative_L2=val,seconds=time.perf_counter()-start)
            history.append(record)
            if val < best:
                best=val
                torch.save(dict(state=model.state_dict(), model_options=dict(
                    coefficient_mean=kmean,coefficient_std=kstd,solution_scale=uscale),
                    args={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
                    best_validation=best,best_epoch=epoch+1),args.out/'model.pt')
            print(json.dumps(record),flush=True)
            (args.out/'history.json').write_text(json.dumps(history,indent=2)+'\n')
    report = dict(epochs=args.epochs,best_validation_relative_L2=best,
                  training_seconds=time.perf_counter()-start,
                  parameters=sum(p.numel() for p in model.parameters()),
                  device=torch.cuda.get_device_name(device),
                  data=json.loads((pairs/'manifest.json').read_text()))
    (args.out/'training.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--epochs',type=int,default=1000)
    p.add_argument('--rhs-per-material',type=int,default=16)
    p.add_argument('--batch',type=int,default=128)
    p.add_argument('--lr',type=float,default=1e-3)
    p.add_argument('--seed',type=int,default=20261104)
    train(p.parse_args())
