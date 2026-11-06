"""Matched training of pinned official neural operators on aligned P1 labels."""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from predictors import PREDICTORS, load_predictor
from run import DarcySplit, tensors, relative, gradient_error, predict, clock, save_json


def main(default_method=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--method',choices=list(PREDICTORS),default=default_method,required=default_method is None)
    p.add_argument('--data',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--epochs',type=int,default=150)
    p.add_argument('--width',type=int,default=24)
    p.add_argument('--depth',type=int,default=4)
    p.add_argument('--seed',type=int,default=20261103)
    args=p.parse_args()
    if args.out.exists():
        p.error('Use a new output directory.')
    args.out.mkdir(parents=True)
    torch.set_num_threads(2)
    torch.manual_seed(args.seed)
    device=torch.device('cuda')
    tr,va=DarcySplit(args.data,'train'),DarcySplit(args.data,'val')
    a,y,xy,mass,mask=tensors(tr,device)
    av,yv,_,_,_=tensors(va,device)
    stats=dict(a_mean=a.log().mean(),a_std=a.log().std().clamp_min(1e-6),
               y_mean=y.mean(0),y_std=y.std(0).clamp_min(.05*y.std()))
    resolution=a.shape[-1]
    if args.method=='MgNO':
        model_options=dict(width=args.width,depth=args.depth)
    elif args.method=='DCNO':
        model_options=dict(resolution=resolution,width=32)
    else:
        model_options=dict(resolution=resolution)
    model=PREDICTORS[args.method](**model_options).to(device)
    opt=torch.optim.Adam(model.parameters(),lr=.001,weight_decay=1e-4)
    sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,args.epochs,eta_min=1e-5)
    history,best=[],float('inf')
    started=clock(device)
    for epoch in range(args.epochs):
        model.train()
        tick=clock(device)
        for ix in torch.randperm(len(tr),device=device).split(8):
            out=predict(model,a[ix],xy,mask,stats)
            loss=relative(out,y[ix],mass).mean()+.1*gradient_error(out,y[ix]).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        sched.step()
        model.eval()
        with torch.no_grad():
            errors,grads=[],[]
            for ix in torch.arange(len(va),device=device).split(8):
                out=predict(model,av[ix],xy,mask,stats)
                errors.append(relative(out,yv[ix],mass))
                grads.append(gradient_error(out,yv[ix]))
            error,grad=float(torch.cat(errors).mean()),float(torch.cat(grads).mean())
        row=dict(epoch=epoch+1,val_L2_percent=100*error,val_gradient_percent=100*grad,
                 seconds=clock(device)-tick)
        history.append(row)
        score=error+.1*grad
        if not np.isfinite(score):
            raise RuntimeError('Nonfinite validation.')
        if score<best:
            best=score
            torch.save(dict(state=model.state_dict(),stats=stats,model_options=model_options,
                 predictor_type=args.method,best_epoch=epoch+1,validation=row,seed=args.seed),args.out/'no.pt')
        if epoch<2 or (epoch+1)%10==0:
            print(json.dumps(row),flush=True)
        save_json(args.out/'training.json',dict(history=history,seconds=clock(device)-started,
                 parameters=sum(p.numel() for p in model.parameters()),best_score=best,
                 protocol='Pinned official '+args.method+' blocks; aligned labels, training normalization and L2+.1 gradient loss; explicit input adapter. Matched retraining, not a paper checkpoint reproduction.'))
    model,stats,_=load_predictor(args.out,device)
    predictions={}
    with torch.no_grad():
        for name in ('train','val','test'):
            split=DarcySplit(args.data,name)
            aa,_,_,_,_=tensors(split,device)
            predictions[name]=np.concatenate([predict(model,aa[i:i+8],xy,mask,stats).flatten(1).cpu().numpy()
                                            for i in range(0,len(split),8)])
    np.savez_compressed(args.out/'predictions.npz',**predictions)


if __name__=='__main__':
    main()
