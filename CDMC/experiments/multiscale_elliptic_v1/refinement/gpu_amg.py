"""Explicit Ruge--Stuben AMG hierarchy with GPU V-cycle application.

PyAMG constructs the material-dependent classical interpolation and Galerkin
matrices. All applications use those explicit matrices on the same GPU as the
learned corrector. Two weighted-Jacobi pre/post sweeps replace CPU Gauss-Seidel;
setup, transfers and terminal factorization are charged in each fresh call.
"""
from __future__ import annotations
import argparse
import csv
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import scipy.sparse as sp
import pyamg
import torch
from run import DarcySplit,tensors,predict,measure,clock,summary,save_json,field_metrics
from predictors import load_predictor
from polynomial_joint import SparseCompiled
from coupled_coarse import mm


def csr(matrix,device):
    a=matrix.tocsr()
    return torch.sparse_csr_tensor(torch.as_tensor(a.indptr.astype('int64'),device=device),
        torch.as_tensor(a.indices.astype('int64'),device=device),
        torch.as_tensor(a.data.astype('float32'),device=device),size=a.shape,device=device)


class ExplicitAMG:
    def __init__(self,matrix,device):
        source=sp.csr_matrix((matrix.values().cpu().numpy().astype('float64'),
                             matrix.col_indices().cpu().numpy(),
                             matrix.crow_indices().cpu().numpy()),shape=matrix.shape)
        ml=pyamg.ruge_stuben_solver(source,max_coarse=128)
        self.a=[csr(level.A,device) for level in ml.levels]
        self.p=[csr(level.P,device) for level in ml.levels[:-1]]
        self.r=[csr(level.R,device) for level in ml.levels[:-1]]
        self.invd=[torch.as_tensor(1/level.A.diagonal(),dtype=torch.float32,device=device)
                   for level in ml.levels]
        terminal=torch.as_tensor(ml.levels[-1].A.toarray(),dtype=torch.float64,device=device)
        self.factor=torch.linalg.cholesky(.5*(terminal+terminal.T))
        self.complexity=float(ml.operator_complexity())

    def vcycle(self,level,rhs,initial):
        if level==len(self.a)-1:
            return torch.cholesky_solve(rhs.double()[:,None],self.factor)[:,0].float()
        x=initial
        for _ in range(2):
            x=x+(2/3)*self.invd[level]*(rhs-mm(self.a[level],x[:,None])[:,0])
        residual=rhs-mm(self.a[level],x[:,None])[:,0]
        coarse=mm(self.r[level],residual[:,None])[:,0]
        dx=self.vcycle(level+1,coarse,torch.zeros_like(coarse))
        x=x+mm(self.p[level],dx[:,None])[:,0]
        for _ in range(2):
            x=x+(2/3)*self.invd[level]*(rhs-mm(self.a[level],x[:,None])[:,0])
        return x

    def solve(self,rhs,initial,cycles):
        x=initial
        for _ in range(cycles):
            x=self.vcycle(0,rhs,x)
        return x


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('data','no-dir','out'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--count',type=int,default=16)
    p.add_argument('--repeats',type=int,default=3)
    args=p.parse_args()
    if args.out.exists():
        p.error('Use a new result directory.')
    args.out.mkdir(parents=True)
    torch.set_num_threads(2)
    device=torch.device('cuda')
    split=DarcySplit(args.data,'test')
    compiled=SparseCompiled(split.mesh,device)
    no,stats,_=load_predictor(args.no_dir,device)
    a,y,xy,_,mask=tensors(split,device)
    rows=[]
    with torch.no_grad():
        for i in range(min(args.count,len(split))):
            def online(k,zero=False):
                initial=(torch.zeros_like(y[i]).flatten() if zero else
                         predict(no,a[i:i+1],xy,mask,stats).flatten())
                system,ae,matrix=compiled.build(a[i].flatten(),torch.zeros_like(initial))
                solver=ExplicitAMG(matrix,device)
                x=solver.solve(system.rhs,initial[system.free],k)
                full=torch.zeros_like(initial)
                full[system.free]=x
                return full
            methods={}
            for k in (1,2,4):
                out,elapsed=measure(lambda:online(k),device,args.repeats)
                metrics=field_metrics(out.cpu().numpy(),split.target[i],split.coefficient[i],split.mesh)
                methods[f'NO_GPU_AMG_K{k}']=dict(metrics,total_ms=elapsed)
            initial=predict(no,a[i:i+1],xy,mask,stats).flatten()
            system,ae,matrix=compiled.build(a[i].flatten(),torch.zeros_like(initial))
            tick=clock(device)
            solver=ExplicitAMG(matrix,device)
            setup=1000*(clock(device)-tick)
            for k in (2,4):
                x,cached=measure(lambda:solver.solve(system.rhs,initial[system.free],k),device,args.repeats)
                methods[f'NO_GPU_AMG_K{k}']['cached_apply_ms']=cached
                methods[f'NO_GPU_AMG_K{k}']['setup_ms_once']=setup
            rows.append(dict(index=i,operator_complexity=solver.complexity,methods=methods))
            aggregated={n:{key:summary([r['methods'][n][key] for r in rows]) for key in methods[n]}
                        for n in methods}
            save_json(args.out/'evaluation.json',dict(summary=aggregated,samples=rows,
                device=torch.cuda.get_device_name(device),pyamg=pyamg.__version__,
                mesh_plan_ms_once=compiled.mesh_plan_ms_once,
                protocol='Classical RS interpolation/Galerkin; explicit GPU matrices; 2/2 weighted Jacobi; terminal FP64 Cholesky. Fresh NO+fine assembly+CPU hierarchy+transfers+factor+GPU cycles per timing call. Cached application also reported.'))
            print(json.dumps(dict(index=i,values={n:v['L2_percent'] for n,v in methods.items()})),flush=True)
    with (args.out/'summary.csv').open('w') as stream:
        writer=csv.writer(stream)
        keys=['L2_percent','gradient_percent','flux_L2_percent','total_ms']
        writer.writerow(['method',*keys])
        for name in aggregated:
            writer.writerow([name,*[aggregated[name][key]['mean'] for key in keys]])


if __name__=='__main__':
    main()
