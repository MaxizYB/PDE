"""CPU mechanism audit on actual aligned P1 data, with train-only POD.

This is NOT an FNO/TB neural benchmark. The reference is a sine space and
the initial iterate is zero. The audit tests whether retained-space defect
learning has held-out benefit and whether derivative/flux gains follow.
"""
from __future__ import annotations
import argparse
import csv
import json
import sys
import time
from pathlib import Path
import numpy as np
import scipy.linalg as la
from scipy.fft import dstn

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from baselines.data.darcy_v2.generate import assemble, element_geometry
from baselines.data.multiscale_elliptic_v1.generate_challenge import consistent_mass


def filtered(a, rhs, x, steps, lower=.001):
    if steps == 0:
        return x.copy()
    upper = float(np.max(np.asarray(abs(a).sum(1)).ravel() / a.diagonal())) * (1+1e-12)
    low = lower * upper / 2
    center, radius = (upper+low)/2, (upper-low)/2
    inverse = 1 / a.diagonal()
    direction = inverse * (rhs-a@x) / center
    out = x + direction
    rho = radius / center
    for _ in range(1, steps):
        next_rho = 1 / (2*center/radius-rho)
        direction = (2*next_rho/radius)*inverse*(rhs-a@out) + (next_rho*rho)*direction
        out += direction
        rho = next_rho
    return out


def factor(a, z):
    az = a @ z
    gram = z.T @ az
    return la.cho_factor((gram+gram.T)/2, lower=True, check_finite=False)


def update(a, rhs, x, z, fact):
    return x + z @ la.cho_solve(fact, z.T @ (rhs-a@x), check_finite=False)


def solve(a, rhs, z, pre=32, post=8, cycles=2):
    fact = factor(a, z)
    x = np.zeros(len(rhs))
    for _ in range(cycles):
        x = filtered(a, rhs, x, pre)
        x = update(a, rhs, x, z, fact)
        x = filtered(a, rhs, x, post, .05)
    return x


def metrics(x, target, coefficient, geometry, cells, mass):
    area, gradients, _ = geometry
    error = x-target
    dg = np.einsum('tic,ti->tc', gradients, error[cells])
    tg = np.einsum('tic,ti->tc', gradients, target[cells])
    ae = coefficient[cells].mean(1)
    ratio = lambda num, den: 100*float(np.sqrt(num/max(den,1e-30)))
    return dict(field_percent=ratio(np.sum(mass*error**2),np.sum(mass*target**2)),
        gradient_percent=ratio(np.sum(area[:,None]*dg**2),np.sum(area[:,None]*tg**2)),
        flux_percent=ratio(np.sum((area*ae**2)[:,None]*dg**2),np.sum((area*ae**2)[:,None]*tg**2)))


def main(args):
    if args.out.exists():
        raise FileExistsError('Choose a new output directory.')
    args.out.mkdir(parents=True)
    mesh = np.load(args.data/'mesh.npz')
    points, cells, boundary, mass = (mesh[k] for k in ('points','cells','boundary','mass'))
    geometry = element_geometry(points,cells)
    consistent = consistent_mass(points,cells,geometry)
    free = np.flatnonzero(~boundary)
    xy = points[free].astype(float)
    pairs = sorted(((i*i+j*j,i,j) for i in range(1,20) for j in range(1,20)))
    raw = np.column_stack([np.sin(np.pi*i*xy[:,0])*np.sin(np.pi*j*xy[:,1])
                           for _,i,j in pairs[:args.reference_rank+args.extra_rank]])
    analytic = np.linalg.qr(raw,mode='reduced')[0]
    q = analytic[:,:args.reference_rank]
    def system(split,i):
        a,_ = assemble(points,cells,split['coefficient'][i].astype(float),geometry)
        a = (a+float(split['reaction'][i])*consistent)[free][:,free].tocsr()
        rhs = geometry[2][free]*split['source'][i,free]
        return a,rhs
    training = np.load(args.data/'train.npz')
    defects, solutions = [], []
    for i in range(len(training['solution'])):
        a,rhs = system(training,i)
        prediction = solve(a,rhs,q,args.pre,args.post,2)
        defect = training['solution'][i,free]-prediction
        # Normalize each training sample, so easy amplitude does not dominate.
        defects.append(defect/max(np.sqrt(np.sum(mass[free]*defect**2)),1e-30))
        y = training['solution'][i,free]
        solutions.append(y/max(np.sqrt(np.sum(mass[free]*y**2)),1e-30))
    def pod(columns):
        data=np.column_stack(columns)
        data-=q@(q.T@data)
        u,s,_=la.svd(data,full_matrices=False,check_finite=False)
        if len(s)<args.extra_rank or s[args.extra_rank-1]<1e-12*s[0]:
            raise RuntimeError('Insufficient training rank for the declared enrichment')
        return u[:,:args.extra_rank],s
    defect_modes,spectrum=pod(defects)
    solution_modes,solution_spectrum=pod(solutions)
    rng=np.random.default_rng(args.seed)
    random=rng.normal(size=(len(free),args.extra_rank))
    random-=q@(q.T@random)
    random=np.linalg.qr(random,mode='reduced')[0]
    spaces=dict(reference=q,analytic_equal_rank=analytic,
        solution_POD=np.column_stack((q,solution_modes)),
        defect_POD=np.column_stack((q,defect_modes)),
        random_equal_rank=np.column_stack((q,random)))
    np.savez_compressed(args.out/'learned_spaces.npz',reference=q,defect=defect_modes,
        solution=solution_modes,defect_singular_values=spectrum,solution_singular_values=solution_spectrum)
    test=np.load(args.data/'test.npz')
    rows=[]
    for i in range(len(test['solution'])):
        a,rhs=system(test,i)
        for name,z in spaces.items():
            solve(a,rhs,z,args.pre,args.post,2)
            times=[]
            for _ in range(args.repeats):
                begin=time.perf_counter()
                x=solve(a,rhs,z,args.pre,args.post,2)
                times.append(1000*(time.perf_counter()-begin))
            out=np.zeros(len(points))
            out[free]=x
            rows.append(dict(index=i,method=name,columns=z.shape[1],
                **metrics(out,test['solution'][i],test['coefficient'][i],geometry,cells,mass),
                correction_ms=float(np.median(times))))
    keys=['field_percent','gradient_percent','flux_percent','correction_ms']
    summary={name:{key:dict(mean=float(np.mean([v[key] for v in rows if v['method']==name])),
                           p95=float(np.percentile([v[key] for v in rows if v['method']==name],95)))
                   for key in keys} for name in spaces}
    result=dict(protocol='CPU mechanism audit, NOT neural/TB benchmark; zero initial; train-only POD; held-out test',
        options={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
        timing='CPU correction only, including space factorization; assembly/reference generation excluded; not comparable to GPU FNO times',
        train_count=len(training['solution']),test_count=len(test['solution']),summary=summary,samples=rows)
    (args.out/'evaluation.json').write_text(json.dumps(result,indent=2)+chr(10))
    with (args.out/'summary.csv').open('w') as f:
        w=csv.writer(f)
        w.writerow(['method',*keys])
        for name,values in summary.items():
            w.writerow([name,*[values[key]['mean'] for key in keys]])
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--reference-rank',type=int,default=40)
    p.add_argument('--extra-rank',type=int,default=8)
    p.add_argument('--pre',type=int,default=32)
    p.add_argument('--post',type=int,default=8)
    p.add_argument('--repeats',type=int,default=3)
    p.add_argument('--seed',type=int,default=20261106)
    main(p.parse_args())
