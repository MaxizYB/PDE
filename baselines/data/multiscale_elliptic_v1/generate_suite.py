"""HANO-family coefficient distributions with aligned, fully reproducible P1 labels.

These are regenerated coefficient-law benchmarks, not the paper's downsampled
label files. All families have exactly the same equation, mesh and data schema.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from baselines.data.darcy_v2.generate import (regular_mesh, element_geometry,
    solve_sample, parent_random_field)
from baselines.data.multiscale_elliptic_v1.generate import coefficient


def sample(points, resolution, rng, family):
    if family == 'trigonometric':
        frequencies = rng.uniform(1, 1.5, 6)*2.**np.arange(6)
        return coefficient(points, frequencies).astype('float32'), frequencies
    rough = family == 'rough'
    field = parent_random_field(rng, resolution, 20. if rough else 9.)
    values = np.where(field >= 0, 12., 2. if rough else 3.).astype('float32')
    return values.ravel(), np.asarray([20. if rough else 9., 12., 2. if rough else 3.])


def generate(args):
    if args.out.exists():
        raise FileExistsError('Choose a new dataset directory.')
    args.out.mkdir(parents=True)
    t0 = time.perf_counter()
    points, cells, boundary = regular_mesh(args.resolution)
    geometry = element_geometry(points, cells)
    np.savez_compressed(args.out/'mesh.npz', points=points.astype('float32'),
                        cells=cells, boundary=boundary, mass=geometry[2].astype('float32'))
    manifest = dict(family=args.family, resolution=args.resolution, seed=args.seed,
        equation='-div(a grad u)=1; homogeneous Dirichlet; unit square',
        discretization='P1 FEM', kind='regular',
        target_protocol='Stored float32 nodal coefficient first; exact same P1 assembly; nodal mean on triangles; float64 direct reference solve; float32 labels.',
        source='https://arxiv.org/html/2210.10890v3#S3.SS1',
        published_labels_reproduced=False,
        coefficient_protocol=('Six-scale HANO law; domain (-1,1)^2 mapped to unit square.'
            if args.family == 'trigonometric' else
            'DCT Neumann GRF approximation to covariance (-Delta+cI)^-2, zero constant mode; sign threshold; HANO c/phase values.'),
        train=args.train, val=args.val, test=args.test, split_indices={})
    index = 0
    for name, count in [('train',args.train),('val',args.val),('test',args.test)]:
        values, targets, parameters, residuals = [], [], [], []
        begin = index
        for _ in range(count):
            rng = np.random.default_rng(args.seed+1000003*index)
            a, par = sample(points, args.resolution, rng, args.family)
            u, residual = solve_sample(points, cells, boundary, a.astype('float64'),
                                       np.zeros(len(points)), geometry)
            values.append(a)
            targets.append(u.astype('float32'))
            parameters.append(par)
            residuals.append(residual)
            index += 1
        target = np.stack(targets)
        np.savez_compressed(args.out/f'{name}.npz', coefficient=np.stack(values),
            solution=target, lift=np.zeros_like(target), sample_index=np.arange(begin,index),
            parameters=np.stack(parameters), relative_residual=residuals)
        manifest['split_indices'][name] = [begin,index]
        manifest[name+'_audit'] = dict(max_reference_residual=max(residuals),
            coefficient_min=float(np.min(values)), coefficient_max=float(np.max(values)))
        print(json.dumps(dict(family=args.family, split=name, count=count,
                              seconds=time.perf_counter()-t0)), flush=True)
    manifest['seconds'] = time.perf_counter()-t0
    (args.out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--family', choices=['smooth','rough','trigonometric'], required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--resolution', type=int, default=129)
    p.add_argument('--train', type=int, default=256)
    p.add_argument('--val', type=int, default=32)
    p.add_argument('--test', type=int, default=32)
    p.add_argument('--seed', type=int, default=20261103)
    args = p.parse_args()
    if min(args.train,args.val,args.test) < 1 or args.resolution < 5:
        p.error('Nonempty splits and resolution >=5 required.')
    generate(args)
