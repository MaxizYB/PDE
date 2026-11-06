"""Aligned scalar elliptic benchmarks with independently recorded difficulty.

PDE: -div(a grad u) + sigma*u = f, u=0 on the unit-square boundary.
Poisson varies f, diffusion varies a, and screened diffusion varies both a
and the nonnegative reaction coefficient. All labels use the stored nodal
inputs and the same P1 stiffness, consistent mass, and mass-lumped load.
The generator needs only the existing NumPy/SciPy environment.
"""
from __future__ import annotations
import argparse
import json
import sys
import time
from pathlib import Path
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from baselines.data.darcy_v2.generate import (regular_mesh, element_geometry,
    assemble, parent_random_field)
from baselines.data.multiscale_elliptic_v1.generate import coefficient


def consistent_mass(points, cells, geometry):
    area = geometry[0]
    template = (np.ones((3, 3)) + np.eye(3)) / 12
    values = area[:, None, None] * template
    rows = np.broadcast_to(cells[:, :, None], values.shape).ravel()
    cols = np.broadcast_to(cells[:, None, :], values.shape).ravel()
    return sp.coo_matrix((values.ravel(), (rows, cols)), shape=(len(points),) * 2).tocsr()


def sample(points, n, rng, family, contrast):
    parameters = {}
    if family == 'trigonometric':
        frequencies = rng.uniform(1, 1.5, 6) * 2. ** np.arange(6)
        a = coefficient(points, frequencies)
        parameters['frequencies'] = frequencies.tolist()
    elif family in ('rough', 'screened'):
        field = parent_random_field(rng, n, 20.)
        a = np.where(field.ravel() >= 0, contrast, 1.)
    elif family == 'poisson':
        a = np.ones(len(points))
    else:
        raise ValueError(family)
    if family in ('poisson', 'screened'):
        # Mixed smooth/rough, signed sources; no manufactured-solution labels.
        source = (parent_random_field(rng, n, 9.) + .3 * parent_random_field(rng, n, 80.))
        source = source.ravel()
        source /= np.sqrt(np.mean(source ** 2))
    else:
        source = np.ones(len(points))
    sigma = float(10. ** rng.uniform(-1, 2)) if family == 'screened' else 0.
    parameters['reaction'] = sigma
    # Rounding precedes assembly, so every saved input defines its own label.
    return a.astype(np.float32), source.astype(np.float32), sigma, parameters


def generate(args):
    if args.out.exists():
        raise FileExistsError('Choose a new dataset directory; existing data are immutable.')
    if args.resolution < 5 or min(args.train, args.val, args.test) < 1 or args.contrast < 1:
        raise ValueError('Invalid resolution, split size, or contrast.')
    args.out.mkdir(parents=True)
    begin = time.perf_counter()
    points, cells, boundary = regular_mesh(args.resolution)
    geometry = element_geometry(points, cells)
    mass = consistent_mass(points, cells, geometry)
    free = np.flatnonzero(~boundary)
    np.savez_compressed(args.out / 'mesh.npz', points=points.astype(np.float32),
        cells=cells, boundary=boundary, mass=geometry[2].astype(np.float32))
    manifest = dict(family=args.family, resolution=args.resolution, kind='regular',
        discretization='P1 FEM', equation='-div(a grad u)+sigma*u=f; homogeneous Dirichlet',
        coefficient_cell_rule='arithmetic mean of stored nodal values',
        source_rule='lumped P1 mass times stored nodal f', reaction_rule='sigma times consistent P1 mass',
        seed=args.seed, contrast=args.contrast, split_indices={},
        reference='float64 sparse direct solve; float64 labels retained',
        training_reference_rule='No coarse PDE labels; train and test use their own fine system',
        literature_inspiration=['HANO arXiv:2210.10890, six-scale coefficient and rough interfaces',
            'DeepONet preconditioning arXiv:2401.02016, JumpDiff contrast stress'],
        published_dataset=False)
    index = 0
    for split, count in [('train', args.train), ('val', args.val), ('test', args.test)]:
        arrays = {k: [] for k in ('coefficient', 'source', 'reaction', 'solution', 'relative_residual', 'parameters')}
        start = index
        for _ in range(count):
            rng = np.random.default_rng(args.seed + 1000003 * index)
            a, f, sigma, par = sample(points, args.resolution, rng, args.family, args.contrast)
            stiffness, _ = assemble(points, cells, a.astype(float), geometry)
            matrix = (stiffness + sigma * mass)[free][:, free].tocsr()
            rhs = geometry[2][free] * f[free]
            u = np.zeros(len(points))
            u[free] = spsolve(matrix.tocsc(), rhs)
            residual = np.linalg.norm(matrix @ u[free] - rhs) / max(np.linalg.norm(rhs), 1e-30)
            if not np.isfinite(residual) or residual > 1e-7:
                raise RuntimeError(f'Reference solve failed: sample {index}, residual {residual}')
            for key, value in dict(coefficient=a, source=f, reaction=sigma, solution=u,
                    relative_residual=residual, parameters=json.dumps(par, sort_keys=True)).items():
                arrays[key].append(value)
            index += 1
        packed = {k: np.asarray(v) for k, v in arrays.items()}
        np.savez_compressed(args.out / (split + '.npz'), **packed,
            lift=np.zeros_like(packed['solution']), sample_index=np.arange(start, index))
        manifest['split_indices'][split] = [start, index]
        manifest[split + '_audit'] = dict(count=count,
            max_reference_residual=float(max(arrays['relative_residual'])),
            coefficient_min=float(packed['coefficient'].min()),
            coefficient_max=float(packed['coefficient'].max()))
        print(json.dumps(dict(split=split, count=count, seconds=time.perf_counter()-begin)), flush=True)
    manifest['seconds'] = time.perf_counter() - begin
    (args.out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + chr(10))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--family', choices=['rough', 'trigonometric', 'poisson', 'screened'], required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--resolution', type=int, default=129)
    p.add_argument('--contrast', type=float, default=100.)
    p.add_argument('--train', type=int, default=256)
    p.add_argument('--val', type=int, default=32)
    p.add_argument('--test', type=int, default=32)
    p.add_argument('--seed', type=int, default=20261106)
    generate(p.parse_args())
