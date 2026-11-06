"""HANO six-scale coefficient law with an explicitly aligned P1-FEM target.

This reproduces the coefficient distribution, not the published downsampled
labels. The original square (-1,1)^2 is mapped to (0,1)^2; u is divided by 4.
All splits use independent parameter vectors and the *stored* nodal coefficient
is the complete input needed to reconstruct the reference stiffness matrix.
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
from baselines.data.darcy_v2.generate import (
    regular_mesh, element_geometry, solve_sample, assemble,
)


def coefficient(points, frequencies):
    x, y = (2 * np.asarray(points, dtype=np.float64) - 1).T
    value = np.ones(len(points), dtype=np.float64)
    for frequency in frequencies:
        value *= (1 + 0.5 * np.cos(frequency * np.pi * (x + y)))
        value *= (1 + 0.5 * np.sin(frequency * np.pi * (y - 3 * x)))
    return value


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--resolution', type=int, default=257)
    p.add_argument('--train', type=int, default=512)
    p.add_argument('--val', type=int, default=64)
    p.add_argument('--test', type=int, default=64)
    p.add_argument('--seed', type=int, default=20261023)
    p.add_argument('--refinement-audit', type=int, default=3)
    args = p.parse_args()
    if min(args.train, args.val, args.test) < 1:
        p.error('all splits must be nonempty')
    args.out.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    points, cells, boundary = regular_mesh(args.resolution)
    geometry = element_geometry(points, cells)
    np.savez_compressed(args.out / 'mesh.npz', points=points.astype('float32'),
                        cells=cells, boundary=boundary, mass=geometry[2].astype('float32'))
    manifest = {
        'equation': '-div(a grad u)=1; u=0 on boundary',
        'coefficient_law': 'HANO six-scale trigonometric product, section 3.1.2',
        'source': 'https://arxiv.org/html/2210.10890',
        'target_protocol': 'same-grid P1 FEM, cell coefficient=mean of 3 stored nodal values',
        'published_labels_reproduced': False,
        'coordinate_map': 'x_original=2*x_unit-1; u_unit=u_original/4',
        'resolution': args.resolution, 'seed': args.seed,
        'train': args.train, 'val': args.val, 'test': args.test,
        'nodes': len(points), 'triangles': len(cells), 'discretization': 'P1 FEM',
        'kind': 'regular', 'split_indices': {},
    }
    index = 0
    for name, count in [('train', args.train), ('val', args.val), ('test', args.test)]:
        values, targets, parameters, residuals, contrasts = [], [], [], [], []
        begin = index
        for _ in range(count):
            rng = np.random.default_rng(args.seed + 1000003 * index)
            frequencies = rng.uniform(1., 1.5, 6) * 2. ** np.arange(6)
            # Quantize before solving, so stored inputs reproduce the reference A.
            a = coefficient(points, frequencies).astype('float32')
            u, residual = solve_sample(points, cells, boundary, a.astype('float64'),
                                       np.zeros(len(points)), geometry)
            values.append(a)
            targets.append(u.astype('float32'))
            parameters.append(frequencies)
            residuals.append(residual)
            contrasts.append(float(a.max() / a.min()))
            index += 1
            if index % 32 == 0:
                print(json.dumps({'generated': index, 'seconds': time.perf_counter()-started}), flush=True)
        target = np.stack(targets)
        np.savez_compressed(args.out / f'{name}.npz', coefficient=np.stack(values),
                            lift=np.zeros_like(target), solution=target,
                            frequencies=np.stack(parameters), sample_index=np.arange(begin, index),
                            relative_residual=np.asarray(residuals))
        manifest['split_indices'][name] = [begin, index]
        manifest[name + '_audit'] = {'max_solve_residual_float64': max(residuals),
                                     'contrast_min': min(contrasts), 'contrast_max': max(contrasts)}
    # A small nested-grid audit distinguishes algebraic alignment from continuum accuracy.
    audit = []
    test = np.load(args.out / 'test.npz')
    for i in range(min(args.refinement_audit, args.test)):
        nf = 2 * (args.resolution - 1) + 1
        fp, fc, fb = regular_mesh(nf)
        fg = element_geometry(fp, fc)
        fa = coefficient(fp, test['frequencies'][i]).astype('float32')
        fu, _ = solve_sample(fp, fc, fb, fa.astype('float64'), np.zeros(len(fp)), fg)
        restricted = fu.reshape(nf, nf)[::2, ::2].ravel()
        stored = test['solution'][i].astype('float64')
        mass = geometry[2]
        relative = np.sqrt(np.sum(mass*(stored-restricted)**2)/np.sum(mass*restricted**2))
        matrix, rhs = assemble(points, cells, test['coefficient'][i].astype('float64'), geometry)
        free = ~boundary
        discrete_residual = np.linalg.norm((matrix @ stored-rhs)[free])/np.linalg.norm(rhs[free])
        audit.append({'test_index': i, 'fine_resolution': nf,
                      'coarse_vs_restricted_fine_L2_percent': 100*float(relative),
                      'stored_float32_relative_residual': float(discrete_residual)})
    manifest['refinement_audit'] = audit
    manifest['seconds'] = time.perf_counter() - started
    (args.out / 'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(json.dumps(manifest), flush=True)


if __name__ == '__main__':
    main()
