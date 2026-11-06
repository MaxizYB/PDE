"""Generate HINTS variable-RHS pairs using the existing fine P1 matrices.

No coarse PDE is used for training labels. Each material is factorized once,
then multiple smooth random source fields are solved in one multiple-RHS call.
Only the train/validation material splits are read here.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates
from scipy.sparse.linalg import splu

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT))
from baselines.data.darcy_v2.generate import assemble, element_geometry


def make_pairs(data, out, rhs_per_material=16, seed=20261104):
    out.mkdir(parents=True, exist_ok=True)
    mesh = dict(np.load(data/'mesh.npz'))
    points, cells, boundary = (mesh[k] for k in ('points', 'cells', 'boundary'))
    geometry = element_geometry(points, cells)
    mass = geometry[2]
    free = np.flatnonzero(~boundary)
    n = int(round(np.sqrt(len(points))))
    if n*n != len(points):
        raise ValueError('This adapter requires the current structured-square datasets.')
    # Fine-grid node queries for the trunk; labels remain fine-P1 solutions.
    stride = max(1, (n-1)//64)
    query = np.arange(n*n).reshape(n,n)[::stride, ::stride].ravel()
    sensor_axis = np.linspace(0, n-1, 31)
    sensor_xy = np.array(np.meshgrid(sensor_axis, sensor_axis, indexing='ij'))
    rng = np.random.default_rng(seed)
    started = time.perf_counter()
    records = {}
    for split in ('train', 'val'):
        raw = np.load(data/f'{split}.npz')
        a = raw['coefficient']
        coefficient, rhs, solutions, material = [], [], [], []
        constant_errors, solve_residuals = [], []
        for i, ai in enumerate(a):
            if np.any(raw['lift'][i] != 0):
                raise ValueError('This HINTS adapter uses homogeneous Dirichlet boundaries.')
            matrix, _ = assemble(points, cells, ai, geometry)
            aff = matrix[free][:, free].tocsc()
            factor = splu(aff)
            fs = np.empty((rhs_per_material, n, n), dtype=np.float64)
            fs[0] = 1.
            for j in range(1, rhs_per_material):
                # Smoothing white noise gives an SE-like GRF of length 0.1.
                fj = gaussian_filter(rng.standard_normal((n,n)),
                                     sigma=0.1*(n-1)/np.sqrt(2), mode='reflect')
                fs[j] = fj/max(np.sqrt(np.mean(fj*fj)), 1e-12)
            load = mass[free, None]*fs.reshape(rhs_per_material,-1)[:,free].T
            solved = factor.solve(load)
            full = np.zeros((rhs_per_material, n*n))
            full[:,free] = solved.T
            constant_errors.append(float(np.linalg.norm(full[0]-raw['solution'][i]) /
                                         max(np.linalg.norm(raw['solution'][i]),1e-30)))
            solve_residuals.append(float(np.linalg.norm(aff@solved-load)/np.linalg.norm(load)))
            coefficient.append(map_coordinates(ai.reshape(n,n), sensor_xy, order=1))
            rhs.extend(map_coordinates(fj, sensor_xy, order=1) for fj in fs)
            solutions.extend(full[:,query])
            material.extend([i]*rhs_per_material)
            if (i+1)%64 == 0:
                print(json.dumps(dict(stage='pairs',split=split,materials=i+1)), flush=True)
        np.savez_compressed(out/f'{split}.npz',
            coefficient=np.asarray(coefficient,dtype=np.float32),
            rhs=np.asarray(rhs,dtype=np.float32),
            solution=np.asarray(solutions,dtype=np.float32),
            material=np.asarray(material,dtype=np.int64),
            coordinates=points[query].astype(np.float32))
        records[split] = dict(materials=len(a), pairs=len(material),
            constant_solution_relative_error_max=max(constant_errors),
            linear_system_relative_residual_max=max(solve_residuals))
        if max(constant_errors) > 1e-4:
            raise RuntimeError('Fine-grid unit-source truth alignment check failed.')
    result = dict(data=str(data),rhs_per_material=rhs_per_material,seed=seed,
        sensors=31,query_resolution=int(round(np.sqrt(len(query)))),fine_resolution=n,
        rhs='One constant plus SE-like GRF sources, correlation length 0.1; lumped-mass load.',
        labels='Fine P1 solve; subsampled query nodes, never a coarsened PDE solution.',
        generation_seconds=time.perf_counter()-started,splits=records)
    (out/'manifest.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)
