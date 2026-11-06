"""Physics-complete, matched P1 plane-strain elasticity preview.

Reuses the project's independently checked elasticity physics, not its old
models or predictions.  Reference labels come from scikit-fem + sparse LU.
The online adapter independently assembles the same B^T C B system on GPU.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
SPEC = importlib.util.spec_from_file_location(
    'validated_elasticity_physics',
    ROOT / 'Nature/physical_response_fidelity/src/elasticity.py')
physics = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(physics)


def generate(out, resolution=65, train=128, val=16, test=16, ood=16, seed=20261028):
    out = Path(out)
    if (out / 'manifest.json').exists():
        raise FileExistsError('Refusing to overwrite a completed dataset')
    out.mkdir(parents=True, exist_ok=True)
    g = physics.geometry(resolution - 1, nu=0.3)
    rng = np.random.default_rng(seed)
    counts = dict(train=train, val=val, test=test, ood=ood)
    start = time.perf_counter()
    checks = []
    # An affine displacement is exactly reproduced by homogeneous P1 elasticity.
    uniform = physics.mature_stiffness(g, np.ones(len(g['triangles'])))
    patch_error = np.max(np.abs(physics.solve(g, uniform) - g['affine']))
    if patch_error > 1e-10:
        raise RuntimeError(f'Affine patch test failed: {patch_error}')
    for label, count in counts.items():
        inputs, coefficients, targets, contrasts, timing = [], [], [], [], []
        for i in range(count):
            e, contrast, _ = physics.sample_material(g, rng, shifted=label == 'ood')
            e = e.astype(np.float32).astype(np.float64)
            tick = time.perf_counter()
            a = physics.mature_stiffness(g, e)
            assembled = time.perf_counter()
            u = physics.solve(g, a)
            solved = time.perf_counter()
            if i == 0:
                independent = physics.stiffness(g, e)
                defect = a - independent
                matrix_error = np.linalg.norm(defect.data) / np.linalg.norm(a.data)
                residual = np.linalg.norm((a @ u)[g['free']]) / max(
                    np.linalg.norm((a @ g['affine'])[g['free']]), 1e-30)
                checks.append(dict(split=label, matrix_relative=matrix_error,
                                   reference_relative_residual=residual))
                if matrix_error > 1e-11 or residual > 1e-10:
                    raise RuntimeError(f'Independent physical check failed: {checks[-1]}')
            inputs.append(physics.coefficient_image(g, e))
            coefficients.append(e)
            targets.append(u.reshape(resolution, resolution, 2))
            contrasts.append(contrast)
            timing.append([(assembled-tick)*1000, (solved-assembled)*1000])
        np.savez_compressed(out / f'{label}.npz',
            inputs=np.asarray(inputs, dtype=np.float32),
            modulus=np.asarray(coefficients, dtype=np.float32),
            target=np.asarray(targets, dtype=np.float64),
            contrast=np.asarray(contrasts), reference_timing_ms=np.asarray(timing))
        print(json.dumps(dict(generated=label, samples=count, elapsed_s=time.perf_counter()-start)), flush=True)
    manifest = dict(pde='2D heterogeneous linear elasticity, plane strain',
        resolution=resolution, nodes=len(g['points']), free_dofs=len(g['free']),
        triangles=len(g['triangles']), poisson_ratio=g['nu'],
        boundary='Full-boundary prescribed displacement (0.01*x, 0); zero body force',
        unknown='interleaved two-component displacement fluctuation about affine lift',
        material='1-3 randomly rotated stiff elliptical inclusions; piecewise constant per triangle',
        id_contrast=[2, 20], ood_contrast=[20, 60], counts=counts, seed=seed,
        reference='scikit-fem P1 vector elasticity + SciPy SuperLU, float64',
        checks=checks, homogeneous_affine_patch_max_error=float(patch_error),
        generation_seconds=time.perf_counter()-start,
        target_alignment='Same stored float32 modulus is used for labels and online assembly')
    # Validate the actual stored coefficients, not just their float64 precursors.
    for label in counts:
        with np.load(out / f'{label}.npz') as stored:
            e = stored['modulus'][0].astype(np.float64)
            target = stored['target'][0].ravel()
            a = physics.stiffness(g, e)
            mismatch = np.linalg.norm((a @ target)[g['free']]) / max(
                np.linalg.norm((a @ g['affine'])[g['free']]), 1e-30)
            manifest['checks'][list(counts).index(label)]['stored_coefficient_residual'] = float(mismatch)
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--resolution', type=int, default=65)
    for key, value in [('train', 128), ('val', 16), ('test', 16), ('ood', 16)]:
        parser.add_argument('--' + key, type=int, default=value)
    parser.add_argument('--seed', type=int, default=20261028)
    args = parser.parse_args()
    print(json.dumps(generate(**vars(args)), indent=2))
