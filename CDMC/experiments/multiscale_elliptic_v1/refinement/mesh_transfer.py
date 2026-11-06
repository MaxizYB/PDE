"""Frozen-corrector transfer to a jittered, Delaunay-retriangulated mesh.

The NO remains on its native Cartesian mesh. An explicit bilinear observation
map supplies its initial field on the new mesh; this is not native irregular
FNO inference. Current-mesh P1 labels are solved independently in float64.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

EXP = Path(__file__).resolve().parents[1]
ROOT = EXP.parents[2]
sys.path.insert(0, str(ROOT))


def generate(args):
    from scipy.spatial import Delaunay
    from baselines.data.darcy_v2.generate import element_geometry, solve_sample

    if args.out.exists():
        raise FileExistsError('Use a new mesh-transfer dataset directory.')
    with np.load(args.source/'mesh.npz') as z:
        points, boundary = z['points'].astype(float), z['boundary']
        original_cells = z['cells'].copy()
    n = round(np.sqrt(len(points)))
    assert n*n == len(points)
    rng = np.random.default_rng(args.seed)
    points[~boundary] += rng.uniform(-args.jitter, args.jitter,
                                   (int((~boundary).sum()), 2))/(n-1)
    # The stored coordinates define the actual discrete reference problem.
    points = points.astype('float32')
    cells = (Delaunay(points).simplices.astype('int32')
             if args.jitter else original_cells)
    geometry = element_geometry(points.astype(float), cells)
    with np.load(args.source/'test.npz') as z:
        coefficient = z['coefficient'][:args.count].copy()
        indices = z['sample_index'][:args.count].copy()
    if args.phase_low is not None:
        if args.phase_low <= 0:
            raise ValueError('The low phase must remain positive.')
        if len(np.unique(coefficient))!=2 or args.phase_low >= coefficient.max():
            raise ValueError('Phase remapping requires a binary field and low < high.')
        coefficient = np.where(coefficient < coefficient.max(),
                               args.phase_low, coefficient.max()).astype('float32')
    targets, residuals = [], []
    for a in coefficient:
        y, residual = solve_sample(points.astype(float), cells, boundary,
            a.astype(float), np.zeros(len(points)), geometry)
        targets.append(y.astype('float32'))
        residuals.append(residual)
    args.out.mkdir(parents=True)
    np.savez_compressed(args.out/'mesh.npz', points=points, cells=cells,
                        boundary=boundary, mass=geometry[2].astype('float32'))
    targets = np.stack(targets)
    np.savez_compressed(args.out/'test.npz', coefficient=coefficient,
                        solution=targets, lift=np.zeros_like(targets),
                        sample_index=indices)
    manifest = dict(kind='unstructured-square' if args.jitter else 'regular', source=str(args.source),
        count=len(targets), seed=args.seed, nodes=len(points), cells=len(cells),
        coordinate_jitter_in_cell_units=args.jitter,
        triangulation='SciPy Delaunay' if args.jitter else 'Original connectivity',
        phase_low=args.phase_low,
        material_protocol='Original nodal phase values transported with vertices; all P1 labels recomputed on the new mesh.',
        initial_protocol=('Original Cartesian NO output bilinearly observed at new coordinates.'
            if args.jitter else 'NO evaluated on the new stored coefficient field on its native grid.'),
        target_protocol='Stored float32 coordinates/coefficient; matching float64 P1 solve; float32 solution.',
        max_reference_residual=max(residuals))
    (args.out/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(json.dumps(manifest), flush=True)


def evaluate(args):
    sys.path.insert(0, str(EXP))
    import torch
    from torch.nn import functional as F
    from run import DarcySplit, tensors, predict, measure, field_metrics, save_json, summary
    from study import load_dictionary, smooth, prepare, cycles
    from predictors import load_predictor
    from polynomial_joint import SparseCompiled
    from coupled_coarse import CoarseContext
    from core import spectral_upper
    from types import SimpleNamespace

    if args.out.exists():
        raise FileExistsError('Use a new evaluation directory.')
    torch.set_num_threads(2)
    device = torch.device('cuda')
    source, target = DarcySplit(args.source, 'test'), DarcySplit(args.data, 'test')
    assert np.array_equal(source.sample_index[:len(target)], target.sample_index)
    no, stats, _ = load_predictor(args.no_dir, device)
    model, saved = load_dictionary(args.checkpoint, device)
    model.eval()
    opts = SimpleNamespace(**saved['options'])
    opts.matrix_bound = True
    compiled = SparseCompiled(target.mesh, device)
    context = CoarseContext(compiled, 1)
    a, _, xy, _, mask = tensors(source, device)
    coefficients = torch.as_tensor(target.coefficient, device=device)
    points = torch.as_tensor(target.mesh.points, device=device)
    observation = (2*points[:, [1, 0]]-1)[None, :, None, :]
    interior = torch.as_tensor(~target.mesh.boundary, device=device)
    rows = []
    with torch.no_grad():
        for i in range(min(args.count, len(target))):
            def solve(name):
                cartesian = predict(no, a[i:i+1], xy, mask, stats)
                full = F.grid_sample(cartesian[:, None], observation,
                    mode='bilinear', padding_mode='border', align_corners=True).flatten()*interior
                if name == 'interpolated_FNO':
                    return full
                system, ae, _ = compiled.build(coefficients[i], torch.zeros_like(full))
                x = smooth(system, compiled.action, ae, full[system.free], opts.pre, opts.lower, opts)
                if name == 'filter_only_K2':
                    x = smooth(system, compiled.action, ae, x,
                               opts.pre+2*opts.post, opts.lower, opts)
                else:
                    prepared = prepare(model(context.features(system, ae, x), context),
                                       context, system, ae, opts)
                    x = cycles(x, prepared, system, ae, context, opts,
                               int(name[-1]), filtered=True)
                out = torch.zeros_like(full)
                out[system.free] = x
                return out

            methods = {}
            for name in ('interpolated_FNO', 'filter_only_K2', 'dictionary_K2', 'dictionary_K4'):
                out, elapsed = measure(lambda: solve(name), device, args.repeats)
                methods[name] = dict(field_metrics(out.cpu().numpy(), target.target[i],
                    target.coefficient[i], target.mesh), total_ms=elapsed)
            system, ae, matrix = compiled.build(coefficients[i], torch.zeros_like(out))
            rows.append(dict(index=i, sample_index=int(target.sample_index[i]),
                spectral_upper=spectral_upper(system, compiled.action), methods=methods))
            aggregated = {name: {key: summary([r['methods'][name][key] for r in rows])
                          for key in methods[name]} for name in methods}
            save_json(args.out/'evaluation.json', dict(samples=rows, summary=aggregated,
                checkpoint=str(args.checkpoint), checkpoint_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
                mesh_plan_ms_once=compiled.mesh_plan_ms_once,
                timing='Fresh Cartesian FNO + GPU interpolation + current-mesh assembly + dictionary + factor + all updates; synchronized batch1 median. Mesh plan excluded and listed once.',
                protocol='Frozen regular-trained corrector; new unstructured Delaunay mesh; no irregular fine-tuning.'))
            print(json.dumps(dict(index=i, field={k:v['L2_percent'] for k,v in methods.items()})), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage', choices=['generate', 'evaluate'])
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--data', type=Path)
    p.add_argument('--no-dir', type=Path)
    p.add_argument('--checkpoint', type=Path)
    p.add_argument('--count', type=int, default=16)
    p.add_argument('--seed', type=int, default=20261104)
    p.add_argument('--jitter', type=float, default=.3)
    p.add_argument('--phase-low', type=float, default=None)
    p.add_argument('--repeats', type=int, default=3)
    args = p.parse_args()
    (generate if args.stage=='generate' else evaluate)(args)


if __name__ == '__main__':
    main()
