"""Offline error/coarse-space diagnosis; never changes trained solvers.

Exact coarse projections use ground-truth errors for diagnosis only. They
measure approximation capacity, not deployable accuracy or runtime.
"""
from __future__ import annotations

import argparse
import json
from types import SimpleNamespace
from pathlib import Path

import numpy as np
import scipy.sparse.linalg as sla
import torch

from run import (DarcySplit, CDMCSetup, compiled_plan, make_example,
                 gates_from_implicit, correct_matrixfree, assemble, save_json)


def positive_ratio(numerator, denominator):
    return float(numerator / max(float(denominator), 1e-30))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--no-dir', type=Path, required=True)
    parser.add_argument('--corrector-dir', type=Path, required=True)
    parser.add_argument('--evaluation', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.manual_seed(20261023)
    source = json.loads(args.evaluation.read_text())
    ordered = sorted(source['samples'], key=lambda row: row['methods']['fixed_MG_K2']['L2_percent'])
    # Typical, largest remaining numerical error, and previously identified
    # learned-smoother stall. Do not report this selection as population means.
    selected = list(dict.fromkeys([ordered[len(ordered)//2]['index'],
                                   ordered[-1]['index'], 30]))
    split = DarcySplit(args.data, 'test')
    device = torch.device(args.device)
    compiled = compiled_plan(split, device, SimpleNamespace(coarse_size=128))
    with np.load(args.no_dir/'predictions.npz') as archive:
        prediction = archive['test'].copy()
    saved = torch.load(args.corrector_dir/'cdmc.pt', map_location=device, weights_only=False)
    setup = CDMCSetup(saved['width']).to(device)
    setup.load_state_dict(saved['state'])
    setup.eval()
    free = compiled.plan.free
    mass = split.mesh.mass[free].astype(np.float64)
    rows = []
    with torch.no_grad():
        for index in selected:
            system, element_a, initial, _ = make_example(split, prediction, index, compiled, device)
            gates, _ = gates_from_implicit(setup, system, compiled.action, element_a, initial)
            outputs = {'NO': initial,
                       'fixed_MG_K2': correct_matrixfree(system, compiled.action, element_a, initial,
                           torch.ones_like(initial), torch.ones_like(initial), None, cycles=2, coarse_steps=0),
                       'actual_K2': correct_matrixfree(system, compiled.action, element_a, initial,
                           gates, torch.ones_like(initial), None, cycles=2, coarse_steps=0)}
            target = split.target[index, free].astype(np.float64)
            errors = np.stack([target-output.cpu().numpy() for output in outputs.values()], axis=1)
            full_a, _ = assemble(split.mesh.points, split.mesh.cells, split.coefficient[index])
            matrix = full_a[free][:, free].tocsr()
            diagonal = matrix.diagonal()
            original_energy = np.einsum('ij,ij->j', errors, matrix@errors)
            original_mass = np.einsum('i,ij,ij->j', mass, errors, errors)
            methods = {}
            for k, name in enumerate(outputs):
                error = errors[:, k]
                residual = matrix@error
                step = system.levels[0].omega*residual/diagonal
                ideal_gate = np.divide(error, step, out=np.ones_like(error), where=np.abs(step)>1e-30)
                ideal_gate = np.clip(ideal_gate, .5, 1.5)
                wrong = error*residual <= 0
                methods[name] = {'field_L2_percent': source['samples'][index]['methods'][name]['L2_percent'],
                                 'algebraic_Rayleigh_exact_diagonal': positive_ratio(
                                     original_energy[k], (diagonal*errors[:, k]**2).sum()),
                                 'positive_local_step_wrong_direction_error_mass_percent': 100*positive_ratio(
                                     (mass[wrong]*error[wrong]**2).sum(), original_mass[k]),
                                 'oracle_bounded_fine_smoothing_mass_remaining_percent': 100*np.sqrt(positive_ratio(
                                     (mass*(error-ideal_gate*step)**2).sum(), original_mass[k])),
                                 'coarse_projection': {}}
            gate_array = gates.cpu().numpy()
            row = {'index': index, 'methods': methods,
                   'gate': {'mean': float(gate_array.mean()), 'std': float(gate_array.std()),
                            'min': float(gate_array.min()), 'max': float(gate_array.max())}}
            # Propagate P^T A e through the same hierarchy while explicitly
            # forming diagnostic Galerkin matrices, never a dense fine matrix.
            coarse_rhs = matrix@errors
            current = matrix
            for depth, (prolongation, restriction) in enumerate(compiled.plan.transfers, 1):
                current = (restriction@current@prolongation).tocsr()
                coarse_rhs = restriction@coarse_rhs
                if current.nnz > 8_000_000:
                    raise RuntimeError('Diagnostic sparse-size safety limit exceeded')
                if depth not in (1, 3, 6, len(compiled.plan.transfers)):
                    continue
                factor = sla.splu(current.tocsc(), permc_spec='MMD_AT_PLUS_A')
                coordinates = factor.solve(coarse_rhs)
                projection = coordinates
                for p, _ in reversed(compiled.plan.transfers[:depth]):
                    projection = p@projection
                remainder = errors-projection
                remainder_energy = np.einsum('ij,ij->j', remainder, matrix@remainder)
                remainder_mass = np.einsum('i,ij,ij->j', mass, remainder, remainder)
                captured = np.einsum('ij,ij->j', coordinates, coarse_rhs)
                orthogonality = np.linalg.norm(current@coordinates-coarse_rhs, axis=0)/np.maximum(np.linalg.norm(coarse_rhs, axis=0), 1e-30)
                for k, name in enumerate(outputs):
                    methods[name]['coarse_projection'][f'level_{depth}'] = {
                        'dimension': current.shape[0],
                        'energy_captured_percent': 100*positive_ratio(captured[k], original_energy[k]),
                        'energy_remaining_percent': 100*positive_ratio(remainder_energy[k], original_energy[k]),
                        'mass_norm_remaining_percent': 100*np.sqrt(positive_ratio(remainder_mass[k], original_mass[k])),
                        'projection_relative_linear_residual': float(orthogonality[k])}
            rows.append(row)
            print(json.dumps(row), flush=True)
    save_json(args.out, {'purpose': __doc__, 'selection': 'Median and maximum fixed-MG-K2 field error among 32 tests, plus prior learned-stall index 30',
                        'projection': 'q=P(P.T A P)^(-1)P.T A e; exact A-orthogonal projection, not the approximate recursive V-cycle',
                        'algebraic_Rayleigh': 'e.T A e / (e.T diag(A) e), exact current fine diagonal',
                        'oracle_fine_step': 'One fine-level Jacobi step only, omega as deployed and 0.5<=g_i<=1.5; true-error-optimal gate, diagnostic not deployable. Wrong direction means e_i*(A e)_i<=0; positive local steps cannot reduce pointwise absolute error there at that step. This does not constrain a full V-cycle.',
                        'interpretation': 'Projection residual is a limitation of that coarse subspace ALONE, not a lower bound for a full MG cycle with smoothing; energy and field norms differ.',
                        'samples': rows})


if __name__ == '__main__':
    main()
