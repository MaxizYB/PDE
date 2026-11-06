"""Offline smoother stability audit; does not modify deployed CDMC or select weights."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from run import (DarcySplit, CDMCSetup, compiled_plan, make_example,
                 gates_from_implicit, field_metrics, save_json, summary)
from solver.numerics import coarsen_gates
from solver.matrixfree_cdmc import _v_cycle, correct_matrixfree


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--no-dir', type=Path, required=True)
    parser.add_argument('--corrector-dir', type=Path, required=True)
    parser.add_argument('--evaluation', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--power-steps', type=int, default=32)
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.manual_seed(20261023)
    device = torch.device(args.device)
    source = json.loads(args.evaluation.read_text())
    name = args.corrector_dir.name
    # Deliberately diagnostic selection: most adverse extension beyond trained K2.
    ranked = sorted(source['samples'], key=lambda row:
                    row['methods'][name+'_K4']['L2_percent'] /
                    row['methods']['fixed_MG_K4']['L2_percent'], reverse=True)
    selected = list(dict.fromkeys([0, *(row['index'] for row in ranked[:2])]))
    split = DarcySplit(args.data, 'test')
    compiled = compiled_plan(split, device, SimpleNamespace(coarse_size=128))
    with np.load(args.no_dir/'predictions.npz') as data:
        prediction = data['test'].copy()
        train_prediction = data['train'].copy()
    checkpoint = torch.load(args.corrector_dir/'cdmc.pt', map_location=device, weights_only=False)
    setup = CDMCSetup(checkpoint['width']).to(device)
    setup.load_state_dict(checkpoint['state'])
    setup.eval()
    rows = []
    with torch.no_grad():
        for index in selected:
            system, element_a, initial, _ = make_example(split, prediction, index, compiled, device)
            gate, _ = gates_from_implicit(setup, system, compiled.action, element_a, initial)
            gates, _ = coarsen_gates(system, gate, torch.ones_like(gate))
            _, coarse_gates = coarsen_gates(system, gate, torch.ones_like(gate))
            cycle_records = []
            x = initial.clone()
            for cycle in range(4):
                residual = system.rhs-compiled.action.action(0, x, element_a)
                direction = _v_cycle(system, compiled.action, element_a, residual,
                                     gates, coarse_gates, None, 0)
                ad = compiled.action.action(0, direction, element_a)
                numerator = residual@direction
                denominator = direction@ad
                alpha = (numerator/denominator.clamp_min(1e-12)).clamp(0, 2)
                cycle_records.append({'cycle': cycle+1, 'residual_dot_direction': float(numerator),
                                      'direction_energy': float(denominator),
                                      'alpha': float(alpha), 'relative_residual': float(residual.norm()/system.rhs.norm())})
                x = x+alpha*direction
            levels = []
            for level, item in enumerate(system.levels[:-1]):
                record = {'level': level, 'size': len(item.diagonal_inverse),
                          'omega': item.omega, 'gate_min': float(gates[level].min()),
                          'gate_max': float(gates[level].max()),
                          'gate_mean': float(gates[level].mean())}
                for mode, scale in [('fixed', torch.ones_like(gates[level])),
                                    ('learned', gates[level])]:
                    root = (item.omega*item.diagonal_inverse*scale).sqrt()
                    vector = torch.randn_like(root)
                    vector /= vector.norm()
                    for _ in range(args.power_steps):
                        action = root*compiled.action.action(level, root*vector, element_a)
                        vector = action/action.norm().clamp_min(1e-30)
                    action = root*compiled.action.action(level, root*vector, element_a)
                    rayleigh = vector@action
                    record[mode+'_lambda_rayleigh'] = float(rayleigh)
                    record[mode+'_eigen_residual'] = float((action-rayleigh*vector).norm()/action.norm())
                levels.append(record)
            rows.append({'index': index, 'coefficient_contrast': float(split.coefficient[index].max()/split.coefficient[index].min()),
                         'errors': {k: v['L2_percent'] for k,v in source['samples'][index]['methods'].items()},
                         'cycles': cycle_records, 'levels': levels})
            print(json.dumps(rows[-1]), flush=True)
    # Calibration uses training inputs only. This control tests whether a spatial
    # graph setup adds value beyond a learned global relaxation scalar.
    train = DarcySplit(args.data, 'train')
    train_means = []
    with torch.no_grad():
        for index in range(checkpoint['training_samples']):
            system, element_a, initial, _ = make_example(train, train_prediction, index, compiled, device)
            gate, _ = gates_from_implicit(setup, system, compiled.action, element_a, initial)
            train_means.append(float(gate.mean()))
        constant = float(np.mean(train_means))
        controls = []
        for index in range(source['test_count']):
            system, element_a, initial, target = make_example(split, prediction, index, compiled, device)
            states = correct_matrixfree(system, compiled.action, element_a, initial,
                                       torch.full_like(initial, constant), torch.ones_like(initial),
                                       None, cycles=4, coarse_steps=0, return_states=True)
            record = {'index': index, 'methods': {}}
            for cycle in (1, 2, 4):
                x = np.zeros_like(split.target[index])
                x[compiled.plan.free] = states[cycle-1].cpu().numpy()
                record['methods'][f'constant_K{cycle}'] = field_metrics(
                    x, split.target[index], split.coefficient[index], split.mesh)
            controls.append(record)
    control_summary = {name: {metric: summary([r['methods'][name][metric] for r in controls])
                              for metric in controls[0]['methods'][name]}
                       for name in controls[0]['methods']}
    save_json(args.out, {'purpose': 'Offline diagnosis of adverse K4 cases plus train-calibrated constant gate control; no weight selection or deployment change',
                        'operator': 'D_g^(1/2) A_l D_g^(1/2), D_g=omega_l*gate_l*approximate_inverse_diagonal_l',
                        'interpretation': 'Rayleigh value >2 proves an expanding Jacobi eigenmode; <=2 alone is not a certified upper bound',
                        'power_steps': args.power_steps, 'samples': rows,
                        'constant_control': {'calibration_samples': len(train_means),
                                             'calibration_split': 'train', 'gate_value': constant,
                                             'summary': control_summary, 'samples': controls}})


if __name__ == '__main__':
    main()
