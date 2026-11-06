"""Validate and summarize the immutable depth-study measurements.

Run with the existing server conda environment for matplotlib figures.
Tables are generated directly from per-sample JSON, never from hand-entered
paper numbers. This script does not import torch or train any model.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


METRICS = ('L2_percent', 'gradient_percent', 'flux_L2_percent')


def read(path):
    return json.loads(path.read_text())


def table(path, rows):
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', required=True, type=Path)
    args = p.parse_args()
    root = args.root
    main_result = read(root/'test32/evaluation.json')
    controls = read(root/'controls_test32/evaluation.json')
    assert len(main_result['samples']) == len(controls['samples']) == 32
    assert [r['sample_index'] for r in main_result['samples']] == [r['sample_index'] for r in controls['samples']]
    rows = []
    for source, data in [('test32', main_result), ('controls_test32', controls)]:
        for method in data['summary']:
            # Controls' original is the two-cycle model, already tabulated.
            if source == 'controls_test32' and method.startswith('original'):
                continue
            values = {key:float(np.mean([r['methods'][method][key] for r in data['samples']]))
                      for key in (*METRICS, 'energy_percent', 'total_ms')}
            assert all(np.isfinite(list(values.values())))
            for key, value in values.items():
                assert np.isclose(value, data['summary'][method][key]['mean'])
            rows.append(dict(method=method, n=32, **values, source=source+'/evaluation.json'))
    table(root/'SUMMARY.csv', rows)
    mechanism = read(root/'mechanism_original/diagnosis.json')['samples']
    mechanism_rows = []
    for method in mechanism[0]['methods']:
        row = dict(method=method, n=len(mechanism))
        row.update({k:np.mean([r['methods'][method][k] for r in mechanism]) for k in METRICS})
        mechanism_rows.append(row)
    table(root/'MECHANISM.csv', mechanism_rows)
    transfer = read(root/'transfer_cycle2/transfer.json')['samples']
    assert len(transfer) == 72
    assert max(r['reference_relative_residual'] for r in transfer) < 1e-8
    transfer_rows = []
    for power in (.75, 1., 1.25):
        for source in ('constant', 'left_source', 'signed_source'):
            group = [r for r in transfer if r['power'] == power and r['source'] == source]
            assert len(group) == 8
            for method in group[0]['methods']:
                transfer_rows.append(dict(power=power, source=source, method=method, n=len(group),
                    **{k:float(np.mean([r['methods'][method][k] for r in group])) for k in (*METRICS, 'apply_ms')}))
    table(root/'TRANSFER.csv', transfer_rows)
    comparisons = []
    for source, data, pairs in [
        ('test32', main_result, [('cycle_trained_K2','original_K2'),
                               ('cycle_trained_K4','original_K4'),
                               ('union144_K2','fixed144_K2'), ('union144_K4','MG_K4')]),
        ('controls_test32', controls, [('original_K2','cycle1_control_K2'),
                                      ('original_K2','material_only_K2')])]:
        for a, b in pairs:
            for k in (*METRICS, 'total_ms'):
                av = np.array([r['methods'][a][k] for r in data['samples']])
                bv = np.array([r['methods'][b][k] for r in data['samples']])
                comparisons.append(dict(source=source, candidate=a, baseline=b, metric=k,
                    candidate_mean=float(av.mean()), baseline_mean=float(bv.mean()),
                    reduction_percent=float(100*(1-av.mean()/bv.mean())), wins=int((av < bv).sum()), n=32))
    table(root/'PAIRED_COMPARISONS.csv', comparisons)
    audit = dict(main_samples=32, control_samples=32, mechanism_samples=len(mechanism),
                 transfer_cases=len(transfer),
                 max_reference_residual=max(r['reference_relative_residual'] for r in transfer),
                 finite_metrics=True, matched_test_indices=True,
                 interpretation='Exploratory single-seed research; repeated timing is not repeated training. All raw comparisons remain available.')
    cold = read(root/'cold_plan_audit/evaluation.json')
    audit['cold_mesh_plan_ms'] = cold['mesh_plan_ms_once']
    audit['cold_assembly_positions_ms'] = cold['assembly_plan_ms_once']
    audit['metadata_correction'] = ('In test32 and controls_test32, mesh_plan_ms_once originally records only attachment to a shared plan. '
        'Use cold_plan_audit for the true cold mesh plan. E2E online timings are unaffected. The source now records the original base plan and attachment separately.')
    spectral = read(root/'spectral_matched4/spectral.json')['samples']
    assert len(spectral) == 4
    assert max(r['max_eigen_relative_residual'] for r in spectral) < 1e-7
    audit['matched_spectral_materials'] = 4
    audit['matched_actual_energy_capture'] = {
        n:float(np.mean([r['actual_filtered_error_energy_captured'][n] for r in spectral]))
        for n in ('learned', 'fixed_sine')}
    (root/'AUDIT.json').write_text(json.dumps(audit, indent=2)+'\n')

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':10, 'svg.fonttype':'none'})
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.7), constrained_layout=True)
    methods = main_result['summary']
    for ax, key, label in zip(axes, METRICS, ('Field error (%)', 'Gradient error (%)', 'Flux error (%)')):
        for prefix, ks, text, color in [('MG', (2,4), 'FNO + numerical MG', '#767676'),
                    ('original', (1,2,4), 'Original learned 72', '#ce9c42'),
                    ('cycle_trained', (1,2,4), 'Cycle-trained learned 72', '#2e6f9e'),
                    ('fixed144', (2,4), 'Fixed sine 144', '#4a9c78'),
                    ('union144', (2,4), 'Fixed + learned 144', '#aa3c52')]:
            ax.plot([methods[f'{prefix}_K{k}']['total_ms']['mean'] for k in ks],
                    [methods[f'{prefix}_K{k}'][key]['mean'] for k in ks], 'o-', color=color, label=text)
        ax.set_yscale('log'); ax.set_xlabel('Full online latency (ms)'); ax.set_ylabel(label)
        ax.grid(True, alpha=.2)
    axes[0].legend(fontsize=7.8)
    fig.savefig(root/'ACCURACY_TIME.svg'); plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.7), constrained_layout=True)
    labels = ['Learned', 'Fixed sine', 'Mismatched state', 'Dynamic inputs removed']
    names = ['learned','fixed_sine','other_material_space','material_only_space']
    captured = [100*np.mean([r['details'][n]['captured_A_energy_fraction'] for r in mechanism]) for n in names]
    axes[0].bar(labels, captured, color=['#2e6f9e','#4a9c78','#999999','#ce9c42'])
    axes[0].set_ylabel('Filtered error energy captured (%)')
    axes[0].tick_params(axis='x', labelrotation=17)
    for n, text, color in [('learned','Learned','#2e6f9e'), ('fixed_sine','Fixed sine','#4a9c78')]:
        values = np.mean([r['fraction_of_each_slow_eigenmode_captured'][n] for r in spectral], axis=0)
        axes[1].plot(np.arange(1,17), 100*values, 'o-', label=text, color=color)
    axes[1].set_xlabel('Slow eigenmode index of A v = lambda D v')
    axes[1].set_ylabel('Eigenmode energy captured (%)'); axes[1].legend()
    axes[1].text(.03, .03, 'Same 4 materials: actual error captured\nLearned 77.70% / fixed 8.56%',
                 transform=axes[1].transAxes, fontsize=8)
    fig.savefig(root/'MECHANISM.svg'); plt.close(fig)
    print(json.dumps(audit))


if __name__ == '__main__':
    main()
