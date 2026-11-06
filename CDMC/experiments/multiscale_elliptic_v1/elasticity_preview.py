"""Matched vector-PDE extension of the current learned response-space solver.

Stages: check, train-no, train-response, evaluate. No scalar-Darcy checkpoint
is reused. Only the physical adapter, input dimension and vector mode output
change; filter / learned space / exact energy projection / filter is retained.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from types import SimpleNamespace
import time

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import splu
import torch
from torch import nn
from torch.nn import functional as F

from run import clock, measure, save_json, summary
from models import SpectralConv
from polynomial_joint import chebyshev, partition_weights
from coupled_coarse import ModeOnlyCompiled, CoarseContext, CoupledCoarseSetup, factor_modes, apply_modes, mm
from adapters.darcy import Mesh
from baselines.data.elasticity_v1.generate import physics


def load_split(root, split):
    with np.load(Path(root) / f'{split}.npz') as z:
        return {k: z[k].copy() for k in z.files}


class ElasticAdapter:
    """Mesh-only CSR assembly plan, with no material coarse matrices.

    P1 engineering strain is (eps_xx, eps_yy, 2 eps_xy).
    Unknown x is the free, interleaved displacement fluctuation u - lift.
    The exact lift enters b = -(K lift)_free, including internal lift nodes.
    """
    def __init__(self, resolution, device):
        start = clock(device)
        self.g = g = physics.geometry(resolution - 1, nu=0.3)
        self.device = device
        self.n = resolution
        self.free_cpu = g['free']
        self.node_free = np.flatnonzero(~g['boundary'])
        tolong = lambda x: torch.as_tensor(x, dtype=torch.long, device=device)
        tovalue = lambda x: torch.as_tensor(x, dtype=torch.float32, device=device)
        self.free = tolong(g['free'])
        self.affine = tovalue(g['affine'])
        self.points = tovalue(g['points'])
        self.xy = self.points.reshape(1, resolution, resolution, 2)
        self.mask = tovalue(~g['boundary']).reshape(1, resolution, resolution, 1)
        self.bmat = tovalue(g['bmat'])
        self.cunit = tovalue(g['cunit'])
        self.dofs = tolong(g['dofs'])
        self.area = tovalue(g['area'])
        self.cells = tolong(g['triangles'])
        mass = np.zeros(len(g['points']))
        np.add.at(mass, g['triangles'].ravel(), np.repeat(g['area'] / 3, 3))
        self.mass = tovalue(mass)
        mesh = Mesh(g['points'], g['triangles'], g['boundary'], mass)
        self.scalar = ModeOnlyCompiled(mesh, device)
        self.context = CoarseContext(self.scalar, start_level=1)
        # Context pooling stays nodal; physical actions below are truly vector coupled.
        self.context.compiled = SimpleNamespace(action=self)
        self.points_free = self.points[self.node_free]
        self.fine = self
        size = len(g['free'])
        position = np.full(2 * len(g['points']), -1, dtype=np.int64)
        position[g['free']] = np.arange(size)
        reduced_rows = position[g['rows']]
        reduced_cols = position[g['cols']]
        internal = (reduced_rows >= 0) & (reduced_cols >= 0)
        keys, inverse = np.unique(reduced_rows[internal] * size + reduced_cols[internal], return_inverse=True)
        rows, cols = keys // size, keys % size
        crow = np.r_[0, np.cumsum(np.bincount(rows, minlength=size))]
        elements = np.repeat(np.arange(len(g['triangles'])), 36)
        weights = g['unit_local'].ravel()
        keep = reduced_rows >= 0
        self.element = tolong(elements[internal])
        self.scatter = tolong(inverse)
        self.weight = torch.as_tensor(weights[internal], dtype=torch.float64, device=device)
        self.lift_element = tolong(elements[keep])
        self.lift_row = tolong(reduced_rows[keep])
        self.lift_weight = torch.as_tensor(
            -weights[keep] * g['affine'][g['cols'][keep]], dtype=torch.float64, device=device)
        self.crow, self.cols = tolong(crow), tolong(cols)
        self.diagonal_position = tolong(np.flatnonzero(rows == cols))
        self.size = size
        # For positive scalar E per element, this local absolute-row ratio
        # bounds every assembled D^-1 K spectrum, independently of E contrast.
        local = g['unit_local']
        self.upper = float(1.01 * np.max(np.abs(local).sum(2) / np.diagonal(local, axis1=1, axis2=2)))
        self.omega = 1.6 / self.upper
        self.rigid = np.zeros((size, 3))
        p = g['points'][self.node_free]
        self.rigid[0::2, 0] = 1
        self.rigid[1::2, 1] = 1
        self.rigid[0::2, 2], self.rigid[1::2, 2] = -p[:, 1], p[:, 0]
        # A fixed mesh must be reusable for the mature FEM baseline too.
        # Rebuilding MeshTri/Basis in every numerical call is not a fair
        # comparison against the learned branch's prepared assembly plan.
        from skfem import MeshTri, Basis, ElementVector, ElementTriP1
        fem_mesh = MeshTri(np.ascontiguousarray(g['points'].T),
                           np.ascontiguousarray(g['triangles'].T), sort_t=False)
        self.fem_basis = Basis(fem_mesh, ElementVector(ElementTriP1()), intorder=2)
        self.plan_ms_once = (clock(device)-start)*1000

    def build(self, modulus):
        e = torch.as_tensor(modulus, dtype=torch.float64, device=self.device)
        values = torch.zeros(self.cols.numel(), dtype=torch.float64, device=self.device)
        values.index_add_(0, self.scatter, e[self.element] * self.weight)
        rhs = torch.zeros(self.size, dtype=torch.float64, device=self.device)
        rhs.index_add_(0, self.lift_row, e[self.lift_element] * self.lift_weight)
        self.matrix = torch.sparse_csr_tensor(self.crow, self.cols, values.float(),
            size=(self.size, self.size), device=self.device)
        diagonal = values[self.diagonal_position].float()
        system = SimpleNamespace(rhs=rhs.float(),
            levels=[SimpleNamespace(diagonal_inverse=diagonal.reciprocal())])
        return system, e.float()

    def action(self, level, x, ae):
        if level != 0:
            raise ValueError('No numerical coarse operators in the response solver')
        return mm(self.matrix, x[:, None])[:, 0]

    def homogeneous_action(self, x, ae):
        return mm(self.matrix, x.T).T

    def features(self, system, ae, x):
        residual = system.rhs-self.action(0, x, ae)
        defect = (system.levels[0].diagonal_inverse*residual).reshape(-1, 2)
        field = x.reshape(-1, 2)
        field = field / field.square().mean().sqrt().clamp_min(1e-12)
        defect = defect / defect.square().mean().sqrt().clamp_min(1e-12)
        # Area-weighted nodal modulus: exact cell E remains in physical assembly.
        nodal = torch.zeros(len(self.points), device=self.device)
        nodal.index_add_(0, self.cells.flatten(), (self.area*ae/3).repeat_interleave(3))
        loge = (nodal/self.mass).log()[self.node_free, None]/np.log(20)
        diagonal = system.levels[0].diagonal_inverse.reciprocal().reshape(-1, 2)
        diagonal = (diagonal/diagonal.mean()).log()
        logmass = (self.mass[self.node_free]/self.mass[self.node_free].mean()).log()[:, None]
        values = torch.cat((loge, diagonal, self.points_free, logmass, field, defect), 1)
        result = [values]
        for restriction in self.context.restrict:
            result.append(mm(restriction, result[-1]))
        return result

    def full(self, fluctuation):
        full = self.affine.clone()
        full[self.free] += fluctuation
        return full

    def ratios(self, x, target, system, ae):
        full = self.full(x)
        error = full-target.flatten()
        e = torch.einsum('tij,tj->ti', self.bmat, error[self.dofs])
        t = torch.einsum('tij,tj->ti', self.bmat, target.flatten()[self.dofs])
        se = (e@self.cunit.T)*ae[:, None]
        st = (t@self.cunit.T)*ae[:, None]
        def weighted_norm(v, weights):
            return (self.area[:, None]*weights*v.square()).sum()
        displacement = (self.mass[:, None]*error.reshape(-1, 2).square()).sum() / (
            self.mass[:, None]*target.reshape(-1, 2).square()).sum().clamp_min(1e-25)
        strain = weighted_norm(e, e.new_tensor([1, 1, .5])) / weighted_norm(t, t.new_tensor([1, 1, .5])).clamp_min(1e-25)
        stress = weighted_norm(se, se.new_tensor([1, 1, 2])) / weighted_norm(st, st.new_tensor([1, 1, 2])).clamp_min(1e-25)
        targetx = (target.flatten()-self.affine)[self.free]
        errx = x-targetx
        energy = (errx*self.action(0, errx, ae)).sum() / (
            targetx*self.action(0, targetx, ae)).sum().clamp_min(1e-25)
        fluctuation = errx.square().sum()/targetx.square().sum().clamp_min(1e-25)
        residual = (system.rhs-self.action(0, x, ae)).square().sum()/system.rhs.square().sum().clamp_min(1e-25)
        return torch.stack((displacement, strain, stress, energy, fluctuation, residual)).clamp_min(1e-25).sqrt()


class VectorFNO(nn.Module):
    """Standard four-block FNO, two coefficient channels to two displacements."""
    def __init__(self, width=24, modes=12):
        super().__init__()
        self.lift = nn.Linear(4, width)
        self.spectral = nn.ModuleList(SpectralConv(width, modes) for _ in range(4))
        self.point = nn.ModuleList(nn.Conv2d(width, width, 1) for _ in range(4))
        self.project = nn.Sequential(nn.Linear(width, 96), nn.GELU(), nn.Linear(96, 2))

    def forward(self, values, xy):
        v = self.lift(torch.cat((values, xy.expand(len(values), -1, -1, -1)), -1))
        v = F.pad(v.permute(0, 3, 1, 2), (0, 8, 0, 8))
        for k, (s, p) in enumerate(zip(self.spectral, self.point)):
            v = s(v)+p(v)
            if k < 3:
                v = F.gelu(v)
        return self.project(v[:, :, :-8, :-8].permute(0, 2, 3, 1))


def no_predict(model, stats, inputs, adapter):
    mean = stats['mean']
    if mean.shape[1:3] != inputs.shape[1:3]:
        mean = F.interpolate(mean.permute(0, 3, 1, 2), inputs.shape[1:3],
                             mode='bilinear', align_corners=True).permute(0, 2, 3, 1)
    fluctuation = (model((inputs-stats['input_mean'])/stats['input_std'], adapter.xy)*stats['scale']+mean)*adapter.mask
    return fluctuation.reshape(len(inputs), -1)[:, adapter.free]


def train_no(args, device):
    tr, va = load_split(args.data, 'train'), load_split(args.data, 'val')
    adapter = ElasticAdapter(tr['target'].shape[1], device)
    a, y = [torch.as_tensor(tr[k], device=device, dtype=torch.float32) for k in ('inputs', 'target')]
    vy, vaa = [torch.as_tensor(va[k], device=device, dtype=torch.float32) for k in ('target', 'inputs')]
    v = y.reshape(len(y), -1)-adapter.affine
    stats = dict(mean=v.reshape_as(y).mean(0, keepdim=True), scale=v.std(),
                 input_mean=a.mean(), input_std=a.std().clamp_min(1e-6))
    model = VectorFNO().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.no_epochs, eta_min=1e-5)
    mass = adapter.mass[adapter.node_free].repeat_interleave(2)
    def loss(pred, target):
        targetx = (target.reshape(len(target), -1)-adapter.affine)[:, adapter.free]
        return ((mass*(pred-targetx).square()).sum(1)/(mass*targetx.square()).sum(1).clamp_min(1e-25)).sqrt().mean()
    best, history = float('inf'), []
    started = clock(device)
    for epoch in range(args.no_epochs):
        model.train()
        losses = []
        for indices in torch.randperm(len(a), device=device).split(8):
            prediction = no_predict(model, stats, a[indices], adapter)
            value = loss(prediction, y[indices])
            opt.zero_grad(set_to_none=True)
            value.backward()
            opt.step()
            losses.append(float(value.detach()))
        scheduler.step()
        model.eval()
        with torch.no_grad():
            values = [loss(no_predict(model, stats, vaa[i:i+8], adapter), vy[i:i+8]) for i in range(0, len(vaa), 8)]
            score = float(torch.stack(values).mean())
        row = dict(epoch=epoch+1, train_relative_fluctuation=float(np.mean(losses)), val_relative_fluctuation=score, elapsed_s=clock(device)-started)
        history.append(row)
        if score < best:
            best = score
            torch.save(dict(state=model.state_dict(), stats=stats, epoch=epoch+1, seed=args.seed), args.out/'elastic_no.pt')
        if epoch == 0 or (epoch+1) % 10 == 0:
            print(json.dumps(row), flush=True)
    save_json(args.out/'no_training.json', dict(history=history, seconds=clock(device)-started, parameters=sum(p.numel() for p in model.parameters())))


def load_no(out, device):
    checkpoint = torch.load(Path(out)/'elastic_no.pt', map_location=device, weights_only=False)
    model = VectorFNO().to(device)
    model.load_state_dict(checkpoint['state'])
    model.eval()
    return model, checkpoint['stats']


def response_model(device, anchor=False):
    model = CoupledCoarseSetup(width=16, rank=16, start_level=1)
    model.lift = nn.Linear(10, 16)
    model.reference_anchor = anchor
    if anchor:
        nn.init.zeros_(model.output.weight)
        nn.init.zeros_(model.output.bias)
    return model.to(device)


def reference_modes(adapter):
    p = adapter.points_free
    base = torch.stack([torch.sin(k*np.pi*p[:, 0])*torch.sin(l*np.pi*p[:, 1])
                        for k, l in ((1, 1), (1, 2), (2, 1), (2, 2))], 1)
    v = torch.zeros((len(p), 2, 8), device=p.device)
    v[:, 0, :4], v[:, 1, 4:] = base, base
    return v.reshape(-1, 8)


def prepare(model, adapter, system, ae, x, fixed=False):
    if fixed:
        modes = reference_modes(adapter)
    else:
        modes = model(adapter.features(system, ae, x), adapter.context).reshape(-1, 2, 8).reshape(-1, 8)
        if getattr(model, 'reference_anchor', False):
            modes = reference_modes(adapter)+modes
    hats = partition_weights(adapter.points_free, 3).repeat_interleave(2, 0)
    modes = (modes[:, :, None]*hats[:, None, :]).flatten(1)
    for _ in range(2):
        modes = modes-adapter.omega*system.levels[0].diagonal_inverse[:, None]*mm(adapter.matrix, modes)
    return factor_modes(modes, adapter.context, ae)


def run_response(model, adapter, system, ae, x, cycles=1, fixed=False, pre=64, post=8):
    prepared = None
    for _ in range(cycles):
        x = chebyshev(system, adapter, ae, x, pre, .001, adapter.upper)
        if prepared is None:
            prepared = prepare(model, adapter, system, ae, x, fixed)
        x = apply_modes(prepared, system, ae, x, adapter.context)
        x = chebyshev(system, adapter, ae, x, post, .05, adapter.upper)
    return x


def train_response(args, device):
    no, stats = load_no(args.out, device)
    tr, va = load_split(args.data, 'train'), load_split(args.data, 'val')
    adapter = ElasticAdapter(tr['target'].shape[1], device)
    def cache(data, count):
        result = []
        with torch.no_grad():
            for i in range(min(count, len(data['target']))):
                system, ae = adapter.build(data['modulus'][i])
                inputs = torch.as_tensor(data['inputs'][i:i+1], device=device)
                x0 = no_predict(no, stats, inputs, adapter)[0]
                x = chebyshev(system, adapter, ae, x0, args.pre, .001, adapter.upper)
                result.append((system, ae, adapter.matrix, x, torch.as_tensor(data['target'][i], dtype=torch.float32, device=device)))
        return result
    trc, vac = cache(tr, 48), cache(va, 12)
    model = response_model(device, args.anchor)
    opt = torch.optim.AdamW(model.parameters(), lr=.002, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.response_epochs, eta_min=1e-5)
    def score(item):
        system, ae, matrix, x, target = item
        adapter.matrix = matrix
        prepared = prepare(model, adapter, system, ae, x)
        corrected = apply_modes(prepared, system, ae, x, adapter.context)
        corrected = chebyshev(system, adapter, ae, corrected, args.post, .05, adapter.upper)
        ratios = adapter.ratios(corrected, target, system, ae)
        return ratios[:3].square().sum(), ratios[:3]
    history, best = [], float('inf')
    suffix = '_anchored' if args.anchor else ''
    started = clock(device)
    if args.anchor:
        with torch.no_grad():
            initial_metrics = torch.stack([score(v)[1] for v in vac]).mean(0)
            best = float(initial_metrics.square().sum())
        history.append(dict(epoch=0, validation_percent=(100*initial_metrics).tolist()))
        torch.save(dict(state=model.state_dict(), epoch=0, seed=args.seed,
                        pre=args.pre, post=args.post, anchor=True),
                   args.out/f'elastic_response{suffix}.pt')
    for epoch in range(args.response_epochs):
        model.train()
        losses = []
        for j in np.random.default_rng(args.seed+epoch).permutation(len(trc)):
            value, _ = score(trc[j])
            opt.zero_grad(set_to_none=True)
            value.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            opt.step()
            losses.append(float(value.detach()))
        scheduler.step()
        model.eval()
        with torch.no_grad():
            metrics = torch.stack([score(v)[1] for v in vac]).mean(0)
            total = float(metrics.square().sum())
        row = dict(epoch=epoch+1, training_loss=float(np.mean(losses)), validation_percent=(100*metrics).tolist(), elapsed_s=clock(device)-started)
        history.append(row)
        if total < best:
            best = total
            torch.save(dict(state=model.state_dict(), epoch=epoch+1, seed=args.seed, pre=args.pre, post=args.post, anchor=args.anchor), args.out/f'elastic_response{suffix}.pt')
        if epoch == 0 or (epoch+1) % 5 == 0:
            print(json.dumps(row), flush=True)
    save_json(args.out/f'response_training{suffix}.json', dict(history=history, seconds=clock(device)-started, training_samples=len(trc), parameters=sum(p.numel() for p in model.parameters())))


METRICS = ['displacement_percent', 'strain_percent', 'stress_percent', 'energy_percent', 'fluctuation_percent', 'residual_percent']


def numerical(adapter, modulus, x0, method, cycles):
    g, free = adapter.g, adapter.free_cpu
    # Mature scikit-fem assembly and PyAMG / SuperLU; all setup is timed.
    from skfem import asm
    from skfem.models.elasticity import linear_elasticity
    modulus = modulus.astype(np.float64)
    nu = g['nu']
    lam = (modulus*nu/((1+nu)*(1-2*nu)))[:, None]
    mu = (modulus/(2*(1+nu)))[:, None]
    matrix = asm(linear_elasticity(Lambda=lam, Mu=mu), adapter.fem_basis).tocsr()
    a = matrix[free][:, free].tocsr()
    b = -(matrix@g['affine'])[free]
    if method == 'LU':
        x = splu(a.tocsc()).solve(b)
    else:
        import pyamg
        hierarchy = pyamg.smoothed_aggregation_solver(a.tobsr(blocksize=(2, 2)),
            B=adapter.rigid, symmetry='symmetric', max_coarse=64,
            presmoother=('block_gauss_seidel', {'sweep': 'symmetric'}),
            postsmoother=('block_gauss_seidel', {'sweep': 'symmetric'}))
        x = hierarchy.solve(b, x0=x0.detach().cpu().numpy().astype(np.float64),
                            maxiter=cycles, tol=0., cycle='V', accel=None)
    return torch.as_tensor(x, dtype=torch.float32, device=adapter.device)


def phase_times(model, no, stats, adapter, inputs, modulus, args):
    phases = {}
    t = clock(adapter.device)
    x = no_predict(no, stats, inputs, adapter)[0]
    phases['NO'] = (clock(adapter.device)-t)*1000
    t = clock(adapter.device)
    system, ae = adapter.build(modulus)
    phases['fine_assembly'] = (clock(adapter.device)-t)*1000
    t = clock(adapter.device)
    x = chebyshev(system, adapter, ae, x, args.pre, .001, adapter.upper)
    phases['prefilter'] = (clock(adapter.device)-t)*1000
    t = clock(adapter.device)
    prepared = prepare(model, adapter, system, ae, x)
    phases['features_graph_space_factor'] = (clock(adapter.device)-t)*1000
    t = clock(adapter.device)
    x = apply_modes(prepared, system, ae, x, adapter.context)
    phases['projection'] = (clock(adapter.device)-t)*1000
    t = clock(adapter.device)
    chebyshev(system, adapter, ae, x, args.post, .05, adapter.upper)
    phases['postfilter'] = (clock(adapter.device)-t)*1000
    return phases


def evaluate(args, device):
    no, stats = load_no(args.out, device)
    model = response_model(device, args.anchor)
    suffix = '_anchored' if args.anchor else ''
    checkpoint = torch.load(args.out/f'elastic_response{suffix}.pt', map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['state'])
    model.eval()
    saved = json.loads((args.data/'manifest.json').read_text())
    if args.training_data is not None:
        training_manifest = json.loads((args.training_data/'manifest.json').read_text())
        if (saved['resolution'] != training_manifest['resolution']
                and saved['seed'] == training_manifest['seed']):
            raise ValueError('Independent cross-grid evaluation requires a distinct material seed; matching shapes can otherwise overlap the training set')
    adapter = ElasticAdapter(saved['resolution'], device)
    rows = []
    with torch.no_grad():
        for split in ('test', 'ood'):
            data = load_split(args.data, split)
            for i in range(min(args.count, len(data['target']))):
                inputs = torch.as_tensor(data['inputs'][i:i+1], device=device)
                target = torch.as_tensor(data['target'][i], device=device, dtype=torch.float32)
                modulus = data['modulus'][i]
                def pipeline(method, cycles=1):
                    if method in ('LU', 'zero-AMG'):
                        zero = torch.zeros(adapter.size, device=device)
                        return numerical(adapter, modulus, zero,
                                         'LU' if method == 'LU' else 'AMG', cycles)
                    x = no_predict(no, stats, inputs, adapter)[0]
                    if method == 'NO':
                        return x
                    if method in ('AMG', 'LU'):
                        return numerical(adapter, modulus, x, method, cycles)
                    system, ae = adapter.build(modulus)
                    if method == 'polynomial':
                        for _ in range(cycles):
                            x = chebyshev(system, adapter, ae, x, args.pre, .001, adapter.upper)
                            x = chebyshev(system, adapter, ae, x, args.post, .05, adapter.upper)
                        return x
                    return run_response(model, adapter, system, ae, x, cycles, method == 'fixed-sine', args.pre, args.post)
                configs = [('NO', 0), ('polynomial', 1), ('polynomial', 2),
                           ('fixed-sine', 1), ('fixed-sine', 2),
                           ('learned', 1), ('learned', 2)]
                if args.anchor:
                    configs = [('learned', 1), ('learned', 2)]
                elif args.only_numerics:
                    configs = [('AMG', 1), ('AMG', 2), ('zero-AMG', 2), ('LU', 1)]
                elif not args.no_numerics:
                    configs += [('AMG', 1), ('AMG', 2), ('zero-AMG', 2), ('LU', 1)]
                methods = {}
                for method, cycles in configs:
                    output, elapsed = measure(lambda: pipeline(method, cycles), device, args.repeats)
                    system, ae = adapter.build(modulus)
                    ratios = (100*adapter.ratios(output, target, system, ae)).tolist()
                    label = 'NO' if method == 'NO' else f'{method}_K{cycles}'
                    if args.anchor:
                        label = f'anchored_K{cycles}'
                    methods[label] = dict(zip(METRICS, ratios), total_ms=elapsed)
                    if not np.isfinite(ratios+[elapsed]).all():
                        raise RuntimeError(f'Nonfinite evaluation: {split} {i} {label}')
                row = dict(split=split, index=i, contrast=float(data['contrast'][i]), methods=methods)
                if i < 2:
                    row['K1_phase_ms'] = phase_times(model, no, stats, adapter, inputs, modulus, args)
                rows.append(row)
                print(json.dumps(dict(split=split, index=i, methods={k:[round(v[q], 5) for q in METRICS[:3]+['total_ms']] for k, v in methods.items()})), flush=True)
    tables = {}
    for split in ('test', 'ood'):
        selected = [r for r in rows if r['split'] == split]
        tables[split] = {name:{key:summary([r['methods'][name][key] for r in selected])
                        for key in METRICS+['total_ms']} for name in selected[0]['methods']}
    if args.only_numerics:
        suffix = '_cached_fem'
    stem = f'elasticity{suffix}_n{adapter.n}'
    result = dict(dataset=str(args.data), training_dataset=str(args.training_data or args.data),
        seed=args.seed, rows=rows, summary=tables, graph_plan_ms_once=adapter.plan_ms_once,
        spectral_bound=adapter.upper, mode_shaping_omega=adapter.omega,
        pre=args.pre, post=args.post, graph_rank=8, regional_columns=72, reference_anchor=args.anchor,
        hardware=torch.cuda.get_device_name() if device.type == 'cuda' else 'CPU',
        repeats=args.repeats, timing='batch1 synchronized wallclock; fresh NO, material assembly and response setup every call; one warmup, repeat median; data IO/metrics/mesh plan excluded; numerical baseline reuses fixed MeshTri/Basis, includes material assembly and solver setup; inputs preloaded on GPU',
        note='K1-trained graph weights and response factor are reused for K2; numerical AMG receives 2 translations + 1 rotation; linear P1 algebraic references, not a continuum convergence claim')
    save_json(args.out/(stem+'.json'), result)
    fields = ['split', 'method']+METRICS+['total_ms']
    with (args.out/(stem+'.csv')).open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for split, methods in tables.items():
            for name, values in methods.items():
                writer.writerow(dict(split=split, method=name, **{k:v['mean'] for k,v in values.items()}))
    print(json.dumps(dict(result=str(args.out/(stem+'.json')), summary=tables), indent=2), flush=True)


def check(args, device):
    adapter = ElasticAdapter(17, device)
    rng = np.random.default_rng(args.seed)
    e, _, _ = physics.sample_material(adapter.g, rng)
    e = e.astype(np.float32)
    system, ae = adapter.build(e)
    matrix = physics.mature_stiffness(adapter.g, e.astype(np.float64))
    free = adapter.free_cpu
    a = matrix[free][:, free].toarray()
    b = -(matrix@adapter.g['affine'])[free]
    gpu = adapter.matrix.to_dense().cpu().numpy()
    scaled = a/np.sqrt(np.diag(a)[:, None]*np.diag(a)[None, :])
    lam = float(np.linalg.eigvalsh(scaled).max())
    matrix_error = np.linalg.norm(a-gpu)/np.linalg.norm(a)
    rhs_error = np.linalg.norm(b-system.rhs.cpu().numpy())/np.linalg.norm(b)
    target = torch.as_tensor(physics.solve(adapter.g, matrix), device=device, dtype=torch.float32)
    x = (target-adapter.affine)[adapter.free]
    model = response_model(device)
    zero = torch.zeros_like(x)
    prepared = prepare(model, adapter, system, ae, zero)
    out = apply_modes(prepared, system, ae, zero, adapter.context)
    before = ((zero-x)*adapter.action(0, zero-x, ae)).sum()
    after = ((out-x)*adapter.action(0, out-x, ae)).sum()
    energy_ratio = float(after/before)
    loss = adapter.ratios(out, target, system, ae)[:3].square().sum()
    loss.backward()
    gradients_finite = all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
    result = dict(matrix_relative=matrix_error, rhs_relative=rhs_error,
        actual_normalized_lambda_max=lam, conservative_upper=adapter.upper,
        projection_energy_ratio=energy_ratio, gradients_finite=bool(gradients_finite),
        feature_width=adapter.features(system, ae, zero)[0].shape[1],
        vector_dofs=len(x), parameters=sum(p.numel() for p in model.parameters()))
    if matrix_error > 2e-6 or rhs_error > 2e-6 or lam > adapter.upper or energy_ratio > 1.00001 or not gradients_finite:
        raise RuntimeError(f'Elasticity smoke check failed: {result}')
    save_json(args.out/'elasticity_smoke.json', result)
    print(json.dumps(result, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['check', 'train-no', 'train-response', 'evaluate'])
    parser.add_argument('--data', type=Path, default=Path('baselines/data/elasticity_v1/generated/n65_s20261028'))
    parser.add_argument('--training-data', type=Path)
    parser.add_argument('--out', type=Path, default=Path('CDMC/experiments/multiscale_elliptic_v1/results/elasticity_20261028'))
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--no-epochs', type=int, default=120)
    parser.add_argument('--response-epochs', type=int, default=30)
    parser.add_argument('--pre', type=int, default=64)
    parser.add_argument('--post', type=int, default=8)
    parser.add_argument('--count', type=int, default=8)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--seed', type=int, default=20261028)
    parser.add_argument('--no-numerics', action='store_true')
    parser.add_argument('--anchor', action='store_true', help='Learn a deformation of the fixed reference space, not a replacement')
    parser.add_argument('--only-numerics', action='store_true', help='Retime only numerical branches after fixing cached FEM geometry')
    args = parser.parse_args()
    if args.anchor and args.only_numerics:
        parser.error('--anchor and --only-numerics are distinct evaluation branches')
    args.out.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.set_num_threads(2)
    {'check':check, 'train-no':train_no, 'train-response':train_response, 'evaluate':evaluate}[args.stage](args, device)


if __name__ == '__main__':
    main()
