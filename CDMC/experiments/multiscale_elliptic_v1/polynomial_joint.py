"""Fixed-work physics filtering and learned complementary response.

The candidate removes *all* material Galerkin construction/coarse solves.
It retains exact fine P1 actions, a shared mesh-only GNN hierarchy and a small
energy calibration. A standard Chebyshev filter is a numerical baseline, not
a claimed new algorithm. Only training/validation select the configuration.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from run import (DarcySplit, compiled_plan, make_example, load_no, tensors,
                 predict, measure, clock, summary, save_json, field_metrics,
                 corrected, CDMCSetup)
from coupled_coarse import (ModeOnlyCompiled, CoarseContext, CoupledCoarseSetup,
                            prepare_modes, apply_modes, mm, harmonize_modes, factor_modes)
from solver.numerics import TorchSystem, TorchLevel
from solver.matrixfree_cdmc import correct_matrixfree
from adapters.darcy_system import compile_gpu_assembly, compiled_gpu_assembly


class FineSparseAction:
    def __init__(self, original):
        self.original = original
        self.prolongations = original.prolongations
        self.restrictions = original.restrictions
        self.level_sizes = original.level_sizes
        self.fine = self
        self.matrix = None

    def action(self, level, x, ae):
        if level != 0:
            raise ValueError('No material coarse operators in this candidate')
        return mm(self.matrix, x[:, None])[:, 0]

    def homogeneous_action(self, x, ae):
        return mm(self.matrix, x.T).T


class SparseCompiled:
    def __init__(self, mesh, device, shared=None):
        self.device = device
        base = ModeOnlyCompiled(mesh, device, shared=shared)
        self.plan, self.points_free = base.plan, base.points_free
        self.mesh_plan_ms_once = base.mesh_plan_ms_once
        self.action = FineSparseAction(base.action)
        started = clock(device)
        self.assembly = compile_gpu_assembly(mesh, self.plan.geometry,
            self.plan.free, self.plan.reference_levels[0], device)
        self.assembly_plan_ms_once = 1000*(clock(device)-started)

    def build(self, coefficient, lift):
        matrix, rhs, feature, ae = compiled_gpu_assembly(self.assembly, coefficient, lift, self.device)
        self.action.matrix = matrix
        diagonal = matrix.values()[self.assembly.fine_diagonal_position]
        system = TorchSystem([TorchLevel(torch.empty(0, device=self.device), diagonal.reciprocal())],
            rhs, self.plan.free, self.plan.graph, feature, self.plan.mass, None)
        return system, ae[None], matrix


class LiftedSparseAction:
    """Unchanged exact factorized Galerkin algebra, with a faster fine CSR matvec."""
    def __init__(self,base,matrix):
        self.base,self.matrix=base,matrix

    def action(self,level,x,ae):
        for depth in reversed(range(level)):
            x=mm(self.base.prolongations[depth],x[:,None])[:,0]
        x=mm(self.matrix,x[:,None])[:,0]
        for depth in range(level):
            x=mm(self.base.restrictions[depth],x[:,None])[:,0]
        return x


def chebyshev(system, action, ae, initial, steps, lower=.001, upper=2.):
    """Standard fixed-degree Chebyshev semi-iteration on D^-1 A.

    For scalar positive-coefficient P1 diffusion on this acute/right mesh,
    the reduced matrix is an M-matrix and the normalized spectrum is <=2.
    lower is a filter cutoff, NOT an assertion about the smallest eigenvalue.
    The polynomial coefficients depend only on the declared interval.
    """
    if steps == 0:
        return initial
    center, radius = (upper+lower)/2, (upper-lower)/2
    invd = system.levels[0].diagonal_inverse
    residual = system.rhs-action.action(0, initial, ae)
    direction = invd*residual/center
    x = initial+direction
    rho = radius/center
    for _ in range(1, steps):
        next_rho = 1/(2*center/radius-rho)
        residual = system.rhs-action.action(0, x, ae)
        direction = (2*next_rho/radius)*(invd*residual)+(next_rho*rho)*direction
        x = x+direction
        rho = next_rho
    return x


class P1Norms:
    def __init__(self, mesh, device):
        from baselines.data.darcy_v2.generate import element_geometry
        area, gradient, _ = element_geometry(mesh.points, mesh.cells)
        self.cells = torch.as_tensor(mesh.cells, dtype=torch.long, device=device)
        self.gradient = torch.as_tensor(gradient, dtype=torch.float32, device=device)
        self.area = torch.as_tensor(area, dtype=torch.float32, device=device)
        self.nodes = len(mesh.points)

    def grad(self, full):
        return torch.einsum('tic,ti->tc', self.gradient, full[self.cells])

    def ratios(self, x, target, system, ae):
        error = torch.zeros_like(target)
        error[system.free] = x-target[system.free]
        dg, tg = self.grad(error), self.grad(target)
        value = (system.mass*error[system.free].square()).sum()/(system.mass*target[system.free].square()).sum().clamp_min(1e-20)
        gradient = (self.area[:, None]*dg.square()).sum()/(self.area[:, None]*tg.square()).sum().clamp_min(1e-20)
        flux_weight = (self.area*ae.flatten().square())[:, None]
        flux = (flux_weight*dg.square()).sum()/(flux_weight*tg.square()).sum().clamp_min(1e-20)
        return torch.stack((value, gradient, flux)).clamp_min(1e-20).sqrt()


def partition_weights(points, grid):
    """Partition of unity for a controlled *space* enrichment, not data tiling."""
    if grid == 1:
        return torch.ones((len(points),1),device=points.device)
    centers=torch.linspace(0,1,grid,device=points.device)
    hats=(1-(points[:, :, None]-centers).abs()*(grid-1)).clamp_min(0)
    return (hats[:,0,:,None]*hats[:,1,None,:]).flatten(1)


def shape_response(modes, context, system, ae, args):
    grid=getattr(args,'partition_grid',1)
    if grid>1:
        weights=partition_weights(context.compiled.points_free,grid)
        modes=(modes[:,:,None]*weights[:,None,:]).flatten(1)
    modes=harmonize_modes(modes,context,system,ae,args.harmonic_steps)
    return factor_modes(modes,context,ae)


def prepare_response(model, features, context, system, ae, args):
    return shape_response(model(features,context),context,system,ae,args)


def candidate(model, context, system, ae, initial, args, prepared=None, cycles=1):
    action = context.compiled.action
    filtered = chebyshev(system, action, ae, initial, args.pre, args.lower)
    if prepared is None:
        features = context.features(system, ae, filtered)
        prepared = prepare_response(model, features, context, system, ae, args)
    for cycle in range(cycles):
        if cycle:
            filtered = chebyshev(system, action, ae, x, args.pre, args.lower)
        x = apply_modes(prepared, system, ae, filtered, context)
        x = chebyshev(system, action, ae, x, args.post, args.post_lower)
    return x


def save_table(path, aggregated):
    fields = ['method','L2_percent','gradient_percent','flux_L2_percent','energy_percent','total_ms']
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for name, result in aggregated.items():
            writer.writerow({'method':name, **{k:result[k]['mean'] for k in fields[1:]}})


def screening(args, device):
    split = DarcySplit(args.data, 'val')
    compiled = SparseCompiled(split.mesh, device)
    with np.load(args.no_dir/'predictions.npz') as z:
        predictions = z['val'].copy()
    rows = []
    with torch.no_grad():
        for i in range(args.count):
            system, ae, matrix = compiled.build(split.coefficient[i], split.lift[i])
            initial = torch.as_tensor(predictions[i], device=device)[system.free]
            methods = {}
            for steps in (16, 32, 64, 128):
                output, elapsed = measure(lambda: chebyshev(system, compiled.action, ae, initial, steps), device, args.repeats)
                full = torch.zeros_like(torch.as_tensor(predictions[i], device=device))
                full[system.free] = output
                methods[str(steps)] = dict(field_metrics(full.cpu().numpy(), split.target[i], split.coefficient[i], split.mesh), apply_ms=elapsed)
            rows.append(dict(index=i, methods=methods))
            print({'screen':i, 'methods':{k:{q:v[q] for q in ('L2_percent','gradient_percent','flux_L2_percent','apply_ms')} for k,v in methods.items()}}, flush=True)
    save_json(args.out/'screening.json', dict(split='val',samples=rows,
        config='Fixed degrees 16/32/64/128, normalized cutoff .001, upper 2. No test-set selection.'))


def train(args, device):
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    train_split, val_split = DarcySplit(args.data,'train'), DarcySplit(args.data,'val')
    compiled = SparseCompiled(train_split.mesh, device)
    context = CoarseContext(compiled, start_level=1)
    norms = P1Norms(train_split.mesh, device)
    with np.load(args.no_dir/'predictions.npz') as z:
        predictions = {s:z[s].copy() for s in ('train','val')}
    def cache(split, label, count):
        result=[]
        with torch.no_grad():
            for i in range(min(count,len(split))):
                system, ae, matrix = compiled.build(split.coefficient[i], split.lift[i])
                target = torch.as_tensor(split.target[i],device=device)
                x0=torch.as_tensor(predictions[label][i],device=device)[system.free]
                x=chebyshev(system,compiled.action,ae,x0,args.pre,args.lower)
                result.append((system,ae,matrix,x,target,context.features(system,ae,x)))
        return result
    started=clock(device)
    tr,va=cache(train_split,'train',args.train_count),cache(val_split,'val',args.val_count)
    # Strong non-neural control, fit ONLY to training defects. No validation/test labels.
    with torch.no_grad():
        errors=torch.stack([item[4][item[0].free]-item[3] for item in tr],1)
        mass_root=compiled.plan.mass.sqrt()
        u,_,_=torch.linalg.svd((mass_root[:,None]*errors).double(),full_matrices=False)
        pod=(u[:,:8].float()/mass_root[:,None])
        torch.save(dict(basis=pod.cpu(),training_indices=list(range(len(tr))),
                        definition='Top-eight mass-weighted POD directions of training FNO errors after fixed prefilter; no test labels.'),args.out/'pod.pt')
    model=CoupledCoarseSetup(width=16,rank=8,start_level=1).to(device)
    if args.initialize:
        old=torch.load(args.initialize,map_location=device,weights_only=False)
        model.load_state_dict(old['state'])
    opt=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=1e-5)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,args.epochs,eta_min=1e-5)
    best=float('inf'); history=[]
    def loss_for(item):
        system,ae,matrix,x,target,features=item
        compiled.action.matrix=matrix
        prepared=prepare_response(model,features,context,system,ae,args)
        output=apply_modes(prepared,system,ae,x,context)
        output=chebyshev(system,compiled.action,ae,output,args.post,args.post_lower)
        return norms.ratios(output,target,system,ae)
    options=dict(pre=args.pre,post=args.post,lower=args.lower,post_lower=args.post_lower,
                 harmonic_steps=args.harmonic_steps,partition_grid=args.partition_grid)
    for epoch in range(args.epochs):
        tick=clock(device); model.train(); train_values=[]
        for i in rng.permutation(len(tr)):
            ratios=loss_for(tr[i])
            loss=ratios.square().mean().sqrt()
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),2.)
            opt.step(); train_values.append(ratios.detach())
        scheduler.step(); model.eval()
        with torch.no_grad():
            vals=torch.stack([loss_for(item) for item in va])
        score=float(vals.square().mean().sqrt())
        if not np.isfinite(score):
            raise RuntimeError('Nonfinite training; do not hide failed runs')
        row=dict(epoch=epoch+1,train_percent=(100*torch.stack(train_values).mean(0)).tolist(),
                 val_percent=(100*vals.mean(0)).tolist(),seconds=clock(device)-tick)
        history.append(row); print(row,flush=True)
        if score<best:
            best=score
            torch.save(dict(state=model.state_dict(),options=options,
                model_options=dict(width=16,rank=8,start_level=1),seed=args.seed,
                epoch=epoch+1,validation=row,training_samples=len(tr),no_dir=str(args.no_dir)),args.out/'model.pt')
        save_json(args.out/'training.json',dict(history=history,best=best,
            total_seconds=clock(device)-started,options=options,
            train_count=len(tr),val_count=len(va),seed=args.seed,
            mesh_plan_ms_once=compiled.mesh_plan_ms_once,
            assembly_plan_ms_once=compiled.assembly_plan_ms_once,
            objective='RMS of normalized P1 field, gradient, and flux errors; current-A projection; fixed pre/post filter; frozen FNO.'))


def evaluate(args,device):
    saved=torch.load(args.checkpoint,map_location=device,weights_only=False)
    for key,value in saved['options'].items():
        setattr(args,key,value)
    split=DarcySplit(args.data,args.split)
    base=compiled_plan(split,device,SimpleNamespace(coarse_size=128))
    compiled=SparseCompiled(split.mesh,device,shared=base)
    context=CoarseContext(compiled,1)
    model=CoupledCoarseSetup(**saved['model_options']).to(device).eval()
    model.load_state_dict(saved['state'])
    no,stats,_=load_no(args.no_dir,device)
    a,y,xy,_,mask=tensors(split,device)
    old=None
    if args.old_checkpoint:
        old_saved=torch.load(args.old_checkpoint,map_location=device,weights_only=False)
        old=CoupledCoarseSetup(**old_saved['options']).to(device).eval()
        old.load_state_dict(old_saved['state'])
    old_joint=None
    if args.old_joint_checkpoint:
        joint_saved=torch.load(args.old_joint_checkpoint,map_location=device,weights_only=False)
        old_joint=CoupledCoarseSetup(**joint_saved['options']).to(device).eval()
        old_joint.load_state_dict(joint_saved['state'])
        joint_context=CoarseContext(base,joint_saved['options']['start_level'])
    old_gate=None
    if args.old_cdmc_checkpoint:
        gate_saved=torch.load(args.old_cdmc_checkpoint,map_location=device,weights_only=False)
        old_gate=CDMCSetup(gate_saved['width']).to(device).eval()
        old_gate.load_state_dict(gate_saved['state'])
    global_model=None
    if args.global_checkpoint:
        global_saved=torch.load(args.global_checkpoint,map_location=device,weights_only=False)
        global_model=CoupledCoarseSetup(**global_saved['model_options']).to(device).eval()
        global_model.load_state_dict(global_saved['state'])
    pod=None
    if args.pod_checkpoint:
        pod=torch.load(args.pod_checkpoint,map_location=device,weights_only=False)['basis'].to(device)
        pod=context.lift(mm(context.restrict[0],pod))
    pairs=sorted(((i,j) for i in range(1,6) for j in range(1,6)),key=lambda t:(t[0]**2+t[1]**2,t))[:8]
    geometric=torch.stack([torch.sin(np.pi*i*compiled.points_free[:,0])*torch.sin(np.pi*j*compiled.points_free[:,1]) for i,j in pairs],1)
    geometric=context.lift(mm(context.restrict[0],geometric))
    rows=[]
    with torch.no_grad():
        for i in range(args.offset,min(args.offset+args.count,len(split))):
            names=['FNO','MG_K1','MG_K2','MG_K4','MG_CSR_K2','MG_CSR_K4',
                   'MG_CSR_zero_K2','MG_CSR_zero_K4','MG_CSR_zero_K8',f'poly{args.pre+args.post}',
                   f'poly{2*(args.pre+args.post)}',f'poly{4*(args.pre+args.post)}',
                   'poly_split','learned_replacement_K1','learned_replacement_K2',
                   'learned_replacement_K4']
            if old is not None:
                names.append('old_modes_no_MG')
            if old_joint is not None:
                names.append('old_MG_joint_K2')
            if old_gate is not None:
                names.append('old_CDMC_K2')
            if global_model is not None:
                names.extend(['global_response_K1','global_response_K2','global_response_K4'])
            names.extend(['geometric_response_K1','geometric_response_K2','geometric_response_K4'])
            if pod is not None:
                names.extend(['POD_response_K1','POD_response_K2','POD_response_K4'])
            if args.methods:
                if set(args.methods)-set(names):
                    raise ValueError('Requested method is unavailable')
                names=args.methods
            def online(name):
                # A pure numerical control must not pay for, or use, FNO.
                full=(torch.zeros_like(y[i]).flatten() if '_zero_' in name else
                      predict(no,a[i:i+1],xy,mask,stats).flatten())
                if name=='FNO': return full
                if name.startswith('MG_') or name.startswith('old_MG_joint') or name=='old_CDMC_K2':
                    sys,ae=base.build(a[i].flatten(),torch.zeros_like(full))
                    if name=='old_CDMC_K2':
                        x=corrected(old_gate,base,(sys,ae,full[sys.free],y[i].flatten()),2)[-1]
                    else:
                        action=base.action
                        if name.startswith('MG_CSR'):
                            _,_,matrix=compiled.build(a[i].flatten(),torch.zeros_like(full))
                            action=LiftedSparseAction(base.action,matrix)
                        gate=joint_saved['prefix_gate'] if name.startswith('old_MG_joint') else 1.
                        x=correct_matrixfree(sys,action,ae,full[sys.free],
                            torch.full_like(full[sys.free],gate),torch.ones_like(full[sys.free]),
                            None,cycles=int(name[-1]),coarse_steps=0)
                        if name.startswith('old_MG_joint'):
                            features=joint_context.features(sys,ae,x)
                            prepared=prepare_modes(old_joint,features,joint_context,sys,ae,joint_saved['harmonic_steps'])
                            x=apply_modes(prepared,sys,ae,x,joint_context)
                else:
                    sys,ae,matrix=compiled.build(a[i].flatten(),torch.zeros_like(full))
                    initial=full[sys.free]
                    if name.startswith('learned_replacement'):
                        x=candidate(model,context,sys,ae,initial,args,cycles=int(name[-1]))
                    elif name.startswith('global_response'):
                        global_args=SimpleNamespace(**dict(vars(args),partition_grid=1))
                        x=candidate(global_model,context,sys,ae,initial,global_args,cycles=int(name[-1]))
                    elif name.startswith(('POD_response','geometric_response')):
                        basis=pod if name.startswith('POD') else geometric
                        prepared=shape_response(basis,context,sys,ae,args)
                        x=candidate(model,context,sys,ae,initial,args,prepared=prepared,cycles=int(name[-1]))
                    elif name=='old_modes_no_MG':
                        features=context.features(sys,ae,initial)
                        prepared=prepare_modes(old,features,context,sys,ae,old_saved['harmonic_steps'])
                        x=apply_modes(prepared,sys,ae,initial,context)
                    elif name=='poly_split':
                        x=chebyshev(sys,compiled.action,ae,initial,args.pre,args.lower)
                        x=chebyshev(sys,compiled.action,ae,x,args.post,args.post_lower)
                    else:
                        x=chebyshev(sys,compiled.action,ae,initial,int(name[4:]),args.lower)
                result=torch.zeros_like(full); result[sys.free]=x
                return result
            methods={}
            for name in names:
                output,elapsed=measure(lambda: online(name),device,args.repeats)
                methods[name]=dict(field_metrics(output.cpu().numpy(),split.target[i],split.coefficient[i],split.mesh),total_ms=elapsed)
            # Independently timed phases start from the same initial state.
            full,t_no=measure(lambda: predict(no,a[i:i+1],xy,mask,stats).flatten(),device,args.repeats)
            (sys,ae,matrix),t_build=measure(lambda: compiled.build(a[i].flatten(),torch.zeros_like(full)),device,args.repeats)
            filtered,t_pre=measure(lambda: chebyshev(sys,compiled.action,ae,full[sys.free],args.pre,args.lower),device,args.repeats)
            features,t_features=measure(lambda: context.features(sys,ae,filtered),device,args.repeats)
            prepared,t_setup=measure(lambda: prepare_response(model,features,context,sys,ae,args),device,args.repeats)
            x,t_apply=measure(lambda: apply_modes(prepared,sys,ae,filtered,context),device,args.repeats)
            _,t_post=measure(lambda: chebyshev(sys,compiled.action,ae,x,args.post,args.post_lower),device,args.repeats)
            rows.append(dict(index=i,methods=methods,profile=dict(FNO=t_no,fine_build=t_build,
                prefilter=t_pre,features=t_features,neural_modes_and_calibration=t_setup,
                mode_apply=t_apply,postfilter=t_post)))
            aggregated={name:{key:summary([r['methods'][name][key] for r in rows])
                         for key in methods[name]} for name in names}
            save_json(args.out/'evaluation.json',dict(summary=aggregated,samples=rows,
                options=saved['options'],checkpoint=str(args.checkpoint),test_count=len(rows),offset=args.offset,split=args.split,
                device=torch.cuda.get_device_name(device),mesh_plan_ms_once=base.mesh_plan_ms_once,
                fine_assembly_plan_ms_once=compiled.assembly_plan_ms_once,
                timing='Batch 1 synchronized wall clock; fresh FNO (omitted for zero-initialized numerical controls), material assembly, setup and correction per call; mesh-only plans separate. Independent stage timings do not define E2E.',
                model_parameters=sum(p.numel() for p in model.parameters())))
            save_table(args.out/'summary.csv',aggregated)
            print({'test':i,'methods':{name:{q:v[q] for q in ('L2_percent','gradient_percent','flux_L2_percent','total_ms')} for name,v in methods.items()}},flush=True)


def space_diagnosis(args,device):
    saved=torch.load(args.checkpoint,map_location=device,weights_only=False)
    for key,value in saved['options'].items():
        setattr(args,key,value)
    split=DarcySplit(args.data,'val')
    compiled=SparseCompiled(split.mesh,device)
    context=CoarseContext(compiled,1)
    model=CoupledCoarseSetup(**saved['model_options']).to(device).eval()
    model.load_state_dict(saved['state'])
    with np.load(args.no_dir/'predictions.npz') as z:
        predictions=z['val'].copy()
    rows=[]
    with torch.no_grad():
        for i in range(args.count):
            system,ae,matrix=compiled.build(split.coefficient[i],split.lift[i])
            initial=torch.as_tensor(predictions[i],device=device)[system.free]
            target=torch.as_tensor(split.target[i],device=device)
            filtered=chebyshev(system,compiled.action,ae,initial,args.pre,args.lower)
            features=context.features(system,ae,filtered)
            methods={}
            for grid in (1,2,3):
                args.partition_grid=grid
                prepared,setup_ms=measure(lambda: prepare_response(model,features,context,system,ae,args),device,args.repeats)
                output=apply_modes(prepared,system,ae,filtered,context)
                output=chebyshev(system,compiled.action,ae,output,args.post,args.post_lower)
                full=torch.zeros_like(target);full[system.free]=output
                metrics=field_metrics(full.cpu().numpy(),split.target[i],split.coefficient[i],split.mesh)
                z=prepared[0].double();m=system.mass.double()
                gram=z.T@(m[:,None]*z)
                values,vectors=torch.linalg.eigh(.5*(gram+gram.T))
                keep=values>values.max()*1e-10
                rhs=z.T@(m*(target[system.free]-filtered).double())
                coeff=vectors[:,keep]@((vectors[:,keep].T@rhs)/values[keep])
                remainder=target[system.free].double()-filtered.double()-z@coeff
                metrics.update(setup_ms=setup_ms,dimension=z.shape[1],effective_rank=int(keep.sum()),
                    oracle_field_before_post_percent=float(100*((m*remainder.square()).sum()/(m*target[system.free].double().square()).sum()).sqrt()))
                methods[f'partition_grid_{grid}']=metrics
            rows.append(dict(index=i,methods=methods))
            save_json(args.out/'space_diagnosis.json',dict(samples=rows,split='val',
                checkpoint=str(args.checkpoint),purpose='Frozen network; only expand response space using a partition of unity. Oracle coefficients are offline diagnostics, not deployed. No new training.'))
            print(rows[-1],flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=['screen','train','evaluate','space'])
    for name in ('data','no-dir','out'):
        p.add_argument('--'+name,required=True,type=Path)
    for name in ('checkpoint','old-checkpoint','initialize','global-checkpoint','pod-checkpoint',
                 'old-joint-checkpoint','old-cdmc-checkpoint'):
        p.add_argument('--'+name,type=Path)
    p.add_argument('--device',default='cuda')
    p.add_argument('--count',type=int,default=4)
    p.add_argument('--offset',type=int,default=0)
    p.add_argument('--split',choices=['val','test'],default='test')
    p.add_argument('--methods',nargs='+')
    p.add_argument('--repeats',type=int,default=3)
    p.add_argument('--pre',type=int,default=64)
    p.add_argument('--post',type=int,default=8)
    p.add_argument('--lower',type=float,default=.001)
    p.add_argument('--post-lower',type=float,default=.05)
    p.add_argument('--harmonic-steps',type=int,default=2)
    p.add_argument('--partition-grid',type=int,default=1)
    p.add_argument('--epochs',type=int,default=20)
    p.add_argument('--train-count',type=int,default=48)
    p.add_argument('--val-count',type=int,default=12)
    p.add_argument('--lr',type=float,default=.001)
    p.add_argument('--seed',type=int,default=20261027)
    args=p.parse_args()
    if args.out.exists():
        p.error('Choose a fresh output directory; preserve every completed run.')
    args.out.mkdir(parents=True)
    torch.set_num_threads(2)
    device=torch.device(args.device)
    {'screen':screening,'train':train,'evaluate':evaluate,'space':space_diagnosis}[args.stage](args,device)


if __name__=='__main__':
    main()
