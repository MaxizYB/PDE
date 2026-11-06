"""Bounded graph transport of a pretrained reference frame.

Z = [I + sum_j diag(g_j) G^j] Q, |g_j| <= rho/(p+1).
G is symmetric normalized mesh adjacency, ||G||_2 <= 1. Consequently the
transport has singular values in [1-rho, 1+rho]; this avoids a collapsed
dictionary without an online QR. The reference Q/GQ/... are mesh-only caches.
This prototype tests a mechanism; it is not an originality claim.
"""
from __future__ import annotations
import argparse
import csv
import json
import sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from stable_space import (torch, DarcySplit, tensors, SparseCompiled, CoarseContext,
                          load_no, predict, smooth, clock, measure, field_metrics,
                          summary, save_json, trunk_spaces, exact_prepare, exact_cycles,
                          load_dictionary, prepare, cycles)
from coupled_coarse import CoupledCoarseSetup
from polynomial_joint import P1Norms
from solver.numerics import sparse_tensor


def mesh_bank(compiled, q, power):
    graph = abs(compiled.plan.reference_levels[0]).tocsr()
    graph.setdiag(0)
    graph.eliminate_zeros()
    degree = np.asarray(graph.sum(1)).ravel()
    inv = 1 / np.sqrt(np.maximum(degree, 1e-30))
    graph = graph.multiply(inv[:, None]).multiply(inv[None, :]).tocsr()
    g = sparse_tensor(graph, compiled.device)
    bank = [q]
    for _ in range(power):
        bank.append(torch.sparse.mm(g, bank[-1]))
    return torch.stack(bank, 0).detach()


class TransportFrame(torch.nn.Module):
    def __init__(self, width=24, power=2, rho=.5):
        super().__init__()
        if not 0 < rho < 1:
            raise ValueError("rho must preserve an invertible near-identity transport")
        self.power, self.rho = power, rho
        self.graph = CoupledCoarseSetup(width=width, rank=power+1, start_level=1)
        torch.nn.init.zeros_(self.graph.output.weight)
        torch.nn.init.zeros_(self.graph.output.bias)

    def forward(self, features, context, bank):
        g = torch.tanh(self.graph(features, context)) * (self.rho/(self.power+1))
        return bank[0] + (g.T[:, :, None] * bank).sum(0)


class NormalFrame(torch.nn.Module):
    """Change the span rather than its coordinates, at fixed output rank.

U=(I-QQ^T)H_theta is orthogonal to the reference. Z=Q+UC therefore
has Z^T Z=I+C^T U^T U C >= I. Frobenius bounds on U and C bound
the largest eigenvalue too. Small-rank graph outputs create new directions
without generating 80/128 mutually redundant columns from scratch.
"""
    def __init__(self, width=24, power=2, rho=2., normal_rank=8):
        super().__init__()
        self.rho = rho
        self.graph = CoupledCoarseSetup(width=width,rank=normal_rank,start_level=1)
        self.coordinate = torch.nn.Parameter(torch.zeros(normal_rank,80))

    def forward(self,features,context,bank):
        q = bank[0].double()
        u = torch.tanh(self.graph(features,context)).double()
        u = u - q @ (q.T @ u)
        u = u / u.norm().clamp_min(1e-10)
        c = self.coordinate.double()
        c = self.rho * c / (1+c.norm())
        return q + u @ c


def new_model(kind, options):
    return (NormalFrame if kind.startswith("normal") else TransportFrame)(**options)


def geometry_features(features):
    result=[]
    for values in features:
        values=values.clone()
        values[:,[0,1,5,6]]=0
        result.append(values)
    return result


def setup(args):
    device = torch.device("cuda:0")
    compiled = SparseCompiled(DarcySplit(args.data, "val").mesh, device)
    context = CoarseContext(compiled, 1)
    no, stats, _ = load_no(args.no_dir, device)
    _, saved = load_dictionary(args.dictionary, device)
    opts = SimpleNamespace(**saved["options"])
    opts.harmonic_steps = 0
    if args.pre is not None:
        opts.pre=args.pre
    if args.cutoff is not None:
        opts.lower=args.cutoff
    split = DarcySplit(args.data, "val")
    _, _, xy, _, mask = tensors(split, device)
    begin = clock(device)
    bases, info = trunk_spaces(args.trunk_model, xy.reshape(-1,2), compiled.plan.free, device)
    bank = mesh_bank(compiled, bases["TB80"], args.power)
    cache_ms = 1000*(clock(device)-begin)
    return device, compiled, context, no, stats, opts, bank, info, cache_ms


def train(args):
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device, compiled, context, no, stats, opts, bank, info, cache_ms = setup(args)
    model_opts = dict(width=args.width, power=args.power, rho=args.rho)
    if args.model_kind.startswith("normal"):
        model_opts["normal_rank"] = args.normal_rank
    model = new_model(args.model_kind,model_opts).to(device)
    if args.initialize:
        previous=torch.load(args.initialize,map_location=device,weights_only=False)
        if previous.get("model_kind") != args.model_kind or previous["model_options"] != model_opts:
            raise ValueError("Initialization must have the identical architecture")
        model.load_state_dict(previous["state"])
    norms = P1Norms(DarcySplit(args.data,"val").mesh, device)
    def cache(split_name, count):
        split = DarcySplit(args.data, split_name)
        a, _, xy, _, mask = tensors(split, device)
        items = []
        with torch.no_grad():
            for i in range(min(count,len(split))):
                full = predict(no,a[i:i+1],xy,mask,stats).flatten()
                system, ae, matrix = compiled.build(a[i].flatten(),torch.zeros_like(full))
                x = smooth(system,compiled.action,ae,full[system.free],opts.pre,opts.lower,opts)
                features=context.features(system,ae,x)
                if args.model_kind == "normal_static":
                    features=geometry_features(features)
                items.append(dict(system=system,ae=ae,matrix=matrix,initial=x,
                    features=features,
                    target=torch.as_tensor(split.target[i],device=device)))
        return items
    started = clock(device)
    tr, va = cache("train",args.train_count), cache("val",args.val_count)
    optimizer = torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,args.epochs,eta_min=args.lr/20)
    def objective(item):
        compiled.action.matrix = item["matrix"]
        z = model(item["features"],context,bank)
        prepared = exact_prepare(z,context,item["system"],item["ae"],opts)
        x, values = item["initial"], []
        for k in range(2):
            if k:
                x = smooth(item["system"],compiled.action,item["ae"],x,opts.pre,opts.lower,opts)
            x = exact_cycles(x,prepared,item["system"],item["ae"],context,opts,1)
            values.append(norms.ratios(x,item["target"],item["system"],item["ae"]))
        return torch.stack(values)
    history, best = [], float("inf")
    for epoch in range(args.epochs+1):
        tick = clock(device)
        train_values=[]
        if epoch:
            model.train()
            for idx in rng.permutation(len(tr)):
                value = objective(tr[idx])
                train_values.append(value.detach())
                selected = value[-1:] if args.target == "K2" else value
                loss = selected.clamp_min(1e-8).log().mean()
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
                optimizer.step()
            scheduler.step()
        model.eval()
        with torch.no_grad():
            values = torch.stack([objective(item) for item in va])
            selected = values[:, -1:] if args.target == "K2" else values
            score = float(selected.clamp_min(1e-8).log().mean())
        if not np.isfinite(score):
            raise RuntimeError("Nonfinite validation score")
        row = dict(epoch=epoch,score=score,val_percent=(100*values.mean(0)).tolist(),seconds=clock(device)-tick)
        if train_values:
            row["train_percent"]=(100*torch.stack(train_values).mean(0)).tolist()
        history.append(row)
        if score < best:
            best = score
            torch.save(dict(state=model.state_dict(),model_options=model_opts,model_kind=args.model_kind,options=vars(opts),
                epoch=epoch,validation=row,arguments={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
                trunk=info,reference="Frozen pretrained TB80; bounded material/NO-conditioned graph transport"),args.out/"model.pt")
        save_json(args.out/"training.json",dict(history=history,seconds=clock(device)-started,
            train_count=len(tr),val_count=len(va),seed=args.seed,cache_ms_once=cache_ms,
            parameters=sum(p.numel() for p in model.parameters())))
        print(json.dumps(row),flush=True)


def evaluate(args):
    device, compiled, context, no, stats, opts, bank, info, cache_ms = setup(args)
    saved = torch.load(args.checkpoint,map_location=device,weights_only=False)
    opts=SimpleNamespace(**saved["options"])
    if args.diagnostic_filter_override:
        if args.pre is not None:
            opts.pre=args.pre
        if args.cutoff is not None:
            opts.lower=args.cutoff
    model = new_model(saved.get("model_kind","transport"),saved["model_options"]).to(device)
    model.load_state_dict(saved["state"])
    model.eval()
    if saved["model_options"]["power"] != args.power:
        raise ValueError("Power does not match trained bank")
    old, old_saved = load_dictionary(args.dictionary,device)
    old.eval()
    old_opts = SimpleNamespace(**old_saved["options"])
    split = DarcySplit(args.data,args.split)
    a, _, xy, _, mask = tensors(split,device)
    static_z = None
    static_setup_ms = 0.
    if saved.get("model_kind") == "normal_static":
        with torch.no_grad():
            started=clock(device)
            system,ae,_=compiled.build(a[0].flatten(),torch.zeros_like(a[0].flatten()))
            # Only coordinate/mass features survive; no sample material, RHS,
            # prediction or truth is used to generate this fixed learned frame.
            features=geometry_features(context.features(system,ae,torch.zeros_like(system.rhs)))
            static_z=model(features,context,bank)
            static_setup_ms=1000*(clock(device)-started)
    methods = [(v,k) for v in args.variants for k in args.cycles]
    rows, stages = [], []
    with torch.no_grad():
        for i in range(min(args.count,len(split))):
            def solve(variant,k,profile=False):
                parts = {}
                last = clock(device) if profile else 0.
                def tick(name):
                    nonlocal last
                    if profile:
                        now=clock(device)
                        parts[name]=1000*(now-last)
                        last=now
                full = predict(no,a[i:i+1],xy,mask,stats).flatten()
                tick("FNO_ms")
                system, ae, _ = compiled.build(a[i].flatten(),torch.zeros_like(full))
                tick("fine_assembly_ms")
                cfg=SimpleNamespace(**vars(old_opts if variant.startswith("CDMC") or variant.startswith("TB") else opts))
                if variant.startswith("TB"):
                    cfg.harmonic_steps=0
                    if variant == "TB_same":
                        cfg=SimpleNamespace(**vars(opts))
                x = smooth(system,compiled.action,ae,full[system.free],cfg.pre,cfg.lower,cfg)
                tick("prefilter_ms")
                if variant == "frame":
                    z = static_z if static_z is not None else model(context.features(system,ae,x),context,bank)
                elif variant.startswith("TB"):
                    z = bank[0]
                else:
                    z = old(context.features(system,ae,x),context)
                tick("space_generation_ms")
                if variant.endswith("old"):
                    prepared = prepare(z,context,system,ae,cfg)
                    tick("space_factor_ms")
                    x = cycles(x,prepared,system,ae,context,cfg,k,filtered=True)
                else:
                    prepared = exact_prepare(z,context,system,ae,cfg)
                    tick("space_factor_ms")
                    x = exact_cycles(x,prepared,system,ae,context,cfg,k)
                tick("correction_ms")
                full = torch.zeros_like(full)
                full[system.free] = x
                return (full,parts) if profile else full
            values = {}
            for variant,k in methods:
                out,ms = measure(lambda:solve(variant,k),device,args.repeats)
                if not bool(torch.isfinite(out).all()):
                    raise RuntimeError(f"Nonfinite {variant} {i}")
                name=f"{variant}_K{k}"
                values[name]=dict(field_metrics(out.cpu().numpy(),split.target[i],split.coefficient[i],split.mesh),total_ms=ms)
                if i < 4:
                    _,parts=solve(variant,k,True)
                    stages.append(dict(index=i,method=name,**parts))
            rows.append(dict(index=i,sample_index=int(split.sample_index[i]),methods=values))
            agg={n:{key:summary([r["methods"][n][key] for r in rows]) for key in values[n]} for n in values}
            save_json(args.out/"evaluation.json",dict(summary=agg,samples=rows,stages=stages,count=len(rows),
                split=args.split,checkpoint=str(args.checkpoint),selected_epoch=saved["epoch"],
                arguments={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
                options=vars(opts),old_options=vars(old_opts),trunk=info,cache_ms_once=cache_ms,
                static_frame_setup_ms_once=static_setup_ms,model_kind=saved.get("model_kind","transport"),
                mesh_plan_ms_once=compiled.mesh_plan_ms_once,assembly_plan_ms_once=compiled.assembly_plan_ms_once,
                device=torch.cuda.get_device_name(device),timing="Batch1; fresh FNO+assembly+space+factor+all corrections; one warmup, synchronized wall median; fixed mesh/bank cached separately"))
            print(i,{n:round(v["L2_percent"],8) for n,v in values.items()},flush=True)
    keys=["L2_percent","gradient_percent","flux_L2_percent","total_ms"]
    with (args.out/"summary.csv").open("w") as f:
        w=csv.writer(f)
        w.writerow(["method",*keys])
        for n in agg:
            w.writerow([n,*[agg[n][key]["mean"] for key in keys]])


def audit(args):
    device,compiled,context,no,stats,opts,bank,info,_=setup(args)
    saved=torch.load(args.checkpoint,map_location=device,weights_only=False)
    model=new_model(saved.get("model_kind","transport"),saved["model_options"]).to(device)
    model.load_state_dict(saved["state"])
    model.eval()
    opts=SimpleNamespace(**saved["options"])
    split=DarcySplit(args.data,"val")
    a,_,xy,_,mask=tensors(split,device)
    rows=[]
    with torch.no_grad():
        for i in range(min(args.count,len(split))):
            initial=predict(no,a[i:i+1],xy,mask,stats).flatten()
            system,ae,_=compiled.build(a[i].flatten(),torch.zeros_like(initial))
            x=smooth(system,compiled.action,ae,initial[system.free],opts.pre,opts.lower,opts)
            q=bank[0].double()
            z=model(context.features(system,ae,x),context,bank).double()
            delta=z-q
            eu=torch.linalg.eigvalsh(z.T@z)
            singular=torch.linalg.svdvals(delta)
            matrix=compiled.action.matrix.double()
            def energy_condition(basis):
                g=basis.T@torch.sparse.mm(matrix,basis)
                eig=torch.linalg.eigvalsh((g+g.T)/2)
                return float(eig[-1]/eig[0])
            rows.append(dict(index=i,euclidean_gram_min=float(eu[0]),euclidean_gram_max=float(eu[-1]),
                euclidean_gram_condition=float(eu[-1]/eu[0]),
                reference_orthogonality_error=float((q.T@q-torch.eye(q.shape[1],device=device)).norm()),
                normal_component_error=float((q.T@delta).norm()),
                relative_frame_change=float(delta.norm()/q.norm()),
                update_rank=int((singular>1e-5*singular[0]).sum()),
                reference_energy_gram_condition=energy_condition(q),
                learned_energy_gram_condition=energy_condition(z)))
    save_json(args.out/"space_audit.json",dict(split="val",count=len(rows),checkpoint=str(args.checkpoint),samples=rows,
        summary={k:summary([v[k] for v in rows]) for k in rows[0] if k!="index"}))



if __name__ == "__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode",choices=["train","evaluate","audit"])
    for name in ("data","no-dir","dictionary","trunk-model","out"):
        p.add_argument("--"+name,type=Path,required=True)
    p.add_argument("--checkpoint",type=Path)
    p.add_argument("--initialize",type=Path)
    p.add_argument("--width",type=int,default=24)
    p.add_argument("--pre",type=int)
    p.add_argument("--cutoff",type=float)
    p.add_argument("--diagnostic-filter-override",action="store_true")
    p.add_argument("--power",type=int,default=2)
    p.add_argument("--rho",type=float,default=.5)
    p.add_argument("--model-kind",choices=["transport","normal","normal_static"],default="transport")
    p.add_argument("--normal-rank",type=int,default=8)
    p.add_argument("--target",choices=["K1K2","K2"],default="K1K2")
    p.add_argument("--seed",type=int,default=20261104)
    p.add_argument("--epochs",type=int,default=30)
    p.add_argument("--lr",type=float,default=.001)
    p.add_argument("--train-count",type=int,default=48)
    p.add_argument("--val-count",type=int,default=8)
    p.add_argument("--split",choices=["val","test"],default="test")
    p.add_argument("--count",type=int,default=32)
    p.add_argument("--cycles",type=int,nargs="+",default=[1,2,4])
    p.add_argument("--repeats",type=int,default=3)
    p.add_argument("--variants",nargs="+",choices=["frame","TB_old","TB_exact","TB_same","CDMC_old","CDMC_exact"],
                   default=["frame","TB_old","TB_exact","CDMC_old","CDMC_exact"])
    args=p.parse_args()
    args.out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(2)
    {"train":train,"evaluate":evaluate,"audit":audit}[args.mode](args)
