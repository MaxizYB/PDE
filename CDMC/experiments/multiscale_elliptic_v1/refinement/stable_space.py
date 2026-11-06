"""Audit exact learned-space projection without changing old solver paths.

FP64 current-A Gram and unregularized Cholesky isolate coordinate-ridge loss.
No full-grid QR, fallback, test labels, or retraining enter prediction.
"""
from __future__ import annotations
import argparse
import csv
import sys
from pathlib import Path
from types import SimpleNamespace

EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXP / "baselines/trunk_basis"))
from evaluate import (torch, DarcySplit, tensors, SparseCompiled, CoarseContext,
                      load_no, predict, load_dictionary, smooth, prepare, cycles,
                      clock, measure, field_metrics, summary, save_json, trunk_spaces)
from core import spectral_upper


def exact_prepare(modes, context, system, ae, opts):
    matrix = context.compiled.action.matrix.double()
    z = modes.double()
    inverse = system.levels[0].diagonal_inverse.double()[:, None]
    upper = spectral_upper(system, context.compiled.action) if opts.matrix_bound else 2.
    for _ in range(opts.harmonic_steps):
        z = z - (4 / (3 * upper)) * inverse * torch.sparse.mm(matrix, z)
    az = torch.sparse.mm(matrix, z)
    scales = (z * az).sum(0).sqrt()
    z, az = z / scales, az / scales
    gram = z.T @ az
    gram = (gram + gram.T) / 2
    return z, torch.linalg.cholesky(gram)


def exact_cycles(x, prepared, system, ae, context, opts, count):
    z, factor = prepared
    for k in range(count):
        if k:
            x = smooth(system, context.compiled.action, ae, x, opts.pre, opts.lower, opts)
        residual = system.rhs - context.compiled.action.action(0, x, ae)
        c = torch.cholesky_solve((z.T @ residual.double())[:, None], factor)
        x = x + (z @ c).squeeze(1).to(x.dtype)
        x = smooth(system, context.compiled.action, ae, x, opts.post, opts.post_lower, opts)
    return x


def main(args):
    torch.set_num_threads(2)
    device = torch.device("cuda:0")
    split = DarcySplit(args.data, args.split)
    a, _, xy, _, mask = tensors(split, device)
    compiled = SparseCompiled(split.mesh, device)
    context = CoarseContext(compiled, 1)
    model, saved = load_dictionary(args.dictionary, device)
    model.eval()
    opts = SimpleNamespace(**saved["options"])
    no, stats, _ = load_no(args.no_dir, device)
    begin = clock(device)
    bases, info = trunk_spaces(args.trunk_model, xy.reshape(-1, 2), compiled.plan.free, device)
    cache_ms = 1000 * (clock(device) - begin)
    methods = [(variant, k) for variant in args.variants for k in args.cycles]
    rows, stages = [], []
    with torch.no_grad():
        for i in range(min(args.count, len(split))):
            def solve(variant, k, profile=False):
                parts = {}
                last = clock(device) if profile else 0.
                def tick(label):
                    nonlocal last
                    if profile:
                        now = clock(device)
                        parts[label] = 1000 * (now-last)
                        last = now
                full = predict(no, a[i:i+1], xy, mask, stats).flatten()
                tick("FNO_ms")
                system, ae, _ = compiled.build(a[i].flatten(), torch.zeros_like(full))
                tick("fine_assembly_ms")
                cfg = SimpleNamespace(**vars(opts))
                x = smooth(system, compiled.action, ae, full[system.free], cfg.pre, cfg.lower, cfg)
                tick("prefilter_ms")
                if variant.startswith("TB"):
                    modes = bases["TB80"]
                    cfg.harmonic_steps = 0
                else:
                    modes = model(context.features(system, ae, x), context)
                tick("space_generation_ms")
                if variant.endswith("exact"):
                    prepared = exact_prepare(modes, context, system, ae, cfg)
                    tick("space_factor_ms")
                    x = exact_cycles(x, prepared, system, ae, context, cfg, k)
                else:
                    prepared = prepare(modes, context, system, ae, cfg)
                    tick("space_factor_ms")
                    x = cycles(x, prepared, system, ae, context, cfg, k, filtered=True)
                tick("correction_ms")
                full = torch.zeros_like(full)
                full[system.free] = x
                return (full, parts) if profile else full
            values = {}
            for variant, k in methods:
                out, ms = measure(lambda: solve(variant, k), device, args.repeats)
                if not bool(torch.isfinite(out).all()):
                    raise RuntimeError(f"Nonfinite output: {variant}, sample {i}")
                name = f"{variant}_K{k}"
                metrics = field_metrics(out.cpu().numpy(), split.target[i], split.coefficient[i], split.mesh)
                values[name] = dict(metrics, total_ms=ms)
                if i < args.profile_count:
                    _, parts = solve(variant, k, True)
                    stages.append(dict(index=i, method=name, **parts))
            rows.append(dict(index=i, sample_index=int(split.sample_index[i]), methods=values))
            agg = {name: {key: summary([r["methods"][name][key] for r in rows])
                          for key in values[name]} for name in values}
            save_json(args.out / "evaluation.json", dict(
                split=args.split, count=len(rows), samples=rows, summary=agg, stages=stages,
                options=vars(opts), arguments={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
                trunk=info, cache_ms_once=cache_ms, device=torch.cuda.get_device_name(device),
                timing="Batch1; FNO+assembly+space+factor+correction; geometry/trunk cache excluded; warmup then sync wall median",
                definition="Exact: FP64 current-A Gram, unregularized Cholesky; common FP32 fine residual/filter; no retraining, full-grid QR or fallback."))
            print(i, {n: round(v["L2_percent"], 8) for n,v in values.items()}, flush=True)
    keys = ["L2_percent", "gradient_percent", "flux_L2_percent", "total_ms"]
    with (args.out / "summary.csv").open("w") as f:
        writer = csv.writer(f)
        writer.writerow(["method", *keys])
        for name in agg:
            writer.writerow([name, *[agg[name][k]["mean"] for k in keys]])


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("data", "no-dir", "dictionary", "trunk-model", "out"):
        p.add_argument("--"+name, type=Path, required=True)
    p.add_argument("--split", choices=["val", "test"], default="val")
    p.add_argument("--count", type=int, default=8)
    p.add_argument("--cycles", type=int, nargs="+", default=[2])
    p.add_argument("--variants", nargs="+", choices=["CDMC_old", "CDMC_exact", "TB_old", "TB_exact"],
                   default=["CDMC_old", "CDMC_exact", "TB_old", "TB_exact"])
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--profile-count", type=int, default=2)
    main(p.parse_args())
