#!/usr/bin/env python3
"""Train and evaluate a Geo-FNO proposal with fixed MG and CDMC correction.

Examples:
  python run.py all --data /path/to/generated/regular --out results/regular
  python run.py evaluate --data /path/to/generated/regular --out results/regular
"""
from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path

# The online coarse inverse is tiny. OpenBLAS's large default thread pool made
# a 121x121 inverse >100 ms on the shared server, versus <1 ms with one thread.
# Set this before NumPy/SciPy are imported; PyTorch thread count is set in main.
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import numpy as np
import torch

from adapters.darcy import DATASETS, DarcySplit, infer_solution, relative_field_error
from adapters.darcy_system import make_mesh_plan, make_system, solve_sample
from solver.correctors import CDMCSetup, cdmc, fixed_mg
from solver.inference import FixedMeshProposal
from solver.matrixfree_cdmc import (
    MeshCompiledCDMC, correct_matrixfree, gates_from_implicit,
)
from solver.model import MODEL_REGISTRY, make_model


def synchronized_clock(device: torch.device) -> float:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return time.perf_counter()


def error_percent(prediction: torch.Tensor, target: torch.Tensor, mass: torch.Tensor) -> float:
    return 100 * float(relative_field_error(
        prediction[None], target[None], mass[None]
    ).item())


def model_scale(split: DarcySplit) -> float:
    return max(float(np.quantile(np.abs(split.target - split.lift), 0.95)), 1e-3)


def load_manifest(root: Path) -> tuple[dict, object]:
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    profile = DATASETS.get(manifest["kind"])
    if profile is None:
        raise ValueError(f"unknown dataset kind: {manifest['kind']}")
    profile.validate(manifest)
    return manifest, profile


def make_proposal(profile, args, device):
    params = dict(profile.DEFAULT_MODEL)
    if args.width is not None:
        params["width"] = args.width
    if args.modes is not None:
        params["modes"] = args.modes
    if args.latent_resolution is not None:
        params["latent_resolution"] = args.latent_resolution
    return make_model(args.model, input_dim=5, **params).to(device), params


def predict_numpy(model, split, scale, device, batch_size=8):
    model.eval()
    batches = []
    with torch.no_grad():
        for start in range(0, len(split), batch_size):
            ids = np.arange(start, min(start + batch_size, len(split)))
            batches.append(infer_solution(model, split, ids, device, scale).detach())
    return torch.cat(batches)


def train_proposal(args, profile, device):
    train = DarcySplit(args.data, "train")
    val = DarcySplit(args.data, "val")
    scale = model_scale(train)
    model, model_kwargs = make_proposal(profile, args, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    best, log = float("inf"), []
    for epoch in range(args.no_epochs):
        model.train()
        order = np.random.permutation(len(train))
        training = []
        for start in range(0, len(order), args.batch_size):
            indices = order[start:start + args.batch_size]
            pred = infer_solution(model, train, indices, device, scale)
            target = train.target_tensor(indices, device)
            mass = torch.as_tensor(train.mesh.mass, device=device).expand(len(indices), -1)
            loss = relative_field_error(pred, target, mass).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            training.append(float(loss.detach()))
        model.eval()
        with torch.no_grad():
            pred = predict_numpy(model, val, scale, device)
            truth = torch.as_tensor(val.target, device=device)
            mass = torch.as_tensor(val.mesh.mass, device=device).expand(len(val), -1)
            validation = float(relative_field_error(pred, truth, mass).mean())
        row = {"epoch": epoch + 1, "train_rel_l2": float(np.mean(training)),
               "val_rel_l2": validation}
        log.append(row)
        print(f"NO epoch {epoch + 1}/{args.no_epochs}: "
              f"train={row['train_rel_l2']:.5f} val={validation:.5f}", flush=True)
        if validation < best:
            best = validation
            torch.save({"state": model.state_dict(), "model": args.model,
                        "model_kwargs": model_kwargs, "scale": scale,
                        "best_val_rel_l2": best, "seed": args.seed},
                       args.out / "proposal.pt")
    (args.out / "proposal_log.json").write_text(json.dumps(log, indent=2) + "\n")
    print(f"best NO validation: {100*best:.3f}%", flush=True)


def load_proposal(args, device):
    checkpoint_root = args.proposal_dir or args.checkpoint_dir or args.out
    checkpoint = torch.load(checkpoint_root / "proposal.pt", map_location=device, weights_only=False)
    model = make_model(
        checkpoint["model"], input_dim=5, **checkpoint["model_kwargs"]
    ).to(device)
    model.load_state_dict(checkpoint["state"])
    model.eval()
    return model, float(checkpoint["scale"])


def prepared_examples(split, model, scale, device, plan):
    predictions = predict_numpy(model, split, scale, device)
    examples = []
    for i in range(len(split)):
        system = make_system(
            split.mesh, split.coefficient[i], split.lift[i], device, plan
        )
        init = predictions[i, system.free].detach()
        target = torch.as_tensor(split.target[i], device=device)
        lift = torch.as_tensor(split.lift[i], device=device)
        mass = torch.as_tensor(split.mesh.mass, device=device)
        examples.append((system, init, target, lift, mass))
    return examples


def corrected_loss(setup, examples, indices, cycles=(1, 2)):
    losses = []
    for i in indices:
        system, init, target, lift, mass = examples[int(i)]
        smooth, coarse = setup(system, init)
        from solver.numerics import correct  # shared kernel; setup remains one call
        total = 0
        for cycle in cycles:
            interior = correct(system, init, cycle, smooth, coarse)
            candidate = lift.clone()
            candidate[system.free] = interior
            numerator = ((candidate - target).square() * mass).sum()
            denominator = (target.square() * mass).sum().clamp_min(1e-12)
            total = total + torch.sqrt(numerator / denominator + 1e-12)
        losses.append(total / len(cycles))
    return torch.stack(losses).mean()


def train_cdmc(args, device):
    model, scale = load_proposal(args, device)
    train = DarcySplit(args.data, "train")
    val = DarcySplit(args.data, "val")
    print("compiling shared mesh plan and preparing NO-conditioned systems", flush=True)
    plan = make_mesh_plan(train.mesh, device, args.coarse_size)
    train_examples = prepared_examples(train, model, scale, device, plan)
    val_examples = prepared_examples(val, model, scale, device, plan)
    setup = CDMCSetup(width=args.setup_width).to(device)
    optimizer = torch.optim.AdamW(setup.parameters(), lr=args.cdmc_lr)
    best, log = float("inf"), []
    for epoch in range(args.cdmc_epochs):
        setup.train()
        order = np.random.permutation(len(train_examples))
        training = []
        for index in order:
            loss = corrected_loss(setup, train_examples, [index])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(setup.parameters(), 5.0)
            optimizer.step()
            training.append(float(loss.detach()))
        setup.eval()
        with torch.no_grad():
            validation = float(corrected_loss(
                setup, val_examples, range(len(val_examples))
            ))
        row = {"epoch": epoch + 1, "train_rel_l2": float(np.mean(training)),
               "val_rel_l2": validation}
        log.append(row)
        print(f"CDMC epoch {epoch + 1}/{args.cdmc_epochs}: "
              f"train={row['train_rel_l2']:.5f} val={validation:.5f}", flush=True)
        if validation < best:
            best = validation
            torch.save({"state": setup.state_dict(), "width": args.setup_width,
                        "best_val_rel_l2": best, "seed": args.seed,
                        "coarse_size": args.coarse_size},
                       args.out / "cdmc.pt")
    (args.out / "cdmc_log.json").write_text(json.dumps(log, indent=2) + "\n")
    print(f"best CDMC validation: {100*best:.3f}%", flush=True)


def describe(values: list[float]):
    return {"mean": float(np.mean(values)), "p95": float(np.percentile(values, 95)),
            "median": float(np.median(values))}


def evaluate(args, device):
    model, scale = load_proposal(args, device)
    checkpoint_root = args.checkpoint_dir or args.out
    checkpoint = torch.load(checkpoint_root / "cdmc.pt", map_location=device, weights_only=False)
    if args.coarse_size is None:
        args.coarse_size = int(checkpoint.get("coarse_size", 28))
    elif "coarse_size" in checkpoint and args.coarse_size != checkpoint["coarse_size"]:
        raise ValueError("evaluation coarse-size differs from the trained CDMC hierarchy")
    setup = CDMCSetup(checkpoint["width"]).to(device)
    setup.load_state_dict(checkpoint["state"])
    setup.eval()
    test = DarcySplit(args.data, "test")
    manifest, _ = load_manifest(args.data)
    t0 = synchronized_clock(device)
    plan = make_mesh_plan(
        test.mesh, device, args.coarse_size,
        gpu_galerkin=args.system_backend != "cpu",
        compiled_assembly=args.system_backend in ("gpu_compiled", "gpu_hybrid", "gpu_affine"),
        hybrid_dense_limit=(args.hybrid_dense_limit if args.system_backend in ("gpu_hybrid", "gpu_affine") else None),
        affine_compiler=args.system_backend == "gpu_affine",
    )
    mesh_plan_ms = 1000 * (synchronized_clock(device) - t0)
    proposal_runner = FixedMeshProposal(
        model, test, scale, device,
        capture_graph=args.proposal_mode == "graph",
        use_prepared=args.proposal_mode != "reference",
    )
    names = ["NO", "MG_zero_K1", "MG_zero_K2", "MG_NO_K1", "MG_NO_K2",
             "CDMC_NO_K1", "CDMC_NO_K2", "FEM_direct"]
    errors = {name: [] for name in names}
    timings = {name: [] for name in names}
    timings["assembly_hierarchy"] = []
    per_sample = []
    # Initialize CUDA kernels and sparse paths before per-sample timing.
    with torch.no_grad():
        warm_system = make_system(
            test.mesh, test.coefficient[0], test.lift[0], device, plan,
            backend=args.system_backend,
            coarse_solver=args.coarse_solver,
        )
        t0 = synchronized_clock(device)
        warm_proposal = proposal_runner.predict(test, 0)
        proposal_warmup_ms = 1000 * (synchronized_clock(device) - t0)
        warm_initial = warm_proposal[warm_system.free]
        fixed_mg(warm_system, warm_initial, 2)
        cdmc(setup, warm_system, warm_initial, 2)
        synchronized_clock(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for i in range(min(args.eval_count or len(test), len(test))):
        t0 = synchronized_clock(device)
        system = make_system(
            test.mesh, test.coefficient[i], test.lift[i], device, plan,
            backend=args.system_backend,
            coarse_solver=args.coarse_solver,
        )
        timings["assembly_hierarchy"].append(
            1000 * (synchronized_clock(device) - t0)
        )
        truth = torch.as_tensor(test.target[i], device=device)
        mass = torch.as_tensor(test.mesh.mass, device=device)
        row = {"sample_index": int(test.sample_index[i])}
        with torch.no_grad():
            t0 = synchronized_clock(device)
            proposal = proposal_runner.predict(test, i)
            timings["NO"].append(1000 * (synchronized_clock(device) - t0))
            init = proposal[system.free]
            zero = torch.zeros_like(init)
            candidates = {"NO": proposal}
            for label, initial, k in (
                ("MG_zero_K1", zero, 1), ("MG_zero_K2", zero, 2),
                ("MG_NO_K1", init, 1), ("MG_NO_K2", init, 2),
            ):
                t0 = synchronized_clock(device)
                interior = fixed_mg(system, initial, k)
                timings[label].append(1000 * (synchronized_clock(device) - t0))
                full = torch.as_tensor(test.lift[i], device=device).clone()
                full[system.free] = interior
                candidates[label] = full
            t0 = synchronized_clock(device)
            smooth, coarse = setup(system, init)
            setup_time = 1000 * (synchronized_clock(device) - t0)
            for label, k in (("CDMC_NO_K1", 1), ("CDMC_NO_K2", 2)):
                t0 = synchronized_clock(device)
                from solver.numerics import correct
                interior = correct(system, init, k, smooth, coarse)
                timings[label].append(setup_time + 1000 * (synchronized_clock(device) - t0))
                full = torch.as_tensor(test.lift[i], device=device).clone()
                full[system.free] = interior
                candidates[label] = full
            for name, candidate in candidates.items():
                value = error_percent(candidate, truth, mass)
                errors[name].append(value)
                row[name + "_error_percent"] = value
        # Mature sparse direct solve on the identical assembled P1 system.
        # Its reported wall time includes assembly, boundary elimination, and solve.
        t0 = time.perf_counter()
        direct, _ = solve_sample(
            test.mesh.points, test.mesh.cells, test.mesh.boundary,
            test.coefficient[i], test.lift[i], plan.geometry,
        )
        timings["FEM_direct"].append(1000 * (time.perf_counter() - t0))
        direct_error = error_percent(
            torch.as_tensor(direct, dtype=truth.dtype, device=device), truth, mass
        )
        errors["FEM_direct"].append(direct_error)
        row["FEM_direct_error_percent"] = direct_error
        per_sample.append(row)
        print(f"test {i + 1}/{min(args.eval_count or len(test), len(test))}: "
              f"NO={row['NO_error_percent']:.3f}% "
              f"CDMC1={row['CDMC_NO_K1_error_percent']:.3f}%", flush=True)
    summary = {name: {"field_error_percent": describe(errors[name]),
                      "apply_ms_sample": describe(timings[name])}
               for name in names}
    for name in names:
        if name in ("NO", "FEM_direct"):
            total = timings[name]
        elif name.startswith("MG_zero"):
            total = [a + b for a, b in zip(
                timings["assembly_hierarchy"], timings[name]
            )]
        else:
            total = [a + b + c for a, b, c in zip(
                timings["assembly_hierarchy"], timings["NO"], timings[name]
            )]
        summary[name]["end_to_end_ms_sample"] = describe(total)
    summary["assembly_hierarchy_ms_sample"] = describe(timings["assembly_hierarchy"])
    summary["online_system_ms_sample"] = summary["assembly_hierarchy_ms_sample"]
    summary["mesh_plan_ms_once"] = mesh_plan_ms
    summary["proposal_compile_ms_once"] = proposal_runner.compile_ms_once
    summary["proposal_warmup_ms_once"] = proposal_warmup_ms
    summary["proposal_startup_ms_once"] = (
        proposal_runner.compile_ms_once + proposal_warmup_ms
    )
    if device.type == "cuda":
        summary["cuda_peak_allocated_mb"] = (
            torch.cuda.max_memory_allocated(device) / (1024 ** 2)
        )
    if proposal_runner.prepared and device.type == "cuda":
        t0 = synchronized_clock(device)
        resident = proposal_runner.stage(test)
        summary["NO_resident_stage_ms_once"] = 1000 * (synchronized_clock(device) - t0)
        resident_times = []
        with torch.no_grad():
            for i in range(min(len(test), 16)):
                proposal_runner.predict(test, i, resident)
            for i in range(min(len(test), 16)):
                t0 = synchronized_clock(device)
                proposal_runner.predict(test, i, resident)
                resident_times.append(1000 * (synchronized_clock(device) - t0))
        summary["NO_gpu_resident_ms_sample"] = describe(resident_times)
    output = {
        "status": "pilot; newly generated data, not an official-paper comparison",
        "dataset": manifest, "proposal": "Geo-FNO geometry Fourier transform + latent FFT",
        "corrector": "shared stable Galerkin V-cycle for fixed MG and learned CDMC setup",
        "coarse_size": args.coarse_size,
        "system_backend": args.system_backend,
        "coarse_solver": args.coarse_solver,
        "hybrid_dense_limit": args.hybrid_dense_limit if args.system_backend in ("gpu_hybrid", "gpu_affine") else None,
        "dense_level_sizes": [
            level.matrix.shape[0] for level in warm_system.levels
            if level.matrix.layout == torch.strided
        ],
        "proposal_mode": args.proposal_mode,
        "proposal_graph_enabled": proposal_runner.graph is not None,
        "device": str(device), "evaluated_samples": len(per_sample),
        "summary": summary, "per_sample": per_sample,
        "timing_note": "Per-sample end-to-end is the repeated-query cost: online system "
                       "assembly/transfer + relevant NO + correction. The one-time "
                       "mesh_plan_ms_once and proposal_startup_ms_once are reported "
                       "separately and must be added for a cold single query. "
                       "NO_gpu_resident_ms_sample stages all dataset inputs and is not "
                       "included in the host-input end-to-end figure. FEM_direct is an independent CPU "
                       "P1 assembly plus sparse direct solve of the same discretization.",
    }
    (args.out / "pilot_result.json").write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)


def evaluate_compiled(args, device):
    """Evaluate the validated mesh-compiled CDMC execution backend."""
    if device.type != "cuda":
        raise RuntimeError("evaluate-compiled requires CUDA for captured inference")
    if args.compiled_coarse_steps < 1 or args.compiled_cycles < 1:
        raise ValueError("compiled cycles and coarse steps must be positive")
    model, scale = load_proposal(args, device)
    checkpoint_root = args.checkpoint_dir or args.out
    checkpoint = torch.load(
        checkpoint_root / "cdmc.pt", map_location=device, weights_only=False
    )
    coarse_size = int(checkpoint["coarse_size"])
    if args.coarse_size is not None and args.coarse_size != coarse_size:
        raise ValueError("evaluation coarse-size differs from the trained hierarchy")
    setup = CDMCSetup(checkpoint["width"]).to(device)
    setup.load_state_dict(checkpoint["state"])
    setup.eval()
    test = DarcySplit(args.data, "test")
    manifest, _ = load_manifest(args.data)
    compiled = MeshCompiledCDMC(test.mesh, device, coarse_size)
    proposal = FixedMeshProposal(
        model, test, scale, device, capture_graph=True, use_prepared=True,
    )
    coefficient = torch.as_tensor(test.coefficient[0], device=device).clone()
    lift = torch.as_tensor(test.lift[0], device=device).clone()
    initial = proposal.predict(test, 0)[compiled.plan.free].clone()

    def solve():
        system, element_a = compiled.build(coefficient, lift)
        if args.compiled_corrector == "cdmc":
            smooth, coarse = gates_from_implicit(
                setup, system, compiled.action, element_a, initial,
            )
        else:
            smooth = coarse = torch.ones_like(initial)
        return correct_matrixfree(
            system, compiled.action, element_a, initial, smooth, coarse,
            compiled.reference_inverse, cycles=args.compiled_cycles,
            coarse_steps=args.compiled_coarse_steps,
        )

    stream = torch.cuda.Stream(device=device)
    stream.wait_stream(torch.cuda.current_stream(device))
    with torch.no_grad(), torch.cuda.stream(stream):
        for _ in range(3):
            solve()
    torch.cuda.current_stream(device).wait_stream(stream)
    synchronized_clock(device)
    graph = torch.cuda.CUDAGraph()
    started = time.perf_counter()
    with torch.no_grad(), torch.cuda.graph(graph):
        output = solve()
    graph.replay()
    graph_capture_ms = 1000 * (synchronized_clock(device) - started)
    reference_plan = None
    if args.compare_assembled:
        reference_plan = make_mesh_plan(
            test.mesh, device, coarse_size,
            gpu_galerkin=True, compiled_assembly=True, hybrid_dense_limit=512,
        )
        from solver.numerics import correct
        with torch.no_grad():
            for _ in range(3):
                reference_system = make_system(
                    test.mesh, test.coefficient[0], test.lift[0], device,
                    reference_plan, backend="gpu_hybrid",
                )
                if args.compiled_corrector == "cdmc":
                    ref_smooth, ref_coarse = setup(reference_system, initial)
                else:
                    ref_smooth = ref_coarse = torch.ones_like(initial)
                correct(
                    reference_system, initial, args.compiled_cycles,
                    ref_smooth, ref_coarse,
                )
        synchronized_clock(device)
    records = []
    count = min(args.eval_count or len(test), len(test))
    mass = torch.as_tensor(test.mesh.mass, device=device)
    free = compiled.plan.free
    with torch.no_grad():
        # One untimed replay after capture also warms host-to-device copies.
        for step in range(count + 1):
            index = 0 if step == 0 else step - 1
            started = synchronized_clock(device)
            coefficient.copy_(torch.as_tensor(test.coefficient[index], device=device))
            lift.copy_(torch.as_tensor(test.lift[index], device=device))
            input_ms = 1000 * (synchronized_clock(device) - started)
            started = synchronized_clock(device)
            proposal_full = proposal.predict(test, index)
            initial.copy_(proposal_full[free])
            no_ms = 1000 * (synchronized_clock(device) - started)
            started = synchronized_clock(device)
            graph.replay()
            graph_ms = 1000 * (synchronized_clock(device) - started)
            if step == 0:
                continue
            truth = torch.as_tensor(test.target[index], device=device)
            denominator = (mass * truth.square()).sum().clamp_min(1e-12)
            no_error = float(
                ((mass * (proposal_full - truth).square()).sum() / denominator).sqrt()
            ) * 100
            field_error = float(
                ((mass[free] * (output - truth[free]).square()).sum()
                 / denominator).sqrt()
            ) * 100
            record = {
                "sample_index": int(test.sample_index[index]),
                "NO_error_percent": no_error,
                "field_error_percent": field_error,
                "input_ms": input_ms,
                "NO_ms": no_ms,
                "corrector_graph_ms": graph_ms,
                "total_ms": input_ms + no_ms + graph_ms,
            }
            if reference_plan is not None:
                started = synchronized_clock(device)
                reference_system = make_system(
                    test.mesh, test.coefficient[index], test.lift[index],
                    device, reference_plan, backend="gpu_hybrid",
                )
                if args.compiled_corrector == "cdmc":
                    ref_smooth, ref_coarse = setup(reference_system, initial)
                else:
                    ref_smooth = ref_coarse = torch.ones_like(initial)
                reference_output = correct(
                    reference_system, initial, args.compiled_cycles,
                    ref_smooth, ref_coarse,
                )
                assembled_ms = 1000 * (synchronized_clock(device) - started)
                record.update({
                    "assembled_error_percent": float(
                        ((mass[free] * (reference_output - truth[free]).square()
                          ).sum() / denominator).sqrt()
                    ) * 100,
                    "assembled_ms": assembled_ms,
                    "assembled_total_ms": input_ms + no_ms + assembled_ms,
                    "max_solution_difference": float(
                        (output - reference_output).abs().max()
                    ),
                })
            records.append(record)
    measured = ("NO_error_percent", "field_error_percent",
                "input_ms", "NO_ms", "corrector_graph_ms", "total_ms")
    if reference_plan is not None:
        measured += ("assembled_error_percent", "assembled_ms",
                     "assembled_total_ms", "max_solution_difference")
    summary = {key: describe([row[key] for row in records]) for key in measured}
    summary["mesh_plan_ms_once"] = compiled.mesh_plan_ms_once
    summary["NO_compile_ms_once"] = proposal.compile_ms_once
    summary["graph_capture_ms_once"] = graph_capture_ms
    summary["peak_allocated_mib"] = torch.cuda.max_memory_allocated(device) / 1024**2
    result = {
        "status": "v2 compiled inference; existing trained weights",
        "dataset": manifest,
        "device": str(device),
        "coarse_size": coarse_size,
        "corrector": args.compiled_corrector,
        "cycles": args.compiled_cycles,
        "terminal_pcg_steps": args.compiled_coarse_steps,
        "compare_assembled": args.compare_assembled,
        "evaluated_samples": len(records),
        "summary": summary,
        "per_sample": records,
        "timing_note": "total_ms includes host input copy, NO, and captured online "
                       "CDMC computation. Mesh plan, NO compilation, and graph "
                       "capture are one-time costs and are reported separately. "
                       "Assembled CDMC is only timed when --compare-assembled is set; "
                       "its timer includes sample-specific system build, gates and "
                       "correction but shares the already timed NO/input.",
    }
    path = args.out / "compiled_result.json"
    path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)
    print(f"result={path}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=(
        "train-no", "train-cdmc", "evaluate", "evaluate-compiled", "all",
    ))
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path,
                        help="load trained weights elsewhere, e.g. for geometry OOD evaluation")
    parser.add_argument("--proposal-dir", type=Path,
                        help="load NO weights separately from the CDMC/output directory")
    parser.add_argument("--proposal-mode", choices=("reference", "prepared", "graph"),
                        default="graph", help="NO evaluation path; training is unchanged")
    parser.add_argument("--system-backend", choices=("auto", "cpu", "gpu", "gpu_compiled", "gpu_hybrid", "gpu_affine"),
                        default="auto", help="evaluation system construction; gpu_affine is experimental (Darcy only)")
    parser.add_argument("--hybrid-dense-limit", type=int, default=512,
                        help="for gpu_hybrid, use dense matrices/transfers when level size <= this limit")
    parser.add_argument("--coarse-solver", choices=("inverse", "cholesky"), default="inverse",
                        help="repeated V-cycle coarse solve: inverse matvec or Cholesky solve")
    parser.add_argument("--model", choices=tuple(MODEL_REGISTRY), default="geofno")
    parser.add_argument("--no-epochs", type=int, default=30)
    parser.add_argument("--cdmc-epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--cdmc-lr", type=float, default=0.001)
    parser.add_argument("--setup-width", type=int, default=32)
    parser.add_argument("--coarse-size", type=int,
                        help="stop aggregation when the coarse matrix has at most this many rows")
    parser.add_argument("--width", type=int)
    parser.add_argument("--modes", type=int)
    parser.add_argument("--latent-resolution", type=int)
    parser.add_argument("--eval-count", type=int, default=0)
    parser.add_argument("--compiled-cycles", type=int, default=2,
                        help="matrix-free correction cycles; validated default: 2")
    parser.add_argument("--compiled-coarse-steps", type=int, default=2,
                        help="terminal fixed PCG steps; validated default: 2")
    parser.add_argument("--compiled-corrector", choices=("cdmc", "mg"),
                        default="cdmc", help="learned CDMC gates or fixed MG gates")
    parser.add_argument("--compare-assembled", action="store_true",
                        help="same-process assembled gpu_hybrid A/B and field check")
    parser.add_argument("--seed", type=int, default=20260927)
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.system_backend == "auto":
        args.system_backend = "gpu_hybrid" if device.type == "cuda" else "cpu"
    args.out.mkdir(parents=True, exist_ok=True)
    manifest, profile = load_manifest(args.data)
    print(f"dataset={profile.DESCRIPTION}, nodes={manifest['nodes']}, device={device}", flush=True)
    if args.command in ("train-no", "all"):
        train_proposal(args, profile, device)
    if args.command in ("train-cdmc", "all"):
        if args.coarse_size is None:
            args.coarse_size = 128
        train_cdmc(args, device)
    if args.command in ("evaluate", "all"):
        evaluate(args, device)
    if args.command == "evaluate-compiled":
        evaluate_compiled(args, device)


if __name__ == "__main__":
    main()
