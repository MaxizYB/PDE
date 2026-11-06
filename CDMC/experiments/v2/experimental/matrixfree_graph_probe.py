"""Fixed-step matrix-free PCG in one CUDA Graph; diagnostic numerical baseline."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from adapters.darcy import DarcySplit
from experimental.matrixfree_cg_probe import pcg
from experimental.matrixfree_two_step import MatrixFreeDarcy, rel_error
from run import describe, load_proposal, synchronized_clock
from solver.inference import FixedMeshProposal
from solver.numerics import build_hierarchy


class FixedReferenceCoarse:
    """SPD additive coarse correction, compiled once from the unit-coefficient mesh.

    The per-sample matrix is never assembled.  Coarse energy is scaled by the
    sample mean coefficient; this is approximate for heterogeneous media, while
    the matrix-free residual and Krylov iteration retain the exact P1 operator.
    """

    def __init__(self, mesh, device, coarse_size=128):
        from baselines.data.darcy_v2.generate import assemble, element_geometry

        points = np.asarray(mesh.points, dtype=np.float64)
        cells = np.asarray(mesh.cells, dtype=np.int32)
        reference, _ = assemble(points, cells, np.ones(len(points)),
                                element_geometry(points, cells))
        free = np.flatnonzero(~mesh.boundary)
        hierarchy = build_hierarchy(reference[free][:, free].tocsr(),
                                    coarse_size=coarse_size)
        prolongation = sp.eye(len(free), format="csr")
        for _, p, _ in hierarchy[:-1]:
            prolongation = (prolongation @ p).tocsr()
        coarse = hierarchy[-1].toarray()
        self.basis = torch.as_tensor(prolongation.toarray(),
                                     dtype=torch.float32, device=device)
        self.inverse = torch.as_tensor(np.linalg.inv(coarse),
                                       dtype=torch.float32, device=device)
        self.size = coarse.shape[0]

    def apply(self, residual, diagonal, scale):
        coarse_rhs = residual @ self.basis
        coarse_direction = (coarse_rhs @ self.inverse) @ self.basis.T
        return residual / diagonal + coarse_direction / scale


def fixed_coarse_pcg(operator, x0, lift, element_a, diagonal, steps,
                     coarse: FixedReferenceCoarse):
    x = x0.clone()
    residual = operator.residual(x, lift, element_a)
    scale = element_a.mean(dim=1, keepdim=True)
    preconditioned = coarse.apply(residual, diagonal, scale)
    direction = preconditioned.clone()
    inner = (residual * preconditioned).sum(dim=1, keepdim=True)
    for _ in range(steps):
        ad = operator.homogeneous_action(direction, element_a)
        alpha = inner / (direction * ad).sum(dim=1, keepdim=True).clamp_min(1e-20)
        x = x + alpha * direction
        residual = residual - alpha * ad
        preconditioned = coarse.apply(residual, diagonal, scale)
        next_inner = (residual * preconditioned).sum(dim=1, keepdim=True)
        direction = preconditioned + next_inner / inner.clamp_min(1e-20) * direction
        inner = next_inner
    return x


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--proposal-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=64)
    parser.add_argument("--count", type=int, default=16)
    parser.add_argument("--initial", choices=("no", "zero"), default="no")
    parser.add_argument("--preconditioner", choices=("jacobi", "fixed-coarse"),
                        default="jacobi")
    args = parser.parse_args()
    args.checkpoint_dir = None
    torch.set_num_threads(4)
    device = torch.device("cuda")
    context_started = time.perf_counter()
    torch.empty(1, device=device)
    torch.cuda.synchronize(device)
    cuda_context_ms_once = 1000 * (time.perf_counter() - context_started)
    split = DarcySplit(args.data, "test")
    model, scale = load_proposal(args, device) if args.initial == "no" else (None, None)
    plan_started = time.perf_counter()
    operator = MatrixFreeDarcy(split.mesh, device)
    coarse = (FixedReferenceCoarse(split.mesh, device)
              if args.preconditioner == "fixed-coarse" else None)
    torch.cuda.synchronize(device)
    mesh_plan_ms_once = 1000 * (time.perf_counter() - plan_started)
    proposal = (FixedMeshProposal(model, split, scale, device,
                                  capture_graph=True, use_prepared=True)
                if model is not None else None)
    coeff_static = torch.as_tensor(split.coefficient[0:1], device=device).clone()
    lift_static = torch.as_tensor(split.lift[0:1], device=device).clone()
    x_static = (proposal.predict(split, 0)[None, operator.free].clone()
                if proposal is not None else
                torch.zeros((1, len(operator.free)), device=device))
    def solve(element_a, diagonal):
        if coarse is None:
            return pcg(operator, x_static, lift_static, element_a, diagonal, args.steps)
        return fixed_coarse_pcg(operator, x_static, lift_static, element_a,
                                diagonal, args.steps, coarse)

    torch.cuda.synchronize(device)
    t0 = time.perf_counter()
    stream = torch.cuda.Stream(device=device)
    stream.wait_stream(torch.cuda.current_stream(device))
    with torch.no_grad(), torch.cuda.stream(stream):
        for _ in range(3):
            element_a, diagonal = operator.prepare(coeff_static)
            solve(element_a, diagonal)
    torch.cuda.current_stream(device).wait_stream(stream)
    torch.cuda.synchronize(device)
    graph = torch.cuda.CUDAGraph()
    with torch.no_grad(), torch.cuda.graph(graph):
        element_a, diagonal = operator.prepare(coeff_static)
        output = solve(element_a, diagonal)
    graph.replay()
    torch.cuda.synchronize(device)
    graph_compile_ms = 1000 * (time.perf_counter() - t0)
    records = []
    with torch.no_grad():
        for i in range(min(args.count, len(split)) + 1):
            index = 0 if i == 0 else i - 1
            t0 = synchronized_clock(device)
            coefficient = torch.as_tensor(split.coefficient[index:index + 1], device=device)
            lift = torch.as_tensor(split.lift[index:index + 1], device=device)
            coeff_static.copy_(coefficient)
            lift_static.copy_(lift)
            input_ms = 1000 * (synchronized_clock(device) - t0)
            if proposal is not None:
                t0 = synchronized_clock(device)
                prediction = proposal.predict(split, index)
                x_static.copy_(prediction[None, operator.free])
                no_ms = 1000 * (synchronized_clock(device) - t0)
            else:
                no_ms = 0.0
            t0 = synchronized_clock(device)
            graph.replay()
            graph_ms = 1000 * (synchronized_clock(device) - t0)
            if i:
                truth = torch.as_tensor(split.target[index:index + 1], device=device)
                error = float(rel_error(output, truth, truth, operator)[0]) * 100
                records.append({"error_percent": error, "input_ms": input_ms,
                                "NO_ms": no_ms, "graph_ms": graph_ms,
                                "total_ms": input_ms + no_ms + graph_ms})
    summary = {key: describe([row[key] for row in records]) for key in
               ("error_percent", "input_ms", "NO_ms", "graph_ms", "total_ms")}
    summary["graph_compile_ms_once"] = graph_compile_ms
    summary["mesh_plan_ms_once"] = mesh_plan_ms_once
    summary["cuda_context_ms_once"] = cuda_context_ms_once
    summary["proposal_compile_ms_once"] = proposal.compile_ms_once if proposal is not None else 0.0
    summary["peak_allocated_mib"] = torch.cuda.max_memory_allocated(device) / 1024**2
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"steps": args.steps, "initial": args.initial,
                                    "preconditioner": args.preconditioner,
                                    "coarse_size": coarse.size if coarse is not None else None,
                                    "summary": summary,
                                    "samples": records}, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
