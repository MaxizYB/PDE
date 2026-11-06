"""Bounded mesh-plan scaling probe; never allocates a large dense Q blindly.

This measures plan construction only, not a model inference or PDE solve.  The
full mode follows PartialGalerkin, while hierarchy mode stops before its dense
cross-level bases and projected-element maps.  No dataset files are generated.
"""
from __future__ import annotations

import argparse
import json
import resource
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE.parents[4]))
from adapters.darcy import Mesh
from adapters.darcy_system import make_mesh_plan, make_system
from baselines.data.darcy_v2.generate import element_geometry, regular_mesh
from experimental.matrixfree_cdmc_partial_probe import (
    PartialGalerkin, compile_diag_terminal_maps,
)
from experimental.matrixfree_cdmc_vcycle_probe import (
    FactorizedGalerkin, ImplicitGalerkin,
)
from experimental.sparse_projection_maps import compile_sparse_affine_maps
from run import synchronized_clock
from solver.matrixfree_cdmc import MeshCompiledCDMC


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resolution", type=int, required=True)
    parser.add_argument("--mode", choices=("full", "hierarchy", "basis", "maps",
                                            "reference", "sparse-full", "scaled-full",
                                            "scaled-terminal-full", "implicit-terminal-full",
                                            "sparse-basis",
                                            "sparse-maps", "compiled-v2"),
                        required=True)
    parser.add_argument("--coarse-size", type=int, default=128)
    parser.add_argument("--max-dense-basis-gib", type=float, default=3.0)
    parser.add_argument("--max-nodes", type=int, default=70000,
                        help="preflight cap for all modes; explicit opt-in to exceed")
    parser.add_argument("--max-map-free-nodes", type=int, default=2048,
                        help="preflight cap for slow full/maps modes")
    parser.add_argument("--gpu-fraction", type=float, default=0.15)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.resolution < 5:
        parser.error("resolution must be at least 5")
    if not (0 < args.gpu_fraction <= 0.5):
        parser.error("gpu-fraction must be in (0, 0.5]")
    if args.resolution**2 > args.max_nodes:
        raise MemoryError(
            f"grid would create {args.resolution**2} nodes, "
            f"exceeding the guarded max-nodes={args.max_nodes}"
        )
    nfree = (args.resolution - 2) ** 2
    if args.mode in ("full", "maps") and nfree > args.max_map_free_nodes:
        raise RuntimeError(
            f"full projected-map compile at {nfree} free nodes is disallowed by "
            f"default max-map-free-nodes={args.max_map_free_nodes}; only override "
            "for a bounded, supervised run"
        )
    # Q_0 plus successively coarser dense bases: ~2 * nfree**2 float32 bytes.
    estimated_gib = 8 * nfree * nfree / 1024**3
    if args.mode in ("full", "basis", "maps") and estimated_gib > args.max_dense_basis_gib:
        raise MemoryError(
            f"full plan preflight rejected: dense bases alone ~{estimated_gib:.2f} GiB "
            f"> budget {args.max_dense_basis_gib:.2f} GiB"
        )
    torch.set_num_threads(4)
    device = torch.device("cuda:0")
    torch.cuda.set_per_process_memory_fraction(args.gpu_fraction, device=device)
    points, cells, boundary = regular_mesh(args.resolution)
    mass = element_geometry(points, cells)[2]
    mesh = Mesh(points.astype(np.float32), cells, boundary, mass.astype(np.float32))
    torch.cuda.reset_peak_memory_stats(device)
    started = synchronized_clock(device)
    if args.mode == "compiled-v2":
        plan = MeshCompiledCDMC(mesh, device, args.coarse_size)
        elapsed_ms = plan.mesh_plan_ms_once
        map_ms = None
        level_sizes = plan.action.level_sizes
    elif args.mode == "full":
        plan = PartialGalerkin(mesh, device, args.coarse_size)
        elapsed_ms = plan.mesh_plan_ms_once
        map_ms = plan.map_compile_ms_once
        level_sizes = plan.implicit.level_sizes
    elif args.mode == "sparse-full":
        plan = PartialGalerkin(mesh, device, args.coarse_size, sparse_plan=True)
        elapsed_ms = plan.mesh_plan_ms_once
        map_ms = plan.map_compile_ms_once
        level_sizes = plan.implicit.level_sizes
    elif args.mode == "scaled-full":
        plan = PartialGalerkin(mesh, device, args.coarse_size, sparse_plan=True,
                               approx_diagonal=True)
        elapsed_ms = plan.mesh_plan_ms_once
        map_ms = plan.map_compile_ms_once
        level_sizes = plan.implicit.level_sizes
    elif args.mode == "scaled-terminal-full":
        plan = PartialGalerkin(mesh, device, args.coarse_size, sparse_plan=True,
                               approx_diagonal=True, approx_terminal=True)
        elapsed_ms = plan.mesh_plan_ms_once
        map_ms = plan.map_compile_ms_once
        level_sizes = plan.implicit.level_sizes
    elif args.mode == "implicit-terminal-full":
        plan = PartialGalerkin(mesh, device, args.coarse_size, sparse_plan=True,
                               approx_diagonal=True, implicit_terminal=True)
        elapsed_ms = plan.mesh_plan_ms_once
        map_ms = plan.map_compile_ms_once
        level_sizes = plan.implicit.level_sizes
    else:
        plan = make_mesh_plan(mesh, device, args.coarse_size)
        hierarchy_ms = 1000 * (synchronized_clock(device) - started)
        print(json.dumps({"stage": "hierarchy", "ms": hierarchy_ms}), flush=True)
        elapsed_ms = hierarchy_ms
        map_ms = None
        level_sizes = [len(plan.free)] + [p.shape[1] for p, _ in plan.transfers]
        if args.mode in ("basis", "maps", "sparse-basis", "sparse-maps"):
            started = synchronized_clock(device)
            implicit = (FactorizedGalerkin(mesh, plan, device)
                        if args.mode.startswith("sparse-")
                        else ImplicitGalerkin(mesh, plan, device))
            basis_ms = 1000 * (synchronized_clock(device) - started)
            print(json.dumps({"stage": "factorized_transfers" if args.mode.startswith("sparse-")
                                       else "dense_bases", "ms": basis_ms}), flush=True)
            elapsed_ms += basis_ms
            if args.mode in ("maps", "sparse-maps"):
                started = synchronized_clock(device)
                if args.mode == "sparse-maps":
                    maps, _ = compile_sparse_affine_maps(
                        mesh, plan.geometry, plan.free, plan.transfers,
                        terminal_mode="gradient",
                        progress=lambda stage: print(json.dumps(stage), flush=True),
                    )
                else:
                    maps = compile_diag_terminal_maps(
                        mesh, plan.geometry, plan.free, plan.transfers, device,
                    )
                map_ms = 1000 * (synchronized_clock(device) - started)
                print(json.dumps({"stage": "projected_maps", "ms": map_ms,
                                  "count": len(maps)}), flush=True)
                elapsed_ms += map_ms
        if args.mode == "reference":
            started = synchronized_clock(device)
            reference = make_system(
                mesh, np.ones(len(points), dtype=np.float32),
                np.zeros(len(points), dtype=np.float32), device, plan,
                backend="cpu",
            )
            reference_ms = 1000 * (synchronized_clock(device) - started)
            print(json.dumps({"stage": "reference_system", "ms": reference_ms,
                              "levels": len(reference.levels)}), flush=True)
            elapsed_ms += reference_ms
    result = {
        "resolution": args.resolution,
        "nodes": len(points),
        "free_nodes": nfree,
        "triangles": len(cells),
        "mode": args.mode,
        "level_sizes": level_sizes,
        "estimated_dense_bases_gib": estimated_gib,
        "plan_ms_once": elapsed_ms,
        "map_compile_ms_once": map_ms,
        "peak_gpu_allocated_mib": torch.cuda.max_memory_allocated(device) / 1024**2,
        "peak_process_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
        "gpu_fraction_cap": args.gpu_fraction,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
