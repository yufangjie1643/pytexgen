"""Small synthetic benchmark for TexGen's Python voxelizer backends.

This benchmark avoids real TexGen geometry so it can run quickly during backend
development. It measures the classification kernel with and without AABB
candidate pruning, then optionally measures torch CPU/GPU if torch is installed.
"""

import argparse
import importlib.util
import sys
import time
import types
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent


def load_voxelizer_module():
    pkg = types.ModuleType("TexGen")
    pkg.__path__ = [str(ROOT / "TexGen")]
    sys.modules["TexGen"] = pkg

    core = types.ModuleType("TexGen.Core")

    class CYarn:
        LINE = 1
        SURFACE = 2
        VOLUME = 4

    class CTextile:
        pass

    core.CYarn = CYarn
    core.CTextile = CTextile
    sys.modules["TexGen.Core"] = core

    spec = importlib.util.spec_from_file_location(
        "TexGen.gpu_voxelizer", ROOT / "TexGen" / "gpu_voxelizer.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def make_centers(resolution, dtype):
    xs = (np.arange(resolution, dtype=np.float64) + 0.5) / resolution
    ys = (np.arange(resolution, dtype=np.float64) + 0.5) / resolution
    zs = (np.arange(resolution, dtype=np.float64) + 0.5) / resolution
    gz, gy, gx = np.meshgrid(zs, ys, xs, indexing="ij")
    return np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=-1).astype(dtype)


def make_snapshots(voxelizer, yarn_grid, dtype):
    section = np.array(
        [
            [-0.035, -0.035],
            [0.035, -0.035],
            [0.035, 0.035],
            [-0.035, 0.035],
            [-0.035, -0.035],
        ],
        dtype=dtype,
    )
    x_nodes = np.linspace(0.0, 1.0, 16, dtype=dtype)
    offsets = np.linspace(0.15, 0.85, yarn_grid, dtype=dtype)
    snapshots = []
    for y in offsets:
        for z in offsets:
            positions = np.column_stack(
                [
                    x_nodes,
                    np.full_like(x_nodes, y),
                    np.full_like(x_nodes, z),
                ]
            )
            tangents = np.tile(np.array([[1.0, 0.0, 0.0]], dtype=dtype), (len(x_nodes), 1))
            ups = np.tile(np.array([[0.0, 0.0, 1.0]], dtype=dtype), (len(x_nodes), 1))
            sides = np.tile(np.array([[0.0, -1.0, 0.0]], dtype=dtype), (len(x_nodes), 1))
            snapshots.append(
                voxelizer.YarnSnapshot(
                    positions=positions,
                    tangents=tangents,
                    ups=ups,
                    sides=sides,
                    section=section,
                    translations=np.zeros((1, 3), dtype=dtype),
                )
            )
    return snapshots


def best_time(fn, repeat):
    timings = []
    result = None
    for _ in range(repeat):
        t0 = time.perf_counter()
        result = fn()
        timings.append(time.perf_counter() - t0)
    return min(timings), result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resolution", type=int, default=32)
    parser.add_argument("--yarn-grid", type=int, default=4, help="Creates yarn_grid^2 straight yarns")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--chunk-voxels", type=int, default=8192)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--dtype", choices=["float32", "float64"], default="float32")
    parser.add_argument("--include-torch", action="store_true")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    voxelizer = load_voxelizer_module()
    dtype = {"float32": np.float32, "float64": np.float64}[args.dtype]
    centers = make_centers(args.resolution, dtype)
    snapshots = make_snapshots(voxelizer, args.yarn_grid, dtype)

    print(
        f"voxels={centers.shape[0]:,} yarns={len(snapshots)} "
        f"dtype={args.dtype} workers={args.workers}"
    )

    unpruned_time, unpruned = best_time(
        lambda: voxelizer._classify_voxels_numpy(
            centers, snapshots, chunk=args.chunk_voxels, workers=args.workers,
            aabb_pruning=False
        ),
        args.repeat,
    )
    pruned_time, pruned = best_time(
        lambda: voxelizer._classify_voxels_numpy(
            centers, snapshots, chunk=args.chunk_voxels, workers=args.workers,
            aabb_pruning=True
        ),
        args.repeat,
    )

    np.testing.assert_array_equal(pruned, unpruned)
    speedup = unpruned_time / pruned_time if pruned_time > 0 else float("inf")
    print(f"numpy unpruned: {unpruned_time:.4f}s")
    print(f"numpy pruned:   {pruned_time:.4f}s  speedup={speedup:.2f}x")
    print(f"occupied cells: {int((pruned >= 0).sum()):,}")

    if not args.include_torch:
        return
    if voxelizer.torch is None:
        print("torch: not installed")
        return

    torch_mod = voxelizer.torch
    torch_dtype = {"float32": torch_mod.float32, "float64": torch_mod.float64}[args.dtype]
    packed = voxelizer._pack_yarns(snapshots, device=args.device, dtype=torch_dtype)
    centers_t = torch_mod.from_numpy(centers).to(device=args.device, dtype=torch_dtype)

    def classify_torch():
        ids = voxelizer._classify_voxels_torch(
            centers_t, packed, chunk=args.chunk_voxels, aabb_pruning=True
        )
        if args.device == "cuda":
            torch_mod.cuda.synchronize()
        elif args.device == "mps" and hasattr(torch_mod, "mps"):
            torch_mod.mps.synchronize()
        return ids

    torch_time, torch_ids = best_time(classify_torch, args.repeat)
    torch_np = torch_ids.detach().cpu().numpy()
    np.testing.assert_array_equal(torch_np, pruned)
    print(f"torch/{args.device} pruned: {torch_time:.4f}s")


if __name__ == "__main__":
    main()
