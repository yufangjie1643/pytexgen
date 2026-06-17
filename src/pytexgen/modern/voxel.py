"""Voxelization entry points for modern textile models."""

from __future__ import annotations

import os

from .compat import load_gpu_voxelizer


def voxelize_model_data(
    model,
    resolution=(64, 64, 64),
    *,
    backend: str = "numpy",
    device=None,
    dtype: str = "float32",
    workers: int | str | None = "auto",
    **kwargs,
):
    """Voxelize a modern textile model and return ``VoxelGridData``.

    ``workers="auto"`` uses a conservative numpy policy: serial for tiny grids
    and at most two workers for 64^3-and-larger grids. Current modern models are
    light enough that wider Python thread pools usually cost more than they save.
    """
    backend = backend.lower()
    if backend == "triton":
        raise NotImplementedError(
            "backend='triton' is reserved; use backend='torch' until a Triton "
            "kernel is implemented"
        )
    if backend not in {"numpy", "torch", "auto"}:
        raise ValueError('backend must be one of "numpy", "torch", "auto", or "triton"')
    textile = model.to_model() if hasattr(model, "to_model") else model
    gv = load_gpu_voxelizer()
    snapshots = textile.to_snapshots(gv)
    nx, ny, nz = (int(value) for value in resolution)
    output = "backend" if backend == "torch" else "numpy"
    workers = _resolve_modern_workers(backend, workers, resolution=(nx, ny, nz))
    return gv.voxelize_snapshots_data(
        snapshots,
        textile.aabb,
        nx=nx,
        ny=ny,
        nz=nz,
        backend=backend,
        device=device,
        dtype=dtype,
        output=output,
        verbose=False,
        workers=workers,
        **kwargs,
    )


def _resolve_modern_workers(
    backend: str,
    workers: int | str | None,
    resolution: tuple[int, int, int],
) -> int | None:
    """Resolve modern API worker policy before calling the legacy voxelizer."""
    if workers is None:
        return None
    if isinstance(workers, str):
        if workers.lower() != "auto":
            raise ValueError('workers must be an integer, None, or "auto"')
        return _auto_numpy_workers(resolution) if backend in {"numpy", "auto"} else None
    if workers < 1:
        raise ValueError("workers must be >= 1")
    return workers


def _auto_numpy_workers(resolution: tuple[int, int, int]) -> int:
    voxel_count = resolution[0] * resolution[1] * resolution[2]
    if voxel_count < 64 ** 3:
        return 1
    return max(1, min(os.cpu_count() or 1, 2))
