"""Voxelization entry points for modern textile models."""

from __future__ import annotations

from .compat import load_gpu_voxelizer


def voxelize_model_data(
    model,
    resolution=(64, 64, 64),
    *,
    backend: str = "numpy",
    device=None,
    dtype: str = "float32",
    **kwargs,
):
    """Voxelize a modern textile model and return ``VoxelGridData``."""
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
        **kwargs,
    )
