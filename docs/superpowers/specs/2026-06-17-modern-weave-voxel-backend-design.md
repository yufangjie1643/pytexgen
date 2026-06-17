# Modern Weave Voxel Backend Design

## Goal

Add a small, cross-platform modern modelling and voxelization layer for woven composite workflows without depending on the legacy C/SWIG/Core build path for new compute code.

## Scope

The first migration slice covers `PlainWeave2D`, a simplified `ShallowCrossLayerToLayer`, structured voxelization, direct voxel data export, and Abaqus `.inp` export. It does not migrate GUI, TG3 compatibility, p4est/octree meshes, full TexGen section interpolation, full binder shaping, or the complete `CTextileLayerToLayer` API.

## Architecture

The new layer lives under `pytexgen.modern` and treats the existing C++ Core as a legacy oracle and optional compatibility source, not as the implementation substrate. Model builders emit flat yarn geometry arrays compatible with the existing `SnapshotBundle` and `VoxelGridData` conventions. Voxelization backends consume those arrays through a stable API:

```text
PlainWeave2D / ShallowCrossLayerToLayer
  -> ModernTextileModel
  -> SnapshotBundle-compatible arrays
  -> voxelize_model_data(...)
  -> VoxelGridData
  -> .npz / .inp / downstream solver
```

## Package Layout

- `src/pytexgen/modern/__init__.py`: public modern API exports.
- `src/pytexgen/modern/geometry.py`: `Section`, `YarnPath`, `ModernTextileModel`, and bundle conversion helpers.
- `src/pytexgen/modern/weave.py`: `PlainWeave2D`, `ShallowCrossLayerToLayer`, and binder-position generation.
- `src/pytexgen/modern/voxel.py`: `voxelize_model_data(...)` with `numpy`, `torch`, and reserved `triton` backend dispatch.
- `src/pytexgen/modern/export.py`: `.inp` writer matching the existing structured node and element order.
- `src/pytexgen/modern/compat.py`: adapters to existing `pytexgen.gpu_voxelizer.VoxelGridData`.
- `test_modern_weave_backend.py`: oracle and backend-alignment tests.

## Public API

The first API is deliberately small:

```python
from pytexgen.modern import PlainWeave2D, ShallowCrossLayerToLayer, voxelize_model_data

model = PlainWeave2D(width=4, height=4, spacing=1.0, thickness=0.2)
model.swap_position(0, 3)
data = voxelize_model_data(model, resolution=(32, 32, 16), backend="torch", device="cuda")
data.save_npz("plain_weave.npz")
```

Legacy-style aliases such as `SwapPosition` may be provided as thin wrappers where they help migration, but new code should use `snake_case`.

## Result Alignment

The migration aligns with TexGen 3.13.1 and current `pytexgen.gpu_voxelizer` contracts:

- `VoxelGridData.resolution == (nx, ny, nz)`.
- `VoxelGridData.grid.shape == (nz, ny, nx)`.
- Flat yarn ids use `ix + iy*nx + iz*nx*ny`.
- Matrix/background voxels use `yarn_id == -1`.
- `material_id()` maps matrix to `0` and yarns to `1..N`.
- `.inp` nodes and hex elements follow the legacy rectangular voxel ordering from `CVoxelMesh::OutputHexElements`.
- `PlainWeave2D` pattern cell ordering follows `CTextileWeave::GetCell(x, y)`.

Legacy Core from the WSL environment `/home/yfj/.venvs/shared-py312-cu130/bin/python` is the oracle for small-grid comparisons where Core import is available. Pure synthetic tests cover environments without Core.

## Backends

`backend="numpy"` is the required portable default. It uses NumPy arrays and remains the fallback on all platforms.

`backend="torch"` is the required GPU-capable backend. It should produce exactly the same yarn ids as NumPy for supported models and resolutions. CUDA is preferred when available; CPU torch is acceptable for parity tests.

`backend="triton"` is reserved in the API but initially raises a clear `ImportError` or `NotImplementedError` unless a Triton kernel is implemented. Triton work starts only after a torch benchmark shows insufficient GPU performance on the agreed workloads.

## Error Handling

Model constructors validate positive dimensions, positive spacing, positive section sizes, and binder offsets within the layer range. Backend selection errors name the valid backends and whether torch or triton is installed. Unsupported legacy features fail explicitly instead of silently approximating them.

## Testing Strategy

Development follows TDD. The first failing tests cover:

- `PlainWeave2D` yarn count, AABB, and pattern swaps against legacy Core on a small model.
- `voxelize_model_data(..., backend="numpy")` output shape, flat order, and material id mapping.
- `backend="torch"` parity with NumPy when torch is installed.
- `.inp` writer node and element order on a `2x2x1` structured grid.
- `ShallowCrossLayerToLayer` binder-position generation and basic yarn/bundle shape for the current SiC/SiC parameter subset.

The WSL Python command for local verification is:

```bash
wsl -d Ubuntu -- /home/yfj/.venvs/shared-py312-cu130/bin/python test_modern_weave_backend.py
```

## Migration Path

Existing scripts keep using legacy Core until each model has a tested modern equivalent. New scripts may opt in through `pytexgen.modern`. After the first slice is stable, `script/sic_sic_shallow_cross_straight.py` can gain a modern-mode flag that writes both legacy and modern voxel outputs for direct comparison.
