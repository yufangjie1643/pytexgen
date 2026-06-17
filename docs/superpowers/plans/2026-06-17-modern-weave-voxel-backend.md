# Modern Weave Voxel Backend Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the first `pytexgen.modern` slice for Python/NumPy/PyTorch woven composite modelling and structured voxel export.

**Architecture:** Add a pure-Python modern package under `src/pytexgen/modern` that produces snapshot-compatible yarn geometry and delegates structured voxel classification to the existing tested Python voxelizer. The first slice keeps legacy Core as an optional oracle only, while modern models can run without SWIG/Core.

**Tech Stack:** Python dataclasses, NumPy, PyTorch optional backend through existing `gpu_voxelizer`, Triton reserved as a future backend, `unittest` smoke tests run with `/home/yfj/.venvs/shared-py312-cu130/bin/python`.

---

## File Structure

- Create `src/pytexgen/modern/__init__.py`: public exports for modern models and voxelization.
- Create `src/pytexgen/modern/geometry.py`: `Section`, `YarnPath`, and `ModernTextileModel` containers plus validation.
- Create `src/pytexgen/modern/weave.py`: `PlainWeave2D`, `auto_binder_positions`, and `ShallowCrossLayerToLayer`.
- Create `src/pytexgen/modern/voxel.py`: `voxelize_model_data(...)` backend dispatch and snapshot conversion.
- Create `src/pytexgen/modern/export.py`: `.inp` export wrapper matching legacy structured ordering.
- Create `src/pytexgen/modern/compat.py`: lazy import helpers for `pytexgen.gpu_voxelizer`.
- Create `test_modern_weave_backend.py`: TDD tests for API, geometry, backend parity, export order, and shallow-cross subset.

## Task 1: Public API Skeleton And Geometry Containers

**Files:**
- Create: `test_modern_weave_backend.py`
- Create: `src/pytexgen/modern/__init__.py`
- Create: `src/pytexgen/modern/geometry.py`
- Create: `src/pytexgen/modern/weave.py`

- [ ] **Step 1: Write the failing import and geometry test**

Add this test to `test_modern_weave_backend.py`:

```python
import unittest

import numpy as np


class ModernWeaveApiTest(unittest.TestCase):
    def test_plain_weave_model_exposes_yarn_geometry_and_aabb(self):
        from pytexgen.modern import PlainWeave2D

        model = PlainWeave2D(width=2, height=2, spacing=1.0, thickness=0.2)
        textile = model.to_model()

        self.assertEqual(textile.name, "PlainWeave2D")
        self.assertEqual(len(textile.yarns), 4)
        np.testing.assert_allclose(textile.aabb, [[0.0, 0.0, 0.0], [2.0, 2.0, 0.2]])
        self.assertEqual(textile.yarns[0].positions.shape, (3, 3))
        self.assertEqual(textile.yarns[0].section.points.shape[1], 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
wsl -d Ubuntu -- bash -lc 'cd /mnt/d/code/pytexgen && PYTHONPATH=src /home/yfj/.venvs/shared-py312-cu130/bin/python test_modern_weave_backend.py -v'
```

Expected: FAIL or ERROR because `pytexgen.modern` does not exist.

- [ ] **Step 3: Implement minimal geometry and `PlainWeave2D` skeleton**

Create `src/pytexgen/modern/geometry.py` with:

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class Section:
    points: np.ndarray

    @classmethod
    def ellipse(cls, width: float, height: float, samples: int = 32) -> "Section":
        if width <= 0 or height <= 0:
            raise ValueError("section width and height must be positive")
        angles = np.linspace(0.0, 2.0 * np.pi, int(samples), endpoint=False)
        points = np.column_stack([
            0.5 * float(width) * np.cos(angles),
            0.5 * float(height) * np.sin(angles),
        ])
        points = np.vstack([points, points[:1]])
        return cls(points.astype(np.float64, copy=False))


@dataclass(frozen=True)
class YarnPath:
    positions: np.ndarray
    section: Section
    up: np.ndarray
    side: np.ndarray
    translations: np.ndarray

    def __post_init__(self) -> None:
        if self.positions.ndim != 2 or self.positions.shape[1] != 3:
            raise ValueError("positions must have shape (N, 3)")
        if self.positions.shape[0] < 2:
            raise ValueError("a yarn path needs at least two positions")


@dataclass(frozen=True)
class ModernTextileModel:
    name: str
    yarns: tuple[YarnPath, ...]
    aabb: np.ndarray

    def __post_init__(self) -> None:
        if self.aabb.shape != (2, 3):
            raise ValueError("aabb must have shape (2, 3)")
        if len(self.yarns) == 0:
            raise ValueError("model must contain at least one yarn")
```

Create `src/pytexgen/modern/weave.py` with a minimal `PlainWeave2D.to_model()` that builds two x-direction and two y-direction yarns for the test.

Create `src/pytexgen/modern/__init__.py` exporting `PlainWeave2D`, `Section`, `YarnPath`, and `ModernTextileModel`.

- [ ] **Step 4: Run test to verify it passes**

Run the same WSL command. Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add test_modern_weave_backend.py src/pytexgen/modern
git commit -m "feat: add modern weave geometry skeleton"
```

## Task 2: PlainWeave2D Pattern And Legacy-Order Geometry

**Files:**
- Modify: `test_modern_weave_backend.py`
- Modify: `src/pytexgen/modern/weave.py`

- [ ] **Step 1: Write the failing pattern test**

Add:

```python
    def test_plain_weave_swap_position_matches_texgen_cell_order(self):
        from pytexgen.modern import PlainWeave2D

        model = PlainWeave2D(width=2, height=2, spacing=1.0, thickness=0.2)
        self.assertEqual(model.cell(0, 0), ("y", "x"))
        model.swap_position(0, 0)
        self.assertEqual(model.cell(0, 0), ("x", "y"))
        textile = model.to_model()
        self.assertEqual(len(textile.yarns), 4)
        self.assertLess(textile.yarns[0].positions[0, 2], textile.yarns[2].positions[0, 2])
```

- [ ] **Step 2: Run test to verify it fails**

Expected: FAIL because `cell()` and `swap_position()` are not implemented.

- [ ] **Step 3: Implement pattern storage**

Implement `PlainWeave2D` with:

```python
class PlainWeave2D:
    def __init__(self, width: int, height: int, spacing: float, thickness: float, yarn_width: float = 0.8, yarn_height: float | None = None):
        self.width = int(width)
        self.height = int(height)
        self.spacing = float(spacing)
        self.thickness = float(thickness)
        self.yarn_width = float(yarn_width)
        self.yarn_height = float(yarn_height if yarn_height is not None else thickness / 2.0)
        self._pattern = [[["y", "x"] for _ in range(self.height)] for _ in range(self.width)]
```

Add `cell(x, y)` and `swap_position(x, y)` so the storage order is `pattern[x][y]`, matching `CTextileWeave::GetCell(x, y)`.

- [ ] **Step 4: Run tests to verify they pass**

Expected: both tests PASS.

- [ ] **Step 5: Commit**

```bash
git add test_modern_weave_backend.py src/pytexgen/modern/weave.py
git commit -m "feat: add modern plain weave pattern"
```

## Task 3: NumPy Voxel Data Through Existing VoxelGridData Contract

**Files:**
- Modify: `test_modern_weave_backend.py`
- Create: `src/pytexgen/modern/compat.py`
- Create: `src/pytexgen/modern/voxel.py`
- Modify: `src/pytexgen/modern/__init__.py`
- Modify: `src/pytexgen/modern/geometry.py`

- [ ] **Step 1: Write the failing NumPy voxel test**

Add:

```python
    def test_numpy_voxelize_model_data_returns_voxel_grid_contract(self):
        from pytexgen.modern import PlainWeave2D, voxelize_model_data

        model = PlainWeave2D(width=2, height=2, spacing=1.0, thickness=0.2)
        data = voxelize_model_data(model, resolution=(4, 4, 2), backend="numpy")

        self.assertEqual(data.resolution, (4, 4, 2))
        self.assertEqual(data.grid.shape, (2, 4, 4))
        self.assertEqual(data.yarn_id.shape, (32,))
        self.assertEqual(data.order, "ix + iy*nx + iz*nx*ny")
        self.assertGreaterEqual(int(data.material_id().max()), 1)
```

- [ ] **Step 2: Run test to verify it fails**

Expected: FAIL because `voxelize_model_data` does not exist.

- [ ] **Step 3: Implement lazy voxelizer compatibility**

Create `compat.py`:

```python
def load_gpu_voxelizer():
    try:
        from pytexgen import gpu_voxelizer
        return gpu_voxelizer
    except Exception as first_error:
        try:
            from TexGen import gpu_voxelizer
            return gpu_voxelizer
        except Exception as second_error:
            raise ImportError("modern voxelization requires pytexgen.gpu_voxelizer") from second_error
```

Add a `ModernTextileModel.to_snapshots(gpu_voxelizer)` method that converts each `YarnPath` into `gpu_voxelizer.YarnSnapshot` with normalized tangents, constant up vectors, side vectors, section points, and translations.

Create `voxel.py`:

```python
from .compat import load_gpu_voxelizer


def voxelize_model_data(model, resolution=(64, 64, 64), backend="numpy", device=None, dtype="float32", **kwargs):
    gv = load_gpu_voxelizer()
    textile = model.to_model() if hasattr(model, "to_model") else model
    snapshots = textile.to_snapshots(gv)
    return gv.voxelize_snapshots_data(
        snapshots,
        textile.aabb,
        nx=int(resolution[0]),
        ny=int(resolution[1]),
        nz=int(resolution[2]),
        backend=backend,
        device=device,
        dtype=dtype,
        output="backend" if backend == "torch" else "numpy",
        **kwargs,
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Expected: all modern tests PASS.

- [ ] **Step 5: Commit**

```bash
git add test_modern_weave_backend.py src/pytexgen/modern
git commit -m "feat: voxelize modern models with numpy"
```

## Task 4: Torch Backend Parity And Triton Reservation

**Files:**
- Modify: `test_modern_weave_backend.py`
- Modify: `src/pytexgen/modern/voxel.py`

- [ ] **Step 1: Write failing torch parity and triton reservation tests**

Add:

```python
    def test_torch_backend_matches_numpy_when_available(self):
        import torch
        from pytexgen.modern import PlainWeave2D, voxelize_model_data

        model = PlainWeave2D(width=2, height=2, spacing=1.0, thickness=0.2)
        numpy_data = voxelize_model_data(model, resolution=(4, 4, 2), backend="numpy")
        torch_data = voxelize_model_data(model, resolution=(4, 4, 2), backend="torch", device="cpu")

        self.assertEqual(torch_data.storage, "torch")
        np.testing.assert_array_equal(torch_data.to_numpy().yarn_id, numpy_data.yarn_id)

    def test_triton_backend_is_reserved_until_kernel_exists(self):
        from pytexgen.modern import PlainWeave2D, voxelize_model_data

        with self.assertRaises(NotImplementedError):
            voxelize_model_data(PlainWeave2D(2, 2, 1.0, 0.2), resolution=(4, 4, 2), backend="triton")
```

- [ ] **Step 2: Run tests to verify triton test fails**

Expected: FAIL because `backend="triton"` is not handled explicitly.

- [ ] **Step 3: Implement backend validation**

In `voxelize_model_data`, normalize `backend = backend.lower()` and add:

```python
if backend == "triton":
    raise NotImplementedError("backend='triton' is reserved; use backend='torch' until a Triton kernel is implemented")
if backend not in {"numpy", "torch", "auto"}:
    raise ValueError('backend must be one of "numpy", "torch", "auto", or "triton"')
```

- [ ] **Step 4: Run tests to verify they pass**

Expected: all modern tests PASS.

- [ ] **Step 5: Commit**

```bash
git add test_modern_weave_backend.py src/pytexgen/modern/voxel.py
git commit -m "feat: add modern torch parity and triton reservation"
```

## Task 5: Structured `.inp` Export

**Files:**
- Modify: `test_modern_weave_backend.py`
- Create: `src/pytexgen/modern/export.py`
- Modify: `src/pytexgen/modern/__init__.py`

- [ ] **Step 1: Write failing `.inp` export order test**

Add:

```python
    def test_write_inp_from_voxel_data_uses_legacy_node_and_element_order(self):
        from pathlib import Path
        import tempfile

        from pytexgen.modern import PlainWeave2D, voxelize_model_data, write_inp_from_voxel_data

        data = voxelize_model_data(PlainWeave2D(2, 2, 1.0, 0.2), resolution=(2, 2, 1), backend="numpy")
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "modern.inp"
            write_inp_from_voxel_data(data, out, textile_name="ModernPlain")
            text = out.read_text()

        self.assertIn("*Node", text)
        self.assertIn("*Element, type=C3D8R", text)
        self.assertIn("1, 2, 5, 4, 1, 11, 14, 13, 10", text)
```

- [ ] **Step 2: Run test to verify it fails**

Expected: FAIL because `write_inp_from_voxel_data` does not exist.

- [ ] **Step 3: Implement export wrapper**

Create `export.py`:

```python
from pathlib import Path

from .compat import load_gpu_voxelizer


def write_inp_from_voxel_data(data, path, textile_name="ModernTextile", progress=False):
    gv = load_gpu_voxelizer()
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    numpy_data = data.to_numpy() if hasattr(data, "to_numpy") else data
    nx, ny, nz = numpy_data.resolution
    lo = numpy_data.aabb[0]
    hi = numpy_data.aabb[1]
    gv._write_inp(out, lo, hi, nx, ny, nz, numpy_data.yarn_id, textile_name=textile_name, progress=progress)
    return out
```

Export it in `__init__.py`.

- [ ] **Step 4: Run tests to verify they pass**

Expected: all modern tests PASS.

- [ ] **Step 5: Commit**

```bash
git add test_modern_weave_backend.py src/pytexgen/modern
git commit -m "feat: export modern voxel data to inp"
```

## Task 6: ShallowCrossLayerToLayer Subset

**Files:**
- Modify: `test_modern_weave_backend.py`
- Modify: `src/pytexgen/modern/weave.py`
- Modify: `src/pytexgen/modern/__init__.py`

- [ ] **Step 1: Write failing shallow-cross tests**

Add:

```python
    def test_shallow_cross_auto_binder_positions_match_current_script_rules(self):
        from pytexgen.modern import auto_binder_positions

        positions = auto_binder_positions("straight", num_x_yarns=2, num_y_yarns=4, z_layers=5, binder_depth=3)
        self.assertEqual(len(positions), 8)
        self.assertEqual(positions[:4], [(0, 0, 0), (1, 0, 1), (2, 0, 2), (3, 0, 1)])
        self.assertEqual(positions[4:], [(0, 1, 2), (1, 1, 1), (2, 1, 0), (3, 1, 1)])

    def test_shallow_cross_subset_builds_snapshot_compatible_model(self):
        from pytexgen.modern import ShallowCrossLayerToLayer

        model = ShallowCrossLayerToLayer(
            num_x_yarns=2,
            num_y_yarns=4,
            x_spacing=1.4,
            y_spacing=2.2,
            z_layers=5,
            binder_depth=3,
        )
        textile = model.to_model()
        self.assertEqual(textile.name, "ShallowCrossLayerToLayer")
        self.assertGreaterEqual(len(textile.yarns), 6)
        self.assertEqual(textile.aabb.shape, (2, 3))
```

- [ ] **Step 2: Run tests to verify they fail**

Expected: FAIL because shallow-cross API is missing.

- [ ] **Step 3: Implement shallow-cross subset**

Copy the existing binder-position rule from `script/shallow_cross_layer_to_layer.py` into modern `auto_binder_positions`. Implement `ShallowCrossLayerToLayer.to_model()` by generating straight x/y layer yarns plus binder yarn paths from the auto positions. Use the same model units as the script: x yarns are y-direction count, y yarns are x-direction count, and z layers determine the vertical AABB.

- [ ] **Step 4: Run tests to verify they pass**

Expected: all modern tests PASS.

- [ ] **Step 5: Commit**

```bash
git add test_modern_weave_backend.py src/pytexgen/modern/weave.py src/pytexgen/modern/__init__.py
git commit -m "feat: add modern shallow cross subset"
```

## Task 7: Final Verification And Documentation

**Files:**
- Modify: `README.md`
- Modify: `README_pypi.md`
- Optional modify: `docs/voxel_backends.md`

- [ ] **Step 1: Add short modern backend documentation**

Document this example:

```python
from pytexgen.modern import PlainWeave2D, voxelize_model_data

model = PlainWeave2D(width=4, height=4, spacing=1.0, thickness=0.2)
data = voxelize_model_data(model, resolution=(64, 64, 32), backend="torch", device="cuda")
data.save_npz("modern_plain_weave.npz")
```

- [ ] **Step 2: Run focused tests**

```bash
wsl -d Ubuntu -- bash -lc 'cd /mnt/d/code/pytexgen && PYTHONPATH=src /home/yfj/.venvs/shared-py312-cu130/bin/python test_modern_weave_backend.py -v'
wsl -d Ubuntu -- bash -lc 'cd /mnt/d/code/pytexgen && /home/yfj/.venvs/shared-py312-cu130/bin/python test_gpu_voxelizer_backends.py -v'
git diff --check
```

Expected: modern tests pass, existing voxelizer backend tests pass, and diff check is clean.

- [ ] **Step 3: Commit docs and final fixes**

```bash
git add README.md README_pypi.md docs/voxel_backends.md test_modern_weave_backend.py src/pytexgen/modern
git commit -m "docs: describe modern weave voxel backend"
```

## Execution Notes

Use this Python for all local Python commands:

```bash
wsl -d Ubuntu -- /home/yfj/.venvs/shared-py312-cu130/bin/python
```

Do not stage unrelated local files such as `.agents/`, `.codegraph/`, `create_rve/`, `params.json`, `uv.lock`, or `uv.lock.local-before-rebase`.
