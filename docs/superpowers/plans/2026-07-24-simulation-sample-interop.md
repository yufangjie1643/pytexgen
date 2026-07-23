# GPU Simulation Sample Interoperability Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a validated, zero-copy-first `SimulationSample` contract that exposes PyTexGen voxel, sparse direction, C21 stiffness, and material-table data to GPU solvers and tensor frameworks, then prove it with a same-device Voxel-ACDM adapter and checked transfer benchmark.

**Architecture:** Keep `VoxelGridData`, `SparseOrientationField`, and `SparseStiffnessField` as the array owners. Add immutable composition and material-table types in `simulation_sample.py`, consolidated persistence in `simulation_io.py`, and explicit solver conversion only at the `acdm_solver.py` boundary. Native fields are views or original arrays; dense layouts, backend changes, dtype changes, and device changes require an explicit allocating operation.

**Tech Stack:** Python dataclasses, NumPy, optional PyTorch/DLPack/CUDA, existing TexGen SWIG bindings, Voxel-ACDM's optional Torch/Triton API, `unittest`, scikit-build-core/CMake.

## Global Constraints

- Work on `agent/gpu-simulation-training`; do not modify the sibling Voxel-ACDM checkout.
- Preserve all existing imports and public behavior. In particular, keep
  `VoxelGridData.to_dlpack()` operational as a documented legacy helper.
- Treat engineering Voigt order as `(xx, yy, zz, yz, xz, xy)` and C21 as the
  existing row-major upper triangle.
- Treat grid shape as `(Nz, Ny, Nx)` and flat order as
  `ix + iy*nx + iz*nx*ny`.
- Never infer stiffness units or equate yarn IDs with material IDs.
- Keep Torch optional. NumPy and Torch-CPU tests run everywhere; CUDA and
  external-solver tests skip only when their dependency is unavailable.
- No `array(..., copy=False)` path may clone, densify, relayout, cast, or move
  data. Such requests raise `ValueError`.
- Each task starts with a failing test, makes the smallest production change
  needed, reruns focused tests, and creates one focused commit.

## File Map

- Create `TexGen/simulation_sample.py`: immutable material/sample contracts,
  validation, stable field registry, conversions, and one-call construction.
- Create `TexGen/simulation_io.py`: versioned directory/NPZ persistence with
  canonical sparse arrays and manifest aliases.
- Modify `TexGen/material_fields.py`: preserve caller-supplied material IDs.
- Modify `TexGen/gpu_voxelizer.py`: document the legacy DLPack capsule boundary.
- Modify `TexGen/acdm_solver.py`: backend-preserving phase IDs, explicit phase
  tables, host-pack policy, and same-device anisotropic solve adapter.
- Modify `Python/CMakeLists.txt`: install the new public modules.
- Create `test_simulation_sample.py`, `test_simulation_io.py`,
  `test_acdm_sample_adapter.py`, and `test_simulation_sample_integration.py`;
  extend the existing focused material, voxel, and ACDM tests.
- Create `bench_simulation_interop.py` and
  `test_simulation_interop_benchmark.py`: correctness-gated transfer and
  speed measurements against the existing TexGen CPU reference.
- Update `README.md`, `README_pypi.md`, `docs/voxel_backends.md`, and
  `docs/voxel_acdm_interface.md`.

## Public API Decisions

The implementation exports these types and functions:

```python
from pytexgen.simulation_sample import (
    MaterialTable,
    SimulationSample,
    voxelize_textile_simulation_sample,
)
from pytexgen.simulation_io import (
    load_simulation_sample,
    save_simulation_sample,
)
from pytexgen.acdm_solver import (
    solve_acdm_anisotropic_from_sample,
    solve_acdm_isotropic_from_voxel_data,
    to_acdm_phase_ids,
)
```

`SimulationSample.array()` supports the version-1 field names in the approved
design. `layout="native"` is zero-copy for resident fields.
`layout="acdm"` is accepted only for `"stiffness.yarn_c21"` and produces
`(1, 6, 6, Nz, Ny, Nx)`; because it densifies, callers must pass `copy=True`.
The derived `"voxel.occupancy"` and `"voxel.material_id"` fields also require
`copy=True`. `as_dict(copy=False)` returns only resident fields, while
`as_dict(copy=True)` includes resident and derived native fields.

Persistence uses one canonical sparse-index array and one canonical sparse
yarn-ID array. Manifest aliases point both orientation and stiffness names to
those files; the values are never written twice.

---

### Task 1: Add the immutable material table

**Files:**

- Create: `TexGen/simulation_sample.py`
- Create: `test_simulation_sample.py`

- [ ] **Step 1: Write the failing validation and identity tests**

Create a source-loader fixture matching the lightweight stubs already used by
`test_material_fields.py`, then add `MaterialTableTest`:

```python
class MaterialTableTest(unittest.TestCase):
    def test_accepts_explicit_non_dense_ids_without_copy(self):
        c21 = np.stack((isotropic_stiffness_c21(3.0, 0.3),
                        isotropic_stiffness_c21(70.0, 0.2)))
        ids = np.array([0, 7], dtype=np.int32)
        table = MaterialTable(c21=c21, material_ids=ids, unit="GPa",
                              names=("matrix", "carbon"))

        self.assertIs(table.c21, c21)
        self.assertIs(table.material_ids, ids)
        self.assertEqual(table.row_for_id(7), 1)
        self.assertTrue(np.shares_memory(table.c21_for_id(0), c21))

    def test_rejects_missing_matrix_duplicate_ids_and_empty_unit(self):
        c21 = np.ones((2, 21), dtype=np.float64)
        with self.assertRaisesRegex(ValueError, "material ID 0"):
            MaterialTable(c21, np.array([1, 2]), "GPa")
        with self.assertRaisesRegex(ValueError, "unique"):
            MaterialTable(c21, np.array([0, 0]), "GPa")
        with self.assertRaisesRegex(ValueError, "unit"):
            MaterialTable(c21, np.array([0, 1]), "")

    def test_positive_definite_validation_is_opt_in(self):
        c21 = np.zeros((1, 21), dtype=np.float64)
        MaterialTable(c21, np.array([0]), "Pa")
        with self.assertRaisesRegex(ValueError, "positive definite"):
            MaterialTable(c21, np.array([0]), "Pa",
                          validate_positive_definite=True)
```

- [ ] **Step 2: Run the focused tests and observe the missing module**

Run:

```bash
python -m unittest test_simulation_sample.MaterialTableTest -v
```

Expected: FAIL because `TexGen.simulation_sample` does not exist.

- [ ] **Step 3: Implement `MaterialTable` and shared conversion helpers**

Add a frozen dataclass with this signature:

```python
@dataclass(frozen=True)
class MaterialTable:
    c21: Any
    material_ids: Any
    unit: str
    names: Optional[Tuple[str, ...]] = None
    validate_positive_definite: bool = field(
        default=False, repr=False, compare=False
    )

    @property
    def storage(self) -> str:
        return "torch" if _is_torch_tensor(self.c21) else "numpy"

    @property
    def device(self) -> str:
        return str(self.c21.device) if _is_torch_tensor(self.c21) else "cpu"

    def row_for_id(self, material_id: int) -> int:
        matches = self.material_ids == int(material_id)
        indices = matches.nonzero(as_tuple=False).reshape(-1) \
            if _is_torch_tensor(matches) else np.flatnonzero(matches)
        if int(indices.shape[0]) != 1:
            raise KeyError(f"unknown material ID {material_id}")
        return int(indices[0].item() if _is_torch_tensor(indices)
                   else indices[0])

    def c21_for_id(self, material_id: int):
        return self.c21[self.row_for_id(material_id)]

    def to(self, storage=None, *, device=None, dtype=None, copy=False):
        target = self.storage if storage is None else str(storage).lower()
        return MaterialTable(
            c21=_convert_float_array(
                self.c21, target, device=device, dtype=dtype, copy=copy
            ),
            material_ids=_convert_integer_array(
                self.material_ids, target, device=device, copy=copy
            ),
            unit=self.unit,
            names=self.names,
            validate_positive_definite=self.validate_positive_definite,
        )
```

`__post_init__` must validate `(M, 21)`, a one-dimensional integer ID array,
same backend/device, finite C21 values, unique non-negative IDs, exactly one
ID `0`, non-empty stripped unit, and name count. Reuse `unpack_c21()` for the
opt-in eigenvalue check. Copy the small `_is_torch_tensor`,
`_convert_float_array`, and `_convert_integer_array` behavior into this module
without importing private names from `material_fields.py`. A NumPy target
accepts only `device=None` or `"cpu"`; reject a CUDA device instead of silently
ignoring it.

- [ ] **Step 4: Test NumPy/Torch conversion behavior**

Add tests proving:

- `.to(copy=False)` returns the original C21 and ID objects;
- `.to(copy=True)` clones both arrays;
- NumPy-to-Torch preserves integer IDs and uses the requested floating dtype;
- Torch-to-NumPy rejects a CUDA target request and produces CPU NumPy only.

Run:

```bash
python -m unittest test_simulation_sample.MaterialTableTest -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add TexGen/simulation_sample.py test_simulation_sample.py
git commit -m "feat: add simulation material tables"
```

---

### Task 2: Add `SimulationSample` validation and resident field registry

**Files:**

- Modify: `TexGen/simulation_sample.py`
- Modify: `test_simulation_sample.py`

- [ ] **Step 1: Add a reusable sample fixture and failing validation tests**

Build a `2 x 2 x 1` NumPy fixture with matrix ID `0`, yarn material IDs `7`
and `9`, and matching sparse orientation/stiffness arrays. Test:

```python
sample = make_sample()
self.assertIs(sample.voxels, voxels)
self.assertIs(sample.orientation, orientation)
self.assertIs(sample.stiffness, stiffness)
self.assertIs(sample.materials, materials)
self.assertEqual(sample.metadata["source"], "unit-test")
self.assertIs(sample.array("orientation.primary"),
              orientation.orientation1)
self.assertIs(sample.array("stiffness.yarn_c21"),
              stiffness.yarn_c21)
self.assertIs(sample.array("material.c21"), materials.c21)
np.testing.assert_array_equal(
    sample.array("voxel.yarn_id").reshape(-1), voxels.yarn_id
)
self.assertTrue(np.shares_memory(
    sample.array("voxel.yarn_id"), voxels.yarn_id
))
```

Add one test per rejected relationship: grid shape, order, backend, Torch
device, sparse indices, sparse yarn IDs, unknown stiffness material ID, matrix
C21 mismatch, and unit mismatch. Mutating the original metadata after
construction must not change `sample.metadata`; attempting to mutate nested
sample metadata must raise `TypeError`. When
`voxels.sparse_orientation` is present and the constructor orientation is
`None`, adopt that object with `object.__setattr__`. If both are present,
require object identity; this prevents two competing direction owners.

- [ ] **Step 2: Run and confirm the missing sample type**

Run:

```bash
python -m unittest test_simulation_sample.SimulationSampleValidationTest -v
```

Expected: FAIL because `SimulationSample` is undefined.

- [ ] **Step 3: Implement immutable metadata and cross-container validation**

Add:

```python
@dataclass(frozen=True)
class SimulationSample:
    voxels: VoxelGridData
    materials: MaterialTable
    orientation: Optional[SparseOrientationField] = None
    stiffness: Optional[SparseStiffnessField] = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def storage(self) -> str:
        return self.voxels.storage

    @property
    def device(self) -> str:
        return self.voxels.device

    @property
    def field_names(self) -> Tuple[str, ...]:
        return tuple(name for name in FIELD_ORDER
                     if self._field_available(name))
```

Use `json.dumps(..., allow_nan=False)` followed by `json.loads()` to validate
and detach metadata. Recursively convert dictionaries to `MappingProxyType`
and lists to tuples. Implement the inverse `_thaw_metadata()` for persistence.

Validate object types first. Then validate `(Nz, Ny, Nx)`, order, storage, and
device across all resident containers. This includes `VoxelGridData.yarn_id`,
`aabb`, optional centers, optional dense directions, and its sparse direction
field. When both sparse fields exist, compare indices and yarn IDs exactly
without moving them to CPU. Require every sparse stiffness material ID to
exist in `MaterialTable`, require
`stiffness.matrix_c21` to match material ID `0`, and require exact unit
equality. Use `rtol=1e-5, atol=1e-6` for FP32 and
`rtol=1e-10, atol=1e-12` for FP64.

Implement `_resident_field(name)` with these mappings:

```text
voxel.yarn_id                  -> voxels.grid
orientation.voxel_indices     -> orientation.voxel_indices
orientation.yarn_ids          -> orientation.yarn_ids
orientation.primary           -> orientation.orientation1
orientation.secondary         -> orientation.orientation2
stiffness.matrix_c21           -> stiffness.matrix_c21
stiffness.voxel_indices       -> stiffness.voxel_indices
stiffness.yarn_ids            -> stiffness.yarn_ids
stiffness.material_ids        -> stiffness.material_ids
stiffness.yarn_c21             -> stiffness.yarn_c21
material.ids                   -> materials.material_ids
material.c21                   -> materials.c21
```

The two derived voxel fields are available only when their inputs exist:
occupancy requires voxels; physical material IDs require stiffness.

- [ ] **Step 4: Run focused tests**

Run:

```bash
python -m unittest \
  test_simulation_sample.SimulationSampleValidationTest -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add TexGen/simulation_sample.py test_simulation_sample.py
git commit -m "feat: validate simulation sample components"
```

---

### Task 3: Implement explicit field, conversion, and DLPack semantics

**Files:**

- Modify: `TexGen/simulation_sample.py`
- Modify: `TexGen/gpu_voxelizer.py`
- Modify: `test_simulation_sample.py`
- Modify: `test_gpu_voxelizer_backends.py`

- [ ] **Step 1: Write failing native/derived/copy tests**

Test all resident field names for object identity or shared storage. Add:

```python
with self.assertRaisesRegex(ValueError, "copy=True"):
    sample.array("voxel.occupancy")
with self.assertRaisesRegex(ValueError, "copy=True"):
    sample.array("voxel.material_id")
with self.assertRaisesRegex(ValueError, "copy=True"):
    sample.array("stiffness.yarn_c21", layout="acdm")

material_grid = sample.array("voxel.material_id", copy=True)
np.testing.assert_array_equal(
    material_grid,
    np.array([[[0, 7], [9, 0]]], dtype=np.int32),
)
dense = sample.array("stiffness.yarn_c21", layout="acdm", copy=True)
self.assertEqual(dense.shape, (1, 6, 6, 1, 2, 2))
np.testing.assert_allclose(dense, sample.stiffness.to_acdm(batch=True))
```

Also test unknown fields, unavailable optional fields, unsupported layout,
`as_dict()` resident-only behavior, and `as_dict(copy=True)` derived fields.

- [ ] **Step 2: Write failing conversion and DLPack alias tests**

For Torch CPU:

```python
torch_sample = sample.to("torch", dtype=torch.float64)
field = torch_sample.array("stiffness.yarn_c21")
shared = torch.from_dlpack(field)
self.assertEqual(shared.data_ptr(), field.data_ptr())
self.assertIs(torch_sample.to(copy=False), torch_sample)
self.assertNotEqual(
    torch_sample.to(copy=True).materials.c21.data_ptr(),
    torch_sample.materials.c21.data_ptr(),
)
self.assertFalse(hasattr(torch_sample, "__dlpack__"))
```

On CUDA, repeat pointer checks for every resident public Torch field. Add a
non-default-stream test that writes the field on a producer stream, imports
the field with `torch.from_dlpack(field)` on a consumer stream, records the
consumer stream, synchronizes once at the assertion boundary, and verifies the
written values. Skip only if CUDA is unavailable.

- [ ] **Step 3: Run and observe copy/layout failures**

Run:

```bash
python -m unittest \
  test_simulation_sample.SimulationSampleFieldTest \
  test_simulation_sample.SimulationSampleDLPackTest -v
```

Expected: FAIL because `array`, `as_dict`, and `to` are incomplete.

- [ ] **Step 4: Implement field access and conversion**

Implement:

```python
def array(self, name: str, *, layout: str = "native", copy: bool = False):
    if name not in self.field_names:
        available = ", ".join(self.field_names)
        raise KeyError(f"unknown or unavailable field {name!r}; "
                       f"available fields: {available}")
    if layout == "native":
        resident = self._resident_field(name)
        if resident is not None:
            return _copy_array(resident) if copy else resident
        if not copy:
            raise ValueError(f"{name!r} is derived; pass copy=True")
        return self._materialize_voxel_field(name)
    if layout == "acdm" and name == "stiffness.yarn_c21":
        if not copy:
            raise ValueError("ACDM layout allocates; pass copy=True")
        return self.stiffness.to_acdm(batch=True)
    raise ValueError(f"layout {layout!r} is not supported for {name!r}")
```

`_materialize_voxel_field("voxel.occupancy")` computes `yarn_id >= 0`.
`_materialize_voxel_field("voxel.material_id")` fills a current-device
integer grid with zero and scatters `stiffness.material_ids` at
`stiffness.voxel_indices`. It never calls the existing yarn-based
`VoxelGridData.material_id()`.

`to()` must return `self` only when storage, device, dtype, and `copy=False`
require no conversion. Otherwise convert voxels first. If orientation is the
voxel container's sparse orientation, reuse the converted voxel container's
orientation. Convert stiffness floating arrays and material IDs while reusing
the converted orientation's sparse indices/yarn IDs; this avoids two copies of
the sparse topology. Convert a standalone orientation independently only when
the voxel container did not own it. Convert the material table and reuse
immutable metadata. `as_dict(copy=False)` returns only
`_resident_field(name) is not None`; `copy=True` returns every available
native field.

- [ ] **Step 5: Mark the old DLPack method as legacy without warning**

Change only the `VoxelGridData.to_dlpack()` docstring to say it is a
compatibility capsule helper and recommend:

```python
field = sample.array("voxel.yarn_id")
tensor = torch.from_dlpack(field)
```

Do not emit a runtime deprecation warning.

- [ ] **Step 6: Run focused tests**

Run:

```bash
python -m unittest \
  test_simulation_sample \
  test_gpu_voxelizer_backends.VoxelizerBackendTest -v
```

Expected: PASS, with CUDA tests skipped only when CUDA is absent.

- [ ] **Step 7: Commit**

```bash
git add TexGen/simulation_sample.py TexGen/gpu_voxelizer.py \
        test_simulation_sample.py test_gpu_voxelizer_backends.py
git commit -m "feat: expose zero-copy simulation fields"
```

---

### Task 4: Preserve explicit material IDs in stiffness construction

**Files:**

- Modify: `TexGen/material_fields.py`
- Modify: `TexGen/simulation_sample.py`
- Modify: `test_material_fields.py`
- Modify: `test_simulation_sample.py`

- [ ] **Step 1: Write failing explicit-ID tests**

Extend `build_stiffness_field()` tests so the default yarn uses ID `7`, yarn
ID `4` uses material ID `19`, and the resulting sparse `material_ids` are
exactly `[7, 19]` in voxel order. Add rejection tests for ID `0`, negative
IDs, duplicate conflicting assignments, a material ID without stiffness, and
a stiffness override without a material ID.

Add a one-call sample test:

```python
sample = voxelize_textile_simulation_sample(
    textile,
    materials=table,
    default_yarn_material_id=7,
    yarn_material_id_by_id={4: 19},
    nx=2, ny=2, nz=1,
    backend="numpy",
)
self.assertIsInstance(sample, SimulationSample)
self.assertEqual(set(sample.stiffness.material_ids.tolist()), {7, 19})
```

Mock `voxelize_textile_data()` in the unit test; reserve a real textile for
Task 8 integration coverage.

- [ ] **Step 2: Run and confirm the new arguments are absent**

Run:

```bash
python -m unittest \
  test_material_fields.StiffnessRotationTest \
  test_simulation_sample.SimulationSampleWorkflowTest -v
```

Expected: FAIL with unexpected keyword arguments or missing workflow function.

- [ ] **Step 3: Extend `build_stiffness_field()` additively**

Add keyword-only arguments:

```python
default_yarn_material_id: Optional[int] = None,
yarn_material_id_by_id: Optional[Mapping[int, int]] = None,
```

Compatibility rules:

- with neither new argument, retain the current generated IDs (`1` for the
  default and `2..` for sorted overrides);
- when either new argument is present, require IDs for every supplied
  stiffness source;
- explicit yarn material IDs must be positive integers;
- use the explicit ID values when filling `material_ids`;
- multiple yarns may intentionally share one material ID;
- one yarn cannot receive conflicting default/override assignments.

When constructing `SparseStiffnessField`, reuse
`orientation.voxel_indices` and `orientation.yarn_ids` directly instead of
calling `_copy_array()`. Add identity assertions to the test. The fields are
frozen and treat these arrays as immutable, so sharing removes redundant
sparse storage without changing values or public behavior.

Extend `voxelize_textile_material_fields()` with the same two explicit-ID
keywords plus `unit: Optional[str] = None`, and forward all three to
`build_stiffness_field()`. Existing callers that omit them retain their current
generated IDs and `unit=None`.

- [ ] **Step 4: Implement the one-call sample workflow**

Add:

```python
def voxelize_textile_simulation_sample(
    textile,
    *,
    materials: MaterialTable,
    default_yarn_material_id: Optional[int] = None,
    yarn_material_id_by_id: Optional[Mapping[int, int]] = None,
    metadata: Optional[Mapping[str, Any]] = None,
    **voxel_kwargs,
) -> SimulationSample:
```

Resolve the matrix from material ID `0`, the default yarn stiffness from
`default_yarn_material_id`, and each per-yarn stiffness from
`yarn_material_id_by_id`. Call
`voxelize_textile_material_fields(..., stiffness_output="sparse")`, passing
the explicit material IDs and `materials.unit` through. Convert `materials` to
the voxel result's backend/device/dtype only when necessary, then construct
`SimulationSample` with `data.sparse_orientation`.

- [ ] **Step 5: Run focused tests**

Run:

```bash
python -m unittest test_material_fields test_simulation_sample -v
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add TexGen/material_fields.py TexGen/simulation_sample.py \
        test_material_fields.py test_simulation_sample.py
git commit -m "feat: preserve explicit material identities"
```

---

### Task 5: Add consolidated sample persistence and memory mapping

**Files:**

- Create: `TexGen/simulation_io.py`
- Create: `test_simulation_io.py`
- Modify: `TexGen/simulation_sample.py`

- [ ] **Step 1: Write failing directory/archive round-trip tests**

Test both a directory and `.npz` target. Assertions must cover values,
dtypes, grid shape/order, units, names, metadata, voxel generation settings,
and optional arrays. For directory loading:

```python
loaded = load_simulation_sample(path, mmap_mode="r")
self.assertIsInstance(loaded.voxels.yarn_id, np.memmap)
self.assertIsInstance(loaded.orientation.orientation1, np.memmap)
self.assertIsInstance(loaded.stiffness.yarn_c21, np.memmap)
self.assertEqual(
    loaded.orientation.voxel_indices.filename,
    loaded.stiffness.voxel_indices.filename,
)
self.assertEqual(
    loaded.orientation.yarn_ids.filename,
    loaded.stiffness.yarn_ids.filename,
)
```

Inspect `manifest.json` and assert the orientation/stiffness aliases refer to
the same canonical keys. Reject unknown schema/version, missing required
arrays, mismatched dtype/shape, and `mmap_mode` with `.npz`.

- [ ] **Step 2: Run and confirm the persistence module is absent**

Run:

```bash
python -m unittest test_simulation_io -v
```

Expected: FAIL because `TexGen.simulation_io` does not exist.

- [ ] **Step 3: Implement schema version 1**

Expose:

```python
def save_simulation_sample(path, sample, *, compressed=True) -> None:
def load_simulation_sample(
    path, *, output="numpy", device=None, mmap_mode=None
) -> SimulationSample:
```

Use schema name `pytexgen.simulation_sample`, version `1`. A directory stores
`manifest.json` and `arrays/<key>.npy`. An `.npz` stores one member per
canonical key plus `_manifest_json`. Canonical keys are:

```text
voxel_yarn_id
voxel_aabb
voxel_centers
sparse_voxel_indices
sparse_yarn_ids
orientation_primary
orientation_secondary
stiffness_matrix_c21
stiffness_material_ids
stiffness_yarn_c21
material_ids
material_c21
```

Omit optional keys that are absent. The manifest maps both
`orientation.voxel_indices` and `stiffness.voxel_indices` to
`sparse_voxel_indices`, and maps both sparse yarn-ID fields to
`sparse_yarn_ids`. Before saving, verify the two source arrays match exactly.

Record each canonical array's filename/member, dtype, and shape. Record
resolution, order, voxel backend settings, unit, material names, frozen user
metadata, schema version, installed PyTexGen version, optional git commit, and
generation parameters. Load and validate the manifest before constructing any
public container. Directory loading passes `mmap_mode` to every `np.load`.
`output="torch"` constructs NumPy first, then calls
`sample.to("torch", device=device)`.

- [ ] **Step 4: Run focused tests**

Run:

```bash
python -m unittest test_simulation_io -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add TexGen/simulation_io.py TexGen/simulation_sample.py \
        test_simulation_io.py
git commit -m "feat: persist simulation samples without duplicate arrays"
```

---

### Task 6: Correct phase mapping and explicit isotropic phase materials

**Files:**

- Modify: `TexGen/acdm_solver.py`
- Modify: `test_acdm_solver_adapter.py`

- [ ] **Step 1: Replace machine-dependent discovery coverage**

Change `test_find_cloned_voxel_acdm_root` to create a temporary fake checkout
containing `README.md` and `femlib/__init__.py`, then pass that explicit path.
This removes dependence on a sibling repository while preserving discovery
coverage.

- [ ] **Step 2: Write failing NumPy/Torch phase tests**

Add tests proving:

- default yarn phase is applied before per-yarn overrides;
- NumPy output stays NumPy and Torch output stays Torch on the same device;
- negative and greater-than-15 mapping values fail before `uint8` conversion;
- batch shapes are `(1, Nz, Ny, Nx)` and unbatched shapes are
  `(Nz, Ny, Nx)`;
- CPU and CUDA phase arrays are bitwise identical.

The regression for the existing override bug is:

```python
phase = to_acdm_phase_ids(
    data,
    yarn_phase=5,
    yarn_phase_by_id={1: 9},
    batch=False,
)
np.testing.assert_array_equal(
    phase,
    np.array([[[0, 5], [9, 0]]], dtype=np.uint8),
)
```

- [ ] **Step 3: Write failing explicit phase-table tests**

Add a private-constructor unit test through the public solve function using a
fake solver class. Call:

```python
phase_materials = {
    0: {"E": 3.0, "Nu": 0.35},
    5: {"E": 70.0, "Nu": 0.20},
    9: {"E": 140.0, "Nu": 0.25},
}
```

Assert the fake receives LUTs of length `16` whose rows `0`, `5`, and `9`
match exactly. Assert missing used phase `9` fails before solver construction.
Assert unused valid table rows are accepted. Keep a test showing legacy
`matrix_material`/`yarn_material` arguments construct rows for
`matrix_phase`/`yarn_phase`.

- [ ] **Step 4: Run and observe failures**

Run:

```bash
python -m unittest test_acdm_solver_adapter -v
```

Expected: FAIL because mapping converts through NumPy, skips the default when
overrides exist, and builds fixed two-row LUTs.

- [ ] **Step 5: Implement backend-preserving phase mapping**

Validate `matrix_phase`, `yarn_phase`, and every override key/value as Python
integers in `0..15` before allocating. Then:

```python
grid = data.grid
if _is_torch_tensor(grid):
    phase = torch.full(grid.shape, matrix_phase, dtype=torch.uint8,
                       device=grid.device)
else:
    phase = np.full(grid.shape, matrix_phase, dtype=np.uint8)
phase[grid >= 0] = yarn_phase
for yarn_id, phase_id in sorted(overrides.items()):
    phase[grid == yarn_id] = phase_id
return phase[None] if batch else phase
```

Do not call `_to_numpy()` in this function.

- [ ] **Step 6: Implement phase-table normalization and LUT construction**

Change the isotropic solve signature additively:

```python
matrix_material: Optional[Mapping[str, float]] = None,
yarn_material: Optional[Mapping[str, float]] = None,
phase_materials: Optional[Mapping[int, Mapping[str, float]]] = None,
allow_host_phase_pack: bool = False,
```

Reject mixing `phase_materials` with legacy material arguments. Normalize
legacy arguments into a phase table keyed by `matrix_phase` and `yarn_phase`.
Validate `E > 0`, `-1 < Nu < 0.5`, finite values, and phase keys `0..15`.

Build fixed length-16 `float64` LUTs. Initialize every row from the lowest
explicit phase row, then overwrite every explicit phase. This keeps unused
rows numerically valid without assigning them semantic meaning. Determine the
actually used phases with backend-native boolean reductions: matrix presence,
default-yarn presence after excluding override yarn IDs, and presence of each
overridden yarn ID. Moving those scalar booleans to Python is allowed; moving
the voxel grid is not. Require table entries for exactly those used phases;
extra rows are allowed.

- [ ] **Step 7: Run focused tests**

Run:

```bash
python -m unittest test_acdm_solver_adapter -v
```

Expected: PASS without importing the real Voxel-ACDM checkout.

- [ ] **Step 8: Commit**

```bash
git add TexGen/acdm_solver.py test_acdm_solver_adapter.py
git commit -m "fix: preserve GPU phase mappings for ACDM"
```

---

### Task 7: Add explicit compact-host policy and dense same-device ACDM path

**Files:**

- Modify: `TexGen/acdm_solver.py`
- Create: `test_acdm_sample_adapter.py`

- [ ] **Step 1: Write failing compact CUDA policy tests**

Use fake solver modules rather than importing Triton. A fake compact class
without capability markers must not be constructed for CUDA phase tensors
when `allow_host_phase_pack=False`. Patch `Tensor.cpu` with a counter and
assert it remains zero. With `allow_host_phase_pack=True`, assert one explicit
host pack, NumPy phase input, `timings["phase_pack_device"] == "cpu"`, and
`timings["phase_pack_bytes"] == phase.numel() * phase.element_size()`.

A capable fake class advertises:

```python
SUPPORTS_CUDA_PHASE_IDS = True
```

Assert it receives the exact CUDA tensor and pointer with no `.cpu()` call.

- [ ] **Step 2: Write failing anisotropic same-device tests**

Create a Torch `SimulationSample` and a fake
`FEMHomogenizerBatched` recording its constructor input:

```python
result = solve_acdm_anisotropic_from_sample(
    sample,
    device=str(sample.stiffness.yarn_c21.device),
    dtype="fp32",
    solver_module=fake_module,
)
self.assertTrue(fake_solver.received.is_cuda)
self.assertEqual(fake_solver.received.device, sample.stiffness.yarn_c21.device)
torch.testing.assert_close(
    fake_solver.received,
    sample.stiffness.to_acdm(batch=True),
)
self.assertIs(result.C_eff_tensor, fake_solver.returned_C_eff)
```

Also test rejection before construction for missing stiffness, NumPy sample,
CPU device, unsupported dtype, mismatched requested device, and an external
class that declares `SUPPORTS_TORCH_C_VOIGT_FIELDS = False`.

- [ ] **Step 3: Run and confirm the new boundary is absent**

Run:

```bash
python -m unittest test_acdm_sample_adapter -v
```

Expected: FAIL because the anisotropic API and host-pack guard do not exist.

- [ ] **Step 4: Implement result preservation and capability checks**

Broaden `ACDMSolveResult.phase_ids` and `C_eff` annotations to `Any`, and add:

```python
C_eff_tensor: Optional[Any] = None
```

Add `effective_stiffness_numpy(copy=True)` that explicitly performs
`detach().cpu().numpy()` only when called. Existing NumPy-returning solver
behavior remains unchanged.

For compact phase construction, treat
`FEMHomogenizerBatchedIsotropicPhases.SUPPORTS_CUDA_PHASE_IDS is True` as the
only positive CUDA capability signal. If absent/false and host packing is not
allowed, raise `RuntimeError` before calling `.cpu()` or constructing the
solver. When host packing is allowed, perform and record the transfer.

- [ ] **Step 5: Implement the general dense adapter**

Expose:

```python
def solve_acdm_anisotropic_from_sample(
    sample: SimulationSample,
    *,
    acdm_root: Optional[str] = None,
    device: Optional[str] = None,
    dtype: str = "fp32",
    precond: str = "fft",
    tol: float = 2e-6,
    max_iter: int = 2000,
    element_type: str = "c3d8",
    hourglass_coefficient: float = 0.1,
    verbose: bool = False,
    solver_module: Any = None,
) -> ACDMSolveResult:
```

`solver_module` is an advanced dependency-injection hook used for compatible
forks and tests; when absent import `femlib.fem_batched`. Require a Torch CUDA
sample and exact requested-device match. Materialize
`sample.array("stiffness.yarn_c21", layout="acdm", copy=True)` on that same
device and pass it directly to `FEMHomogenizerBatched`. Do not call NumPy.
Pass voxel size, grid shape, dtype, element type, and hourglass coefficient.
Apply the same validated preconditioner choices as the isotropic path.

If the external result is a Torch tensor, retain it as both `C_eff` and
`C_eff_tensor`; do not create NumPy. If it is NumPy, keep `C_eff_tensor=None`.
Record input/output devices and dense-allocation bytes in timings.

- [ ] **Step 6: Run focused tests**

Run:

```bash
python -m unittest \
  test_acdm_solver_adapter \
  test_acdm_sample_adapter -v
```

Expected: PASS; CUDA-specific tests skip only if CUDA is unavailable.

- [ ] **Step 7: Commit**

```bash
git add TexGen/acdm_solver.py test_acdm_sample_adapter.py
git commit -m "feat: connect simulation samples to GPU ACDM"
```

---

### Task 8: Package, document, and integrate the public contract

**Files:**

- Modify: `Python/CMakeLists.txt`
- Modify: `README.md`
- Modify: `README_pypi.md`
- Modify: `docs/voxel_backends.md`
- Modify: `docs/voxel_acdm_interface.md`
- Create: `test_simulation_sample_integration.py`

- [ ] **Step 1: Write failing installed-package smoke coverage**

Add integration coverage that builds the repository's small real weave fixture,
calls `voxelize_textile_simulation_sample()`, and validates sparse direction,
stiffness, material IDs, shape, order, unit, and ACDM dense values against
`SparseStiffnessField.to_acdm()`. Keep the grid small enough for CPU CI.

Add a subprocess import test:

```bash
python -c "from pytexgen.simulation_sample import MaterialTable, SimulationSample"
python -c "from pytexgen.simulation_io import save_simulation_sample, load_simulation_sample"
```

Expected before packaging changes: installed-wheel imports fail.

- [ ] **Step 2: Install both new modules in wheel and legacy modes**

Add `simulation_sample.py` and `simulation_io.py` beside the current
`gpu_voxelizer.py`, `material_fields.py`, and `acdm_solver.py` wheel entries.
For the legacy install branch, install the complete interoperable group
(`gpu_voxelizer.py`, `material_fields.py`, `simulation_sample.py`,
`simulation_io.py`, `acdm_solver.py`, and `torch_periodic_mesher.py`) together
so the new modules never have missing sibling dependencies.

- [ ] **Step 3: Document framework and solver handoffs**

Add concise examples for:

- native NumPy access;
- `.to("torch", device="cuda")`;
- `torch.from_dlpack(sample.array(...))`;
- the equivalent JAX/Warp/CuPy consumer pattern without claiming those
  optional libraries are test dependencies;
- physical `voxel.material_id` versus legacy yarn-based
  `VoxelGridData.material_id()`;
- explicit `copy=True` for dense ACDM layout;
- explicit `allow_host_phase_pack=True` for current compact Voxel-ACDM;
- sample persistence and NumPy memory mapping.

Update the old DLPack text so the sample/array protocol is preferred and the
capsule helper is labeled legacy-compatible.

- [ ] **Step 4: Run focused integration and packaging checks**

Run:

```bash
python -m unittest test_simulation_sample_integration -v
python -m py_compile \
  TexGen/simulation_sample.py \
  TexGen/simulation_io.py \
  TexGen/acdm_solver.py
python -m build
```

Create a temporary virtual environment outside the repository, install the
new wheel, run the two import commands, and remove only that temporary
environment afterward.

Expected: integration PASS, compilation PASS, wheel/sdist build PASS, and
installed imports print no errors.

- [ ] **Step 5: Commit**

```bash
git add Python/CMakeLists.txt README.md README_pypi.md \
        docs/voxel_backends.md docs/voxel_acdm_interface.md \
        test_simulation_sample_integration.py
git commit -m "docs: publish simulation sample interoperability"
```

---

### Task 9: Add a checked zero-copy and accuracy benchmark

**Files:**

- Create: `bench_simulation_interop.py`
- Create: `test_simulation_interop_benchmark.py`
- Modify: `docs/voxel_backends.md`

- [ ] **Step 1: Write failing benchmark contract tests**

Import the benchmark as a module and test its JSON schema:

```text
schema
device
resolution
num_voxels
resident_sparse_bytes
acdm_dense_bytes
handoff_host_transfer_bytes
dlpack_pointer_shared
phase_equal_numpy
acdm_dense_max_abs_error
pytexgen_seconds
texgen_cpu_seconds
speedup_vs_texgen_cpu
accepted
```

Use fake CPU/Torch arrays to make the test deterministic. Acceptance must be
false if the DLPack pointer differs, a phase value differs, dense error exceeds
the dtype tolerance, `handoff_host_transfer_bytes` is at least the handed-off
field size, or the measured speedup is below a configurable threshold.

- [ ] **Step 2: Run and confirm the benchmark is absent**

Run:

```bash
python -m unittest test_simulation_interop_benchmark -v
```

Expected: FAIL because `bench_simulation_interop` does not exist.

- [ ] **Step 3: Implement measurement and acceptance**

The benchmark must:

1. construct the same representative textile for PyTexGen and TexGen CPU;
2. warm up both paths separately;
3. synchronize CUDA immediately before and after each timed GPU section;
4. time at least three repeats and report the median;
5. compare phase IDs bit-for-bit and dense C21/ACDM values using
   FP32 `rtol=1e-5, atol=1e-6` or FP64 `rtol=1e-10, atol=1e-12`;
6. import each resident GPU field with `torch.from_dlpack(field)` and compare
   `data_ptr()`;
7. report handoff transfer bytes as zero only when source/consumer device and
   pointer match and the handoff contains no explicit `.cpu()`, `.numpy()`, or
   storage conversion; otherwise report the full moved field byte count;
8. write JSON when `--json-out` is supplied;
9. exit nonzero under `--check` when any accuracy, transfer, or speed threshold
   fails.

Expose command-line options:

```text
--resolution
--repeat
--device
--dtype
--min-speedup
--json-out
--check
```

Default `--min-speedup` to `5.0` for the checked performance run, while unit
tests pass an injected timing result instead of asserting wall-clock speed.
Reuse the textile builders and TexGen CPU reference timing conventions from
`bench_gpu_material_fields.py` so the comparison is the same workload rather
than a synthetic proxy.

- [ ] **Step 4: Run unit and available hardware checks**

Run:

```bash
python -m unittest test_simulation_interop_benchmark -v
python bench_simulation_interop.py \
  --resolution 16 --repeat 3 --device cpu \
  --min-speedup 0 --check \
  --json-out build/simulation_interop_cpu.json
```

When CUDA is available, also run:

```bash
python bench_simulation_interop.py \
  --resolution 64 --repeat 5 --device cuda \
  --min-speedup 5.0 --check \
  --json-out build/simulation_interop_cuda.json
```

Expected: CPU smoke PASS. CUDA acceptance PASS with zero full-field host
handoff bytes and speedup greater than `5.0`; if no CUDA device exists, record
that the hardware acceptance run remains not executed rather than fabricating
a result.

- [ ] **Step 5: Document reproducible benchmark commands**

Add the commands, JSON fields, warm-up policy, median timing policy, accuracy
tolerances, and the meaning of `handoff_host_transfer_bytes` to
`docs/voxel_backends.md`.

- [ ] **Step 6: Commit**

```bash
git add bench_simulation_interop.py test_simulation_interop_benchmark.py \
        docs/voxel_backends.md
git commit -m "bench: verify zero-copy simulation handoff"
```

---

### Task 10: Final regression, artifact, and scope verification

**Files:**

- Modify only if a failing test exposes a defect in files already listed above.

- [ ] **Step 1: Run the full repository suite**

Run:

```bash
python -m unittest discover -s . -p 'test_*.py' -v
```

Expected: all tests PASS; only dependency/device-specific tests SKIP.

- [ ] **Step 2: Run source, build, and install verification**

Run:

```bash
python -m py_compile \
  TexGen/gpu_voxelizer.py \
  TexGen/material_fields.py \
  TexGen/simulation_sample.py \
  TexGen/simulation_io.py \
  TexGen/acdm_solver.py \
  bench_simulation_interop.py
python -m build
```

Install the wheel into a fresh temporary environment and run:

```bash
python -c "import pytexgen.simulation_sample, pytexgen.simulation_io, pytexgen.acdm_solver"
python -m unittest test_simulation_sample test_simulation_io -v
```

Expected: compilation, build, imports, and tests PASS.

- [ ] **Step 3: Inspect scope and generated artifacts**

Run:

```bash
git status --short
git diff --check
git log --oneline origin/main..HEAD
```

Confirm generated `build/`, `dist/`, benchmark JSON, temporary environments,
and solver outputs are ignored or absent. Confirm the unrelated untracked
`AGENTS.md` remains unstaged and unchanged. Confirm no sibling Voxel-ACDM file
was modified.

- [ ] **Step 4: Run the required hardware acceptance when CUDA is present**

Run the checked CUDA benchmark from Task 9 and a real compatible
Voxel-ACDM anisotropic smoke solve. Record GPU model, CUDA/Torch versions,
PyTexGen commit, Voxel-ACDM commit, tolerances, host-transfer bytes, timings,
and speedup in the benchmark JSON.

Expected: accurate phase/dense fields, same-device handoff, no full-field D2H
copy, and speedup above the accepted threshold. Do not claim GPU performance
acceptance if this hardware run was skipped.

- [ ] **Step 5: Commit only genuine final fixes**

If verification required source fixes, stage only their exact files and use:

```bash
git commit -m "fix: satisfy simulation interop acceptance"
```

If no fixes were needed, do not create an empty commit.

## Plan Self-Review Checklist

- [ ] Every approved stable field name has construction, access, and test
  coverage.
- [ ] Physical material IDs never use `VoxelGridData.material_id()`.
- [ ] Material ID `0`, explicit non-dense IDs, C21 order, flat voxel order,
  units, backend, device, and sparse identity are validated.
- [ ] DLPack is consumed from individual arrays, not from `SimulationSample`.
- [ ] All implicit allocation/transfer paths reject under `copy=False`.
- [ ] Directory and archive persistence write each canonical array once and
  directory loading preserves `np.memmap`.
- [ ] Compact phase packing has an explicit host opt-in; general anisotropic
  ACDM stays on the current CUDA device.
- [ ] Accuracy and speed claims are gated by checked tests/benchmarks.
- [ ] Training datasets, batching, sharding, augmentation, additional solvers,
  and classifier-kernel optimization remain outside this first project.
- [ ] The plan contains no unresolved placeholder markers or instruction to
  modify the external Voxel-ACDM checkout.
