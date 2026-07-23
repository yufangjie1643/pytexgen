# GPU Sparse Material Fields Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add exact sparse yarn orientation fields and 21-component engineering-Voigt stiffness fields whose classification, material lookup, and rotation remain on Torch/CUDA, with compact persistence and measured speedups over TexGen CPU export.

**Architecture:** `gpu_voxelizer.py` captures winning tangent/up vectors during classification and attaches a `SparseOrientationField` to `VoxelGridData`. A new dependency-free `material_fields.py` owns C21 conventions, sparse containers, NumPy/Torch rotation, dense views, persistence, and one-call orchestration. The existing dense orientation API remains compatible.

**Tech Stack:** Python 3.9+, NumPy 1.21+, optional PyTorch, `unittest`, scikit-build-core/CMake, TexGen C++/SWIG bindings.

## Global Constraints

- Do not add Voxel-ACDM or another runtime dependency.
- Engineering-Voigt order is exactly `(xx, yy, zz, yz, xz, xy)`.
- C21 is the row-major upper triangle of a symmetric `6 x 6` matrix.
- Matrix voxels have no direction entries and share one `matrix_c21`.
- Sparse Torch classification and stiffness construction must not call NumPy conversion helpers.
- Existing `include_orientations=True` dense output remains backward compatible.
- GPU rotation is chunked; no whole-grid `6 x 6` temporary is allowed.
- Persistence is an explicit GPU-to-CPU boundary.
- The reference RTX 5090 acceptance gate is at least 5x median speedup for both approved models and both comparison modes at `128^3` and `256^3`.
- Do not stage or commit the unrelated untracked root `AGENTS.md`.

## File Map

- Create `TexGen/material_fields.py`: C21 math, sparse field types, rotations, material construction, dense views, and persistence.
- Modify `TexGen/gpu_voxelizer.py`: Torch orientation capture, sparse output selection, `VoxelGridData` conversion and persistence integration.
- Modify `Python/CMakeLists.txt`: install `material_fields.py` in wheels.
- Create `test_material_fields.py`: focused material-field and persistence tests.
- Modify `test_gpu_voxelizer_backends.py`: sparse NumPy/Torch classifier and compatibility tests.
- Create `bench_gpu_material_fields.py`: correctness-gated CPU/GPU performance benchmark and JSON reporting.
- Modify `README.md`, `README_pypi.md`, and `docs/voxel_backends.md`: public usage, conventions, memory behavior, and benchmark instructions.

---

### Task 1: C21 and Engineering-Constant Primitives

**Files:**
- Create: `TexGen/material_fields.py`
- Create: `test_material_fields.py`

**Interfaces:**
- Produces: `VOIGT_COMPONENTS`, `C21_INDICES`, `pack_voigt_c21(matrix)`, `unpack_c21(c21)`, `isotropic_stiffness_c21(E, nu)`, and `orthotropic_stiffness_c21(E1, E2, E3, nu12, nu13, nu23, G12, G13, G23)`.

- [ ] **Step 1: Write failing C21 and material-construction tests**

```python
class C21UtilitiesTest(unittest.TestCase):
    def test_pack_unpack_uses_documented_upper_triangle_order(self):
        C = np.arange(36, dtype=np.float64).reshape(6, 6)
        C = np.triu(C)
        C = C + np.triu(C, 1).T
        packed = mf.pack_voigt_c21(C)
        np.testing.assert_array_equal(
            packed,
            C[np.triu_indices(6)],
        )
        np.testing.assert_array_equal(mf.unpack_c21(packed), C)

    def test_pack_rejects_nonsymmetric_matrix(self):
        C = np.eye(6)
        C[0, 1] = 1.0
        with self.assertRaisesRegex(ValueError, "symmetric"):
            mf.pack_voigt_c21(C)

    def test_isotropic_helper_recovers_lame_coefficients(self):
        C = mf.unpack_c21(mf.isotropic_stiffness_c21(70.0, 0.25))
        self.assertAlmostEqual(C[3, 3], 28.0)
        self.assertAlmostEqual(C[0, 1], 28.0)
        self.assertAlmostEqual(C[0, 0], 84.0)

    def test_orthotropic_helper_inverts_engineering_compliance(self):
        c21 = mf.orthotropic_stiffness_c21(
            150.0, 10.0, 12.0, 0.25, 0.20, 0.30, 5.0, 6.0, 4.0
        )
        C = mf.unpack_c21(c21)
        S = np.linalg.inv(C)
        self.assertAlmostEqual(S[0, 0], 1.0 / 150.0)
        self.assertAlmostEqual(S[5, 5], 1.0 / 5.0)
```

- [ ] **Step 2: Run the tests and verify missing-module failure**

Run: `python -m unittest test_material_fields.C21UtilitiesTest -v`  
Expected: FAIL because `TexGen.material_fields` does not exist.

- [ ] **Step 3: Implement backend-preserving packing and NumPy material helpers**

```python
VOIGT_COMPONENTS = ("xx", "yy", "zz", "yz", "xz", "xy")
C21_INDICES = tuple((i, j) for i in range(6) for j in range(i, 6))

def pack_voigt_c21(matrix, *, symmetry_rtol=1e-10, symmetry_atol=1e-12):
    if tuple(matrix.shape[-2:]) != (6, 6):
        raise ValueError(f"matrix must end with shape (6, 6), got {matrix.shape}")
    transpose = matrix.transpose(-1, -2) if _is_torch_tensor(matrix) else np.swapaxes(matrix, -1, -2)
    if not _allclose(matrix, transpose, rtol=symmetry_rtol, atol=symmetry_atol):
        raise ValueError("matrix must be symmetric")
    return _stack([matrix[..., i, j] for i, j in C21_INDICES], axis=-1)

def unpack_c21(c21):
    if c21.shape[-1] != 21:
        raise ValueError(f"c21 must end with length 21, got {c21.shape}")
    out = _zeros(c21.shape[:-1] + (6, 6), like=c21)
    for k, (i, j) in enumerate(C21_INDICES):
        out[..., i, j] = c21[..., k]
        out[..., j, i] = c21[..., k]
    return out
```

Implement `isotropic_stiffness_c21` from Lamé coefficients and
`orthotropic_stiffness_c21` by constructing and inverting the documented
engineering compliance matrix. Reject non-finite inputs, non-positive moduli,
and isotropic `nu <= -1` or `nu >= 0.5`.

- [ ] **Step 4: Run the focused tests**

Run: `python -m unittest test_material_fields.C21UtilitiesTest -v`  
Expected: all tests PASS.

- [ ] **Step 5: Commit Task 1**

```bash
git add TexGen/material_fields.py test_material_fields.py
git -c user.name=yufangjie1643 \
    -c user.email=yufangjie1643@users.noreply.github.com \
    commit -m "feat: add C21 material utilities"
```

### Task 2: Sparse Field Containers and Dense Views

**Files:**
- Modify: `TexGen/material_fields.py`
- Modify: `test_material_fields.py`

**Interfaces:**
- Consumes: Task 1 C21 functions.
- Produces: `SparseOrientationField`, `SparseStiffnessField`, `.to(...)`, `to_dense_c21()`, `to_dense_voigt()`, and `to_acdm(batch=True)`.

- [ ] **Step 1: Add failing sparse-container tests**

```python
class SparseFieldContainerTest(unittest.TestCase):
    def make_orientation(self):
        return mf.SparseOrientationField(
            voxel_indices=np.array([1, 3], dtype=np.int64),
            yarn_ids=np.array([0, 2], dtype=np.int32),
            orientation1=np.array([[1., 0., 0.], [0., 1., 0.]]),
            orientation2=np.array([[0., 0., 1.], [0., 0., 1.]]),
            grid_shape=(1, 2, 2),
        )

    def test_orientation_rejects_unsorted_or_duplicate_indices(self):
        field = self.make_orientation()
        with self.assertRaisesRegex(ValueError, "strictly increasing"):
            dataclasses.replace(field, voxel_indices=np.array([3, 3]))

    def test_sparse_stiffness_materializes_matrix_and_yarns(self):
        yarn = np.stack([np.arange(21), np.arange(21) + 100.0])
        field = mf.SparseStiffnessField(
            matrix_c21=np.full(21, -1.0),
            voxel_indices=np.array([1, 3]),
            yarn_ids=np.array([0, 2], dtype=np.int32),
            material_ids=np.array([1, 2], dtype=np.int32),
            yarn_c21=yarn,
            grid_shape=(1, 2, 2),
        )
        dense = field.to_dense_c21()
        self.assertEqual(dense.shape, (1, 2, 2, 21))
        np.testing.assert_array_equal(dense.reshape(-1, 21)[0], -np.ones(21))
        np.testing.assert_array_equal(dense.reshape(-1, 21)[1], yarn[0])
        np.testing.assert_array_equal(field.to_dense_voigt().shape, (6, 6, 1, 2, 2))
        np.testing.assert_array_equal(field.to_acdm().shape, (1, 6, 6, 1, 2, 2))
```

- [ ] **Step 2: Verify the tests fail for missing types**

Run: `python -m unittest test_material_fields.SparseFieldContainerTest -v`  
Expected: FAIL because the sparse classes are undefined.

- [ ] **Step 3: Implement validated immutable data classes**

```python
@dataclass(frozen=True)
class SparseOrientationField:
    voxel_indices: Any
    yarn_ids: Any
    orientation1: Any
    orientation2: Any
    grid_shape: Tuple[int, int, int]
    order: str = "ix + iy*nx + iz*nx*ny"

    def to(self, storage=None, *, device=None, dtype=None, copy=False):
        return _convert_orientation_field(
            self, storage=storage, device=device, dtype=dtype, copy=copy
        )

@dataclass(frozen=True)
class SparseStiffnessField:
    matrix_c21: Any
    voxel_indices: Any
    yarn_ids: Any
    material_ids: Any
    yarn_c21: Any
    grid_shape: Tuple[int, int, int]
    unit: Optional[str] = None
    order: str = "ix + iy*nx + iz*nx*ny"

    def to_dense_c21(self):
        total = math.prod(self.grid_shape)
        flat = self.matrix_c21.expand(total, 21).clone() if _is_torch_tensor(
            self.matrix_c21
        ) else np.broadcast_to(self.matrix_c21, (total, 21)).copy()
        flat[self.voxel_indices] = self.yarn_c21
        return flat.reshape(self.grid_shape + (21,))
```

Validate equal leading lengths, numeric arrays, finite floats, integer IDs,
index range, strict index ordering, C21 shapes, and positive grid dimensions.
Implement backend/device-preserving conversions. `to_dense_voigt()` unpacks
C21 and permutes from `(Nz,Ny,Nx,6,6)` to `(6,6,Nz,Ny,Nx)`;
`to_acdm(batch=True)` prepends the batch dimension.

- [ ] **Step 4: Run NumPy and optional Torch container tests**

Run: `python -m unittest test_material_fields.SparseFieldContainerTest -v`  
Expected: all tests PASS; Torch-specific tests skip if Torch is absent.

- [ ] **Step 5: Commit Task 2**

```bash
git add TexGen/material_fields.py test_material_fields.py
git -c user.name=yufangjie1643 \
    -c user.email=yufangjie1643@users.noreply.github.com \
    commit -m "feat: add sparse material field containers"
```

### Task 3: Chunked NumPy/Torch Stiffness Rotation

**Files:**
- Modify: `TexGen/material_fields.py`
- Modify: `test_material_fields.py`

**Interfaces:**
- Consumes: `SparseOrientationField` and Task 1 material functions.
- Produces: `rotate_stiffness_c21(local_c21, orientation1, orientation2, *, chunk_voxels=65536, eps=1e-12)` and `build_stiffness_field(data, *, matrix_stiffness, default_yarn_stiffness=None, yarn_stiffness_by_id=None, output="sparse", chunk_voxels=65536, validate_positive_definite=False, unit=None)`.

- [ ] **Step 1: Add failing rotation and material-map tests**

```python
class StiffnessRotationTest(unittest.TestCase):
    def test_identity_frame_preserves_general_c21(self):
        local = mf.pack_voigt_c21(self.symmetric_positive_definite_matrix())
        result = mf.rotate_stiffness_c21(
            local[None, :],
            np.array([[1., 0., 0.]]),
            np.array([[0., 1., 0.]]),
            chunk_voxels=1,
        )
        np.testing.assert_allclose(result[0], local, rtol=1e-12, atol=1e-12)

    def test_ninety_degree_frame_matches_fourth_order_reference(self):
        local = mf.orthotropic_stiffness_c21(
            150., 10., 12., .25, .20, .30, 5., 6., 4.
        )
        actual = mf.rotate_stiffness_c21(
            local[None, :],
            np.array([[0., 1., 0.]]),
            np.array([[-1., 0., 0.]]),
        )
        expected = self.rotate_with_explicit_fourth_order_tensor(
            mf.unpack_c21(local), self.rotation_z_90()
        )
        np.testing.assert_allclose(
            mf.unpack_c21(actual[0]), expected, rtol=1e-10, atol=1e-10
        )

    def test_builder_selects_default_and_per_yarn_materials(self):
        data = types.SimpleNamespace(sparse_orientation=self.make_three_voxel_field())
        result = mf.build_stiffness_field(
            data,
            matrix_stiffness=np.ones(21),
            default_yarn_stiffness=np.full(21, 2.0),
            yarn_stiffness_by_id={7: np.full(21, 3.0)},
            chunk_voxels=2,
        )
        self.assertEqual(result.yarn_c21.shape, (3, 21))
        self.assertEqual(result.material_ids.tolist(), [1, 2, 1])
```

- [ ] **Step 2: Verify rotation tests fail**

Run: `python -m unittest test_material_fields.StiffnessRotationTest -v`  
Expected: FAIL because rotation and builder functions are undefined.

- [ ] **Step 3: Implement orthonormal frames and Mandel rotation**

```python
def _orthonormal_frames(orientation1, orientation2, eps):
    e1 = orientation1 / _norm(orientation1, axis=-1, keepdims=True)
    projected = orientation2 - _sum(orientation2 * e1, axis=-1, keepdims=True) * e1
    projected_norm = _norm(projected, axis=-1, keepdims=True)
    if _any(projected_norm <= eps):
        raise ValueError("orientation vectors are zero or collinear")
    e2 = projected / projected_norm
    e3 = _cross(e1, e2)
    return _stack((e1, e2, e3), axis=-1)
```

Create the six orthonormal Mandel basis tensors once per backend/device. Build
`Q[...,i,j] = basis_i : (R @ basis_j @ R.T)`, then calculate
`C_global_m = Q @ C_local_m @ Q.T`. Convert engineering Voigt to/from Mandel
using weights `(1,1,1,sqrt(2),sqrt(2),sqrt(2))`. Process slices of at most
`chunk_voxels` and append packed C21 outputs.

- [ ] **Step 4: Implement material lookup and sparse field construction**

Convert all input `(21,)` or `(6,6)` values to a C21 table on the orientation
field's backend/device. Assign material ID 1 to the default and stable IDs
starting at 2 to sorted overrides. Use tensor-native comparison and indexed
assignment. If a yarn lacks both override and default, raise an error listing
its `yarn_id`. Dense output delegates to `SparseStiffnessField.to_dense_c21()`.

- [ ] **Step 5: Run rotation, chunk invariance, validation, and Torch parity tests**

Run: `python -m unittest test_material_fields.StiffnessRotationTest -v`  
Expected: all tests PASS; FP64 NumPy/Torch relative Frobenius error `<=1e-10`.

- [ ] **Step 6: Commit Task 3**

```bash
git add TexGen/material_fields.py test_material_fields.py
git -c user.name=yufangjie1643 \
    -c user.email=yufangjie1643@users.noreply.github.com \
    commit -m "feat: rotate sparse C21 stiffness fields"
```

### Task 4: GPU Sparse Orientation Capture

**Files:**
- Modify: `TexGen/gpu_voxelizer.py`
- Modify: `test_gpu_voxelizer_backends.py`

**Interfaces:**
- Consumes: `SparseOrientationField`.
- Produces: `orientation_storage={"dense","sparse"}` on `voxelize_snapshots_data`, `voxelize_snapshot_bundle_data`, `voxelize_textile_data`, and `VoxelizationCache.voxelize`; adds `VoxelGridData.sparse_orientation`.

- [ ] **Step 1: Add failing sparse NumPy and Torch classifier tests**

```python
def test_numpy_sparse_orientation_matches_dense_yarn_entries(self):
    self.patch_extract_snapshots()
    dense = self.voxelizer.voxelize_textile_data(
        FakeTextile(), nx=4, ny=4, nz=4, backend="numpy",
        include_orientations=True, orientation_storage="dense",
        workers=1, verbose=False,
    )
    sparse = self.voxelizer.voxelize_textile_data(
        FakeTextile(), nx=4, ny=4, nz=4, backend="numpy",
        include_orientations=True, orientation_storage="sparse",
        workers=1, verbose=False,
    )
    indices = np.flatnonzero(dense.yarn_id >= 0)
    np.testing.assert_array_equal(sparse.sparse_orientation.voxel_indices, indices)
    np.testing.assert_allclose(
        sparse.sparse_orientation.orientation1, dense.orientation1.reshape(-1, 3)[indices]
    )

@unittest.skipIf(torch is None, "torch is optional")
def test_torch_sparse_orientation_stays_on_selected_device(self):
    self.patch_extract_snapshots()
    data = self.voxelizer.voxelize_textile_data(
        FakeTextile(), nx=4, ny=4, nz=4, backend="torch", device="cpu",
        output="backend", include_orientations=True,
        orientation_storage="sparse", verbose=False,
    )
    self.assertEqual(data.sparse_orientation.orientation1.device.type, "cpu")
    self.assertIsNone(data.orientation1)
```

- [ ] **Step 2: Verify current Torch rejection and missing parameter failures**

Run: `python -m unittest test_gpu_voxelizer_backends.VoxelizerBackendTest.test_numpy_sparse_orientation_matches_dense_yarn_entries -v`  
Expected: FAIL with unexpected `orientation_storage`.

- [ ] **Step 3: Capture winning directions inside `_classify_voxels_torch`**

Extend the signature with `include_orientations=False` and
`orientation_storage="dense"`. For each chunk allocate `best_orientation1` and
`best_orientation2` only when requested. Every `best_yarn` update must update
both direction arrays using the same boolean mask. Dense mode writes flat
whole-grid arrays. Sparse mode appends:

```python
mask = best_yarn >= 0
index_parts.append(torch.arange(v0, v1, device=device, dtype=torch.int64)[mask])
yarn_parts.append(best_yarn[mask])
orientation1_parts.append(best_orientation1[mask])
orientation2_parts.append(best_orientation2[mask])
```

Return `(yarn_id, orientation_payload)` when orientations are enabled, where
the payload is either two dense flat tensors or the four sparse tensors.

- [ ] **Step 4: Integrate sparse output across public voxelization paths**

Validate `orientation_storage` even when orientations are disabled. Remove the
Torch rejection. Wrap sparse tensor/array payloads in
`SparseOrientationField(grid_shape=(nz,ny,nx))`. For NumPy, derive compact
arrays from the existing dense classifier result and release dense temporaries.
Keep default `orientation_storage="dense"` to preserve existing behavior.

- [ ] **Step 5: Run backend tests**

Run: `python -m unittest test_gpu_voxelizer_backends.VoxelizerBackendTest -v`  
Expected: all existing and new tests PASS.

- [ ] **Step 6: Commit Task 4**

```bash
git add TexGen/gpu_voxelizer.py test_gpu_voxelizer_backends.py
git -c user.name=yufangjie1643 \
    -c user.email=yufangjie1643@users.noreply.github.com \
    commit -m "feat: capture sparse orientations on GPU"
```

### Task 5: Conversion and Persistence Integration

**Files:**
- Modify: `TexGen/gpu_voxelizer.py`
- Modify: `TexGen/material_fields.py`
- Modify: `test_gpu_voxelizer_backends.py`
- Modify: `test_material_fields.py`

**Interfaces:**
- Consumes: sparse classes and voxel output.
- Produces: sparse-aware `VoxelGridData.to_numpy`, `to_torch`, `save_npz`, `load_npz`, `save_npy_dir`, `load_npy_dir`; `save_material_field_bundle` and `load_material_field_bundle`.

- [ ] **Step 1: Add failing conversion and round-trip tests**

Test NumPy→Torch→NumPy preservation of indices, yarn IDs, vectors, dtype, shape,
and order. Test `VoxelGridData` `.npz` and `.npy` round trips with sparse
orientations. Test both combined-bundle `.npz` and directory `.npy` schemas,
the exact directory file names, and memory-mapped NumPy load.

```python
with tempfile.TemporaryDirectory() as tmp:
    path = Path(tmp) / "fields"
    mf.save_material_field_bundle(path, orientation, stiffness)
    loaded_o, loaded_c = mf.load_material_field_bundle(path, mmap_mode="r")
    self.assertIsInstance(loaded_o.orientation1, np.memmap)
    np.testing.assert_array_equal(loaded_c.yarn_c21, stiffness.yarn_c21)
```

- [ ] **Step 2: Verify tests fail because sparse persistence is absent**

Run: `python -m unittest test_material_fields.PersistenceTest -v`  
Expected: FAIL on missing persistence functions or missing metadata arrays.

- [ ] **Step 3: Extend `VoxelGridData` conversion and persistence**

Add `sparse_orientation=None` to the data class. Delegate conversion to
`SparseOrientationField.to`. Store sparse arrays using unambiguous names:
`orientation_voxel_indices`, `orientation_yarn_ids`,
`sparse_orientation1`, and `sparse_orientation2`. Increment directory metadata
to format version 2 while accepting version 1 dense archives.

- [ ] **Step 4: Implement combined field schema**

Write format `pytexgen.sparse_material_fields`, version 1, with the exact files
from the approved design. Provide a single `.npz` variant with the same logical
array names and a JSON metadata scalar. Validate manifest fields, shapes, index
order/range, matching lengths, finiteness, and C21 conventions before
constructing objects. For `output="torch"`, load on CPU once and upload directly
to the requested device. Reject `mmap_mode` with Torch output.

- [ ] **Step 5: Run persistence and backend regression tests**

Run: `python -m unittest test_material_fields.PersistenceTest test_gpu_voxelizer_backends.VoxelizerBackendTest -v`  
Expected: all tests PASS, including version-1 dense archive compatibility.

- [ ] **Step 6: Commit Task 5**

```bash
git add TexGen/gpu_voxelizer.py TexGen/material_fields.py \
        test_gpu_voxelizer_backends.py test_material_fields.py
git -c user.name=yufangjie1643 \
    -c user.email=yufangjie1643@users.noreply.github.com \
    commit -m "feat: persist sparse material fields"
```

### Task 6: One-Call API, Packaging, and Documentation

**Files:**
- Modify: `TexGen/material_fields.py`
- Modify: `Python/CMakeLists.txt`
- Modify: `README.md`
- Modify: `README_pypi.md`
- Modify: `docs/voxel_backends.md`
- Modify: `test_material_fields.py`

**Interfaces:**
- Produces: `voxelize_textile_material_fields(textile, *, matrix_stiffness, default_yarn_stiffness=None, yarn_stiffness_by_id=None, orientation_storage="sparse", stiffness_output="sparse", **voxel_kwargs) -> Tuple[VoxelGridData, SparseStiffnessField]`.

- [ ] **Step 1: Add a failing orchestration test**

Patch `voxelize_textile_data` to return a known sparse field and assert the
one-call function forwards `backend`, `device`, resolution, dtype,
`include_orientations=True`, and `orientation_storage="sparse"`, then calls
`build_stiffness_field` with the exact material inputs.

- [ ] **Step 2: Verify the one-call test fails**

Run: `python -m unittest test_material_fields.OrchestrationTest -v`  
Expected: FAIL because `voxelize_textile_material_fields` is undefined.

- [ ] **Step 3: Implement orchestration without circular imports**

Import `voxelize_textile_data` inside the function body. Reject attempts to
override `include_orientations=False`. Return `(data, field)` and record
`stiffness_build` timing after synchronizing the selected Torch device.

- [ ] **Step 4: Install the new module**

Add this wheel-mode install entry next to `gpu_voxelizer.py`:

```cmake
install(FILES "${CMAKE_CURRENT_SOURCE_DIR}/../TexGen/material_fields.py"
    DESTINATION ${PYTEXGEN_INSTALL_DIR})
```

- [ ] **Step 5: Document the exact public workflow**

Add examples for C21 creation, per-yarn overrides, sparse GPU generation,
dense C21/ACDM views, file persistence, and the CPU-transfer boundary. State
the C21 and Voigt orders verbatim in all technical documentation. Keep
`README.md` and `README_pypi.md` package-facing sections synchronized.

- [ ] **Step 6: Run orchestration tests and build syntax checks**

Run:

```bash
python -m unittest test_material_fields.OrchestrationTest -v
python -m py_compile TexGen/material_fields.py TexGen/gpu_voxelizer.py
cmake -S . -B /tmp/pytexgen-cmake-check \
  -DBUILD_PYTHON_INTERFACE=OFF -DBUILD_RENDERER=OFF -DBUILD_GUI=OFF
```

Expected: tests PASS, Python compiles, and CMake configuration succeeds.

- [ ] **Step 7: Commit Task 6**

```bash
git add TexGen/material_fields.py Python/CMakeLists.txt \
        README.md README_pypi.md docs/voxel_backends.md test_material_fields.py
git -c user.name=yufangjie1643 \
    -c user.email=yufangjie1643@users.noreply.github.com \
    commit -m "feat: expose GPU material field workflow"
```

### Task 7: Correctness-Gated CPU/GPU Benchmark

**Files:**
- Create: `bench_gpu_material_fields.py`
- Create: `test_gpu_material_fields_benchmark.py`
- Modify: `README.md`
- Modify: `docs/voxel_backends.md`

**Interfaces:**
- Produces: CLI options `--resolutions`, `--repeat`, `--warmup`, `--device`, `--dtype`, `--json-out`, `--skip-cpu`, and `--keep-output`; JSON records per case/resolution/mode with correctness, timing, memory, environment, and speedup data.

- [ ] **Step 1: Add failing benchmark utility tests**

```python
class MaterialFieldBenchmarkTest(unittest.TestCase):
    def test_summary_uses_median_p90_and_speedup(self):
        result = bench.summarize_timings([5., 1., 3., 2., 4.], cpu_median=20.)
        self.assertEqual(result["median_s"], 3.0)
        self.assertAlmostEqual(result["p90_s"], 4.6)
        self.assertEqual(result["speedup"], 20.0 / 3.0)

    def test_gate_requires_correctness_and_every_large_case(self):
        records = self.acceptance_records(speedup=5.1)
        self.assertTrue(bench.evaluate_acceptance(records)["passed"])
        records[0]["occupancy_mismatch_fraction"] = 0.006
        self.assertFalse(bench.evaluate_acceptance(records)["passed"])
```

- [ ] **Step 2: Verify benchmark tests fail**

Run: `python -m unittest test_gpu_material_fields_benchmark -v`  
Expected: FAIL because the benchmark module is absent.

- [ ] **Step 3: Implement deterministic cases and timing harness**

Build `plain_2x2` and `multi_yarn_8x8` textiles with fixed dimensions,
sections, resolution, and default domains. Implement:

```python
def timed_cuda(fn, repeat, warmup):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    values = []
    for _ in range(repeat):
        torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        result = fn()
        torch.cuda.synchronize()
        values.append(time.perf_counter() - start)
    return result, values, torch.cuda.max_memory_allocated()
```

The compute CPU baseline uses `CTextile.GetPointInformation` on identical
structured centers and extracts `iYarnIndex`, `Orientation`, and `Up`. The
practical baseline uses `CRectangularVoxelMesh.SaveVoxelMesh` plus `.ori/.eld`
parsing in a temporary directory. The GPU tracks direct sparse generation and
combined `.npy` persistence.

- [ ] **Step 4: Add correctness and resource reporting**

Reject speedup records when occupancy/yarn mismatch exceeds `0.005`, matched
axis dot product is below `0.999`, FP64 relative stiffness error exceeds
`1e-10`, or FP32 error exceeds `5e-5`. Record median, P90, per-phase timings,
RSS, peak GPU allocated/reserved memory, output bytes, yarn count, CPU/GPU,
driver, Python, Torch, pytexgen, precision, and git commit.

- [ ] **Step 5: Run utility tests and a `16^3` smoke benchmark**

Run:

```bash
python -m unittest test_gpu_material_fields_benchmark -v
python bench_gpu_material_fields.py \
  --resolutions 16 --repeat 1 --warmup 1 --device cuda \
  --json-out build/material_fields_smoke.json
```

Expected: tests PASS; JSON contains both models and both modes, and all
correctness gates pass. No speed threshold applies at `16^3`.

- [ ] **Step 6: Commit benchmark implementation**

```bash
git add bench_gpu_material_fields.py test_gpu_material_fields_benchmark.py \
        README.md docs/voxel_backends.md
git -c user.name=yufangjie1643 \
    -c user.email=yufangjie1643@users.noreply.github.com \
    commit -m "bench: compare GPU material fields with TexGen CPU"
```

### Task 8: Full Verification and Performance Acceptance

**Files:**
- Modify only when evidence identifies a correctness or performance defect:
  `TexGen/gpu_voxelizer.py`, `TexGen/material_fields.py`,
  `bench_gpu_material_fields.py`, and their focused tests.

**Interfaces:**
- Consumes: all prior tasks.
- Produces: passing repository tests, build/import smoke check, and an accepted reference benchmark JSON.

- [ ] **Step 1: Create a CUDA-capable development environment**

Run:

```bash
./build.sh
.venv/bin/python -m pip install torch
.venv/bin/python -c \
  "import torch; assert torch.cuda.is_available(); print(torch.__version__, torch.cuda.get_device_name(0))"
```

Expected: pytexgen imports from `.venv`; Torch reports
`NVIDIA GeForce RTX 5090`.

- [ ] **Step 2: Run all focused and repository tests**

Run:

```bash
.venv/bin/python -m unittest \
  test_material_fields \
  test_gpu_voxelizer_backends \
  test_gpu_material_fields_benchmark -v
.venv/bin/python -m unittest discover -s . -p 'test_*.py'
```

Expected: all applicable tests PASS; optional-dependency tests skip only when
their dependency is unavailable.

- [ ] **Step 3: Run wheel build and import smoke check**

Run:

```bash
.venv/bin/python -m build
.venv/bin/python -c \
  "from pytexgen.material_fields import SparseStiffnessField, build_stiffness_field; print('material fields import OK')"
```

Expected: wheel and sdist build successfully and installed-package imports work.

- [ ] **Step 4: Run the reference performance suite**

Run:

```bash
.venv/bin/python bench_gpu_material_fields.py \
  --resolutions 64 128 256 \
  --repeat 5 --warmup 2 \
  --device cuda --dtype float32 \
  --json-out build/material_fields_rtx5090.json
```

Expected: correctness passes for every record. At both `128^3` and `256^3`,
both models have compute and practical-export median speedup `>=5.0`.

- [ ] **Step 5: Diagnose any failed gate before changing code**

If correctness fails, invoke `superpowers:systematic-debugging`, reduce the
failing case to the smallest grid, and add a failing regression test before
editing. If performance fails while correctness passes, capture
`torch.profiler` CPU/CUDA traces and phase timings, identify the largest
measured phase, then optimize only that phase under a new regression or
benchmark test. Re-run Step 4 after every measured optimization; do not weaken
the approved thresholds.

- [ ] **Step 6: Verify the final diff and artifact hygiene**

Run:

```bash
git diff --check
git status --short
git log --oneline -10
```

Expected: no whitespace errors; no generated meshes, benchmark output,
`.venv`, or root `AGENTS.md` is staged.

- [ ] **Step 7: Commit any evidence-driven final fixes**

```bash
git add TexGen/gpu_voxelizer.py TexGen/material_fields.py \
        bench_gpu_material_fields.py test_material_fields.py \
        test_gpu_voxelizer_backends.py test_gpu_material_fields_benchmark.py
git diff --cached --quiet || \
git -c user.name=yufangjie1643 \
    -c user.email=yufangjie1643@users.noreply.github.com \
    commit -m "perf: meet GPU material field acceptance"
```
