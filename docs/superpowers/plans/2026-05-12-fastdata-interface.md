# Fastdata Interface Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a stable Python-side data interface that future C++/nanobind `_fastdata` providers can satisfy without changing the public voxelizer API.

**Architecture:** Keep SWIG as the broad object API. Add a structure-of-arrays snapshot bundle and provider hook in `TexGen/gpu_voxelizer.py`, with the existing SWIG extraction as fallback. Extend `VoxelGridData` with a DLPack bridge for tensor consumers.

**Tech Stack:** Python dataclasses, numpy, optional torch DLPack, existing unittest smoke tests.

---

### Task 1: Snapshot Bundle Tests

**Files:**
- Modify: `test_gpu_voxelizer_backends.py`

- [ ] **Step 1: Write failing tests**

Add tests that call `SnapshotBundle.from_snapshots(...)`, `SnapshotBundle.to_snapshots()`, `voxelize_snapshot_bundle_data(...)`, and `extract_snapshot_bundle(..., provider=...)`.

- [ ] **Step 2: Run tests to verify failure**

Run:

```powershell
.\.venv\Scripts\python.exe test_gpu_voxelizer_backends.py
```

Expected: failures because `SnapshotBundle`, `voxelize_snapshot_bundle_data`, and `extract_snapshot_bundle` are not defined.

### Task 2: Snapshot Bundle Implementation

**Files:**
- Modify: `TexGen/gpu_voxelizer.py`

- [ ] **Step 1: Implement `SnapshotBundle`**

Add a dataclass with flat arrays and offset arrays:

```python
positions, tangents, ups, sides, node_offsets
sections, section_offsets
translations, translation_offsets
aabb
```

- [ ] **Step 2: Implement conversion helpers**

Add `SnapshotBundle.from_snapshots(...)`, `SnapshotBundle.to_snapshots()`, `_coerce_snapshot_bundle(...)`, and `voxelize_snapshot_bundle_data(...)`.

- [ ] **Step 3: Run tests**

Run:

```powershell
.\.venv\Scripts\python.exe test_gpu_voxelizer_backends.py
```

Expected: new snapshot tests pass.

### Task 3: Fastdata Provider Hook

**Files:**
- Modify: `TexGen/gpu_voxelizer.py`

- [ ] **Step 1: Implement provider loader**

Add `_load_fastdata_provider()` that tries package-local `_fastdata` and legacy `TexGen._fastdata`.

- [ ] **Step 2: Implement `extract_snapshot_bundle(...)`**

Accept an explicit provider for tests. If provider returns bundle-like data, coerce it. If no provider exists, call existing `extract_snapshots(...)`.

- [ ] **Step 3: Run provider tests**

Run:

```powershell
.\.venv\Scripts\python.exe test_gpu_voxelizer_backends.py
```

Expected: provider hook test passes and normal fallback remains unchanged.

### Task 4: DLPack Bridge

**Files:**
- Modify: `TexGen/gpu_voxelizer.py`
- Modify: `test_gpu_voxelizer_backends.py`

- [ ] **Step 1: Write failing DLPack test**

Test `VoxelGridData.to_dlpack("yarn_id")`; if torch is absent, assert a clear `ImportError`.

- [ ] **Step 2: Implement `VoxelGridData.to_dlpack(...)`**

Support `field="yarn_id"`, `field="material_id"`, and `field="occupancy"` by converting to torch if needed, then returning `torch.utils.dlpack.to_dlpack(...)`.

- [ ] **Step 3: Run tests**

Run:

```powershell
.\.venv\Scripts\python.exe test_gpu_voxelizer_backends.py
```

Expected: DLPack test passes.

### Task 5: Documentation and Verification

**Files:**
- Modify: `docs/voxel_backends.md`
- Modify: `docs/cross_language_modernization_report.md`

- [ ] **Step 1: Document the architecture boundary**

Describe SWIG object API, optional `_fastdata` provider, `SnapshotBundle`, and DLPack handoff.

- [ ] **Step 2: Run smoke tests**

Run:

```powershell
.\.venv\Scripts\python.exe test_gpu_voxelizer_backends.py
.\.venv\Scripts\python.exe test_tetrahedral_mesh.py
```

Expected: tests pass; native TetGen export may remain skipped unless `PYTEXGEN_RUN_TETGEN_NATIVE=1`.
