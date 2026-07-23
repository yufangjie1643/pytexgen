# GPU Sparse Orientation and Stiffness Fields

**Status:** Approved design  
**Date:** 2026-07-23

## Summary

pytexgen will generate per-voxel yarn orientation fields and material stiffness
fields without falling back from Torch to NumPy. The primary representation is
exact and sparse: matrix voxels have no orientation entries and share one
matrix stiffness, while yarn voxels store flat voxel indices, two direction
vectors, yarn/material identifiers, and 21 independent engineering-Voigt
stiffness coefficients.

The TexGen geometry snapshot remains a CPU/C++ operation. Once the snapshot is
uploaded, Torch classification, orientation capture, material lookup, stiffness
rotation, and optional dense materialization remain on the selected Torch
device. CPU transfer occurs only when the caller explicitly requests NumPy
storage or saves files.

## Goals

- Generate sparse yarn orientation fields directly on Torch CPU, CUDA, or other
  supported Torch devices.
- Generate rotated global stiffness coefficients on the same device.
- Support a default yarn material plus per-`yarn_id` local material overrides.
- Store stiffness as the 21 independent coefficients of a symmetric `6 x 6`
  engineering-Voigt matrix.
- Avoid direction work for matrix voxels and avoid repeating matrix stiffness
  per voxel in the default representation.
- Materialize dense C21 or full Voigt fields only when requested.
- Persist portable compact fields as `.npz` or directory-based `.npy` data.
- Demonstrate at least a 5x median speedup over the original TexGen CPU
  workflow at both `128^3` and `256^3` on the reference RTX 5090 system.

## Non-goals

- Assemble element stiffness matrices or a global finite-element stiffness
  matrix.
- Add a solver dependency such as Voxel-ACDM to pytexgen.
- Quantize or cluster continuous orientations.
- Store a fourth-order `3 x 3 x 3 x 3` tensor per voxel.
- Move TexGen's C++ geometry construction and snapshot extraction to the GPU.

## Current State

`VoxelGridData` supports dense `orientation1` and `orientation2` fields, each
shaped `(Nz, Ny, Nx, 3)`. Matrix voxels contain zeros. Requesting orientations
with the Torch classification backend currently raises an error and forces
callers to classify with NumPy before converting the resulting arrays to Torch.

The Torch classifier already computes nearest-node tangent and up vectors while
testing yarn membership, but discards them. Voxel-ACDM contains a related GPU
stiffness-field builder, but pytexgen must provide this capability independently
and expose a stable package-level data contract.

## Chosen Approach

The implementation uses an exact sparse representation rather than a dense
field or an orientation lookup table. A dense representation is simple but
wastes direction storage in matrix regions and repeats the same matrix
stiffness. An orientation table would reduce storage further only by quantizing
continuous directions or by creating nearly one table entry per yarn voxel.
Both outcomes are unsuitable for the general export API.

An optional lazy form may retain orientation data, material identifiers, and
the local C21 material table without immediately rotating every yarn voxel.
The normal stiffness-field export materializes rotated sparse `yarn_C21`.

## Architecture

### `TexGen/gpu_voxelizer.py`

The voxelizer remains responsible only for geometry classification. Its Torch
classifier gains orientation capture and can return either:

- the existing dense direction fields for backward compatibility; or
- a `SparseOrientationField` containing yarn voxels only.

For each classification chunk, the classifier keeps the winning yarn,
tangent, and up vector together. In sparse mode it selects the yarn mask,
appends global flat indices and directions, then releases chunk temporaries.
It does not allocate dense direction arrays.

### `TexGen/material_fields.py`

A new standalone module owns:

- sparse orientation and stiffness data classes;
- C21 packing and unpacking;
- isotropic and orthotropic engineering-constant helpers;
- direction-frame validation and orthonormalization;
- NumPy and Torch stiffness rotation;
- per-yarn material lookup;
- sparse-to-dense conversion; and
- compact persistence.

`Python/CMakeLists.txt` installs the module into the `pytexgen` wheel.

### `VoxelGridData`

Existing `orientation1` and `orientation2` fields remain valid for dense mode.
A new optional `sparse_orientation` field holds compact output.
`VoxelGridData.to("numpy" | "torch")`, `save_npz`, `save_npy_dir`, and their
load counterparts preserve the sparse field and its metadata.

## Data Model

### `SparseOrientationField`

```text
voxel_indices  int64  (Nyarn,)
yarn_ids       int32  (Nyarn,)
orientation1   float  (Nyarn, 3)
orientation2   float  (Nyarn, 3)
grid_shape            (Nz, Ny, Nx)
order                 "ix + iy*nx + iz*nx*ny"
```

Indices are strictly increasing and refer to the same flat order as
`VoxelGridData.yarn_id`. `orientation1` is the local material 1-axis and yarn
tangent. `orientation2` is the reference up direction used to reconstruct the
local 2-axis.

### `SparseStiffnessField`

```text
matrix_c21       float  (21,)
voxel_indices    int64  (Nyarn,)
yarn_ids         int32  (Nyarn,)
material_ids     int32  (Nyarn,)
yarn_c21         float  (Nyarn, 21)
grid_shape
order
dtype/device/unit metadata
```

The field owns its index arrays so it can be saved and consumed independently.
Material ID zero is reserved for matrix. Yarn material IDs are stable table
indices derived from the default material and sorted explicit `yarn_id`
overrides.

## Numerical Conventions

The engineering-Voigt component order is:

```text
(xx, yy, zz, yz, xz, xy)
```

C21 stores the upper triangle of the symmetric `6 x 6` stiffness matrix in
row-major order:

```text
C11 C12 C13 C14 C15 C16
    C22 C23 C24 C25 C26
        C33 C34 C35 C36
            C44 C45 C46
                C55 C56
                    C66
```

Input stiffness may be `(21,)` or symmetric `(6, 6)`. Convenience functions
construct C21 from isotropic `(E, nu)` or orthotropic
`(E1,E2,E3,nu12,nu13,nu23,G12,G13,G23)` engineering constants.

For every yarn voxel:

1. normalize `orientation1` as local `e1`;
2. remove the `e1` component from `orientation2` and normalize the remainder
   as local `e2`;
3. compute `e3 = e1 x e2`; and
4. rotate local stiffness through Mandel space before converting back to
   engineering Voigt and packing C21.

Mandel-space rotation avoids ambiguous engineering-shear scaling. Rotation is
performed in configurable chunks. Full `6 x 6` tensors exist only as chunk
temporaries and are packed immediately to 21 components.

Inputs must be finite. A supplied `6 x 6` matrix must be symmetric within a
documented tolerance. Zero or collinear direction frames are errors and report
the affected flat indices. Positive-definiteness validation is optional because
research workflows may intentionally use semidefinite or degraded materials.
Units are caller-defined but an optional unit label is preserved in metadata.

## Public API

Sparse direction generation:

```python
data = voxelize_textile_data(
    textile,
    nx=128,
    ny=128,
    nz=128,
    backend="torch",
    device="cuda",
    output="backend",
    include_orientations=True,
    orientation_storage="sparse",
)
```

Stiffness generation:

```python
field = build_stiffness_field(
    data,
    matrix_stiffness=matrix_c21,
    default_yarn_stiffness=default_yarn_c21,
    yarn_stiffness_by_id={0: warp_c21, 1: weft_c21},
    output="sparse",
    chunk_voxels=65536,
    validate_positive_definite=False,
    unit="Pa",
)
```

An optional `voxelize_textile_material_fields(...)` convenience function
returns `(data, field)` without changing the two-stage implementation boundary.

Dense views are explicit:

```python
dense_c21 = field.to_dense_c21()       # (Nz, Ny, Nx, 21)
dense_c66 = field.to_dense_voigt()     # (6, 6, Nz, Ny, Nx)
acdm_c66 = field.to_acdm(batch=True)   # (1, 6, 6, Nz, Ny, Nx)
```

`output="backend"` preserves tensors on the classification device.
`output="torch"` accepts a target device. `output="numpy"` is an explicit CPU
conversion. No unsupported device silently falls back to NumPy.

## Persistence

The compact directory format is:

```text
material_fields/
  metadata.json
  voxel_indices.npy
  yarn_ids.npy
  material_ids.npy
  orientation1.npy
  orientation2.npy
  matrix_c21.npy
  yarn_c21.npy
```

`metadata.json` records a format name and version, shape, flat order, Voigt
order, C21 packing order, dtype, original device, unit, and array manifest.
A single-file `.npz` variant stores the same logical fields. Saving is
documented as a GPU-to-CPU boundary. Loading can return NumPy arrays, memory-map
directory arrays, or upload directly to a selected Torch device.
`save_material_field_bundle(path, data.sparse_orientation, field)` and its
matching loader own this combined schema; the existing `VoxelGridData` save
methods continue to support geometry plus direction data without requiring a
stiffness field.

The implementation uses ordinary contiguous index arrays rather than
`torch.sparse_coo_tensor`. The compact payload has multiple vector/tensor
components per structured voxel and is consumed primarily through indexed
gather/scatter, for which sorted indices are simpler and more portable.

## Error Handling

- Torch output without Torch installed raises an actionable `ImportError`.
- A requested CUDA device that is unavailable raises instead of falling back.
- Stiffness generation without sparse or dense orientations raises
  `ValueError`.
- Missing per-yarn entries use `default_yarn_stiffness`; omitting both an
  override and a default is an error listing the missing yarn IDs.
- Device and dtype conversions are explicit. All returned arrays belonging to
  one field must share a backend and compatible device.
- Dense materialization checks allocation size and includes the requested
  shape and estimated bytes in an out-of-memory diagnostic.
- Loaders validate schema version, shapes, sorted unique indices, index range,
  finite values, and matching array lengths.

## Correctness Testing

Unit and integration coverage includes:

- exact C21/symmetric-Voigt packing round trips;
- isotropic and orthotropic engineering-constant reference values;
- identity rotation invariance;
- a known 90-degree orthotropic rotation;
- NumPy/Torch CPU FP64 parity;
- optional CUDA FP32 and FP64 parity;
- sparse-to-dense equality against a dense reference;
- exclusion of matrix voxels from direction data;
- matrix C21 recovery after dense materialization;
- default and per-yarn material selection;
- chunk-size invariance;
- `.npz` and `.npy` round trips;
- backward compatibility for dense orientations; and
- a guard proving that sparse Torch classification and stiffness construction
  do not invoke NumPy conversion helpers.

Real-model comparisons run only after synthetic numerical tests pass. For
voxels assigned to the same yarn, normalized direction agreement is checked
using a dot product of at least `0.999` for each corresponding oriented axis;
axis signs are not discarded because a general C21 tensor can contain coupling
terms that are sign-sensitive. Occupancy and yarn-ID discrepancies against the
C++ reference are reported; a benchmark result is invalid if more than 0.5% of
voxels disagree, because speed comparisons between materially different fields
are not meaningful. Relative Frobenius error for rotated stiffness is at most
`1e-10` in FP64 and `5e-5` in FP32.

## Performance Validation

A new benchmark uses both a simple plain weave and a complex multi-yarn RVE at
`64^3`, `128^3`, and `256^3`.

Two comparisons are reported:

1. **Compute path:** original TexGen CPU point classification/orientation
   queries versus GPU classification, sparse orientation capture, and C21
   rotation, excluding disk I/O.
2. **Practical export path:** original `CRectangularVoxelMesh.SaveVoxelMesh`
   `.inp/.ori/.eld` generation and parsing versus GPU compact field generation
   and `.npy` persistence.

The benchmark:

- uses identical textiles, domains, resolutions, and material definitions;
- performs GPU warm-up;
- synchronizes CUDA around every timed GPU phase;
- runs at least five measured repetitions;
- reports median and P90 duration;
- reports snapshot, upload, classification, orientation, rotation, and save
  phases separately;
- records peak process RAM, peak allocated/reserved VRAM, yarn voxel count, and
  output bytes; and
- emits machine-readable JSON including CPU, GPU, driver, Torch, pytexgen,
  precision, and commit metadata.

`64^3` is informational because launch and transfer overhead may dominate. On
the reference NVIDIA GeForce RTX 5090 system, both real models must achieve at
least a 5x median speedup for the compute path and the practical export path at
both `128^3` and `256^3`. No speedup is claimed unless the corresponding
correctness checks pass. If the threshold is missed, profiling and optimization
continue or the unmet acceptance criterion is reported explicitly.

## Acceptance Criteria

1. Torch classification directly returns sparse orientations on the selected
   device.
2. Matrix voxels consume no direction storage or direction-rotation work.
3. Per-yarn material overrides and a default yarn material work together.
4. Sparse stiffness stores exactly 21 global coefficients per yarn voxel and
   one matrix C21 value.
5. Rotation is chunked and creates no whole-grid `6 x 6` intermediate.
6. Dense C21 and Voxel-ACDM-compatible Voigt layouts can be materialized on
   demand.
7. Compact persistence preserves every semantic field and validates on load.
8. Existing dense orientation behavior remains compatible.
9. CPU, Torch CPU, and available CUDA correctness tests pass.
10. The reference RTX 5090 benchmarks satisfy the approved 5x thresholds at
    `128^3` and `256^3` for both test textiles and both comparison modes.
