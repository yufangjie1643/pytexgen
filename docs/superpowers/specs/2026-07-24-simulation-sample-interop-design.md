# GPU Simulation Sample Interoperability Design

## Purpose

PyTexGen already produces GPU-resident voxel labels, sparse yarn directions,
and rotated engineering-Voigt stiffness fields. The next interface layer must
let GPU solvers and neural-network pipelines consume those results without
TexGen text export, implicit host transfers, or solver-specific data models.

This design introduces one validated, framework-neutral sample contract and
uses PyTorch and Voxel-ACDM as the first end-to-end consumers. It is the first
of four projects:

1. sample and tensor interoperability;
2. additional GPU solver adapters;
3. batched training datasets;
4. GPU classifier kernel and spatial-index optimization.

The later projects build on this contract but are not silently folded into
this implementation. The narrow Voxel-ACDM work in this design exists only to
prove the contract against one real GPU solver and correct the current
PyTexGen adapter; broader solver coverage belongs to project 2.

## Evidence and Design Influences

The current `VoxelGridData.to_dlpack()` returns a legacy, single-consumption
capsule and supports only labels and occupancy. PyTorch now recommends passing
objects implementing `__dlpack__` directly to `from_dlpack`; the Python Array
API additionally defines device, copy, and stream semantics. NVIDIA Warp uses
the same protocol for PyTorch, JAX, CuPy, and other consumers and recommends
reusing shared array objects rather than recreating conversion wrappers in hot
loops.

The local Voxel-ACDM implementations provide two useful lessons:

- dense anisotropic operators already accept CUDA `torch.Tensor` inputs;
- compact phase-LUT paths still normalize through NumPy and therefore force a
  CUDA-to-host-to-CUDA round trip.

Voxel-ACDM's experimental `VoxelInput` is useful as a schema reference, but it
coerces all fields through NumPy and cannot be the shared GPU contract.

References:

- <https://docs.pytorch.org/docs/stable/dlpack.html>
- <https://data-apis.org/array-api/2024.12/API_specification/generated/array_api.array.__dlpack__.html>
- <https://nvidia.github.io/warp/latest/user_guide/interoperability.html>

## Chosen Approach

Add a small `SimulationSample` composition object instead of expanding
`VoxelGridData` into a solver/training monolith or adding one conversion
function per consumer.

```text
SimulationSample
├── voxels: VoxelGridData
├── orientation: SparseOrientationField | None
├── stiffness: SparseStiffnessField | None
├── materials: MaterialTable
└── metadata: immutable provenance and user metadata
```

Existing containers remain the authoritative owners of arrays. The new object
validates their relationship and exposes stable field names. It does not copy
arrays during construction.

Alternatives rejected:

- **Adapter functions only:** initially smaller, but field names, layouts,
  validation, and copy behavior would diverge across every solver.
- **TensorDict or xarray as the public contract:** TensorDict makes PyTorch a
  mandatory dependency; xarray is oriented toward CPU labelled arrays. Either
  choice weakens the NumPy/Torch/JAX/Warp-neutral core.

## Public Types

### `MaterialTable`

`MaterialTable` is immutable and contains:

- `c21`: `(M, 21)` local engineering-Voigt stiffness values;
- `material_ids`: `(M,)` unique non-negative integer identifiers;
- `unit`: required non-empty string;
- `names`: optional tuple of `M` stable display names.

Rows are addressed by explicit IDs, not by assuming that IDs are dense or
equal to row indices. Material ID `0` is reserved for the matrix and must occur
exactly once. Construction rejects duplicate IDs, non-finite values, incorrect
C21 shape, and a missing unit. Positive-definite validation remains opt-in
because research workflows may use deliberately degraded materials.

### `SimulationSample`

The constructor accepts the five components shown above. It validates:

- identical `(Nz, Ny, Nx)` shape and TexGen flat-index order;
- identical storage backend and device for all resident tensor fields;
- matching sparse voxel indices and yarn IDs between direction and stiffness;
- every stiffness `material_id` exists in `MaterialTable`;
- `stiffness.matrix_c21` matches the material-table row whose ID is `0` within
  the dtype's existing stiffness tolerance;
- unit equality between stiffness and material table;
- metadata is JSON-compatible and copied on construction.

The sample exposes:

```python
sample.field_names
sample.array(name, *, layout="native", copy=False)
sample.to(storage=None, *, device=None, dtype=None, copy=False)
sample.as_dict(*, layout="native", copy=False)
```

Stable field names in version 1 are:

```text
voxel.yarn_id
voxel.material_id
voxel.occupancy
orientation.voxel_indices
orientation.yarn_ids
orientation.primary
orientation.secondary
stiffness.matrix_c21
stiffness.voxel_indices
stiffness.yarn_ids
stiffness.material_ids
stiffness.yarn_c21
material.ids
material.c21
```

`array()` returns the actual NumPy or Torch array whenever the requested field
already exists. A derived field may allocate only when its documented layout
requires materialization. Unknown fields and unavailable optional fields raise
clear errors listing the available names.

`voxel.material_id` is a physical material field, not an alias for
`VoxelGridData.material_id()`. When stiffness is present it is constructed by
filling matrix voxels with material ID `0` and scattering
`SparseStiffnessField.material_ids` at the sparse voxel indices. This preserves
shared yarn materials and per-yarn overrides. When stiffness is absent,
`voxel.material_id` is unavailable rather than silently treating
`yarn_id + 1` as a material identity.

`layout="native"` preserves current storage:

- voxel grids: `(Nz, Ny, Nx)`;
- sparse directions: `(N, 3)`;
- sparse stiffness: `(N, 21)`;
- material stiffness: `(M, 21)`.

Later training and solver adapters may request explicit layouts such as
`"channels_first"` or `"acdm"`. Layout conversion is never described as
zero-copy.

## Interoperability and Copy Semantics

The sample itself must not implement `__dlpack__`: DLPack represents one array,
while a sample contains multiple arrays. Instead:

```python
field = sample.array("stiffness.yarn_c21")
tensor = torch.from_dlpack(field)
```

NumPy and Torch fields already implement the standard array-level protocol.
This also lets Warp, JAX, and CuPy consume the same returned object.

Rules:

- same backend, device, dtype, and native layout with `copy=False` returns the
  original array object;
- a same-device DLPack consumer shares the allocation;
- `array(..., copy=False)` never performs cross-device transfer, dtype
  conversion, densification, stacking, or relayout;
- `sample.to(...)` is the explicit conversion boundary and may allocate when
  its requested storage, device, or dtype differs; its `copy` argument only
  forces a clone when no conversion would otherwise be required;
- if `copy=False` cannot be honored, raise `ValueError` instead of silently
  allocating;
- CUDA stream ordering is delegated to each array's standard `__dlpack__`
  implementation; PyTexGen must not create a legacy capsule that bypasses the
  consumer-provided stream;
- the existing `VoxelGridData.to_dlpack()` remains available for compatibility
  during this phase but is documented as legacy.

## Voxel-ACDM Corrections and Adapter Boundary

`to_acdm_phase_ids()` becomes backend-preserving:

- NumPy input returns NumPy;
- Torch input returns Torch on the same device;
- validation occurs on Python material mappings before any `uint8` cast;
- result shape is `(B, Nz, Ny, Nx)` when `batch=True`;
- IDs outside `0..15` are rejected before solver construction.

The isotropic API replaces the ambiguous two-material arguments internally
with an explicit phase table:

```python
phase_materials = {
    0: {"E": matrix_E, "Nu": matrix_nu},
    1: {"E": yarn_E, "Nu": yarn_nu},
}
```

Legacy `matrix_material` and `yarn_material` calls remain supported and build
this table. Every phase appearing in the voxel tensor must have a table row;
unused table rows are allowed. The adapter constructs LUTs through an explicit
phase-to-row mapping rather than assuming two entries.

The anisotropic adapter accepts `SimulationSample`, requests an explicit ACDM
dense layout on the current device, and passes the resulting Torch tensor
directly to a Voxel-ACDM constructor that accepts Torch. This phase does not
modify the external Voxel-ACDM repository. If the installed solver version
cannot accept a Torch tensor, the adapter raises a compatibility error instead
of silently moving through NumPy.

The current compact Voxel-ACDM phase-LUT constructor still calls
`numpy.asarray()` even when handed a Torch tensor. PyTexGen therefore separates
the verified GPU phase mapping from external solver construction:

- the general dense-stiffness Voxel-ACDM path is the first required
  end-to-end same-device integration;
- compact isotropic construction accepts an explicit
  `allow_host_phase_pack=False` option;
- with the default `False`, an external solver lacking CUDA phase support
  raises a compatibility error before any transfer;
- setting it to `True` opts into the legacy host packing path and records
  `phase_pack_device="cpu"` in timings/provenance;
- when a Voxel-ACDM version advertises CUDA phase support, the same adapter
  passes the CUDA tensor without changing the public PyTexGen API.

Solver outputs retain a GPU `C_eff_tensor` when the external solver produces
one. A NumPy `C_eff` convenience view is created only on explicit request or
when the external solver itself returns NumPy.

## Accuracy and Error Handling

Accuracy is part of the interface, not only a benchmark concern:

- C21 packing is fixed to the existing row-major upper triangle of engineering
  Voigt order `(xx, yy, zz, yz, xz, xy)`;
- all conversions preserve voxel order `ix + iy*nx + iz*nx*ny`;
- no adapter may infer units;
- orientation/stiffness sparse indices must match exactly;
- GPU phase mapping must equal the NumPy reference bit-for-bit;
- GPU C21 layouts must match existing dense views within the current dtype
  tolerances;
- an adapter reports unsupported dtype/device/layout before external solver
  construction.

Errors use `TypeError` for unsupported object types and `ValueError` for
invalid values, shapes, mappings, devices, and layouts. Optional dependency
errors name the missing package and the installation extra.

## Persistence and Provenance

Version 1 persistence reuses the existing NumPy directory and archive
mechanisms. `SimulationSample` adds one manifest tying voxel, direction,
stiffness, and material data together without duplicating array files.

The manifest records:

- schema name and version;
- field filenames and dtypes;
- grid shape, voxel order, and units;
- PyTexGen version and optional git commit;
- caller metadata and generation parameters.

Loading validates the manifest before constructing a sample. Memory-mapped
NumPy arrays remain supported. Direct GPU storage, sharded training datasets,
checksums, dataset splits, and asynchronous prefetch belong to the subsequent
training-data project.

## Testing and Acceptance

Unit coverage must demonstrate:

1. construction performs no copy and rejects inconsistent components;
2. stable field names return the original arrays;
3. `copy=False` rejects unavoidable conversions;
4. NumPy and Torch `to()` conversions preserve identifiers and metadata;
5. DLPack import aliases the original CPU and CUDA allocation;
6. a non-default CUDA stream consumes a freshly produced field correctly;
7. phase validation precedes `uint8` conversion;
8. per-yarn phase mappings build complete material LUTs;
9. sparse C21 and ACDM dense layouts match the existing reference;
10. save/load round trips preserve the schema and permit NumPy memory mapping.

Integration coverage must demonstrate:

- a real TexGen textile produces a valid `SimulationSample`;
- NumPy and CUDA phase grids are identical;
- PyTorch consumes every GPU-resident public field without a host copy;
- the general Voxel-ACDM path receives a same-device dense-stiffness tensor;
- compact isotropic integration either receives a same-device tensor from a
  capable external version or rejects it without transferring when
  `allow_host_phase_pack=False`;
- the anisotropic path matches the existing CPU/TexGen material field within
  current FP32/FP64 tolerances;
- all existing repository tests remain green.

The CUDA tests skip only when CUDA or the external optional solver is absent;
core NumPy/Torch-CPU contract tests always run. A checked benchmark reports
host-device transfer bytes and fails if the direct CUDA handoff performs a
full field-sized device-to-host copy.

## Compatibility and Rollout

This is an additive change:

- no existing public class or persistence format is removed;
- legacy import paths continue to work;
- wheel installation includes the new module;
- README and backend documentation show NumPy, Torch, DLPack, Warp/JAX-style,
  and Voxel-ACDM handoff examples;
- legacy `to_dlpack()` is marked as such but not deprecated with a runtime
  warning until consumers have migrated.

After this contract is implemented and verified, the next design will add
`SimulationDataset`, batching/collation, sharded storage, pinned-memory
prefetch, deterministic splits, and physics-consistent augmentation.
