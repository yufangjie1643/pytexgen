# GPU Training Data Pipeline Design

## Goal

Add a versioned, physics-aware training-data layer on top of
`SimulationSample` so PyTexGen datasets can be generated, audited, split,
memory-mapped, batched, asynchronously transferred to a GPU, and consumed by
neural networks without losing material identity, stiffness convention, units,
or provenance.

The first verified learning loop predicts the 21 independent entries of the
effective engineering-Voigt stiffness tensor from a fixed-resolution woven
voxel RVE. The underlying field registry and batch contract remain generic so
later displacement, stress, damage, and other spatial targets do not require a
new storage format.

## Context and References

The existing `SimulationSample` contract validates one voxel/direction/C21
sample and exposes its resident arrays without implicit conversion.
`simulation_io.py` persists one sample in a directory of `.npy` arrays or an
`.npz` archive. It deliberately leaves sharding, checksums, dataset splits,
prefetch, and augmentation to this project.

Related local repositories reveal recurring problems:

- Voxel-ACDM production data uses compressed `chunk_*.npz` files and
  project-specific scripts rather than a reusable dataset API.
- ViT_FEM and WMGN repeat one-off `Dataset`, normalization, random split, and
  `DataLoader` code. Several paths load whole samples before field selection.
- random per-sample splitting cannot enforce parent-geometry or augmentation
  group isolation.
- existing rotation augmentation does not provide a shared proof that spatial
  inputs and anisotropic C21 labels transform together.

The design adopts the reader/transform/collate/prefetch separation used by
[NVIDIA PhysicsNeMo Datapipes](https://docs.nvidia.com/physicsnemo/latest/physicsnemo/api/datapipes/physicsnemo.datapipes.html),
the custom-batch `pin_memory()` contract supported by
[PyTorch DataLoader](https://docs.pytorch.org/docs/stable/data.html), and the
memory-mapped structured-field approach documented by
[TensorDict](https://docs.pytorch.org/tensordict/stable/storage.html). PyTexGen
does not require those packages: NumPy remains the storage dependency and
PyTorch remains optional. Zarr v3 is deferred because its chunk/shard layer
would add a required dependency and duplicate the first native backend; a
future reader may target the same public example/batch contract.

## Scope

Version 1 includes:

- immutable dataset, field, example, ragged-array, and batch contracts;
- native uncompressed `.npy` shards with fixed and ragged fields;
- crash-safe streaming writes and resumable staging;
- immutable sample index, group-safe splits, checksums, label provenance, and
  rejection records;
- selective memory-mapped reads;
- PyTorch collation, pinned memory, non-blocking device transfer, and
  main-process CUDA-stream prefetch;
- deterministic proper cubic rotations that transform spatial fields,
  direction vectors, global stiffness fields, and effective C21 labels;
- train-only normalization statistics and explicit standardization;
- an end-to-end C21 regression smoke loop and checked loader benchmark.

Version 1 does not include:

- online TexGen generation from a `DataLoader` worker;
- mixed spatial resolutions in one dataset;
- padding or resampling;
- reflections or arbitrary-angle interpolation;
- distributed object storage, Zarr, HDF5, TensorDict, or PhysicsNeMo as a
  required backend;
- a production neural-network architecture or training framework;
- solver-label generation orchestration beyond accepting and validating
  caller-provided targets and provenance.

## Module Boundaries

Create three focused public modules:

1. `training_data.py`
   - immutable schemas and containers;
   - ragged representation and collation;
   - C21 semantics, normalization, deterministic group splitting, and cubic
     rotations.
2. `training_io.py`
   - version-1 directory schema;
   - streaming writer, resume journal, atomic publication, checksum validation,
     audit, and selective memory-mapped dataset reader.
3. `torch_training.py`
   - optional PyTorch `DataLoader` factory;
   - recursive custom-batch pinning and device transfer;
   - CUDA stream prefetch and transfer accounting.

`SimulationSample` remains the owner and validator of one physical sample.
Training targets are not added to it. `simulation_io.py` remains the portable
single-sample format and is not changed into a dataset store.

## Public Contracts

### Field and dataset schema

```python
@dataclass(frozen=True)
class TrainingFieldSpec:
    name: str
    role: Literal["input", "target"]
    layout: Literal["fixed", "ragged"]
    dtype: str
    shape: tuple[int, ...]
    unit: str | None = None
    semantic: str | None = None
    ragged_group: str | None = None


@dataclass(frozen=True)
class TrainingDatasetSchema:
    inputs: tuple[TrainingFieldSpec, ...]
    targets: tuple[TrainingFieldSpec, ...]
    grid_shape: tuple[int, int, int]
    voxel_order: str
    shard_size: int
    statistics_fields: tuple[str, ...]
    geometry_digest_field: str = "voxel.material_id"
```

`shape` excludes the sample/batch axis. For ragged fields it describes the
value trailing shape: orientation values use `(3,)`, C21 values use `(21,)`,
and sparse indices use `()`. Fields in one `ragged_group` share exactly one
offset array. The built-in groups are:

- `yarn_voxels`: sparse voxel indices, yarn IDs, material IDs, directions, and
  global yarn C21;
- `materials`: material table IDs and material C21 rows.

The first target spec is:

```python
TrainingFieldSpec(
    name="effective_c21",
    role="target",
    layout="fixed",
    dtype="float64",
    shape=(21,),
    unit="GPa",
    semantic="engineering_voigt_c21",
)
```

The semantic fixes component order to `(xx, yy, zz, yz, xz, xy)` and C21
packing to the row-major upper triangle already used by
`material_fields.py`. Units are never inferred. All C21 fields and targets
must declare a non-empty unit.

### Examples, ragged values, and batches

```python
@dataclass(frozen=True)
class RaggedArray:
    values: Any
    offsets: Any


@dataclass(frozen=True)
class TrainingExample:
    inputs: Mapping[str, Any]
    targets: Mapping[str, Any]
    sample_id: str
    group_id: str
    split: Literal["train", "validation", "test"]
    metadata: Mapping[str, Any]


@dataclass
class SimulationBatch:
    inputs: Mapping[str, Any]
    targets: Mapping[str, Any]
    sample_ids: tuple[str, ...]
    group_ids: tuple[str, ...]
    metadata: tuple[Mapping[str, Any], ...]

    def pin_memory(self) -> "SimulationBatch": ...
    def to(self, device, *, non_blocking=False) -> "SimulationBatch": ...
    def as_dict(self) -> Mapping[str, Any]: ...
    @property
    def nbytes(self) -> int: ...
```

On disk, `TrainingExample` values are read-only NumPy views or memmap slices.
Collation makes one owned, contiguous CPU batch allocation:

- fixed fields stack to `(B, *shape)`;
- ragged fields concatenate values and return batch offsets of length `B + 1`;
- metadata stays on the CPU;
- no field outside the requested input/target selection is opened.

`SimulationBatch` contains Torch CPU tensors when produced by the public
PyTorch collator. Each tensor can be handed directly to a DLPack consumer;
the batch itself does not implement `__dlpack__`.

### Writer

```python
writer = SimulationDatasetWriter.create(
    path,
    schema=schema,
    quality=DatasetQualityPolicy(...),
    resume=False,
)
writer.append(
    sample,
    targets={"effective_c21": c_eff_c21},
    sample_id="woven-000123",
    group_id="geometry-seed-42",
    split="train",
    provenance=solver_info,
    metadata={"weave": "plain"},
)
writer.reject(
    sample_id="woven-000124",
    stage="label",
    reason="PCG residual exceeds 1e-8",
    metadata=parameters,
)
writer.finalize()
```

The writer is a context manager. `create()` refuses an existing published
target. It writes to a sibling staging directory whose name contains the
target name and `.incomplete`. Each complete shard is flushed, checksummed,
and appended to `journal.jsonl`. `resume=True` accepts only a staging directory
whose serialized schema and generation configuration digest match the current
request.

`append()`:

1. validates the `SimulationSample`;
2. selects only schema fields;
3. explicitly materializes derived fields such as physical
   `voxel.material_id`;
4. validates targets and provenance;
5. computes the configured geometry digest;
6. rejects duplicate sample IDs and geometry digests;
7. enforces one split per `group_id`;
8. updates train-only statistics;
9. buffers one shard.

`finalize()` writes the last shard, validates the complete staging dataset,
writes the final manifest, fsyncs metadata, and atomically renames staging to
the requested target. A failed run never publishes a partial dataset.

### Reader and audit

```python
dataset = SimulationDataset(
    path,
    split="train",
    inputs=("voxel.material_id", "material.c21"),
    targets=("effective_c21",),
    verify="shard",
    transform=None,
)

example = dataset[0]
report = audit_simulation_dataset(path, verify="sample")
```

Verification levels are:

- `manifest`: schema, paths, dtypes, shapes, sample index, split/group
  isolation, and declared shard metadata;
- `shard`: manifest checks plus lazy SHA-256 verification of each requested
  field/offset file once per worker; this is the default and does not open
  unselected field arrays;
- `sample`: all shard files are verified and every configured geometry digest
  is recomputed; intended for dataset publication audits.

The reader is map-style and picklable. It stores paths and index metadata, not
open file handles, so each DataLoader worker opens its own mmap handles lazily.
`set_epoch(epoch)` propagates deterministic transform state. Missing requested
fields, unsupported schema versions, unsafe paths, checksum mismatches, and
group leakage fail before a bad example is returned.

### Splits and normalization

```python
split_by_group = deterministic_group_split(
    group_ids,
    ratios={"train": 0.8, "validation": 0.1, "test": 0.1},
    seed=42,
    strata=stratum_by_group,
)
```

The function hashes `(seed, group_id)` rather than relying on input order.
Every group has one optional stratum. It deterministically shuffles within
each stratum, assigns complete groups, and guarantees that every nonzero split
receives a group when the stratum has enough groups. The writer remains the
authority and revalidates group isolation.

Statistics use an online float64 Welford accumulator and only samples whose
stored split is `train`. Per-component mean, variance, standard deviation,
minimum, maximum, count, source split, and field unit are stored. A zero
standard deviation is recorded as `1` for division and flagged by a constant
mask. `StandardizeFields` requires matching field units and never changes
integer identifiers.

## Directory Format

The published layout is:

```text
dataset/
├── dataset.json
├── samples.jsonl
├── rejections.jsonl
└── shards/
    ├── shard_00000/
    │   ├── fields/
    │   │   ├── voxel_material_id.npy
    │   │   ├── sparse_voxel_indices.values.npy
    │   │   ├── orientation_primary.values.npy
    │   │   ├── yarn_voxels.offsets.npy
    │   │   ├── material_ids.values.npy
    │   │   ├── material_c21.values.npy
    │   │   └── materials.offsets.npy
    │   ├── targets/
    │   │   └── effective_c21.npy
    │   └── shard.json
    └── shard_00001/
```

The root manifest uses schema `pytexgen.simulation_dataset`, version `1`. It
contains:

- serialized `TrainingDatasetSchema`;
- complete input/target field registry and aliases;
- grid shape and voxel order;
- shard path, row count, byte count, and SHA-256 for every file;
- sample and rejection counts;
- split and group counts;
- train-only field statistics;
- quality policy;
- dataset-generation configuration and its canonical SHA-256;
- PyTexGen version/commit and creation timestamp;
- C21 convention and units.

`samples.jsonl` contains one JSON object per accepted sample:

- sample ID, group ID, split;
- shard number and row;
- geometry digest;
- selected JSON metadata;
- target provenance including solver commit, element formulation, arithmetic
  dtype, tolerance, maximum residual, iteration count, and wall time.

No pickle or object-dtype array is allowed. JSON encoding rejects NaN and
infinity. Field names map to safe relative paths through the manifest; callers
never construct paths from unvalidated names.

Fixed fields use `(S, *shape)`. A ragged group with sample lengths `n_i` uses
one int64 offsets array `[0, n_0, n_0+n_1, ...]` and one values array per group
field. Orientation and stiffness topology aliases share the same stored sparse
index/yarn-ID arrays, matching the single-sample contract.

## Quality Policy

```python
@dataclass(frozen=True)
class DatasetQualityPolicy:
    validate_target_positive_definite: bool = True
    maximum_solver_residual: float | None = 1e-8
    require_solver_provenance: bool = True
    require_unique_geometry: bool = True
```

For `engineering_voigt_c21` targets the writer requires:

- shape `(21,)`, floating dtype, finite values, and exact unit match;
- reconstructable symmetric `(6, 6)` engineering-Voigt matrix;
- positive eigenvalues when enabled;
- residual no larger than the configured threshold when provenance is
  required.

The writer does not silently symmetrize, repair, cast, or unit-convert a label.
Callers must explicitly correct or reject invalid solver output.

## Physics-Consistent Augmentation

`CubicRotation(seed, probability=1.0)` samples only the 24 orientation-
preserving signed permutation matrices of a cube. Reflections and arbitrary
interpolation are excluded. Version 1 requires `Nz == Ny == Nx`; applying this
transform to a non-cubic grid raises `ValueError` rather than changing the
declared field shape or silently restricting the rotation set.

The random choice is derived from a stable hash of
`(global_seed, epoch, sample_id)` and is therefore independent of item order,
worker count, process ID, and Python's randomized hash seed.

For each selected rotation:

- dense voxel fields are transposed/flipped in `(Nz, Ny, Nx)` space;
- flattened sparse voxel indices are decoded, transformed, re-encoded, and
  resorted;
- direction vectors are multiplied by the same rotation matrix;
- global per-voxel C21 fields are unpacked, rotated as fourth-order elasticity
  tensors, and repacked;
- a global anisotropic matrix C21 value is rotated by the same rule;
- `effective_c21` targets are rotated by the same tensor rule;
- local material-table C21 values are unchanged because sparse material
  directions define their placement in the global frame;
- AABB/domain extents are permuted if present.

The transform records the rotation ID and matrix in example metadata.
Applying a rotation and its inverse must recover integer fields exactly and
floating fields within the existing FP32/FP64 stiffness tolerances.

## PyTorch Loading and CUDA Prefetch

```python
loader = make_simulation_dataloader(
    dataset,
    batch_size=8,
    shuffle=True,
    num_workers=4,
    pin_memory=True,
    persistent_workers=True,
    prefetch_factor=2,
    seed=42,
)

for batch in CudaPrefetcher(loader, device="cuda"):
    prediction = model(batch.inputs["voxel.material_id"])
```

The factory imports PyTorch lazily and installs the repository collator.
`SimulationBatch.pin_memory()` recursively pins fixed tensors and ragged
values/offsets, enabling PyTorch's documented custom-batch pinning behavior.
It leaves identifiers and metadata unchanged.

`SimulationBatch.to()` explicitly transfers all tensors. `CudaPrefetcher`:

1. runs only in the main process;
2. creates one dedicated stream on the target CUDA device;
3. transfers the next pinned batch with `non_blocking=True`;
4. records an event after transfer and optional device transform;
5. makes the current compute stream wait for that event before yielding;
6. calls `record_stream()` on yielded tensors so allocator lifetime is safe;
7. exposes transferred logical bytes and wait timing.

CPU and non-CUDA targets fall back to synchronous iteration. No dataset worker
initializes CUDA. Device movement, dtype conversion, one-hot encoding, and
dense stiffness construction remain explicit transforms.

## Error Model

- `TypeError`: unsupported object or field value type.
- `ValueError`: invalid schema values, field shape/dtype/unit, split, group,
  target, quality policy, or transform request.
- `FileExistsError`: published target already exists.
- `DatasetFormatError(ValueError)`: unsupported/corrupt manifest, unsafe path,
  invalid offsets, or incompatible version.
- `DatasetIntegrityError(RuntimeError)`: checksum/digest mismatch or published
  split/group leakage.
- `ImportError`: a Torch-only API is used without PyTorch; the message names
  the GPU extra.

Errors include sample ID, field, shard, and expected/observed values when
available. Integrity failures are never downgraded to warnings.

## Testing and Acceptance

### Unit tests

1. field schemas reject invalid roles, layouts, dtypes, units, names, and
   ragged groups;
2. fixed and ragged examples collate into exact shapes and offsets;
3. custom batches report bytes, pin recursively, transfer explicitly, and
   preserve metadata;
4. group splits are input-order independent, stratified, deterministic, and
   leakage-free;
5. writer rejects duplicate sample IDs/digests, cross-split groups, bad units,
   non-finite/non-SPD C21, and excessive residuals;
6. multi-shard write/read preserves all selected values and shared ragged
   offsets;
7. unselected field files are never opened;
8. interrupted staging resumes only with the exact schema/config digest;
9. finalize is atomic and refuses overwrite;
10. manifest, path, shape, dtype, offsets, checksum, and sample digest
    corruption are detected at the promised verification level;
11. train statistics exclude validation/test values and constant fields divide
    safely;
12. all 24 rotations are unique, proper, invertible, and match an independent
    fourth-order tensor reference;
13. sparse indices, directions, global C21, and effective C21 rotate together;
14. transform choices match for `num_workers=0` and `2` at the same epoch and
    change deterministically across epochs;
15. CUDA prefetch values/devices match synchronous transfer, use a non-default
    stream safely, and account for exactly the selected logical tensor bytes.

### Integration tests

- generate at least three real TexGen weaves with distinct group IDs;
- obtain one real Voxel-ACDM anisotropic effective-C21 label and use validated
  synthetic SPD labels for the remaining fast tests;
- write multiple shards, reopen with mmap, select inputs/targets, and iterate
  with two workers;
- prefetch a batch to CUDA and run a small 3-D CNN forward/backward optimizer
  step;
- reconstruct the target `(6, 6)` matrix and verify finite gradients and loss;
- consume a batch tensor with DLPack and verify same-device pointer identity;
- install the wheel in a clean environment and import all three new modules.

### Performance benchmark

`bench_training_data.py` builds identical representative 64-cubed records in
the native shard format and the current compressed `chunk_*.npz` convention.
It reports:

- samples/second and logical MB/second for cold and warm reads;
- median and P90 batch wait time;
- peak RSS and pinned-memory bytes;
- explicit H2D bytes and CUDA transfer time;
- overlapped prefetch versus synchronous iteration;
- selected and total stored field bytes;
- CPU, storage, PyTorch, CUDA, GPU, PyTexGen, and commit metadata.

CLI options include `--dataset`, `--batch-size`, `--num-workers`, `--repeat`,
`--device`, `--min-read-speedup`, `--min-prefetch-speedup`, `--json-out`, and
`--check`. Unit tests inject timings. Hardware acceptance defaults to at least
`1.5x` read throughput over compressed NPZ and no prefetch regression; the
thresholds remain configurable because storage devices differ. `--check`
fails on any value mismatch, checksum failure, unexpected transferred byte,
non-finite training step, or missed performance threshold.

## Packaging and Documentation

The wheel installs `training_data.py`, `training_io.py`, and
`torch_training.py`. PyTorch-only symbols fail lazily when PyTorch is absent.
README examples show:

- dataset generation from `SimulationSample`;
- deterministic grouped splits;
- selective mmap reading;
- a standard PyTorch DataLoader;
- CUDA prefetch;
- effective C21 target reconstruction;
- how a PhysicsNeMo or JAX/Warp consumer receives the plain field mapping or
  individual DLPack tensors.

The documentation explicitly distinguishes:

- physical material IDs from yarn IDs;
- local material C21 from global rotated voxel C21;
- sparse resident fields from explicit dense model encodings;
- storage reads, CPU collation copies, expected H2D transfer, and forbidden
  hidden transfers;
- solver residual/label quality from neural-network prediction error.

## Compatibility

All changes are additive. Existing voxelization, material-field,
`SimulationSample`, single-sample persistence, and ACDM APIs remain unchanged.
Version-1 datasets are immutable after publication. Future backends must
produce the same `TrainingExample` and `SimulationBatch` semantics rather than
changing model-facing code.
