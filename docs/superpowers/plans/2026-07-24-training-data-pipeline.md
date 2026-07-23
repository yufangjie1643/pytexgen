# GPU Training Data Pipeline Implementation Plan

> **Execution note:** Run this plan inline on
> `agent/gpu-simulation-training`. The current collaboration policy forbids
> sub-agent delegation. Follow `superpowers:test-driven-development` for every
> behavior change and `superpowers:verification-before-completion` before each
> completion claim.

**Goal:** Add a versioned, selective, memory-mapped training dataset pipeline
that preserves PyTexGen material/orientation/C21 semantics and feeds
PyTorch/CUDA training without hidden host/device conversion.

**Architecture:** `TexGen/training_data.py` owns immutable schemas, examples,
batches, splitting, statistics, and physical augmentation.
`TexGen/training_io.py` owns crash-safe native `.npy` shards, manifests,
checksums, resume, mmap reads, and audit. `TexGen/torch_training.py` owns the
optional Torch loader and main-process CUDA prefetch. Existing
`SimulationSample` and single-sample persistence remain unchanged.

**Tech Stack:** Python 3.9+, NumPy, optional PyTorch/CUDA, `unittest`,
scikit-build-core/CMake, JSON/JSONL, SHA-256, atomic filesystem rename.

**Design source:**
`docs/superpowers/specs/2026-07-24-training-data-pipeline-design.md`

---

## Task 1: Define immutable training-data contracts

**Files:**

- Create: `TexGen/training_data.py`
- Create: `test_training_data.py`

### Step 1: Write failing schema and container tests

Add `TrainingSchemaTest` covering:

```python
field = TrainingFieldSpec(
    name="effective_c21",
    role="target",
    layout="fixed",
    dtype="float64",
    shape=(21,),
    unit="GPa",
    semantic="engineering_voigt_c21",
)
self.assertEqual(field.dtype, np.dtype("float64").str)
self.assertRaises(ValueError, TrainingFieldSpec, "../x", "input",
                  "fixed", "float32", (1,))
self.assertRaises(ValueError, TrainingFieldSpec, "x", "input",
                  "ragged", "float32", (3,))
self.assertRaises(ValueError, TrainingFieldSpec, "c", "target",
                  "fixed", "float64", (21,), None,
                  "engineering_voigt_c21")
```

Test duplicate names across roles, non-positive grid/shard dimensions,
unsupported voxel order, invalid statistics field references, mixed
`ragged_group` layouts, and mismatched trailing shapes. Test
`RaggedArray` offset validation and `TrainingExample` split/JSON-metadata
validation.

Run:

```bash
python -m unittest test_training_data.TrainingSchemaTest -v
```

Expected: FAIL because `TexGen.training_data` does not exist.

### Step 2: Implement field, schema, example, and ragged contracts

Implement these public types:

```python
@dataclass(frozen=True)
class TrainingFieldSpec:
    name: str
    role: str
    layout: str
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
    statistics_fields: tuple[str, ...] = ()
    geometry_digest_field: str = "voxel.material_id"

@dataclass(frozen=True)
class DatasetQualityPolicy:
    validate_target_positive_definite: bool = True
    maximum_solver_residual: float | None = 1e-8
    require_solver_provenance: bool = True
    require_unique_geometry: bool = True

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
    split: str
    metadata: Mapping[str, Any] = field(default_factory=dict)
```

Canonicalize NumPy dtype strings with `np.dtype(dtype).str`, freeze JSON
metadata through a round trip with `allow_nan=False`, and expose
`schema.fields`, `schema.field(name)`, `schema.to_dict()`, and
`TrainingDatasetSchema.from_dict()`. Reject object/string dtypes, unsafe names,
zero/negative shapes, C21 fields without units, and a ragged field without a
group. Export public symbols through `__all__`.

### Step 3: Prove serialization and immutability

Add round-trip tests that compare `schema == from_dict(schema.to_dict())`,
reject unknown keys/schema mutations, prove nested example metadata is
immutable, and prove array objects are retained without copying.

Run:

```bash
python -m unittest test_training_data.TrainingSchemaTest -v
```

Expected: PASS.

### Step 4: Commit

```bash
git add TexGen/training_data.py test_training_data.py
git commit -m "feat: define training data contracts"
```

---

## Task 2: Add owned collation and explicit batch movement

**Files:**

- Modify: `TexGen/training_data.py`
- Modify: `test_training_data.py`

### Step 1: Write failing fixed/ragged collation tests

Create two examples containing a fixed `(2, 2, 2)` material-ID grid and
ragged `orientation.primary` / `orientation.voxel_indices` values sharing the
`yarn_voxels` offsets. Assert:

```python
batch = collate_training_examples(examples, schema)
self.assertEqual(batch.inputs["voxel.material_id"].shape, (2, 2, 2, 2))
ragged = batch.inputs["orientation.primary"]
np.testing.assert_array_equal(ragged.offsets, [0, 2, 3])
self.assertTrue(batch.inputs["voxel.material_id"].flags.owndata)
self.assertTrue(batch.inputs["voxel.material_id"].flags.c_contiguous)
self.assertEqual(batch.sample_ids, ("s0", "s1"))
```

Reject missing/extra fields, wrong fixed shape/dtype, different offsets within
a ragged group, and unsafe implicit casting.

Run:

```bash
python -m unittest test_training_data.TrainingCollationTest -v
```

Expected: FAIL because the collator and batch are absent.

### Step 2: Implement NumPy collation and `SimulationBatch`

Add:

```python
@dataclass
class SimulationBatch:
    inputs: Mapping[str, Any]
    targets: Mapping[str, Any]
    sample_ids: tuple[str, ...]
    group_ids: tuple[str, ...]
    metadata: tuple[Mapping[str, Any], ...]

    def pin_memory(self) -> "SimulationBatch":
        return _map_batch_tensors(self, lambda value: value.pin_memory())

    def to(self, device, *, non_blocking: bool = False) -> "SimulationBatch":
        return _map_batch_tensors(
            self,
            lambda value: value.to(device, non_blocking=non_blocking),
        )

    def as_dict(self) -> Mapping[str, Any]:
        return {
            "inputs": self.inputs,
            "targets": self.targets,
            "sample_ids": self.sample_ids,
            "group_ids": self.group_ids,
            "metadata": self.metadata,
        }

    @property
    def nbytes(self) -> int:
        return _logical_nbytes(self.inputs) + _logical_nbytes(self.targets)
```

`collate_training_examples()` first allocates one owned C-contiguous NumPy
array per fixed field and one values array plus one `int64` offsets array per
ragged group. Do not call `np.stack` on mmap views. Preserve immutable
metadata separately.

### Step 3: Add optional Torch conversion tests and implementation

Add `as_torch_batch(batch)` using lazy Torch import and `torch.from_numpy()` on
owned writable arrays. Test:

- fixed and ragged tensors share memory with the owned collated arrays;
- identifiers remain integer tensors;
- `nbytes` counts tensor storage logically once;
- `.to("cpu")` and `.as_dict()` preserve metadata;
- without Torch, only Torch-specific APIs raise an actionable `ImportError`.

Gate pinning tests on CUDA availability because CPU-only Torch builds cannot
allocate pinned memory.

Run:

```bash
python -m unittest test_training_data.TrainingCollationTest -v
```

Expected: PASS.

### Step 4: Commit

```bash
git add TexGen/training_data.py test_training_data.py
git commit -m "feat: collate owned simulation batches"
```

---

## Task 3: Implement deterministic splitting and train-only statistics

**Files:**

- Modify: `TexGen/training_data.py`
- Modify: `test_training_data.py`

### Step 1: Write failing group-split tests

Test the exact public API:

```python
result = deterministic_group_split(
    ["g3", "g1", "g2", "g1"],
    ratios={"train": 0.5, "validation": 0.25, "test": 0.25},
    seed=42,
)
self.assertEqual(
    result,
    deterministic_group_split(
        ["g2", "g1", "g3"],
        ratios={"test": 0.25, "train": 0.5, "validation": 0.25},
        seed=42,
    ),
)
```

Add strata tests, minimum-one-group behavior, conflicting stratum rejection,
ratio validation, and a check that no group enters two splits.

### Step 2: Implement stable group assignment

Canonicalize ratios in `("train", "validation", "test")` order. Deduplicate
group IDs, validate one stratum per group, sort each stratum by
`sha256(f"{seed}\\0{group_id}")`, and allocate integer group counts by largest
remainder while ensuring each feasible nonzero split gets one group. Return
`dict[str, str]`.

### Step 3: Write failing statistics/normalization tests

Exercise a float64 Welford accumulator with train, validation, and test
outliers. Assert stored count, mean, population variance, standard deviation,
min/max, unit, source split, and constant mask. Verify integer fields and
unselected fields are untouched and a unit mismatch raises.

### Step 4: Implement statistics and `StandardizeFields`

Expose:

```python
class RunningFieldStatistics:
    def update(self, value: Any) -> None:
        self._combine(np.asarray(value, dtype=np.float64))

    def finalize(self, *, unit: str | None) -> Mapping[str, Any]:
        return self._serialized_statistics(unit=unit)

@dataclass(frozen=True)
class StandardizeFields:
    statistics: Mapping[str, Mapping[str, Any]]
    fields: tuple[str, ...]

    def __call__(
        self, example: TrainingExample, schema: TrainingDatasetSchema
    ) -> TrainingExample:
        return _standardize_selected_fields(
            example, schema, self.statistics, self.fields
        )
```

Use float64 Welford combination over the leading observations while retaining
each field's trailing component shape. Store a division standard deviation of
`1.0` for constant components and a separate Boolean mask.

Run:

```bash
python -m unittest \
  test_training_data.GroupSplitTest \
  test_training_data.StatisticsTest -v
```

Expected: PASS.

### Step 5: Commit

```bash
git add TexGen/training_data.py test_training_data.py
git commit -m "feat: add deterministic training splits"
```

---

## Task 4: Add physics-consistent cubic rotation

**Files:**

- Modify: `TexGen/training_data.py`
- Modify: `test_training_data.py`

### Step 1: Write an independent tensor-reference test

In the test only, independently map engineering-Voigt matrices to
`C_ijkl`, including engineering shear factors, rotate with:

```python
rotated = np.einsum(
    "ia,jb,kc,ld,abcd->ijkl",
    rotation,
    rotation,
    rotation,
    rotation,
    tensor,
)
```

and map back to C21. Do not import the implementation's private conversion
helpers. Assert all generated rotation matrices:

- total exactly 24 and are unique;
- contain only `-1`, `0`, `1`;
- satisfy `R @ R.T == I` and `det(R) == 1`;
- reproduce the independent rotated anisotropic C21 reference.

Run:

```bash
python -m unittest test_training_data.CubicRotationTest -v
```

Expected: FAIL because rotation support is absent.

### Step 2: Implement the 24 proper cube rotations

Add `proper_cubic_rotations()` and:

```python
class CubicRotation:
    def __init__(self, seed: int, probability: float = 1.0):
        self.seed = int(seed)
        self.probability = float(probability)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __call__(
        self, example: TrainingExample, schema: TrainingDatasetSchema
    ) -> TrainingExample:
        rotation_id = self._rotation_id(example.sample_id)
        return _rotate_example(example, schema, rotation_id)
```

Select both apply/no-apply and rotation ID from a SHA-256 digest of canonical
`(seed, epoch, sample_id)`. Reject non-cubic grids. Implement exact
transpose/flip mapping for dense `(Nz, Ny, Nx, ...)` fields and matching
decode/rotate/re-encode/sort logic for flattened sparse indices using the
schema's voxel order.

### Step 3: Rotate all coupled physical fields

Use field semantics, not name prefixes, to transform:

- dense spatial inputs;
- sparse indices and their row-coupled directions/C21 values;
- direction vectors;
- global voxel C21 and anisotropic matrix C21;
- `effective_c21`;
- AABB/domain extents.

Keep `material.c21` unchanged. Record `rotation_id` and a JSON-safe matrix in
metadata. Reject an unknown direction/C21 semantic rather than guessing.

### Step 4: Prove inverse recovery and worker independence

Add tests for exact integer recovery, FP64/FP32 tolerance recovery, sparse
sorting, input/label co-rotation, matrix material invariance, identical
choices across input order, and deterministic epoch changes. Add a
non-cubic-grid rejection test.

Run:

```bash
python -m unittest test_training_data.CubicRotationTest -v
```

Expected: PASS.

### Step 5: Commit

```bash
git add TexGen/training_data.py test_training_data.py
git commit -m "feat: rotate training fields consistently"
```

---

## Task 5: Stream validated samples into native shards

**Files:**

- Create: `TexGen/training_io.py`
- Create: `test_training_io.py`

### Step 1: Write failing writer and quality-gate tests

Reuse the small real `SimulationSample` fixture from
`test_simulation_sample.py`. Define a schema with physical material IDs,
shared orientation/stiffness topology, material-table C21, and
`effective_c21`. Assert:

```python
writer = SimulationDatasetWriter.create(
    target, schema=schema, quality=DatasetQualityPolicy()
)
writer.append(
    sample,
    targets={"effective_c21": effective},
    sample_id="s0",
    group_id="g0",
    split="train",
    provenance=valid_solver_provenance(),
)
writer.finalize()
self.assertTrue((target / "dataset.json").is_file())
```

Write rejection tests for duplicate sample IDs, duplicate physical geometry
digests, one group in multiple splits, unit mismatch, non-finite/non-SPD
target, excessive residual, missing provenance, wrong grid/order, and Torch
CUDA source arrays. The latter must fail explicitly; writing requires the
caller to request CPU/NumPy conversion.

Run:

```bash
python -m unittest test_training_io.WriterTest -v
```

Expected: FAIL because the writer is absent.

### Step 2: Implement canonical metadata and field extraction

Add:

```python
class DatasetFormatError(ValueError):
    """Dataset metadata or array layout is structurally invalid."""

class DatasetIntegrityError(RuntimeError):
    """Dataset content does not match its published digest or split contract."""

class SimulationDatasetWriter:
    @classmethod
    def create(
        cls,
        path,
        *,
        schema: TrainingDatasetSchema,
        quality: DatasetQualityPolicy | None = None,
        generation: Mapping[str, Any] | None = None,
        resume: bool = False,
    ) -> "SimulationDatasetWriter":
        return cls(
            path,
            schema=schema,
            quality=quality,
            generation=generation,
            resume=resume,
        )

    def append(
        self,
        sample: SimulationSample,
        *,
        targets: Mapping[str, Any],
        sample_id: str,
        group_id: str,
        split: str,
        provenance: Mapping[str, Any],
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        record = self._validate_sample_and_target(
            sample, targets, sample_id, group_id, split, provenance, metadata
        )
        self._buffer_record(record)
```

Serialize JSON with sorted keys, compact separators, `allow_nan=False`, UTF-8,
and a terminal newline. Extract fields only through
`sample.array(name, copy=True)`. Materialize `voxel.material_id` explicitly;
never substitute yarn IDs. Validate every shape/dtype/unit without casting.
Hash canonical bytes of the configured geometry field with its dtype/shape.

### Step 3: Implement shard buffering and shared ragged offsets

At `schema.shard_size`, write one sibling `.incomplete` shard directory.
Fixed fields become `(S, *shape)`. For each ragged group, concatenate values
and write one `int64` offsets file referenced by every group field. Store one
canonical topology array and manifest aliases for orientation/stiffness
voxel-index and yarn-ID names. Use `np.save(..., allow_pickle=False)`.

Compute SHA-256 and byte count by streaming each completed file. Write
`shard.json` only after every array is closed and synced. Update train-only
statistics after validation, never for rejected or non-train samples.

### Step 4: Implement rejection records and context management

Add:

```python
def reject(
    self,
    *,
    sample_id: str,
    stage: str,
    reason: str,
    metadata: Mapping[str, Any] | None = None,
) -> None:
    self._append_rejection_record(sample_id, stage, reason, metadata)
```

Append deterministic JSONL rejection records. A normal context-manager exit
calls `finalize()`; an exceptional exit leaves resumable staging state and
never publishes the target.

Run:

```bash
python -m unittest test_training_io.WriterTest -v
```

Expected: PASS.

### Step 5: Commit

```bash
git add TexGen/training_io.py test_training_io.py
git commit -m "feat: write native training shards"
```

---

## Task 6: Make publication crash-safe and resumable

**Files:**

- Modify: `TexGen/training_io.py`
- Modify: `test_training_io.py`

### Step 1: Write failing interruption/resume tests

Simulate interruption after:

1. a complete shard;
2. array creation but before `shard.json`;
3. journal append but before root manifest;
4. final manifest but before rename.

Assert `resume=True` removes only the incomplete trailing shard, reconstructs
accepted IDs/digests/groups/statistics from complete journal records, and
continues at the correct shard number. Assert schema or generation digest
mismatch fails without modifying staging.

### Step 2: Implement generation digest and append-only journal

Name staging `<target.name>.incomplete` beside the target. Store
`staging.json` with serialized schema, quality, generation configuration, and
their canonical digest before accepting samples. Append one `journal.jsonl`
record only after a complete shard has been flushed and checksummed.

On resume:

- require the exact target-derived staging path;
- validate staging and generation digests;
- ignore/remove only a non-journaled trailing shard;
- verify all journaled shard metadata and checksums;
- rebuild in-memory uniqueness/group/statistics state from journal content.

### Step 3: Implement atomic finalize and overwrite protection

`finalize()` flushes the partial shard, validates the complete staging
dataset, writes `samples.jsonl`, `rejections.jsonl`, and `dataset.json` through
temporary files plus `os.replace`, fsyncs files and directories on platforms
that support it, then atomically renames staging to target. Refuse both an
existing target and a target appearing during finalize.

### Step 4: Test finalization idempotence and failure safety

Assert repeated `finalize()` is harmless on the same writer, `append()` after
finalize fails, overwrite raises `FileExistsError`, target is absent after an
injected write failure, and no `.tmp` file remains after successful publish.

Run:

```bash
python -m unittest test_training_io.ResumeAndFinalizeTest -v
```

Expected: PASS.

### Step 5: Commit

```bash
git add TexGen/training_io.py test_training_io.py
git commit -m "feat: resume atomic dataset publication"
```

---

## Task 7: Add selective mmap reading, verification, and audit

**Files:**

- Modify: `TexGen/training_io.py`
- Modify: `test_training_io.py`

### Step 1: Write failing multi-shard reader tests

Publish five samples with `shard_size=2`, then assert:

```python
dataset = SimulationDataset(
    target,
    split="train",
    inputs=("voxel.material_id", "orientation.primary"),
    targets=("effective_c21",),
    verify="shard",
)
self.assertEqual(len(dataset), expected_train_count)
self.assertIsInstance(dataset[0].inputs["voxel.material_id"].base, np.memmap)
pickle.loads(pickle.dumps(dataset))[0]
```

Patch `np.load` and SHA helpers to prove unselected field paths are never
opened or verified at `verify="shard"`. Assert ragged slices expose sample
offsets `[0, n]` without copying values.

### Step 2: Implement manifest/path validation

Add:

```python
class SimulationDataset:
    def __init__(
        self,
        path,
        *,
        split: str | None = None,
        inputs: tuple[str, ...] | None = None,
        targets: tuple[str, ...] | None = None,
        verify: str = "shard",
        transform: Any = None,
    ):
        self._initialize_index(path, split, inputs, targets, verify, transform)

    def __len__(self) -> int:
        return len(self._selected_samples)

    def __getitem__(self, index: int) -> TrainingExample:
        return self._read_selected_example(self._selected_samples[index])

    def set_epoch(self, epoch: int) -> None:
        self._epoch = int(epoch)
        if hasattr(self.transform, "set_epoch"):
            self.transform.set_epoch(self._epoch)
```

Resolve all paths under the dataset root and reject absolute paths, `..`,
symlinks escaping the root, duplicate manifest paths, unsupported
schema/version, invalid index rows, mismatched split counts, group leakage,
and unsafe NumPy dtypes. Store paths/index metadata only. Lazily open mmap
handles per process ID and discard caches in `__getstate__`.

### Step 3: Implement promised verification levels

- `manifest`: validate JSON, schema, paths, declared shapes/dtypes/counts.
- `shard`: on first selected file access per worker, hash that field file and
  its shared offsets file once.
- `sample`: eagerly verify every file, then recompute every configured
  physical geometry digest.

Raise `DatasetFormatError` for structural corruption and
`DatasetIntegrityError` for hashes, digests, or split leakage. Include field,
shard, expected, and observed values in messages.

### Step 4: Add full audit and corruption matrix

Expose:

```python
def audit_simulation_dataset(
    path, *, verify: str = "sample"
) -> Mapping[str, Any]:
    dataset = SimulationDataset(path, verify=verify)
    return dataset.audit_report()
```

Return schema version, sample/rejection/shard/split/group counts, stored and
logical bytes, checked files/samples, and `ok=True`. Test corrupt JSON, unsafe
path, field shape/dtype, offsets monotonicity/end value, checksum, sample
digest, sample row, and group split at each documented verification level.

### Step 5: Prove transform epoch propagation

Attach a recording transform, call `dataset.set_epoch(7)`, and assert
`transform.set_epoch(7)` runs and examples contain epoch-7 output. Pickle the
reader and repeat to mimic a worker-local copy.

Run:

```bash
python -m unittest test_training_io.ReaderAndAuditTest -v
```

Expected: PASS.

### Step 6: Commit

```bash
git add TexGen/training_io.py test_training_io.py
git commit -m "feat: mmap and audit training datasets"
```

---

## Task 8: Add the optional PyTorch loader and CUDA prefetch

**Files:**

- Create: `TexGen/torch_training.py`
- Create: `test_torch_training.py`

### Step 1: Write failing DataLoader tests

Using a published multi-shard fixture, compare `num_workers=0` and `2`:

```python
loader = make_simulation_dataloader(
    dataset,
    batch_size=2,
    shuffle=False,
    num_workers=2,
    pin_memory=False,
    seed=11,
)
batch = next(iter(loader))
self.assertIsInstance(batch, SimulationBatch)
self.assertIsInstance(batch.inputs["voxel.material_id"], torch.Tensor)
```

Assert exact values/offsets, owned writable backing before tensor conversion,
deterministic shuffle for a fixed seed, persistent-worker defaults, and no
CUDA initialization from worker code. Test absence-of-Torch behavior via an
isolated module load.

### Step 2: Implement lazy loader factory

Add:

```python
def make_simulation_dataloader(
    dataset,
    *,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    pin_memory: bool,
    persistent_workers: bool | None = None,
    prefetch_factor: int | None = None,
    seed: int = 0,
    drop_last: bool = False,
):
    return _build_torch_dataloader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
        seed=seed,
        drop_last=drop_last,
    )
```

Import Torch only inside the public function. Use a top-level picklable
collator calling `collate_training_examples()` then `as_torch_batch()`. Seed a
Torch generator. Pass `prefetch_factor` and `persistent_workers` only when
`num_workers > 0`. Do not touch `torch.cuda` in the dataset, collator, or
worker initialization path.

### Step 3: Write failing CPU/CUDA prefetch tests

CPU tests require identity values and synchronous byte accounting. CUDA-gated
tests require:

- all selected tensors are on the requested resolved device;
- values match `batch.to(device)`;
- transfer runs on a non-default stream;
- current stream waits on the recorded event;
- recursive `record_stream()` is called;
- `transferred_bytes == sum(batch.nbytes)`;
- DLPack round trip preserves device pointer;
- no unselected field bytes are counted.

### Step 4: Implement `CudaPrefetcher`

Add:

```python
class CudaPrefetcher:
    def __init__(self, loader, *, device, transform=None):
        self.loader = loader
        self.device = device
        self.transform = transform
        self.transferred_bytes = 0
        self.wait_seconds = 0.0

    def __iter__(self):
        return _prefetch_batches(
            self.loader, self.device, self.transform, counters=self
        )
```

For CUDA, create one dedicated stream in the main process, enqueue the next
batch with `non_blocking=True`, run the optional device transform on that
stream, record an event, wait from the current compute stream, recursively
call `record_stream()` before yield, and prefetch one batch ahead. For CPU,
iterate synchronously and do not create a CUDA stream. Reject CUDA use in a
non-main process.

Run:

```bash
python -m unittest test_torch_training -v
```

Expected: PASS, with CUDA tests skipped only when CUDA is unavailable.

### Step 5: Commit

```bash
git add TexGen/torch_training.py test_torch_training.py
git commit -m "feat: prefetch training batches to CUDA"
```

---

## Task 9: Package and document the complete model-facing workflow

**Files:**

- Modify: `Python/CMakeLists.txt`
- Modify: `README.md`
- Modify: `README_pypi.md`
- Modify: `docs/voxel_backends.md`
- Create: `test_training_pipeline_integration.py`

### Step 1: Write a failing wheel/package smoke assertion

Extend a source-tree test to verify the install list contains
`training_data.py`, `training_io.py`, and `torch_training.py`. Add import checks
for legacy `TexGen.*` and installed `pytexgen.*` module names without importing
Torch on `pytexgen.training_data` import.

### Step 2: Update package installation

Append the three modules to `PYTEXGEN_INTEROP_MODULE_FILES` in
`Python/CMakeLists.txt`. Do not modify SWIG outputs.

### Step 3: Add real TexGen-to-dataset integration coverage

Build at least three small real woven samples with distinct group IDs, assign
validated synthetic SPD C21 labels to fast cases, write multiple shards, mmap
read selected fields, iterate with two DataLoader workers, and assert the
physical material-ID grid—not yarn IDs—reaches the batch.

Add a CUDA-gated small 3-D CNN step:

```python
prediction = model(encoded_material_ids)
loss = torch.nn.functional.mse_loss(prediction, target_c21)
optimizer.zero_grad(set_to_none=True)
loss.backward()
optimizer.step()
self.assertTrue(torch.isfinite(loss))
```

Assert finite gradients and DLPack pointer identity for one prefetched tensor.

### Step 4: Add real Voxel-ACDM label coverage

When the sibling Voxel-ACDM checkout and CUDA are available, solve one small
anisotropic sample through the existing `SimulationSample` ACDM adapter,
validate solver provenance/residual, write its 21-value effective C21 label,
read it back, reconstruct `(6, 6)`, and compare to the solver result. Skip
with an explicit dependency reason otherwise; never replace this test with a
mock label.

### Step 5: Publish concise usage documentation

Document:

- schema creation and grouped split assignment;
- writer append/reject/finalize and label provenance;
- selective mmap reading;
- standard DataLoader and `CudaPrefetcher`;
- C21 reconstruction and units;
- PhysicsNeMo/JAX/Warp mappings or per-tensor DLPack;
- physical material IDs versus yarn IDs;
- local material C21 versus rotated global/effective C21;
- expected collation/H2D copies and forbidden hidden conversion.

Run:

```bash
python -m unittest test_training_pipeline_integration -v
```

Expected: PASS, with dependency-specific skips explained.

### Step 6: Commit

```bash
git add Python/CMakeLists.txt README.md README_pypi.md \
  docs/voxel_backends.md test_training_pipeline_integration.py
git commit -m "docs: publish GPU training data workflow"
```

---

## Task 10: Add checked storage and prefetch benchmarks

**Files:**

- Create: `bench_training_data.py`
- Create: `test_training_data_benchmark.py`

### Step 1: Write failing injected-timing tests

Factor threshold evaluation into a pure function and test:

```python
report = evaluate_benchmark(
    native_samples_per_second=300.0,
    npz_samples_per_second=100.0,
    synchronous_wait_seconds=0.4,
    prefetch_wait_seconds=0.2,
    expected_h2d_bytes=4096,
    observed_h2d_bytes=4096,
    min_read_speedup=1.5,
    min_prefetch_speedup=1.0,
)
self.assertTrue(report["passed"])
```

Test exact-threshold pass, read regression fail, prefetch regression fail,
byte mismatch fail, non-finite training loss fail, and JSON serialization.

### Step 2: Implement benchmark CLI and comparable fixtures

Provide options:

```text
--dataset --batch-size --num-workers --repeat --device
--min-read-speedup --min-prefetch-speedup --json-out --check
```

Generate or load identical representative 64-cubed records. Persist the same
logical fields in native uncompressed shards and compressed `chunk_*.npz`.
Measure cold/warm samples/s and logical MB/s, median/P90 batch wait, peak RSS,
pinned bytes, H2D bytes/time, selected/stored bytes, synchronous/prefetched
iteration, and one finite model step. Record platform, CPU, storage path,
NumPy, Torch, CUDA, GPU, PyTexGen, and commit metadata.

### Step 3: Enforce benchmark correctness before speed

`--check` first compares all native and NPZ values, audits checksums, verifies
exact H2D byte accounting, and requires a finite forward/backward step. It
then enforces default native read speedup `>= 1.5` and prefetch speedup
`>= 1.0`; both remain CLI-configurable. Exit nonzero with individual failed
metrics.

Run:

```bash
python -m unittest test_training_data_benchmark -v
python bench_training_data.py --help
```

Expected: PASS.

### Step 4: Commit

```bash
git add bench_training_data.py test_training_data_benchmark.py
git commit -m "bench: verify GPU training data throughput"
```

---

## Task 11: Verify source, wheel, CUDA, and real-solver acceptance

**Files:**

- Modify only files required by failures found below.

### Step 1: Run focused training pipeline tests

```bash
python -m unittest \
  test_training_data \
  test_training_io \
  test_torch_training \
  test_training_pipeline_integration \
  test_training_data_benchmark -v
```

Expected: all available tests PASS; optional dependency skips name the missing
dependency/device.

### Step 2: Run the full source regression suite

```bash
python -m unittest discover -s . -p 'test_*.py'
```

Expected: all tests PASS, including existing simulation/ACDM tests.

### Step 3: Build distributions and test a clean wheel

```bash
python -m build
python -m venv /tmp/pytexgen-training-wheel
/tmp/pytexgen-training-wheel/bin/pip install dist/pytexgen-*.whl
/tmp/pytexgen-training-wheel/bin/python -c \
  "import pytexgen.training_data, pytexgen.training_io, pytexgen.torch_training"
```

Expected: wheel and sdist build; all three imports succeed. Remove only the
explicit temporary environment after recording results.

### Step 4: Run hardware acceptance

On the available RTX 5090:

```bash
python bench_training_data.py \
  --batch-size 8 --num-workers 4 --repeat 5 --device cuda \
  --min-read-speedup 1.5 --min-prefetch-speedup 1.0 \
  --json-out training_data_benchmark.json --check
python -m unittest \
  test_training_pipeline_integration.TrainingPipelineIntegrationTest.test_real_acdm_label_and_cuda_training_step \
  -v
```

Expected:

- native shard reads are at least `1.5x` compressed NPZ;
- CUDA prefetch does not regress synchronous wait time;
- transferred bytes exactly equal selected batch tensor bytes;
- ACDM C21 round trip matches;
- CNN loss and gradients are finite.

Do not commit `training_data_benchmark.json` or generated datasets.

### Step 5: Inspect the final diff and history

```bash
git diff --check
git status --short
git log --oneline origin/main..HEAD
```

Expected: no whitespace errors; only intended tracked changes plus the user's
untracked `AGENTS.md`; coherent focused commits.

### Step 6: Commit verification-only fixes if needed

If verification required code changes, rerun the failing focused test first,
then the full affected suite. Stage the exact modified tracked paths shown by
`git status --short` (never `AGENTS.md`) and commit them with
`git commit -m "fix: harden training data verification"`.

If no tracked changes remain, do not create an empty commit.
