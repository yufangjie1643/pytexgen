"""Tests for native sharded simulation training datasets."""

import dataclasses
import json
import pickle
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from test_simulation_sample import load_simulation_sample


SIMULATION_SAMPLE, MATERIAL_FIELDS = load_simulation_sample()
GPU_VOXELIZER = sys.modules["TexGen.gpu_voxelizer"]

from TexGen.training_data import (
    DatasetQualityPolicy,
    TrainingDatasetSchema,
    TrainingFieldSpec,
    VOXEL_ORDER,
)
from TexGen.training_io import (
    DatasetFormatError,
    DatasetIntegrityError,
    SimulationDataset,
    SimulationDatasetWriter,
    audit_simulation_dataset,
)

try:
    import torch
except ImportError:
    torch = None


def make_sample():
    shape = (2, 2, 2)
    voxel_indices = np.array([1, 5], dtype=np.int64)
    yarn_ids = np.array([0, 1], dtype=np.int32)
    primary = np.array(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float64
    )
    secondary = np.array(
        [[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64
    )
    orientation = MATERIAL_FIELDS.SparseOrientationField(
        voxel_indices=voxel_indices,
        yarn_ids=yarn_ids,
        orientation1=primary,
        orientation2=secondary,
        grid_shape=shape,
        order=VOXEL_ORDER,
    )
    matrix_c21 = MATERIAL_FIELDS.isotropic_stiffness_c21(3.0, 0.30)
    yarn_c21 = np.stack(
        (
            MATERIAL_FIELDS.isotropic_stiffness_c21(70.0, 0.20),
            MATERIAL_FIELDS.isotropic_stiffness_c21(50.0, 0.25),
        )
    )
    material_ids = np.array([7, 9], dtype=np.int32)
    stiffness = MATERIAL_FIELDS.SparseStiffnessField(
        matrix_c21=matrix_c21,
        voxel_indices=voxel_indices,
        yarn_ids=yarn_ids,
        material_ids=material_ids,
        yarn_c21=yarn_c21,
        grid_shape=shape,
        unit="GPa",
        order=VOXEL_ORDER,
    )
    materials = SIMULATION_SAMPLE.MaterialTable(
        c21=np.vstack((matrix_c21, yarn_c21)),
        material_ids=np.array([0, 7, 9], dtype=np.int32),
        unit="GPa",
        names=("matrix", "warp", "weft"),
        validate_positive_definite=True,
    )
    voxels = GPU_VOXELIZER.VoxelGridData(
        yarn_id=np.array(
            [-1, 0, -1, -1, -1, 1, -1, -1], dtype=np.int32
        ),
        aabb=np.array([[0.0, 0.0, 0.0], [2.0, 2.0, 2.0]]),
        resolution=(2, 2, 2),
        backend="cpu",
        device="cpu",
        workers=1,
        dtype="float64",
        timings={"voxelize": 0.01},
        sparse_orientation=orientation,
        storage="numpy",
        order=VOXEL_ORDER,
    )
    return SIMULATION_SAMPLE.SimulationSample(
        voxels=voxels,
        materials=materials,
        orientation=orientation,
        stiffness=stiffness,
        metadata={"generator": {"seed": 11}},
    )


def make_schema(shard_size=2):
    return TrainingDatasetSchema(
        inputs=(
            TrainingFieldSpec(
                "voxel.material_id",
                "input",
                "fixed",
                "int32",
                (2, 2, 2),
                semantic="material_id_grid",
            ),
            TrainingFieldSpec(
                "orientation.voxel_indices",
                "input",
                "ragged",
                "int64",
                (),
                semantic="flat_voxel_index",
                ragged_group="yarn_voxels",
            ),
            TrainingFieldSpec(
                "stiffness.voxel_indices",
                "input",
                "ragged",
                "int64",
                (),
                semantic="flat_voxel_index",
                ragged_group="yarn_voxels",
            ),
            TrainingFieldSpec(
                "orientation.primary",
                "input",
                "ragged",
                "float64",
                (3,),
                semantic="direction_vector",
                ragged_group="yarn_voxels",
            ),
            TrainingFieldSpec(
                "stiffness.yarn_c21",
                "input",
                "ragged",
                "float64",
                (21,),
                "GPa",
                "global_engineering_voigt_c21",
                "yarn_voxels",
            ),
            TrainingFieldSpec(
                "material.ids",
                "input",
                "ragged",
                "int32",
                (),
                semantic="material_id",
                ragged_group="materials",
            ),
            TrainingFieldSpec(
                "material.c21",
                "input",
                "ragged",
                "float64",
                (21,),
                "GPa",
                "local_engineering_voigt_c21",
                "materials",
            ),
        ),
        targets=(
            TrainingFieldSpec(
                "effective_c21",
                "target",
                "fixed",
                "float64",
                (21,),
                "GPa",
                "engineering_voigt_c21",
            ),
        ),
        grid_shape=(2, 2, 2),
        voxel_order=VOXEL_ORDER,
        shard_size=shard_size,
        statistics_fields=("effective_c21",),
    )


def effective_c21(scale=1.0):
    return MATERIAL_FIELDS.isotropic_stiffness_c21(
        12.0 * scale, 0.22
    )


def valid_provenance(**overrides):
    result = {
        "solver_commit": "acdm-abc123",
        "element_formulation": "voxel-periodic",
        "arithmetic_dtype": "float64",
        "tolerance": 1e-10,
        "maximum_residual": 2e-10,
        "iteration_count": 18,
        "wall_time_seconds": 0.25,
        "target_units": {"effective_c21": "GPa"},
    }
    result.update(overrides)
    return result


def append_sample(
    writer,
    *,
    sample_id,
    group_id,
    split="train",
    scale=1.0,
):
    writer.append(
        make_sample(),
        targets={"effective_c21": effective_c21(scale)},
        sample_id=sample_id,
        group_id=group_id,
        split=split,
        provenance=valid_provenance(),
    )


class WriterTest(unittest.TestCase):
    def test_writes_validated_fields_shared_offsets_aliases_and_statistics(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "dataset"
            writer = SimulationDatasetWriter.create(
                target,
                schema=make_schema(),
                quality=DatasetQualityPolicy(),
                generation={"family": "plain-weave", "seed": 5},
            )
            writer.append(
                make_sample(),
                targets={"effective_c21": effective_c21()},
                sample_id="s0",
                group_id="geometry-0",
                split="train",
                provenance=valid_provenance(),
                metadata={"weave": "plain"},
            )
            writer.reject(
                sample_id="s-rejected",
                stage="label",
                reason="solver did not converge",
                metadata={"seed": 9},
            )
            writer.finalize()

            self.assertTrue(target.is_dir())
            self.assertFalse(
                (Path(directory) / "dataset.incomplete").exists()
            )
            manifest = json.loads(
                (target / "dataset.json").read_text(encoding="utf-8")
            )
            self.assertEqual(
                manifest["schema"], "pytexgen.simulation_dataset"
            )
            self.assertEqual(manifest["version"], 1)
            self.assertEqual(manifest["sample_count"], 1)
            self.assertEqual(manifest["rejection_count"], 1)
            self.assertEqual(manifest["shard_count"], 1)
            self.assertEqual(
                manifest["statistics"]["effective_c21"]["count"], 1
            )
            shard = manifest["shards"][0]
            orientation_entry = shard["fields"][
                "orientation.voxel_indices"
            ]
            stiffness_entry = shard["fields"][
                "stiffness.voxel_indices"
            ]
            self.assertEqual(
                orientation_entry["values"],
                stiffness_entry["values"],
            )
            self.assertEqual(
                orientation_entry["offsets"],
                shard["fields"]["orientation.primary"]["offsets"],
            )
            self.assertEqual(
                shard["fields"]["material.ids"]["offsets"],
                shard["fields"]["material.c21"]["offsets"],
            )
            stored_material_id = np.load(
                target
                / shard["fields"]["voxel.material_id"]["values"],
                allow_pickle=False,
            )
            np.testing.assert_array_equal(
                stored_material_id[0],
                make_sample().array(
                    "voxel.material_id", copy=True
                ),
            )
            samples = [
                json.loads(line)
                for line in (target / "samples.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            self.assertEqual(samples[0]["sample_id"], "s0")
            self.assertEqual(
                samples[0]["provenance"]["solver_commit"],
                "acdm-abc123",
            )
            rejections = (
                target / "rejections.jsonl"
            ).read_text(encoding="utf-8")
            self.assertIn("solver did not converge", rejections)

    def test_context_manager_finalizes_on_clean_exit(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "dataset"
            with SimulationDatasetWriter.create(
                target, schema=make_schema()
            ) as writer:
                writer.append(
                    make_sample(),
                    targets={"effective_c21": effective_c21()},
                    sample_id="s0",
                    group_id="g0",
                    split="train",
                    provenance=valid_provenance(),
                )

            self.assertTrue((target / "dataset.json").is_file())

    def test_rejects_duplicate_sample_and_geometry_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            writer = SimulationDatasetWriter.create(
                Path(directory) / "dataset", schema=make_schema()
            )
            kwargs = {
                "sample": make_sample(),
                "targets": {"effective_c21": effective_c21()},
                "sample_id": "s0",
                "group_id": "g0",
                "split": "train",
                "provenance": valid_provenance(),
            }
            writer.append(**kwargs)
            with self.assertRaisesRegex(ValueError, "sample_id"):
                writer.append(**kwargs)
            kwargs["sample_id"] = "s1"
            with self.assertRaisesRegex(ValueError, "geometry"):
                writer.append(**kwargs)

    def test_rejects_one_group_in_multiple_splits(self):
        with tempfile.TemporaryDirectory() as directory:
            writer = SimulationDatasetWriter.create(
                Path(directory) / "dataset",
                schema=make_schema(),
                quality=DatasetQualityPolicy(
                    require_unique_geometry=False
                ),
            )
            writer.append(
                make_sample(),
                targets={"effective_c21": effective_c21()},
                sample_id="s0",
                group_id="g0",
                split="train",
                provenance=valid_provenance(),
            )
            with self.assertRaisesRegex(ValueError, "group.*split"):
                writer.append(
                    make_sample(),
                    targets={"effective_c21": effective_c21(1.1)},
                    sample_id="s1",
                    group_id="g0",
                    split="test",
                    provenance=valid_provenance(),
                )

    def test_rejects_invalid_target_dtype_shape_values_and_unit(self):
        bad_cases = (
            (
                np.ones(21, dtype=np.float32),
                valid_provenance(),
                "dtype",
            ),
            (
                np.ones(20, dtype=np.float64),
                valid_provenance(),
                "shape",
            ),
            (
                np.full(21, np.nan, dtype=np.float64),
                valid_provenance(),
                "finite",
            ),
            (
                np.zeros(21, dtype=np.float64),
                valid_provenance(),
                "positive definite",
            ),
            (
                effective_c21(),
                valid_provenance(
                    target_units={"effective_c21": "Pa"}
                ),
                "unit",
            ),
            (
                effective_c21(),
                valid_provenance(maximum_residual=1e-5),
                "residual",
            ),
        )
        for target_value, provenance, message in bad_cases:
            with self.subTest(message=message):
                with tempfile.TemporaryDirectory() as directory:
                    writer = SimulationDatasetWriter.create(
                        Path(directory) / "dataset",
                        schema=make_schema(),
                    )
                    with self.assertRaisesRegex(ValueError, message):
                        writer.append(
                            make_sample(),
                            targets={
                                "effective_c21": target_value
                            },
                            sample_id="s0",
                            group_id="g0",
                            split="train",
                            provenance=provenance,
                        )

    def test_rejects_missing_provenance_target_and_wrong_grid(self):
        cases = (
            (
                make_schema(),
                {},
                valid_provenance(),
                "target",
            ),
            (
                make_schema(),
                {"effective_c21": effective_c21()},
                {},
                "provenance",
            ),
            (
                TrainingDatasetSchema(
                    inputs=make_schema().inputs,
                    targets=make_schema().targets,
                    grid_shape=(3, 3, 3),
                    voxel_order=VOXEL_ORDER,
                    shard_size=2,
                ),
                {"effective_c21": effective_c21()},
                valid_provenance(),
                "grid",
            ),
        )
        for schema, targets, provenance, message in cases:
            with self.subTest(message=message):
                with tempfile.TemporaryDirectory() as directory:
                    writer = SimulationDatasetWriter.create(
                        Path(directory) / "dataset", schema=schema
                    )
                    with self.assertRaisesRegex(ValueError, message):
                        writer.append(
                            make_sample(),
                            targets=targets,
                            sample_id="s0",
                            group_id="g0",
                            split="train",
                            provenance=provenance,
                        )

    def test_quality_policy_can_disable_solver_provenance_requirement(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "dataset"
            writer = SimulationDatasetWriter.create(
                target,
                schema=make_schema(),
                quality=DatasetQualityPolicy(
                    require_solver_provenance=False,
                    maximum_solver_residual=None,
                ),
            )

            writer.append(
                make_sample(),
                targets={"effective_c21": effective_c21()},
                sample_id="s0",
                group_id="g0",
                split="train",
                provenance={},
            )
            writer.finalize()

            record = json.loads(
                (target / "samples.jsonl")
                .read_text(encoding="utf-8")
                .strip()
            )
            self.assertEqual(record["provenance"], {})

    @unittest.skipIf(torch is None, "PyTorch is not installed")
    def test_rejects_torch_samples_instead_of_hiding_host_transfer(self):
        with tempfile.TemporaryDirectory() as directory:
            writer = SimulationDatasetWriter.create(
                Path(directory) / "dataset", schema=make_schema()
            )
            with self.assertRaisesRegex(ValueError, "CPU NumPy"):
                writer.append(
                    make_sample().to("torch"),
                    targets={"effective_c21": effective_c21()},
                    sample_id="s0",
                    group_id="g0",
                    split="train",
                    provenance=valid_provenance(),
                )


class ResumeAndFinalizeTest(unittest.TestCase):
    @staticmethod
    def resumable_quality():
        return DatasetQualityPolicy(require_unique_geometry=False)

    def test_resume_restores_complete_shards_identity_groups_and_statistics(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "dataset"
            schema = make_schema(shard_size=1)
            quality = self.resumable_quality()
            writer = SimulationDatasetWriter.create(
                target,
                schema=schema,
                quality=quality,
                generation={"seed": 4},
            )
            append_sample(
                writer, sample_id="s0", group_id="g0", scale=1.0
            )
            self.assertTrue(
                (
                    target.with_name("dataset.incomplete")
                    / "shards"
                    / "shard_00000"
                    / "shard.json"
                ).is_file()
            )

            resumed = SimulationDatasetWriter.create(
                target,
                schema=schema,
                quality=quality,
                generation={"seed": 4},
                resume=True,
            )
            with self.assertRaisesRegex(ValueError, "sample_id"):
                append_sample(
                    resumed,
                    sample_id="s0",
                    group_id="g1",
                    scale=1.1,
                )
            with self.assertRaisesRegex(ValueError, "group.*split"):
                append_sample(
                    resumed,
                    sample_id="s-leak",
                    group_id="g0",
                    split="test",
                    scale=1.1,
                )
            append_sample(
                resumed, sample_id="s1", group_id="g1", scale=2.0
            )
            resumed.finalize()

            manifest = json.loads(
                (target / "dataset.json").read_text(encoding="utf-8")
            )
            self.assertEqual(manifest["sample_count"], 2)
            self.assertEqual(manifest["shard_count"], 2)
            expected = (
                effective_c21(1.0) + effective_c21(2.0)
            ) / 2.0
            np.testing.assert_allclose(
                manifest["statistics"]["effective_c21"]["mean"],
                expected,
            )

    def test_resume_removes_only_unjournaled_trailing_shard(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "dataset"
            schema = make_schema(shard_size=1)
            quality = self.resumable_quality()
            writer = SimulationDatasetWriter.create(
                target, schema=schema, quality=quality
            )
            append_sample(writer, sample_id="s0", group_id="g0")
            staging = target.with_name("dataset.incomplete")
            partial = staging / "shards" / "shard_00001"
            partial.mkdir()
            (partial / "partial.npy").write_bytes(b"incomplete")

            resumed = SimulationDatasetWriter.create(
                target,
                schema=schema,
                quality=quality,
                resume=True,
            )

            self.assertFalse(partial.exists())
            self.assertTrue(
                (staging / "shards" / "shard_00000").is_dir()
            )
            append_sample(
                resumed, sample_id="s1", group_id="g1", scale=1.2
            )
            resumed.finalize()
            self.assertTrue(
                (target / "shards" / "shard_00001").is_dir()
            )

    def test_resume_after_final_rename_failure_preserves_rejections(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "dataset"
            schema = make_schema(shard_size=1)
            quality = self.resumable_quality()
            writer = SimulationDatasetWriter.create(
                target,
                schema=schema,
                quality=quality,
                generation={"seed": 8},
            )
            append_sample(writer, sample_id="s0", group_id="g0")
            writer.reject(
                sample_id="bad",
                stage="label",
                reason="residual",
                metadata={"seed": 99},
            )
            real_replace = __import__("os").replace

            def fail_final_rename(source, destination):
                if Path(destination) == target:
                    raise OSError("injected final rename failure")
                return real_replace(source, destination)

            with mock.patch(
                "TexGen.training_io.os.replace",
                side_effect=fail_final_rename,
            ):
                with self.assertRaisesRegex(OSError, "injected"):
                    writer.finalize()
            self.assertFalse(target.exists())
            self.assertTrue(
                (
                    target.with_name("dataset.incomplete")
                    / "dataset.json"
                ).is_file()
            )

            resumed = SimulationDatasetWriter.create(
                target,
                schema=schema,
                quality=quality,
                generation={"seed": 8},
                resume=True,
            )
            resumed.finalize()

            manifest = json.loads(
                (target / "dataset.json").read_text(encoding="utf-8")
            )
            self.assertEqual(manifest["rejection_count"], 1)
            rejection = json.loads(
                (target / "rejections.jsonl")
                .read_text(encoding="utf-8")
                .strip()
            )
            self.assertEqual(rejection["sample_id"], "bad")

    def test_resume_refuses_changed_configuration_without_mutation(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "dataset"
            schema = make_schema(shard_size=1)
            quality = self.resumable_quality()
            writer = SimulationDatasetWriter.create(
                target,
                schema=schema,
                quality=quality,
                generation={"seed": 1},
            )
            append_sample(writer, sample_id="s0", group_id="g0")
            staging = target.with_name("dataset.incomplete")

            def snapshot():
                return {
                    path.relative_to(staging).as_posix(): path.read_bytes()
                    for path in staging.rglob("*")
                    if path.is_file()
                }

            before = snapshot()
            mismatches = (
                {
                    "schema": dataclasses.replace(
                        schema, shard_size=2
                    ),
                    "quality": quality,
                    "generation": {"seed": 1},
                },
                {
                    "schema": schema,
                    "quality": DatasetQualityPolicy(
                        require_unique_geometry=False,
                        maximum_solver_residual=1e-6,
                    ),
                    "generation": {"seed": 1},
                },
                {
                    "schema": schema,
                    "quality": quality,
                    "generation": {"seed": 2},
                },
            )
            for kwargs in mismatches:
                with self.subTest(kwargs=kwargs):
                    with self.assertRaisesRegex(
                        ValueError, "configuration"
                    ):
                        SimulationDatasetWriter.create(
                            target, resume=True, **kwargs
                        )
                    self.assertEqual(snapshot(), before)

    def test_resume_verifies_journaled_shard_checksums(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "dataset"
            schema = make_schema(shard_size=1)
            quality = self.resumable_quality()
            writer = SimulationDatasetWriter.create(
                target, schema=schema, quality=quality
            )
            append_sample(writer, sample_id="s0", group_id="g0")
            staging = target.with_name("dataset.incomplete")
            journal = json.loads(
                (staging / "journal.jsonl")
                .read_text(encoding="utf-8")
                .strip()
            )
            first_path = next(iter(journal["shard"]["files"]))
            array_path = staging / first_path
            with array_path.open("r+b") as stream:
                stream.seek(-1, 2)
                byte = stream.read(1)
                stream.seek(-1, 2)
                stream.write(bytes([byte[0] ^ 0xFF]))

            with self.assertRaisesRegex(
                DatasetIntegrityError, "checksum"
            ):
                SimulationDatasetWriter.create(
                    target,
                    schema=schema,
                    quality=quality,
                    resume=True,
                )

    def test_finalize_is_idempotent_non_overwriting_and_cleans_temps(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "dataset"
            writer = SimulationDatasetWriter.create(
                target, schema=make_schema()
            )
            append_sample(writer, sample_id="s0", group_id="g0")
            writer.finalize()
            writer.finalize()
            with self.assertRaisesRegex(RuntimeError, "finalize"):
                append_sample(
                    writer, sample_id="s1", group_id="g1"
                )
            with self.assertRaises(FileExistsError):
                SimulationDatasetWriter.create(
                    target, schema=make_schema()
                )
            self.assertEqual(
                [path for path in target.rglob("*") if ".tmp" in path.name],
                [],
            )

            other = Path(directory) / "appeared"
            blocked = SimulationDatasetWriter.create(
                other, schema=make_schema()
            )
            append_sample(
                blocked, sample_id="s2", group_id="g2"
            )
            other.mkdir()
            with self.assertRaises(FileExistsError):
                blocked.finalize()
            self.assertTrue(
                other.with_name("appeared.incomplete").exists()
            )


def publish_dataset(path):
    schema = make_schema(shard_size=2)
    writer = SimulationDatasetWriter.create(
        path,
        schema=schema,
        quality=DatasetQualityPolicy(require_unique_geometry=False),
        generation={"fixture": "reader"},
    )
    splits = ("train", "validation", "train", "test", "train")
    for index, split in enumerate(splits):
        append_sample(
            writer,
            sample_id=f"s{index}",
            group_id=f"g{index}",
            split=split,
            scale=index + 1.0,
        )
    writer.reject(
        sample_id="bad",
        stage="geometry",
        reason="invalid parameters",
    )
    writer.finalize()
    return schema


class RecordingTransform:
    def __init__(self):
        self.epoch = -1

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __call__(self, example, schema):
        metadata = dict(example.metadata)
        metadata["transform_epoch"] = self.epoch
        return dataclasses.replace(example, metadata=metadata)


class ReaderAndAuditTest(unittest.TestCase):
    def test_reads_selected_fields_from_multiple_mmap_shards_and_pickles(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "dataset"
            publish_dataset(target)

            dataset = SimulationDataset(
                target,
                split="train",
                inputs=(
                    "voxel.material_id",
                    "orientation.primary",
                ),
                targets=("effective_c21",),
                verify="shard",
            )

            self.assertEqual(len(dataset), 3)
            first = dataset[0]
            self.assertEqual(
                set(first.inputs),
                {"voxel.material_id", "orientation.primary"},
            )
            self.assertEqual(set(first.targets), {"effective_c21"})
            self.assertIsInstance(
                first.inputs["voxel.material_id"], np.memmap
            )
            orientation = first.inputs["orientation.primary"]
            self.assertIsInstance(orientation.values, np.memmap)
            np.testing.assert_array_equal(
                orientation.offsets, [0, 2]
            )
            self.assertFalse(
                first.inputs["voxel.material_id"].flags.writeable
            )
            self.assertEqual(first.sample_id, "s0")
            self.assertEqual(first.metadata["provenance"][
                "solver_commit"
            ], "acdm-abc123")

            restored = pickle.loads(pickle.dumps(dataset))
            self.assertEqual(restored[1].sample_id, "s2")
            np.testing.assert_array_equal(
                restored[2].inputs["voxel.material_id"],
                make_sample().array(
                    "voxel.material_id", copy=True
                ),
            )

    def test_shard_verification_opens_and_hashes_only_selected_files(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "dataset"
            publish_dataset(target)
            import TexGen.training_io as training_io

            with mock.patch.object(
                training_io.np, "load", wraps=np.load
            ) as load, mock.patch.object(
                training_io,
                "_sha256_file",
                wraps=training_io._sha256_file,
            ) as checksum:
                dataset = SimulationDataset(
                    target,
                    split="train",
                    inputs=(
                        "voxel.material_id",
                        "orientation.primary",
                    ),
                    targets=("effective_c21",),
                    verify="shard",
                )
                self.assertEqual(load.call_count, 0)
                dataset[0]

            opened = {
                Path(call.args[0]).name for call in load.call_args_list
            }
            hashed = {
                Path(call.args[0]).name
                for call in checksum.call_args_list
            }
            self.assertIn("voxel_material_id.npy", opened)
            self.assertIn("orientation_primary.values.npy", opened)
            self.assertIn("yarn_voxels.offsets.npy", opened)
            self.assertIn("effective_c21.npy", opened)
            self.assertNotIn("material_c21.values.npy", opened)
            self.assertEqual(opened, hashed)

    def test_epoch_propagates_to_transform_and_survives_pickle(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "dataset"
            publish_dataset(target)
            transform = RecordingTransform()
            dataset = SimulationDataset(
                target,
                split="train",
                inputs=("voxel.material_id",),
                targets=("effective_c21",),
                verify="manifest",
                transform=transform,
            )

            dataset.set_epoch(7)
            self.assertEqual(dataset[0].metadata["transform_epoch"], 7)
            restored = pickle.loads(pickle.dumps(dataset))
            self.assertEqual(
                restored[0].metadata["transform_epoch"], 7
            )

    def test_sample_audit_reports_counts_bytes_and_all_checks(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "dataset"
            publish_dataset(target)

            report = audit_simulation_dataset(
                target, verify="sample"
            )

            self.assertTrue(report["ok"])
            self.assertEqual(report["sample_count"], 5)
            self.assertEqual(report["rejection_count"], 1)
            self.assertEqual(report["shard_count"], 3)
            self.assertEqual(
                report["split_counts"],
                {"train": 3, "validation": 1, "test": 1},
            )
            self.assertEqual(report["group_count"], 5)
            self.assertEqual(report["checked_samples"], 5)
            self.assertGreater(report["checked_files"], 0)
            self.assertGreater(report["stored_bytes"], 0)
            self.assertGreater(report["logical_bytes"], 0)

    def test_detects_unsafe_paths_metadata_and_group_leakage_at_manifest(self):
        mutations = (
            "unsafe",
            "dtype",
            "group",
            "topology alias",
            "ragged offsets",
        )
        for mutation in mutations:
            with self.subTest(mutation=mutation):
                with tempfile.TemporaryDirectory() as directory:
                    target = Path(directory) / "dataset"
                    publish_dataset(target)
                    manifest_path = target / "dataset.json"
                    manifest = json.loads(
                        manifest_path.read_text(encoding="utf-8")
                    )
                    expected_error = DatasetFormatError
                    message = mutation
                    if mutation == "unsafe":
                        manifest["shards"][0]["fields"][
                            "voxel.material_id"
                        ]["values"] = "../outside.npy"
                        manifest_path.write_text(
                            json.dumps(manifest), encoding="utf-8"
                        )
                    elif mutation == "dtype":
                        relative = manifest["shards"][0]["fields"][
                            "voxel.material_id"
                        ]["values"]
                        manifest["shards"][0]["files"][relative][
                            "dtype"
                        ] = "<f8"
                        manifest_path.write_text(
                            json.dumps(manifest), encoding="utf-8"
                        )
                    elif mutation == "group":
                        samples_path = target / "samples.jsonl"
                        samples = [
                            json.loads(line)
                            for line in samples_path.read_text(
                                encoding="utf-8"
                            ).splitlines()
                        ]
                        samples[1]["group_id"] = samples[0]["group_id"]
                        samples_path.write_text(
                            "\n".join(
                                json.dumps(value) for value in samples
                            )
                            + "\n",
                            encoding="utf-8",
                        )
                        expected_error = DatasetIntegrityError
                        message = "group"
                    elif mutation == "topology alias":
                        shard = manifest["shards"][0]
                        original = shard["fields"][
                            "orientation.voxel_indices"
                        ]["values"]
                        duplicate = original.replace(
                            ".values.npy", "_copy.values.npy"
                        )
                        (target / duplicate).write_bytes(
                            (target / original).read_bytes()
                        )
                        shard["files"][duplicate] = dict(
                            shard["files"][original]
                        )
                        shard["fields"][
                            "stiffness.voxel_indices"
                        ]["values"] = duplicate
                        manifest_path.write_text(
                            json.dumps(manifest), encoding="utf-8"
                        )
                        message = "alias"
                    else:
                        shard = manifest["shards"][0]
                        original = shard["fields"][
                            "material.ids"
                        ]["offsets"]
                        duplicate = original.replace(
                            ".offsets.npy", "_copy.offsets.npy"
                        )
                        (target / duplicate).write_bytes(
                            (target / original).read_bytes()
                        )
                        shard["files"][duplicate] = dict(
                            shard["files"][original]
                        )
                        shard["fields"]["material.c21"][
                            "offsets"
                        ] = duplicate
                        manifest_path.write_text(
                            json.dumps(manifest), encoding="utf-8"
                        )
                        message = "ragged group"
                    with self.assertRaisesRegex(
                        expected_error, message
                    ):
                        SimulationDataset(
                            target, verify="manifest"
                        )

    def test_checksum_offsets_and_sample_digest_fail_at_promised_levels(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "checksum"
            publish_dataset(target)
            manifest = json.loads(
                (target / "dataset.json").read_text(encoding="utf-8")
            )
            relative = manifest["shards"][0]["fields"][
                "voxel.material_id"
            ]["values"]
            path = target / relative
            with path.open("r+b") as stream:
                stream.seek(-1, 2)
                byte = stream.read(1)
                stream.seek(-1, 2)
                stream.write(bytes([byte[0] ^ 0xFF]))
            dataset = SimulationDataset(
                target,
                inputs=("voxel.material_id",),
                targets=("effective_c21",),
                verify="shard",
            )
            with self.assertRaisesRegex(
                DatasetIntegrityError, "checksum"
            ):
                dataset[0]

        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "offsets"
            publish_dataset(target)
            manifest = json.loads(
                (target / "dataset.json").read_text(encoding="utf-8")
            )
            relative = manifest["shards"][0]["fields"][
                "orientation.primary"
            ]["offsets"]
            np.save(
                target / relative,
                np.array([0, 2, 1], dtype=np.int64),
                allow_pickle=False,
            )
            dataset = SimulationDataset(
                target,
                inputs=("orientation.primary",),
                targets=("effective_c21",),
                verify="manifest",
            )
            with self.assertRaisesRegex(
                DatasetFormatError, "offset"
            ):
                dataset[0]

        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "digest"
            publish_dataset(target)
            samples_path = target / "samples.jsonl"
            samples = [
                json.loads(line)
                for line in samples_path.read_text(
                    encoding="utf-8"
                ).splitlines()
            ]
            samples[0]["geometry_digest"] = "0" * 64
            samples_path.write_text(
                "\n".join(json.dumps(value) for value in samples)
                + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                DatasetIntegrityError, "geometry digest"
            ):
                SimulationDataset(target, verify="sample")

    def test_unselected_corruption_is_lazy_for_shard_but_not_sample(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "dataset"
            publish_dataset(target)
            manifest = json.loads(
                (target / "dataset.json").read_text(encoding="utf-8")
            )
            relative = manifest["shards"][0]["fields"][
                "material.c21"
            ]["values"]
            path = target / relative
            with path.open("r+b") as stream:
                stream.seek(-1, 2)
                byte = stream.read(1)
                stream.seek(-1, 2)
                stream.write(bytes([byte[0] ^ 0xFF]))

            selected = SimulationDataset(
                target,
                inputs=("voxel.material_id",),
                targets=("effective_c21",),
                verify="shard",
            )
            selected[0]
            with self.assertRaisesRegex(
                DatasetIntegrityError, "checksum"
            ):
                SimulationDataset(target, verify="sample")


if __name__ == "__main__":
    unittest.main()
