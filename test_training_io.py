"""Tests for native sharded simulation training datasets."""

import json
import sys
import tempfile
import unittest
from pathlib import Path

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
from TexGen.training_io import SimulationDatasetWriter

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


if __name__ == "__main__":
    unittest.main()
