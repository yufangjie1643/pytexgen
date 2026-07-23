"""Tests for framework-neutral training data contracts."""

import dataclasses
import unittest

import numpy as np

from TexGen.training_data import (
    DatasetQualityPolicy,
    RaggedArray,
    SimulationBatch,
    TrainingDatasetSchema,
    TrainingExample,
    TrainingFieldSpec,
    as_torch_batch,
    collate_training_examples,
)

try:
    import torch
except ImportError:
    torch = None


VOXEL_ORDER = "ix + iy*nx + iz*nx*ny"


def make_schema():
    return TrainingDatasetSchema(
        inputs=(
            TrainingFieldSpec(
                name="voxel.material_id",
                role="input",
                layout="fixed",
                dtype="int32",
                shape=(2, 2, 2),
                semantic="material_id_grid",
            ),
            TrainingFieldSpec(
                name="orientation.voxel_indices",
                role="input",
                layout="ragged",
                dtype="int64",
                shape=(),
                semantic="flat_voxel_index",
                ragged_group="yarn_voxels",
            ),
            TrainingFieldSpec(
                name="orientation.primary",
                role="input",
                layout="ragged",
                dtype="float64",
                shape=(3,),
                semantic="direction_vector",
                ragged_group="yarn_voxels",
            ),
        ),
        targets=(
            TrainingFieldSpec(
                name="effective_c21",
                role="target",
                layout="fixed",
                dtype="float64",
                shape=(21,),
                unit="GPa",
                semantic="engineering_voigt_c21",
            ),
        ),
        grid_shape=(2, 2, 2),
        voxel_order=VOXEL_ORDER,
        shard_size=2,
        statistics_fields=("effective_c21",),
    )


class TrainingSchemaTest(unittest.TestCase):
    def test_field_canonicalizes_dtype_and_preserves_semantics(self):
        field = TrainingFieldSpec(
            name="effective_c21",
            role="target",
            layout="fixed",
            dtype=np.float64,
            shape=(21,),
            unit=" GPa ",
            semantic="engineering_voigt_c21",
        )

        self.assertEqual(field.dtype, np.dtype(np.float64).str)
        self.assertEqual(field.shape, (21,))
        self.assertEqual(field.unit, "GPa")

    def test_field_rejects_unsafe_or_inconsistent_values(self):
        valid = {
            "name": "x",
            "role": "input",
            "layout": "fixed",
            "dtype": "float32",
            "shape": (1,),
        }
        cases = (
            ({"name": "../x"}, "name"),
            ({"name": "x/y"}, "name"),
            ({"role": "feature"}, "role"),
            ({"layout": "dense"}, "layout"),
            ({"dtype": "object"}, "dtype"),
            ({"dtype": "U2"}, "dtype"),
            ({"shape": (0,)}, "shape"),
            ({"shape": (-1,)}, "shape"),
            ({"layout": "ragged"}, "ragged_group"),
            ({"ragged_group": "values"}, "ragged_group"),
        )
        for override, message in cases:
            with self.subTest(override=override):
                values = dict(valid)
                values.update(override)
                with self.assertRaisesRegex(ValueError, message):
                    TrainingFieldSpec(**values)

    def test_c21_semantic_requires_shape_and_unit(self):
        with self.assertRaisesRegex(ValueError, "unit"):
            TrainingFieldSpec(
                "effective_c21",
                "target",
                "fixed",
                "float64",
                (21,),
                semantic="engineering_voigt_c21",
            )
        with self.assertRaisesRegex(ValueError, r"\(21,\)"):
            TrainingFieldSpec(
                "effective_c21",
                "target",
                "fixed",
                "float64",
                (6, 6),
                "GPa",
                "engineering_voigt_c21",
            )

    def test_schema_round_trips_and_exposes_fields(self):
        schema = make_schema()
        restored = TrainingDatasetSchema.from_dict(schema.to_dict())

        self.assertEqual(restored, schema)
        self.assertEqual(
            tuple(field.name for field in schema.fields),
            (
                "voxel.material_id",
                "orientation.voxel_indices",
                "orientation.primary",
                "effective_c21",
            ),
        )
        self.assertEqual(schema.field("effective_c21").role, "target")
        with self.assertRaisesRegex(KeyError, "unknown field"):
            schema.field("missing")

    def test_schema_rejects_invalid_structure(self):
        field = TrainingFieldSpec(
            "x", "input", "fixed", "float32", (1,)
        )
        target_named_input = dataclasses.replace(field, role="target")
        cases = (
            (
                {"inputs": (target_named_input,)},
                "role",
            ),
            (
                {"inputs": (field,), "targets": (target_named_input,)},
                "duplicate",
            ),
            ({"grid_shape": (1, 2)}, "grid_shape"),
            ({"grid_shape": (1, 0, 2)}, "grid_shape"),
            ({"voxel_order": "unknown"}, "voxel_order"),
            ({"shard_size": 0}, "shard_size"),
            ({"statistics_fields": ("missing",)}, "statistics"),
            ({"geometry_digest_field": "missing"}, "geometry"),
        )
        base = {
            "inputs": (field,),
            "targets": (),
            "grid_shape": (1, 1, 1),
            "voxel_order": VOXEL_ORDER,
            "shard_size": 1,
            "geometry_digest_field": "x",
        }
        for override, message in cases:
            with self.subTest(override=override):
                values = dict(base)
                values.update(override)
                with self.assertRaisesRegex(ValueError, message):
                    TrainingDatasetSchema(**values)

    def test_schema_from_dict_rejects_unknown_keys(self):
        serialized = make_schema().to_dict()
        serialized["unexpected"] = True

        with self.assertRaisesRegex(ValueError, "unknown schema keys"):
            TrainingDatasetSchema.from_dict(serialized)

    def test_quality_policy_rejects_invalid_residual(self):
        self.assertEqual(
            DatasetQualityPolicy().maximum_solver_residual,
            1e-8,
        )
        for residual in (0.0, -1.0, float("inf"), float("nan")):
            with self.subTest(residual=residual):
                with self.assertRaisesRegex(ValueError, "residual"):
                    DatasetQualityPolicy(
                        maximum_solver_residual=residual
                    )


class TrainingContainerTest(unittest.TestCase):
    def test_ragged_array_retains_valid_arrays(self):
        values = np.arange(9, dtype=np.float64).reshape(3, 3)
        offsets = np.array([0, 2, 3], dtype=np.int64)

        ragged = RaggedArray(values=values, offsets=offsets)

        self.assertIs(ragged.values, values)
        self.assertIs(ragged.offsets, offsets)

    def test_ragged_array_rejects_invalid_offsets(self):
        values = np.ones((3, 2), dtype=np.float32)
        invalid = (
            np.array([[0, 3]], dtype=np.int64),
            np.array([1, 3], dtype=np.int64),
            np.array([0, 3, 2], dtype=np.int64),
            np.array([0, 2], dtype=np.int64),
            np.array([0.0, 3.0], dtype=np.float64),
        )
        for offsets in invalid:
            with self.subTest(offsets=offsets):
                with self.assertRaisesRegex(ValueError, "offset"):
                    RaggedArray(values=values, offsets=offsets)

    def test_example_freezes_mappings_and_json_metadata_without_copying_arrays(self):
        material_id = np.zeros((2, 2, 2), dtype=np.int32)
        effective = np.ones(21, dtype=np.float64)
        metadata = {"solver": {"iterations": [3, 4]}}
        example = TrainingExample(
            inputs={"voxel.material_id": material_id},
            targets={"effective_c21": effective},
            sample_id="sample-1",
            group_id="geometry-1",
            split="train",
            metadata=metadata,
        )

        self.assertIs(example.inputs["voxel.material_id"], material_id)
        self.assertIs(example.targets["effective_c21"], effective)
        metadata["solver"]["iterations"].append(5)
        self.assertEqual(example.metadata["solver"]["iterations"], (3, 4))
        with self.assertRaises(TypeError):
            example.inputs["new"] = material_id
        with self.assertRaises(TypeError):
            example.metadata["solver"]["iterations"] = ()

    def test_example_rejects_invalid_identity_split_and_metadata(self):
        base = {
            "inputs": {},
            "targets": {},
            "sample_id": "s",
            "group_id": "g",
            "split": "train",
        }
        cases = (
            ({"sample_id": ""}, "sample_id"),
            ({"group_id": " "}, "group_id"),
            ({"split": "dev"}, "split"),
            ({"inputs": []}, "inputs"),
            ({"targets": []}, "targets"),
            ({"metadata": {"bad": float("nan")}}, "metadata"),
        )
        for override, message in cases:
            with self.subTest(override=override):
                values = dict(base)
                values.update(override)
                with self.assertRaisesRegex((TypeError, ValueError), message):
                    TrainingExample(**values)


def make_examples():
    first = TrainingExample(
        inputs={
            "voxel.material_id": np.arange(
                8, dtype=np.int32
            ).reshape(2, 2, 2),
            "orientation.voxel_indices": RaggedArray(
                np.array([1, 5], dtype=np.int64),
                np.array([0, 2], dtype=np.int64),
            ),
            "orientation.primary": RaggedArray(
                np.array(
                    [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
                    dtype=np.float64,
                ),
                np.array([0, 2], dtype=np.int64),
            ),
        },
        targets={
            "effective_c21": np.arange(21, dtype=np.float64),
        },
        sample_id="s0",
        group_id="g0",
        split="train",
        metadata={"case": 0},
    )
    second = TrainingExample(
        inputs={
            "voxel.material_id": np.full(
                (2, 2, 2), 7, dtype=np.int32
            ),
            "orientation.voxel_indices": RaggedArray(
                np.array([3], dtype=np.int64),
                np.array([0, 1], dtype=np.int64),
            ),
            "orientation.primary": RaggedArray(
                np.array([[0.0, 0.0, 1.0]], dtype=np.float64),
                np.array([0, 1], dtype=np.int64),
            ),
        },
        targets={
            "effective_c21": np.arange(
                21, dtype=np.float64
            ) + 100.0,
        },
        sample_id="s1",
        group_id="g1",
        split="validation",
        metadata={"case": 1},
    )
    return first, second


class TrainingCollationTest(unittest.TestCase):
    def test_collates_fixed_and_shared_ragged_fields_into_owned_arrays(self):
        examples = make_examples()

        batch = collate_training_examples(examples, make_schema())

        self.assertIsInstance(batch, SimulationBatch)
        fixed = batch.inputs["voxel.material_id"]
        self.assertEqual(fixed.shape, (2, 2, 2, 2))
        self.assertTrue(fixed.flags.owndata)
        self.assertTrue(fixed.flags.c_contiguous)
        self.assertTrue(fixed.flags.writeable)
        np.testing.assert_array_equal(fixed[0], examples[0].inputs[
            "voxel.material_id"
        ])
        indices = batch.inputs["orientation.voxel_indices"]
        directions = batch.inputs["orientation.primary"]
        self.assertIs(indices.offsets, directions.offsets)
        np.testing.assert_array_equal(indices.offsets, [0, 2, 3])
        np.testing.assert_array_equal(indices.values, [1, 5, 3])
        self.assertTrue(indices.values.flags.owndata)
        self.assertTrue(directions.values.flags.owndata)
        self.assertEqual(batch.targets["effective_c21"].shape, (2, 21))
        self.assertEqual(batch.sample_ids, ("s0", "s1"))
        self.assertEqual(batch.group_ids, ("g0", "g1"))
        self.assertEqual(batch.metadata[1]["case"], 1)
        self.assertEqual(batch.nbytes, 520)

    def test_collation_copies_read_only_sources(self):
        examples = list(make_examples())
        readonly = examples[0].inputs["voxel.material_id"]
        readonly.flags.writeable = False

        batch = collate_training_examples(examples, make_schema())

        self.assertTrue(batch.inputs["voxel.material_id"].flags.writeable)
        self.assertFalse(
            np.shares_memory(batch.inputs["voxel.material_id"], readonly)
        )

    def test_collation_rejects_missing_extra_shape_dtype_and_offsets(self):
        schema = make_schema()
        first, second = make_examples()
        cases = []

        inputs = dict(first.inputs)
        del inputs["orientation.primary"]
        cases.append(
            (
                dataclasses.replace(first, inputs=inputs),
                "missing",
            )
        )
        inputs = dict(first.inputs)
        inputs["extra"] = np.ones(1)
        cases.append(
            (
                dataclasses.replace(first, inputs=inputs),
                "extra",
            )
        )
        inputs = dict(first.inputs)
        inputs["voxel.material_id"] = np.zeros((2, 2), dtype=np.int32)
        cases.append(
            (
                dataclasses.replace(first, inputs=inputs),
                "shape",
            )
        )
        inputs = dict(first.inputs)
        inputs["voxel.material_id"] = np.zeros(
            (2, 2, 2), dtype=np.int64
        )
        cases.append(
            (
                dataclasses.replace(first, inputs=inputs),
                "dtype",
            )
        )
        inputs = dict(first.inputs)
        inputs["orientation.primary"] = RaggedArray(
            np.ones((1, 3), dtype=np.float64),
            np.array([0, 1], dtype=np.int64),
        )
        cases.append(
            (
                dataclasses.replace(first, inputs=inputs),
                "ragged group",
            )
        )

        for bad, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    collate_training_examples((bad, second), schema)

    def test_batch_mapping_is_read_only_and_as_dict_is_complete(self):
        batch = collate_training_examples(make_examples(), make_schema())

        with self.assertRaises(TypeError):
            batch.inputs["new"] = np.ones(1)
        exported = batch.as_dict()
        self.assertEqual(
            set(exported),
            {"inputs", "targets", "sample_ids", "group_ids", "metadata"},
        )
        self.assertIs(exported["inputs"], batch.inputs)

    @unittest.skipIf(torch is None, "PyTorch is not installed")
    def test_torch_conversion_shares_owned_cpu_memory_and_preserves_types(self):
        numpy_batch = collate_training_examples(
            make_examples(), make_schema()
        )

        batch = as_torch_batch(numpy_batch)

        fixed = batch.inputs["voxel.material_id"]
        self.assertIsInstance(fixed, torch.Tensor)
        self.assertEqual(fixed.dtype, torch.int32)
        self.assertEqual(
            fixed.data_ptr(),
            numpy_batch.inputs["voxel.material_id"].ctypes.data,
        )
        indices = batch.inputs["orientation.voxel_indices"]
        directions = batch.inputs["orientation.primary"]
        self.assertIs(indices.offsets, directions.offsets)
        self.assertEqual(
            indices.values.data_ptr(),
            numpy_batch.inputs[
                "orientation.voxel_indices"
            ].values.ctypes.data,
        )
        self.assertEqual(batch.nbytes, numpy_batch.nbytes)
        moved = batch.to("cpu", non_blocking=True)
        self.assertEqual(moved.sample_ids, batch.sample_ids)
        self.assertEqual(moved.metadata, batch.metadata)

    @unittest.skipUnless(
        torch is not None and torch.cuda.is_available(),
        "CUDA is required for pinned-memory allocation",
    )
    def test_batch_pins_all_tensors_without_changing_metadata(self):
        batch = as_torch_batch(
            collate_training_examples(make_examples(), make_schema())
        )

        pinned = batch.pin_memory()

        self.assertTrue(pinned.inputs["voxel.material_id"].is_pinned())
        self.assertTrue(
            pinned.inputs["orientation.primary"].values.is_pinned()
        )
        self.assertTrue(
            pinned.inputs["orientation.primary"].offsets.is_pinned()
        )
        self.assertEqual(pinned.metadata, batch.metadata)


if __name__ == "__main__":
    unittest.main()
