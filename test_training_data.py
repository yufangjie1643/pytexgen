"""Tests for framework-neutral training data contracts."""

import dataclasses
import unittest

import numpy as np

from TexGen.training_data import (
    DatasetQualityPolicy,
    RaggedArray,
    TrainingDatasetSchema,
    TrainingExample,
    TrainingFieldSpec,
)


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


if __name__ == "__main__":
    unittest.main()
