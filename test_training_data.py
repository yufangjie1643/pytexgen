"""Tests for framework-neutral training data contracts."""

import dataclasses
import unittest

import numpy as np

from TexGen.training_data import (
    DatasetQualityPolicy,
    CubicRotation,
    RaggedArray,
    RunningFieldStatistics,
    SimulationBatch,
    StandardizeFields,
    TrainingDatasetSchema,
    TrainingExample,
    TrainingFieldSpec,
    as_torch_batch,
    collate_training_examples,
    compute_training_statistics,
    deterministic_group_split,
    proper_cubic_rotations,
    rotate_engineering_voigt_c21,
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

    def test_example_accepts_already_frozen_nested_metadata(self):
        first = TrainingExample(
            inputs={},
            targets={},
            sample_id="s0",
            group_id="g0",
            split="train",
            metadata={"nested": {"values": [1, 2]}},
        )

        second = TrainingExample(
            inputs={},
            targets={},
            sample_id="s1",
            group_id="g0",
            split="train",
            metadata=first.metadata,
        )

        self.assertEqual(
            second.metadata["nested"]["values"], (1, 2)
        )


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


class GroupSplitTest(unittest.TestCase):
    def test_split_is_deterministic_and_input_order_independent(self):
        ratios = {"train": 0.5, "validation": 0.25, "test": 0.25}

        first = deterministic_group_split(
            ["g3", "g1", "g2", "g1"],
            ratios=ratios,
            seed=42,
        )
        second = deterministic_group_split(
            ["g2", "g1", "g3"],
            ratios={
                "test": 0.25,
                "train": 0.5,
                "validation": 0.25,
            },
            seed=42,
        )

        self.assertEqual(first, second)
        self.assertEqual(set(first), {"g1", "g2", "g3"})
        self.assertEqual(
            sorted(first.values()),
            ["test", "train", "validation"],
        )
        self.assertEqual(
            first,
            deterministic_group_split(
                ["g1", "g2", "g3"],
                ratios=ratios,
                seed=42,
            ),
        )

    def test_strata_allocate_whole_groups_to_every_split(self):
        groups = [f"g{index}" for index in range(6)]
        strata = {
            group: ("plain" if index < 3 else "twill")
            for index, group in enumerate(groups)
        }

        result = deterministic_group_split(
            groups,
            ratios={
                "train": 1 / 3,
                "validation": 1 / 3,
                "test": 1 / 3,
            },
            seed=7,
            strata=strata,
        )

        for stratum in ("plain", "twill"):
            assigned = {
                result[group]
                for group in groups
                if strata[group] == stratum
            }
            self.assertEqual(
                assigned, {"train", "validation", "test"}
            )

    def test_split_rejects_invalid_ratios_groups_and_strata(self):
        cases = (
            (
                {"group_ids": [], "ratios": {"train": 1.0}},
                "group",
            ),
            (
                {
                    "group_ids": ["g"],
                    "ratios": {"train": 0.8, "test": 0.1},
                },
                "sum",
            ),
            (
                {
                    "group_ids": ["g"],
                    "ratios": {"train": 1.0, "holdout": 0.0},
                },
                "split",
            ),
            (
                {
                    "group_ids": ["g"],
                    "ratios": {"train": -1.0, "test": 2.0},
                },
                "ratio",
            ),
            (
                {
                    "group_ids": ["g", ""],
                    "ratios": {"train": 1.0},
                },
                "group",
            ),
            (
                {
                    "group_ids": ["g1", "g2"],
                    "ratios": {"train": 1.0},
                    "strata": {"g1": "plain"},
                },
                "strata",
            ),
        )
        for kwargs, message in cases:
            with self.subTest(kwargs=kwargs):
                kwargs.setdefault("seed", 1)
                with self.assertRaisesRegex(ValueError, message):
                    deterministic_group_split(**kwargs)


class StatisticsTest(unittest.TestCase):
    def test_welford_statistics_are_componentwise_and_constant_safe(self):
        accumulator = RunningFieldStatistics(component_shape=(3,))
        accumulator.update(
            np.array([[1.0, 5.0, -1.0], [3.0, 5.0, 1.0]])
        )
        accumulator.update(np.array([[5.0, 5.0, 3.0]]))

        result = accumulator.finalize(unit="GPa")

        np.testing.assert_allclose(result["mean"], [3.0, 5.0, 1.0])
        np.testing.assert_allclose(
            result["variance"], [8 / 3, 0.0, 8 / 3]
        )
        np.testing.assert_allclose(
            result["standard_deviation"],
            [np.sqrt(8 / 3), 1.0, np.sqrt(8 / 3)],
        )
        self.assertEqual(result["constant_mask"], [False, True, False])
        self.assertEqual(result["minimum"], [1.0, 5.0, -1.0])
        self.assertEqual(result["maximum"], [5.0, 5.0, 3.0])
        self.assertEqual(result["count"], 3)
        self.assertEqual(result["source_split"], "train")
        self.assertEqual(result["unit"], "GPa")

    def test_dataset_statistics_exclude_validation_and_test(self):
        schema = make_schema()
        first, second = make_examples()
        train_two = dataclasses.replace(
            first,
            sample_id="s2",
            targets={
                "effective_c21": np.full(
                    21, 2.0, dtype=np.float64
                )
            },
        )
        validation = dataclasses.replace(
            second,
            targets={
                "effective_c21": np.full(
                    21, 1000.0, dtype=np.float64
                )
            },
        )
        test = dataclasses.replace(
            validation,
            sample_id="s3",
            split="test",
            targets={
                "effective_c21": np.full(
                    21, -1000.0, dtype=np.float64
                )
            },
        )

        statistics = compute_training_statistics(
            (first, train_two, validation, test), schema
        )

        expected_mean = (
            np.arange(21, dtype=np.float64) + 2.0
        ) / 2.0
        np.testing.assert_allclose(
            statistics["effective_c21"]["mean"], expected_mean
        )
        self.assertEqual(
            statistics["effective_c21"]["count"], 2
        )

    def test_standardization_uses_declared_units_and_leaves_metadata(self):
        schema = make_schema()
        example = make_examples()[0]
        statistics = {
            "effective_c21": {
                "mean": np.arange(21, dtype=np.float64).tolist(),
                "variance": np.ones(21).tolist(),
                "standard_deviation": np.full(21, 2.0).tolist(),
                "minimum": np.zeros(21).tolist(),
                "maximum": np.ones(21).tolist(),
                "constant_mask": [False] * 21,
                "count": 2,
                "source_split": "train",
                "unit": "GPa",
            }
        }
        transform = StandardizeFields(
            statistics=statistics,
            fields=("effective_c21",),
        )

        transformed = transform(example, schema)

        np.testing.assert_allclose(
            transformed.targets["effective_c21"], np.zeros(21)
        )
        self.assertIs(
            transformed.inputs["voxel.material_id"],
            example.inputs["voxel.material_id"],
        )
        self.assertEqual(transformed.metadata, example.metadata)

        bad_unit = {
            "effective_c21": dict(
                statistics["effective_c21"], unit="Pa"
            )
        }
        with self.assertRaisesRegex(ValueError, "unit"):
            StandardizeFields(
                bad_unit, ("effective_c21",)
            )(example, schema)
        with self.assertRaisesRegex(ValueError, "floating"):
            StandardizeFields(
                statistics={"voxel.material_id": {}},
                fields=("voxel.material_id",),
            )(example, schema)

    def test_statistics_reject_nonfinite_or_wrong_component_shape(self):
        accumulator = RunningFieldStatistics(component_shape=(2,))
        with self.assertRaisesRegex(ValueError, "shape"):
            accumulator.update(np.ones((3,), dtype=np.float64))
        with self.assertRaisesRegex(ValueError, "finite"):
            accumulator.update(
                np.array([[1.0, np.nan]], dtype=np.float64)
            )
        with self.assertRaisesRegex(ValueError, "empty"):
            accumulator.finalize(unit=None)


def _pack_c21_reference(matrix):
    return np.stack(
        [
            matrix[..., row, column]
            for row in range(6)
            for column in range(row, 6)
        ],
        axis=-1,
    )


def _unpack_c21_reference(c21):
    result = np.zeros(c21.shape[:-1] + (6, 6), dtype=c21.dtype)
    index = 0
    for row in range(6):
        for column in range(row, 6):
            result[..., row, column] = c21[..., index]
            result[..., column, row] = c21[..., index]
            index += 1
    return result


def _rotate_c21_reference(c21, rotation):
    pairs = (
        (0, 0),
        (1, 1),
        (2, 2),
        (1, 2),
        (0, 2),
        (0, 1),
    )
    matrix = _unpack_c21_reference(np.asarray(c21))
    tensor = np.zeros(
        matrix.shape[:-2] + (3, 3, 3, 3), dtype=matrix.dtype
    )
    for row, (i, j) in enumerate(pairs):
        for column, (k, ell) in enumerate(pairs):
            value = matrix[..., row, column]
            tensor[..., i, j, k, ell] = value
            tensor[..., j, i, k, ell] = value
            tensor[..., i, j, ell, k] = value
            tensor[..., j, i, ell, k] = value
    rotated = np.einsum(
        "iI,jJ,kK,lL,...IJKL->...ijkl",
        rotation,
        rotation,
        rotation,
        rotation,
        tensor,
    )
    result = np.empty(matrix.shape, dtype=matrix.dtype)
    for row, (i, j) in enumerate(pairs):
        for column, (k, ell) in enumerate(pairs):
            result[..., row, column] = rotated[..., i, j, k, ell]
    return _pack_c21_reference(result)


def make_rotation_schema():
    yarn_group = "yarn_voxels"
    material_group = "materials"
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
                ragged_group=yarn_group,
            ),
            TrainingFieldSpec(
                "orientation.primary",
                "input",
                "ragged",
                "float64",
                (3,),
                semantic="direction_vector",
                ragged_group=yarn_group,
            ),
            TrainingFieldSpec(
                "stiffness.yarn_c21",
                "input",
                "ragged",
                "float64",
                (21,),
                "GPa",
                "global_engineering_voigt_c21",
                yarn_group,
            ),
            TrainingFieldSpec(
                "stiffness.matrix_c21",
                "input",
                "fixed",
                "float64",
                (21,),
                "GPa",
                "global_engineering_voigt_c21",
            ),
            TrainingFieldSpec(
                "stiffness.voxel_c21",
                "input",
                "fixed",
                "float64",
                (2, 2, 2, 21),
                "GPa",
                "global_engineering_voigt_c21",
            ),
            TrainingFieldSpec(
                "voxel.direction",
                "input",
                "fixed",
                "float64",
                (2, 2, 2, 3),
                semantic="direction_vector_field",
            ),
            TrainingFieldSpec(
                "domain.aabb",
                "input",
                "fixed",
                "float64",
                (2, 3),
                semantic="aabb",
            ),
            TrainingFieldSpec(
                "material.ids",
                "input",
                "ragged",
                "int32",
                (),
                semantic="material_id",
                ragged_group=material_group,
            ),
            TrainingFieldSpec(
                "material.c21",
                "input",
                "ragged",
                "float64",
                (21,),
                "GPa",
                "local_engineering_voigt_c21",
                material_group,
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
        shard_size=2,
    )


def make_rotation_example():
    base = np.arange(1.0, 37.0).reshape(6, 6)
    general = base @ base.T + 10.0 * np.eye(6)
    c21 = _pack_c21_reference(general)
    matrix = _pack_c21_reference(
        np.diag([2.0, 3.0, 5.0, 7.0, 11.0, 13.0])
    )
    offsets = np.array([0, 2], dtype=np.int64)
    material_offsets = np.array([0, 2], dtype=np.int64)
    return TrainingExample(
        inputs={
            "voxel.material_id": np.arange(
                8, dtype=np.int32
            ).reshape(2, 2, 2),
            "orientation.voxel_indices": RaggedArray(
                np.array([0, 3], dtype=np.int64), offsets
            ),
            "orientation.primary": RaggedArray(
                np.array(
                    [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
                ),
                offsets,
            ),
            "stiffness.yarn_c21": RaggedArray(
                np.stack((c21, matrix)), offsets
            ),
            "stiffness.matrix_c21": matrix.copy(),
            "material.ids": RaggedArray(
                np.array([0, 7], dtype=np.int32),
                material_offsets,
            ),
            "material.c21": RaggedArray(
                np.stack((matrix, c21)), material_offsets
            ),
        },
        targets={"effective_c21": c21.copy()},
        sample_id="rotation-sample",
        group_id="rotation-group",
        split="train",
        metadata={"source": {"seed": 4}},
    )


class CubicRotationTest(unittest.TestCase):
    @staticmethod
    def rotation_z_90():
        return np.array(
            [[0, -1, 0], [1, 0, 0], [0, 0, 1]],
            dtype=np.int8,
        )

    def test_rotation_set_contains_24_unique_proper_matrices(self):
        rotations = proper_cubic_rotations()

        self.assertEqual(len(rotations), 24)
        self.assertEqual(
            len({tuple(rotation.reshape(-1)) for rotation in rotations}),
            24,
        )
        for rotation in rotations:
            self.assertEqual(rotation.dtype, np.int8)
            np.testing.assert_array_equal(
                rotation @ rotation.T, np.eye(3, dtype=np.int8)
            )
            self.assertEqual(round(np.linalg.det(rotation)), 1)
            self.assertTrue(np.isin(rotation, (-1, 0, 1)).all())

    def test_c21_rotation_matches_independent_fourth_order_reference(self):
        base = np.arange(1.0, 37.0).reshape(6, 6) / 11.0
        matrix = base @ base.T + 2.0 * np.eye(6)
        c21 = _pack_c21_reference(matrix)

        for rotation in proper_cubic_rotations():
            with self.subTest(rotation=rotation.tolist()):
                actual = rotate_engineering_voigt_c21(c21, rotation)
                expected = _rotate_c21_reference(c21, rotation)
                np.testing.assert_allclose(
                    actual, expected, rtol=1e-12, atol=1e-12
                )

    def test_explicit_rotation_couples_grid_sparse_direction_and_c21(self):
        schema = make_rotation_schema()
        example = make_rotation_example()
        transform = CubicRotation(seed=3)
        rotation = self.rotation_z_90()
        rotation_id = next(
            index
            for index, candidate in enumerate(proper_cubic_rotations())
            if np.array_equal(candidate, rotation)
        )

        rotated = transform.apply(example, schema, rotation_id)

        expected_grid = np.empty((2, 2, 2), dtype=np.int32)
        for z in range(2):
            for y in range(2):
                for x in range(2):
                    old = np.array([x, y, z])
                    new = rotation @ (2 * old - 1)
                    new = ((new + 1) // 2).astype(int)
                    expected_grid[new[2], new[1], new[0]] = (
                        example.inputs["voxel.material_id"][z, y, x]
                    )
        np.testing.assert_array_equal(
            rotated.inputs["voxel.material_id"], expected_grid
        )

        old_indices = example.inputs[
            "orientation.voxel_indices"
        ].values
        expected_indices = []
        for index in old_indices:
            x = int(index) % 2
            y = (int(index) // 2) % 2
            z = int(index) // 4
            new = rotation @ (2 * np.array([x, y, z]) - 1)
            new = ((new + 1) // 2).astype(int)
            expected_indices.append(
                new[0] + 2 * new[1] + 4 * new[2]
            )
        order = np.argsort(expected_indices)
        np.testing.assert_array_equal(
            rotated.inputs["orientation.voxel_indices"].values,
            np.asarray(expected_indices)[order],
        )
        expected_directions = (
            example.inputs["orientation.primary"].values @ rotation.T
        )[order]
        np.testing.assert_allclose(
            rotated.inputs["orientation.primary"].values,
            expected_directions,
        )
        expected_yarn_c21 = rotate_engineering_voigt_c21(
            example.inputs["stiffness.yarn_c21"].values,
            rotation,
        )[order]
        np.testing.assert_allclose(
            rotated.inputs["stiffness.yarn_c21"].values,
            expected_yarn_c21,
        )
        np.testing.assert_allclose(
            rotated.inputs["stiffness.matrix_c21"],
            rotate_engineering_voigt_c21(
                example.inputs["stiffness.matrix_c21"], rotation
            ),
        )
        np.testing.assert_allclose(
            rotated.targets["effective_c21"],
            rotate_engineering_voigt_c21(
                example.targets["effective_c21"], rotation
            ),
        )
        self.assertIs(
            rotated.inputs["material.ids"].values,
            example.inputs["material.ids"].values,
        )
        self.assertIs(
            rotated.inputs["material.c21"].values,
            example.inputs["material.c21"].values,
        )
        self.assertEqual(rotated.metadata["rotation_id"], rotation_id)
        self.assertEqual(
            rotated.metadata["rotation_matrix"],
            tuple(tuple(int(value) for value in row) for row in rotation),
        )

    def test_rotation_and_inverse_recover_every_field(self):
        schema = make_rotation_schema()
        original = make_rotation_example()
        rotations = proper_cubic_rotations()
        rotation_id = 7
        inverse_id = next(
            index
            for index, candidate in enumerate(rotations)
            if np.array_equal(candidate, rotations[rotation_id].T)
        )
        transform = CubicRotation(seed=9)

        rotated = transform.apply(original, schema, rotation_id)
        recovered = transform.apply(rotated, schema, inverse_id)

        for name, original_value in original.inputs.items():
            recovered_value = recovered.inputs[name]
            if isinstance(original_value, RaggedArray):
                if np.issubdtype(original_value.values.dtype, np.integer):
                    np.testing.assert_array_equal(
                        recovered_value.values, original_value.values
                    )
                else:
                    np.testing.assert_allclose(
                        recovered_value.values,
                        original_value.values,
                        rtol=1e-12,
                        atol=1e-12,
                    )
            elif np.issubdtype(original_value.dtype, np.integer):
                np.testing.assert_array_equal(
                    recovered_value, original_value
                )
            else:
                np.testing.assert_allclose(
                    recovered_value,
                    original_value,
                    rtol=1e-12,
                    atol=1e-12,
                )
        np.testing.assert_allclose(
            recovered.targets["effective_c21"],
            original.targets["effective_c21"],
            rtol=1e-12,
            atol=1e-12,
        )

    def test_rotates_dense_direction_c21_and_domain_extents(self):
        schema = make_rotation_schema()
        original = make_rotation_example()
        rotation = self.rotation_z_90()
        rotation_id = next(
            index
            for index, candidate in enumerate(proper_cubic_rotations())
            if np.array_equal(candidate, rotation)
        )
        dense_direction = np.arange(
            24, dtype=np.float64
        ).reshape(2, 2, 2, 3)
        base_c21 = original.targets["effective_c21"]
        dense_c21 = np.stack(
            [base_c21 + index for index in range(8)]
        ).reshape(2, 2, 2, 21)
        inputs = dict(original.inputs)
        inputs.update(
            {
                "voxel.direction": dense_direction,
                "stiffness.voxel_c21": dense_c21,
                "domain.aabb": np.array(
                    [[0.0, 0.0, 0.0], [2.0, 4.0, 6.0]]
                ),
            }
        )
        example = dataclasses.replace(original, inputs=inputs)

        rotated = CubicRotation(seed=1).apply(
            example, schema, rotation_id
        )

        expected_direction = np.empty_like(dense_direction)
        expected_c21 = np.empty_like(dense_c21)
        for z in range(2):
            for y in range(2):
                for x in range(2):
                    old = np.array([x, y, z])
                    new = rotation @ (2 * old - 1)
                    new = ((new + 1) // 2).astype(int)
                    destination = (new[2], new[1], new[0])
                    expected_direction[destination] = (
                        rotation @ dense_direction[z, y, x]
                    )
                    expected_c21[destination] = (
                        _rotate_c21_reference(
                            dense_c21[z, y, x], rotation
                        )
                    )
        np.testing.assert_allclose(
            rotated.inputs["voxel.direction"], expected_direction
        )
        np.testing.assert_allclose(
            rotated.inputs["stiffness.voxel_c21"], expected_c21
        )
        np.testing.assert_allclose(
            rotated.inputs["domain.aabb"],
            [[-1.0, 1.0, 0.0], [3.0, 3.0, 6.0]],
        )

    def test_preserves_float32_and_rejects_unknown_physical_semantics(self):
        c21 = make_rotation_example().targets[
            "effective_c21"
        ].astype(np.float32)
        rotated = rotate_engineering_voigt_c21(
            c21, self.rotation_z_90()
        )
        self.assertEqual(rotated.dtype, np.float32)

        unknown = TrainingFieldSpec(
            "orientation.unknown",
            "input",
            "fixed",
            "float64",
            (3,),
            semantic="direction_cosines",
        )
        schema = dataclasses.replace(
            make_rotation_schema(),
            inputs=make_rotation_schema().inputs + (unknown,),
        )
        example = make_rotation_example()
        inputs = dict(example.inputs)
        inputs[unknown.name] = np.array([1.0, 0.0, 0.0])
        example = dataclasses.replace(example, inputs=inputs)

        with self.assertRaisesRegex(ValueError, "semantic"):
            CubicRotation(seed=1).apply(example, schema, 0)

    def test_hash_choice_is_reproducible_across_order_and_epochs(self):
        schema = make_rotation_schema()
        first = make_rotation_example()
        second = dataclasses.replace(
            first, sample_id="another-sample"
        )
        left = CubicRotation(seed=42)
        right = CubicRotation(seed=42)

        ids_left = [
            left(example, schema).metadata["rotation_id"]
            for example in (first, second)
        ]
        ids_right = [
            right(example, schema).metadata["rotation_id"]
            for example in (second, first)
        ]
        self.assertEqual(ids_left, list(reversed(ids_right)))

        epoch_ids = []
        for epoch in range(4):
            left.set_epoch(epoch)
            epoch_ids.append(
                left(first, schema).metadata["rotation_id"]
            )
        self.assertGreater(len(set(epoch_ids)), 1)

    def test_rejects_non_cubic_schema_and_invalid_rotation_request(self):
        schema = dataclasses.replace(
            make_rotation_schema(), grid_shape=(2, 2, 3)
        )
        with self.assertRaisesRegex(ValueError, "cubic"):
            CubicRotation(seed=1)(make_rotation_example(), schema)
        with self.assertRaisesRegex(ValueError, "rotation_id"):
            CubicRotation(seed=1).apply(
                make_rotation_example(),
                make_rotation_schema(),
                24,
            )


if __name__ == "__main__":
    unittest.main()
