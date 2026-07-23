"""Tests for sparse orientation and stiffness field utilities."""

import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
MODULE_PATH = ROOT / "TexGen" / "material_fields.py"


def load_material_fields():
    """Load the source module without requiring an installed wheel."""
    if not MODULE_PATH.exists():
        raise AssertionError("TexGen/material_fields.py has not been created")
    spec = importlib.util.spec_from_file_location(
        "material_fields_under_test", MODULE_PATH
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class C21UtilitiesTest(unittest.TestCase):
    def setUp(self):
        self.mf = load_material_fields()

    def test_pack_unpack_uses_documented_upper_triangle_order(self):
        matrix = np.arange(36, dtype=np.float64).reshape(6, 6)
        matrix = np.triu(matrix)
        matrix = matrix + np.triu(matrix, 1).T

        packed = self.mf.pack_voigt_c21(matrix)

        np.testing.assert_array_equal(packed, matrix[np.triu_indices(6)])
        np.testing.assert_array_equal(self.mf.unpack_c21(packed), matrix)
        self.assertEqual(
            self.mf.VOIGT_COMPONENTS,
            ("xx", "yy", "zz", "yz", "xz", "xy"),
        )

    def test_pack_supports_batches(self):
        matrices = np.stack([np.eye(6), 2.0 * np.eye(6)])

        packed = self.mf.pack_voigt_c21(matrices)
        restored = self.mf.unpack_c21(packed)

        self.assertEqual(packed.shape, (2, 21))
        np.testing.assert_array_equal(restored, matrices)

    def test_pack_rejects_nonsymmetric_or_nonfinite_matrix(self):
        nonsymmetric = np.eye(6)
        nonsymmetric[0, 1] = 1.0
        with self.assertRaisesRegex(ValueError, "symmetric"):
            self.mf.pack_voigt_c21(nonsymmetric)

        nonfinite = np.eye(6)
        nonfinite[0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "finite"):
            self.mf.pack_voigt_c21(nonfinite)

    def test_unpack_rejects_wrong_shape(self):
        with self.assertRaisesRegex(ValueError, "length 21"):
            self.mf.unpack_c21(np.zeros(20))

    def test_isotropic_helper_recovers_lame_coefficients(self):
        matrix = self.mf.unpack_c21(
            self.mf.isotropic_stiffness_c21(70.0, 0.25)
        )

        self.assertAlmostEqual(matrix[3, 3], 28.0)
        self.assertAlmostEqual(matrix[0, 1], 28.0)
        self.assertAlmostEqual(matrix[0, 0], 84.0)

    def test_isotropic_helper_rejects_invalid_parameters(self):
        with self.assertRaisesRegex(ValueError, "positive"):
            self.mf.isotropic_stiffness_c21(0.0, 0.25)
        with self.assertRaisesRegex(ValueError, "Poisson"):
            self.mf.isotropic_stiffness_c21(70.0, 0.5)

    def test_orthotropic_helper_inverts_engineering_compliance(self):
        c21 = self.mf.orthotropic_stiffness_c21(
            150.0, 10.0, 12.0,
            0.25, 0.20, 0.30,
            5.0, 6.0, 4.0,
        )

        compliance = np.linalg.inv(self.mf.unpack_c21(c21))

        self.assertAlmostEqual(compliance[0, 0], 1.0 / 150.0)
        self.assertAlmostEqual(compliance[1, 1], 1.0 / 10.0)
        self.assertAlmostEqual(compliance[5, 5], 1.0 / 5.0)
        self.assertAlmostEqual(compliance[0, 1], -0.25 / 150.0)

    def test_orthotropic_helper_rejects_nonpositive_modulus(self):
        with self.assertRaisesRegex(ValueError, "positive"):
            self.mf.orthotropic_stiffness_c21(
                150.0, 10.0, 12.0,
                0.25, 0.20, 0.30,
                0.0, 6.0, 4.0,
            )


class SparseFieldContainerTest(unittest.TestCase):
    def setUp(self):
        self.mf = load_material_fields()
        self.assertTrue(
            hasattr(self.mf, "SparseOrientationField"),
            "SparseOrientationField is not implemented",
        )
        self.assertTrue(
            hasattr(self.mf, "SparseStiffnessField"),
            "SparseStiffnessField is not implemented",
        )

    def make_orientation(self):
        return self.mf.SparseOrientationField(
            voxel_indices=np.array([1, 3], dtype=np.int64),
            yarn_ids=np.array([0, 2], dtype=np.int32),
            orientation1=np.array(
                [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
                dtype=np.float64,
            ),
            orientation2=np.array(
                [[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]],
                dtype=np.float64,
            ),
            grid_shape=(1, 2, 2),
        )

    def test_orientation_stores_valid_compact_arrays(self):
        field = self.make_orientation()

        self.assertEqual(field.grid_shape, (1, 2, 2))
        self.assertEqual(field.num_yarn_voxels, 2)
        self.assertEqual(field.storage, "numpy")
        np.testing.assert_array_equal(field.voxel_indices, [1, 3])

    def test_orientation_rejects_unsorted_or_duplicate_indices(self):
        with self.assertRaisesRegex(ValueError, "strictly increasing"):
            self.mf.SparseOrientationField(
                voxel_indices=np.array([3, 3], dtype=np.int64),
                yarn_ids=np.array([0, 2], dtype=np.int32),
                orientation1=np.ones((2, 3)),
                orientation2=np.ones((2, 3)),
                grid_shape=(1, 2, 2),
            )

    def test_orientation_rejects_inconsistent_shapes_and_range(self):
        with self.assertRaisesRegex(ValueError, "same leading length"):
            self.mf.SparseOrientationField(
                voxel_indices=np.array([1, 3], dtype=np.int64),
                yarn_ids=np.array([0], dtype=np.int32),
                orientation1=np.ones((2, 3)),
                orientation2=np.ones((2, 3)),
                grid_shape=(1, 2, 2),
            )
        with self.assertRaisesRegex(ValueError, "out of range"):
            self.mf.SparseOrientationField(
                voxel_indices=np.array([1, 4], dtype=np.int64),
                yarn_ids=np.array([0, 2], dtype=np.int32),
                orientation1=np.ones((2, 3)),
                orientation2=np.ones((2, 3)),
                grid_shape=(1, 2, 2),
            )

    def test_orientation_to_numpy_changes_float_dtype_not_integer_dtype(self):
        converted = self.make_orientation().to("numpy", dtype=np.float32)

        self.assertEqual(converted.orientation1.dtype, np.float32)
        self.assertEqual(converted.orientation2.dtype, np.float32)
        self.assertEqual(converted.voxel_indices.dtype, np.int64)
        self.assertEqual(converted.yarn_ids.dtype, np.int32)

    def make_stiffness(self):
        yarn = np.stack(
            [
                np.arange(21, dtype=np.float64),
                np.arange(21, dtype=np.float64) + 100.0,
            ]
        )
        return self.mf.SparseStiffnessField(
            matrix_c21=np.full(21, -1.0),
            voxel_indices=np.array([1, 3], dtype=np.int64),
            yarn_ids=np.array([0, 2], dtype=np.int32),
            material_ids=np.array([1, 2], dtype=np.int32),
            yarn_c21=yarn,
            grid_shape=(1, 2, 2),
            unit="Pa",
        )

    def test_sparse_stiffness_materializes_matrix_and_yarns(self):
        field = self.make_stiffness()

        dense = field.to_dense_c21()

        self.assertEqual(dense.shape, (1, 2, 2, 21))
        np.testing.assert_array_equal(
            dense.reshape(-1, 21)[0], -np.ones(21)
        )
        np.testing.assert_array_equal(
            dense.reshape(-1, 21)[1], field.yarn_c21[0]
        )
        np.testing.assert_array_equal(
            dense.reshape(-1, 21)[3], field.yarn_c21[1]
        )

    def test_sparse_stiffness_materializes_voigt_and_acdm_layouts(self):
        field = self.make_stiffness()

        dense_voigt = field.to_dense_voigt()
        acdm = field.to_acdm(batch=True)
        no_batch = field.to_acdm(batch=False)

        self.assertEqual(dense_voigt.shape, (6, 6, 1, 2, 2))
        self.assertEqual(acdm.shape, (1, 6, 6, 1, 2, 2))
        self.assertEqual(no_batch.shape, (6, 6, 1, 2, 2))
        np.testing.assert_array_equal(acdm[0], dense_voigt)

    def test_sparse_stiffness_rejects_mismatched_arrays(self):
        with self.assertRaisesRegex(ValueError, "same leading length"):
            self.mf.SparseStiffnessField(
                matrix_c21=np.ones(21),
                voxel_indices=np.array([1, 3]),
                yarn_ids=np.array([0, 2]),
                material_ids=np.array([1]),
                yarn_c21=np.ones((2, 21)),
                grid_shape=(1, 2, 2),
            )


if __name__ == "__main__":
    unittest.main()
