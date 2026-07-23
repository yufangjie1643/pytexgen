"""Tests for sparse orientation and stiffness field utilities."""

import importlib.util
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


if __name__ == "__main__":
    unittest.main()
