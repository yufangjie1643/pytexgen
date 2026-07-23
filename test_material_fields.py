"""Tests for sparse orientation and stiffness field utilities."""

import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np

try:
    import torch
except ImportError:
    torch = None


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


class StiffnessRotationTest(unittest.TestCase):
    def setUp(self):
        self.mf = load_material_fields()
        self.assertTrue(
            hasattr(self.mf, "rotate_stiffness_c21"),
            "rotate_stiffness_c21 is not implemented",
        )
        self.assertTrue(
            hasattr(self.mf, "build_stiffness_field"),
            "build_stiffness_field is not implemented",
        )

    @staticmethod
    def symmetric_positive_definite_matrix():
        base = np.arange(1.0, 37.0).reshape(6, 6) / 36.0
        return base @ base.T + 2.0 * np.eye(6)

    @staticmethod
    def rotation_z_90():
        return np.array(
            [
                [0.0, -1.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0],
            ]
        )

    @staticmethod
    def rotate_with_explicit_fourth_order_tensor(matrix, rotation):
        pairs = ((0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1))
        tensor = np.zeros((3, 3, 3, 3), dtype=np.float64)
        for row, (i, j) in enumerate(pairs):
            for column, (k, ell) in enumerate(pairs):
                value = matrix[row, column]
                tensor[i, j, k, ell] = value
                tensor[j, i, k, ell] = value
                tensor[i, j, ell, k] = value
                tensor[j, i, ell, k] = value
        rotated = np.einsum(
            "iI,jJ,kK,lL,IJKL->ijkl",
            rotation,
            rotation,
            rotation,
            rotation,
            tensor,
        )
        result = np.empty((6, 6), dtype=np.float64)
        for row, (i, j) in enumerate(pairs):
            for column, (k, ell) in enumerate(pairs):
                result[row, column] = rotated[i, j, k, ell]
        return result

    def make_three_voxel_field(self, storage="numpy", device=None):
        kwargs = {
            "voxel_indices": np.arange(3, dtype=np.int64),
            "yarn_ids": np.array([0, 7, 0], dtype=np.int32),
            "orientation1": np.tile([1.0, 0.0, 0.0], (3, 1)),
            "orientation2": np.tile([0.0, 1.0, 0.0], (3, 1)),
            "grid_shape": (1, 1, 3),
        }
        field = self.mf.SparseOrientationField(**kwargs)
        return field.to(storage, device=device) if storage != "numpy" else field

    def test_identity_frame_preserves_general_c21(self):
        local = self.mf.pack_voigt_c21(
            self.symmetric_positive_definite_matrix()
        )

        result = self.mf.rotate_stiffness_c21(
            local[None, :],
            np.array([[1.0, 0.0, 0.0]]),
            np.array([[0.0, 1.0, 0.0]]),
            chunk_voxels=1,
        )

        np.testing.assert_allclose(result[0], local, rtol=1e-12, atol=1e-12)

    def test_ninety_degree_frame_matches_fourth_order_reference(self):
        local = self.mf.orthotropic_stiffness_c21(
            150.0, 10.0, 12.0,
            0.25, 0.20, 0.30,
            5.0, 6.0, 4.0,
        )

        actual = self.mf.rotate_stiffness_c21(
            local[None, :],
            np.array([[0.0, 1.0, 0.0]]),
            np.array([[-1.0, 0.0, 0.0]]),
        )
        expected = self.rotate_with_explicit_fourth_order_tensor(
            self.mf.unpack_c21(local), self.rotation_z_90()
        )

        np.testing.assert_allclose(
            self.mf.unpack_c21(actual[0]),
            expected,
            rtol=1e-10,
            atol=1e-10,
        )

    def test_rotation_is_chunk_size_invariant(self):
        local = np.broadcast_to(
            self.mf.pack_voigt_c21(
                self.symmetric_positive_definite_matrix()
            ),
            (5, 21),
        ).copy()
        orientation1 = np.tile([1.0, 0.0, 0.0], (5, 1))
        orientation2 = np.tile([0.0, 1.0, 0.0], (5, 1))

        chunked = self.mf.rotate_stiffness_c21(
            local, orientation1, orientation2, chunk_voxels=2
        )
        single = self.mf.rotate_stiffness_c21(
            local, orientation1, orientation2, chunk_voxels=100
        )

        np.testing.assert_allclose(chunked, single, rtol=0.0, atol=0.0)

    def test_rotation_rejects_collinear_frame(self):
        local = self.mf.isotropic_stiffness_c21(70.0, 0.25)
        with self.assertRaisesRegex(ValueError, "zero or collinear"):
            self.mf.rotate_stiffness_c21(
                local[None, :],
                np.array([[1.0, 0.0, 0.0]]),
                np.array([[2.0, 0.0, 0.0]]),
            )

    def test_builder_selects_default_and_per_yarn_materials(self):
        data = type(
            "VoxelData",
            (),
            {"sparse_orientation": self.make_three_voxel_field()},
        )()

        result = self.mf.build_stiffness_field(
            data,
            matrix_stiffness=np.ones(21),
            default_yarn_stiffness=np.full(21, 2.0),
            yarn_stiffness_by_id={7: np.full(21, 3.0)},
            chunk_voxels=2,
            unit="Pa",
        )

        self.assertEqual(result.yarn_c21.shape, (3, 21))
        self.assertEqual(result.material_ids.tolist(), [1, 2, 1])
        np.testing.assert_allclose(result.yarn_c21[0], 2.0)
        np.testing.assert_allclose(result.yarn_c21[1], 3.0)
        self.assertEqual(result.unit, "Pa")

    def test_builder_requires_material_for_every_yarn(self):
        data = type(
            "VoxelData",
            (),
            {"sparse_orientation": self.make_three_voxel_field()},
        )()
        with self.assertRaisesRegex(ValueError, "missing.*0"):
            self.mf.build_stiffness_field(
                data,
                matrix_stiffness=np.ones(21),
                yarn_stiffness_by_id={7: np.full(21, 3.0)},
            )

    def test_builder_optional_positive_definite_validation(self):
        data = type(
            "VoxelData",
            (),
            {"sparse_orientation": self.make_three_voxel_field()},
        )()
        with self.assertRaisesRegex(ValueError, "positive definite"):
            self.mf.build_stiffness_field(
                data,
                matrix_stiffness=-np.eye(6),
                default_yarn_stiffness=np.eye(6),
                validate_positive_definite=True,
            )

    @unittest.skipIf(torch is None, "torch is optional")
    def test_torch_cpu_matches_numpy_and_preserves_device(self):
        local = self.mf.pack_voigt_c21(
            self.symmetric_positive_definite_matrix()
        )
        orientation1 = np.array([[0.0, 1.0, 0.0]])
        orientation2 = np.array([[-1.0, 0.0, 0.0]])
        expected = self.mf.rotate_stiffness_c21(
            local[None, :], orientation1, orientation2
        )

        actual = self.mf.rotate_stiffness_c21(
            torch.as_tensor(local[None, :], dtype=torch.float64),
            torch.as_tensor(orientation1, dtype=torch.float64),
            torch.as_tensor(orientation2, dtype=torch.float64),
        )

        self.assertEqual(actual.device.type, "cpu")
        self.assertEqual(actual.dtype, torch.float64)
        np.testing.assert_allclose(
            actual.numpy(), expected, rtol=1e-10, atol=1e-10
        )

    @unittest.skipUnless(
        torch is not None and torch.cuda.is_available(),
        "CUDA is optional",
    )
    def test_cuda_builder_keeps_outputs_on_gpu(self):
        orientation = self.make_three_voxel_field(
            storage="torch", device="cuda"
        )
        data = type(
            "VoxelData",
            (),
            {"sparse_orientation": orientation},
        )()

        result = self.mf.build_stiffness_field(
            data,
            matrix_stiffness=np.ones(21),
            default_yarn_stiffness=np.full(21, 2.0),
        )

        self.assertEqual(result.device, "cuda:0")
        self.assertEqual(result.yarn_c21.device.type, "cuda")


if __name__ == "__main__":
    unittest.main()
