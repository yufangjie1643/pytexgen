"""Tests for the optional Voxel-ACDM adapter.

These tests cover conversion and discovery helpers without requiring Triton or
running the Voxel-ACDM CUDA solver.
"""

import importlib.util
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

try:
    import torch
except ImportError:
    torch = None


ROOT = Path(__file__).resolve().parent


def load_adapter_module():
    """Load TexGen.acdm_solver with the tiny Core stub used by voxelizer tests."""
    pkg = types.ModuleType("TexGen")
    pkg.__path__ = [str(ROOT / "TexGen")]
    sys.modules["TexGen"] = pkg

    core = types.ModuleType("TexGen.Core")

    class CYarn:
        LINE = 1
        SURFACE = 2
        VOLUME = 4

    class CTextile:
        pass

    core.CYarn = CYarn
    core.CTextile = CTextile
    sys.modules["TexGen.Core"] = core

    spec = importlib.util.spec_from_file_location(
        "TexGen.acdm_solver", ROOT / "TexGen" / "acdm_solver.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class ACDMSolverAdapterTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.adapter = load_adapter_module()
        cls.voxelizer = sys.modules["TexGen.gpu_voxelizer"]

    def make_data(self):
        yarn_id = np.array(
            [
                -1, 0,
                1, -1,
            ],
            dtype=np.int32,
        )
        return self.voxelizer.VoxelGridData(
            yarn_id=yarn_id,
            aabb=np.array([[1.0, 2.0, 3.0], [3.0, 6.0, 4.0]], dtype=np.float64),
            resolution=(2, 2, 1),
            backend="numpy",
            device="cpu",
            workers=1,
            dtype="float32",
            timings={},
        )

    def test_acdm_shape_and_voxel_size(self):
        data = self.make_data()

        self.assertEqual(self.adapter.acdm_grid_shape(data), (1, 2, 2))
        self.assertEqual(self.adapter.acdm_voxel_size(data), (1.0, 2.0, 1.0))

    def test_default_phase_mapping(self):
        data = self.make_data()
        phase = self.adapter.to_acdm_phase_ids(data)

        self.assertEqual(phase.shape, (1, 1, 2, 2))
        np.testing.assert_array_equal(
            phase[0],
            np.array([[[0, 1], [1, 0]]], dtype=np.uint8),
        )

    def test_custom_yarn_phase_mapping(self):
        data = self.make_data()
        phase = self.adapter.to_acdm_phase_ids(
            data,
            matrix_phase=0,
            yarn_phase_by_id={0: 2, 1: 3},
            batch=False,
        )

        np.testing.assert_array_equal(
            phase,
            np.array([[[0, 2], [3, 0]]], dtype=np.uint8),
        )

    def test_partial_override_keeps_default_yarn_phase(self):
        phase = self.adapter.to_acdm_phase_ids(
            self.make_data(),
            yarn_phase=5,
            yarn_phase_by_id={1: 9},
            batch=False,
        )

        np.testing.assert_array_equal(
            phase,
            np.array([[[0, 5], [9, 0]]], dtype=np.uint8),
        )

    def test_phase_mapping_rejects_out_of_range(self):
        data = self.make_data()
        with self.assertRaisesRegex(ValueError, "0..15"):
            self.adapter.to_acdm_phase_ids(data, yarn_phase=17)
        with self.assertRaisesRegex(ValueError, "0..15"):
            self.adapter.to_acdm_phase_ids(data, matrix_phase=-1)
        with self.assertRaisesRegex(ValueError, "0..15"):
            self.adapter.to_acdm_phase_ids(
                data,
                yarn_phase_by_id={1: -2},
            )

    def test_find_cloned_voxel_acdm_root(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "Voxel-ACDM"
            (root / "femlib").mkdir(parents=True)
            (root / "README.md").write_text("fake test checkout")

            found = self.adapter.find_voxel_acdm_root(str(root))

        self.assertEqual(found, root.resolve())

    @unittest.skipIf(torch is None, "PyTorch is not installed")
    def test_torch_phase_mapping_preserves_backend_device_and_values(self):
        numpy_data = self.make_data()
        torch_data = numpy_data.to("torch", device="cpu")

        phase = self.adapter.to_acdm_phase_ids(
            torch_data,
            yarn_phase=5,
            yarn_phase_by_id={1: 9},
        )

        self.assertIsInstance(phase, torch.Tensor)
        self.assertEqual(phase.device.type, "cpu")
        self.assertEqual(phase.dtype, torch.uint8)
        np.testing.assert_array_equal(
            phase.numpy(),
            np.array([[[[0, 5], [9, 0]]]], dtype=np.uint8),
        )

    @unittest.skipUnless(
        torch is not None and torch.cuda.is_available(),
        "CUDA is not available",
    )
    def test_cuda_phase_mapping_matches_numpy_bit_for_bit(self):
        numpy_data = self.make_data()
        cuda_data = numpy_data.to("torch", device="cuda")
        expected = self.adapter.to_acdm_phase_ids(
            numpy_data,
            yarn_phase=5,
            yarn_phase_by_id={1: 9},
        )

        actual = self.adapter.to_acdm_phase_ids(
            cuda_data,
            yarn_phase=5,
            yarn_phase_by_id={1: 9},
        )

        self.assertTrue(actual.is_cuda)
        np.testing.assert_array_equal(actual.cpu().numpy(), expected)

    def fake_acdm_modules(self):
        calls = {"constructed": 0}

        class FakeSolver:
            @classmethod
            def from_E_nu(
                cls,
                phase_ids,
                E_lut,
                nu_lut,
                voxel_size,
                grid_shape,
                **kwargs,
            ):
                calls["constructed"] += 1
                calls["phase_ids"] = phase_ids
                calls["E_lut"] = E_lut
                calls["nu_lut"] = nu_lut
                calls["voxel_size"] = voxel_size
                calls["grid_shape"] = grid_shape
                calls["kwargs"] = kwargs
                return cls()

            def compute_effective_stiffness(self, **kwargs):
                calls["solve_kwargs"] = kwargs
                return np.eye(6, dtype=np.float64)[None], [["ok"]]

        femlib = types.ModuleType("femlib")
        femlib.extract_engineering_constants = lambda _C: {}
        batched = types.ModuleType("femlib.fem_batched")
        batched.FEMHomogenizerBatchedIsotropicPhases = FakeSolver
        return calls, femlib, batched

    def test_explicit_phase_materials_build_complete_raw_phase_luts(self):
        calls, femlib, batched = self.fake_acdm_modules()
        phase_materials = {
            0: {"E": 3.0, "Nu": 0.35},
            5: {"E": 70.0, "Nu": 0.20},
            9: {"E": 140.0, "Nu": 0.25},
            12: {"E": 210.0, "Nu": 0.22},
        }

        with mock.patch.object(
            self.adapter,
            "import_voxel_acdm",
            return_value=femlib,
        ), mock.patch.dict(
            sys.modules,
            {"femlib": femlib, "femlib.fem_batched": batched},
        ):
            result = self.adapter.solve_acdm_isotropic_from_voxel_data(
                self.make_data(),
                phase_materials=phase_materials,
                matrix_phase=0,
                yarn_phase=5,
                yarn_phase_by_id={1: 9},
                precond="none",
            )

        self.assertEqual(calls["E_lut"].shape, (16,))
        self.assertEqual(calls["nu_lut"].shape, (16,))
        self.assertEqual(calls["E_lut"][0], 3.0)
        self.assertEqual(calls["E_lut"][5], 70.0)
        self.assertEqual(calls["E_lut"][9], 140.0)
        self.assertEqual(calls["E_lut"][12], 210.0)
        self.assertEqual(calls["nu_lut"][9], 0.25)
        self.assertEqual(calls["constructed"], 1)
        np.testing.assert_array_equal(
            result.phase_ids,
            np.array([[[[0, 5], [9, 0]]]], dtype=np.uint8),
        )

    def test_missing_used_phase_fails_before_solver_construction(self):
        calls, femlib, batched = self.fake_acdm_modules()

        with mock.patch.object(
            self.adapter,
            "import_voxel_acdm",
            return_value=femlib,
        ), mock.patch.dict(
            sys.modules,
            {"femlib": femlib, "femlib.fem_batched": batched},
        ):
            with self.assertRaisesRegex(ValueError, "used phase 9"):
                self.adapter.solve_acdm_isotropic_from_voxel_data(
                    self.make_data(),
                    phase_materials={
                        0: {"E": 3.0, "Nu": 0.35},
                        5: {"E": 70.0, "Nu": 0.20},
                    },
                    matrix_phase=0,
                    yarn_phase=5,
                    yarn_phase_by_id={1: 9},
                    precond="none",
                )

        self.assertEqual(calls["constructed"], 0)

    def test_legacy_materials_are_placed_at_explicit_phase_rows(self):
        calls, femlib, batched = self.fake_acdm_modules()

        with mock.patch.object(
            self.adapter,
            "import_voxel_acdm",
            return_value=femlib,
        ), mock.patch.dict(
            sys.modules,
            {"femlib": femlib, "femlib.fem_batched": batched},
        ):
            self.adapter.solve_acdm_isotropic_from_voxel_data(
                self.make_data(),
                matrix_material={"E": 3.0, "Nu": 0.35},
                yarn_material={"E": 70.0, "Nu": 0.20},
                matrix_phase=2,
                yarn_phase=5,
                precond="none",
            )

        self.assertEqual(calls["E_lut"][2], 3.0)
        self.assertEqual(calls["E_lut"][5], 70.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
