"""Tests for the optional Voxel-ACDM adapter.

These tests cover conversion and discovery helpers without requiring Triton or
running the Voxel-ACDM CUDA solver.
"""

import importlib.util
import sys
import types
import unittest
from pathlib import Path

import numpy as np


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

    def test_phase_mapping_rejects_out_of_range(self):
        data = self.make_data()
        with self.assertRaisesRegex(ValueError, "0..15"):
            self.adapter.to_acdm_phase_ids(data, yarn_phase=17)

    def test_find_cloned_voxel_acdm_root(self):
        root = self.adapter.find_voxel_acdm_root(str(ROOT.parent / "Voxel-ACDM"))
        self.assertEqual(root.name, "Voxel-ACDM")


if __name__ == "__main__":
    unittest.main(verbosity=2)
