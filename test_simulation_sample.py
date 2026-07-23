"""Tests for the framework-neutral simulation sample contract."""

import importlib.util
import sys
import types
import unittest
from pathlib import Path

import numpy as np

try:
    import torch
except ImportError:
    torch = None


ROOT = Path(__file__).resolve().parent


def _load_source_module(name):
    path = ROOT / "TexGen" / f"{name.rsplit('.', 1)[-1]}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_simulation_sample():
    """Load source modules without requiring an installed wheel."""
    package = types.ModuleType("TexGen")
    package.__path__ = [str(ROOT / "TexGen")]
    sys.modules["TexGen"] = package

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

    material_fields = _load_source_module("TexGen.material_fields")
    _load_source_module("TexGen.gpu_voxelizer")
    simulation_sample = _load_source_module("TexGen.simulation_sample")
    return simulation_sample, material_fields


class MaterialTableTest(unittest.TestCase):
    def setUp(self):
        self.sample_module, self.material_fields = load_simulation_sample()

    def make_table(self):
        c21 = np.stack(
            (
                self.material_fields.isotropic_stiffness_c21(3.0, 0.30),
                self.material_fields.isotropic_stiffness_c21(70.0, 0.20),
            )
        )
        ids = np.array([0, 7], dtype=np.int32)
        table = self.sample_module.MaterialTable(
            c21=c21,
            material_ids=ids,
            unit="GPa",
            names=("matrix", "carbon"),
        )
        return table, c21, ids

    def test_accepts_explicit_non_dense_ids_without_copy(self):
        table, c21, ids = self.make_table()

        self.assertIs(table.c21, c21)
        self.assertIs(table.material_ids, ids)
        self.assertEqual(table.row_for_id(7), 1)
        self.assertTrue(np.shares_memory(table.c21_for_id(0), c21))
        self.assertEqual(table.names, ("matrix", "carbon"))
        self.assertEqual(table.storage, "numpy")
        self.assertEqual(table.device, "cpu")

    def test_rejects_invalid_identifiers_shapes_and_units(self):
        c21 = np.ones((2, 21), dtype=np.float64)
        MaterialTable = self.sample_module.MaterialTable

        with self.assertRaisesRegex(ValueError, "material ID 0"):
            MaterialTable(c21, np.array([1, 2]), "GPa")
        with self.assertRaisesRegex(ValueError, "unique"):
            MaterialTable(c21, np.array([0, 0]), "GPa")
        with self.assertRaisesRegex(ValueError, "non-negative"):
            MaterialTable(c21, np.array([0, -1]), "GPa")
        with self.assertRaisesRegex(ValueError, r"\(M, 21\)"):
            MaterialTable(np.ones((2, 20)), np.array([0, 1]), "GPa")
        with self.assertRaisesRegex(ValueError, "one-dimensional integer"):
            MaterialTable(c21, np.array([0.0, 1.0]), "GPa")
        with self.assertRaisesRegex(ValueError, "unit"):
            MaterialTable(c21, np.array([0, 1]), " ")
        with self.assertRaisesRegex(ValueError, "names"):
            MaterialTable(c21, np.array([0, 1]), "GPa", names=("matrix",))

    def test_rejects_nonfinite_and_unknown_material(self):
        table, c21, ids = self.make_table()
        c21 = c21.copy()
        c21[1, 3] = np.nan

        with self.assertRaisesRegex(ValueError, "finite"):
            self.sample_module.MaterialTable(c21, ids, "GPa")
        with self.assertRaisesRegex(KeyError, "unknown material ID 99"):
            table.row_for_id(99)

    def test_positive_definite_validation_is_opt_in(self):
        c21 = np.zeros((1, 21), dtype=np.float64)
        ids = np.array([0], dtype=np.int32)

        self.sample_module.MaterialTable(c21, ids, "Pa")
        with self.assertRaisesRegex(ValueError, "positive definite"):
            self.sample_module.MaterialTable(
                c21,
                ids,
                "Pa",
                validate_positive_definite=True,
            )

    @unittest.skipIf(torch is None, "PyTorch is not installed")
    def test_conversion_preserves_ids_and_copy_semantics(self):
        table, c21, ids = self.make_table()

        same = table.to(copy=False)
        copied = table.to(copy=True)
        torch_table = table.to("torch", dtype=torch.float32)

        self.assertIs(same.c21, c21)
        self.assertIs(same.material_ids, ids)
        self.assertIsNot(copied.c21, c21)
        self.assertIsNot(copied.material_ids, ids)
        np.testing.assert_array_equal(copied.material_ids, ids)
        self.assertEqual(torch_table.c21.dtype, torch.float32)
        self.assertEqual(torch_table.material_ids.dtype, torch.int32)
        self.assertEqual(torch_table.device, "cpu")
        np.testing.assert_array_equal(torch_table.material_ids.numpy(), ids)

        same_torch = torch_table.to("torch", copy=False)
        copied_torch = torch_table.to("torch", copy=True)
        self.assertIs(same_torch.c21, torch_table.c21)
        self.assertEqual(
            same_torch.material_ids.data_ptr(),
            torch_table.material_ids.data_ptr(),
        )
        self.assertNotEqual(
            copied_torch.c21.data_ptr(),
            torch_table.c21.data_ptr(),
        )

    @unittest.skipIf(torch is None, "PyTorch is not installed")
    def test_rejects_mixed_backends_and_numpy_cuda_target(self):
        c21 = np.ones((2, 21), dtype=np.float64)

        with self.assertRaisesRegex(ValueError, "same storage backend"):
            self.sample_module.MaterialTable(
                c21,
                torch.tensor([0, 1], dtype=torch.int32),
                "GPa",
            )

        table, _, _ = self.make_table()
        with self.assertRaisesRegex(ValueError, "NumPy.*CPU"):
            table.to("numpy", device="cuda")


if __name__ == "__main__":
    unittest.main(verbosity=2)
