"""Lightweight backend tests for TexGen's Python voxelizer.

These tests avoid building a real CTextile. They patch ``extract_snapshots`` to
return a synthetic straight yarn, then exercise the public voxelizer path for
numpy, adaptive numpy, and torch when torch is installed.
"""

import importlib.util
import sys
import tempfile
import types
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent


def load_voxelizer_module():
    """Load TexGen.gpu_voxelizer with a tiny Core stub."""
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
        "TexGen.gpu_voxelizer", ROOT / "TexGen" / "gpu_voxelizer.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def synthetic_snapshot(voxelizer, dtype=np.float32):
    """Straight yarn through the unit cube, square section around y/z center."""
    section = np.array(
        [
            [-0.24, -0.24],
            [0.24, -0.24],
            [0.24, 0.24],
            [-0.24, 0.24],
            [-0.24, -0.24],
        ],
        dtype=dtype,
    )
    return voxelizer.YarnSnapshot(
        positions=np.array([[0.0, 0.5, 0.5], [1.0, 0.5, 0.5]], dtype=dtype),
        tangents=np.array([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=dtype),
        ups=np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]], dtype=dtype),
        sides=np.array([[0.0, -1.0, 0.0], [0.0, -1.0, 0.0]], dtype=dtype),
        section=section,
        translations=np.zeros((1, 3), dtype=dtype),
    )


class FakeTextile:
    def GetName(self):
        return "SyntheticBackendTest"


class VoxelizerBackendTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.voxelizer = load_voxelizer_module()
        cls.aabb = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], dtype=np.float64)

    def patch_extract_snapshots(self):
        old_extract = self.voxelizer.extract_snapshots
        snap = synthetic_snapshot(self.voxelizer)

        def fake_extract(_textile):
            return [snap], self.aabb.copy()

        self.voxelizer.extract_snapshots = fake_extract
        self.addCleanup(lambda: setattr(self.voxelizer, "extract_snapshots", old_extract))

    def test_numpy_structured_public_path(self):
        self.patch_extract_snapshots()
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "numpy.inp"
            info = self.voxelizer.voxelize_textile(
                FakeTextile(),
                nx=4, ny=4, nz=4,
                out_inp=str(out),
                backend="numpy",
                workers=1,
                chunk_voxels=16,
                verbose=False,
            )

            self.assertEqual(info["backend"], "numpy")
            self.assertEqual(info["device"], "cpu")
            self.assertFalse(info["adaptive"])
            self.assertEqual(info["yarn_id"].shape, (64,))
            self.assertEqual(int((info["yarn_id"] >= 0).sum()), 16)
            self.assertIn("*Element, type=C3D8R", out.read_text())

    def test_aabb_pruning_matches_unpruned_numpy(self):
        self.patch_extract_snapshots()
        with tempfile.TemporaryDirectory() as tmp:
            pruned = self.voxelizer.voxelize_textile(
                FakeTextile(),
                nx=6, ny=6, nz=6,
                out_inp=str(Path(tmp) / "pruned.inp"),
                backend="numpy",
                workers=1,
                chunk_voxels=32,
                aabb_pruning=True,
                verbose=False,
            )
            unpruned = self.voxelizer.voxelize_textile(
                FakeTextile(),
                nx=6, ny=6, nz=6,
                out_inp=str(Path(tmp) / "unpruned.inp"),
                backend="numpy",
                workers=1,
                chunk_voxels=32,
                aabb_pruning=False,
                verbose=False,
            )

            np.testing.assert_array_equal(pruned["yarn_id"], unpruned["yarn_id"])

    def test_numpy_adaptive_public_path(self):
        self.patch_extract_snapshots()
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "adaptive.inp"
            info = self.voxelizer.voxelize_textile(
                FakeTextile(),
                nx=2, ny=2, nz=2,
                out_inp=str(out),
                backend="numpy",
                workers=1,
                chunk_voxels=64,
                adaptive=True,
                adaptive_levels=1,
                verbose=False,
            )

            self.assertTrue(info["adaptive"])
            self.assertGreater(info["num_cells"], 8)
            self.assertEqual(info["num_cells"], info["mesh"]["elements"])
            self.assertGreaterEqual(int(info["levels"].max()), 1)
            text = out.read_text()
            self.assertIn("Hanging-node constraints", text)
            self.assertIn("*Element, type=C3D8R", text)

    def test_torch_public_path_or_missing_error(self):
        self.patch_extract_snapshots()
        if self.voxelizer.torch is None:
            with tempfile.TemporaryDirectory() as tmp:
                with self.assertRaisesRegex(ImportError, "Torch backend requested"):
                    self.voxelizer.voxelize_textile(
                        FakeTextile(),
                        nx=4, ny=4, nz=4,
                        out_inp=str(Path(tmp) / "torch.inp"),
                        backend="torch",
                        verbose=False,
                    )
            return

        with tempfile.TemporaryDirectory() as tmp:
            numpy_info = self.voxelizer.voxelize_textile(
                FakeTextile(),
                nx=4, ny=4, nz=4,
                out_inp=str(Path(tmp) / "numpy.inp"),
                backend="numpy",
                workers=1,
                chunk_voxels=16,
                verbose=False,
            )
            torch_info = self.voxelizer.voxelize_textile(
                FakeTextile(),
                nx=4, ny=4, nz=4,
                out_inp=str(Path(tmp) / "torch.inp"),
                backend="torch",
                device="cpu",
                chunk_voxels=16,
                verbose=False,
            )

            self.assertEqual(torch_info["backend"], "torch")
            self.assertEqual(torch_info["device"], "cpu")
            np.testing.assert_array_equal(torch_info["yarn_id"], numpy_info["yarn_id"])

    def test_adaptive_rejects_torch(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, "adaptive=True currently supports only"):
                self.voxelizer.voxelize_textile(
                    FakeTextile(),
                    nx=2, ny=2, nz=2,
                    out_inp=str(Path(tmp) / "adaptive_torch.inp"),
                    backend="torch",
                    adaptive=True,
                    verbose=False,
                )


if __name__ == "__main__":
    unittest.main(verbosity=2)
