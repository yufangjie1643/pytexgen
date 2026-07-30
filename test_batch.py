"""Regression tests for prepared geometry and streaming file batches."""

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent


def load_batch_module():
    """Load the source batch module against the installed pytexgen package."""
    import pytexgen

    name = "pytexgen.batch_under_test"
    spec = importlib.util.spec_from_file_location(
        name,
        ROOT / "TexGen" / "batch.py",
    )
    module = importlib.util.module_from_spec(spec)
    module.__package__ = "pytexgen"
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def synthetic_bundle(batch, dtype=np.float32):
    """Straight yarn crossing a unit cube."""
    voxelizer = sys.modules["pytexgen.gpu_voxelizer"]
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
    snapshot = voxelizer.YarnSnapshot(
        positions=np.array(
            [[0.0, 0.5, 0.5], [1.0, 0.5, 0.5]],
            dtype=dtype,
        ),
        tangents=np.array(
            [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            dtype=dtype,
        ),
        ups=np.array(
            [[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]],
            dtype=dtype,
        ),
        sides=np.array(
            [[0.0, -1.0, 0.0], [0.0, -1.0, 0.0]],
            dtype=dtype,
        ),
        section=section,
        translations=np.zeros((1, 3), dtype=dtype),
    )
    return voxelizer.SnapshotBundle.from_snapshots(
        [snapshot],
        np.array(
            [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]],
            dtype=np.float64,
        ),
    )


class PreparedGeometryTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.batch = load_batch_module()

    def test_ptgb_checksum_and_mmap_round_trip(self):
        bundle = synthetic_bundle(self.batch)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sample.ptgb"
            self.batch.save_prepared_geometry(
                bundle,
                path,
                metadata={"case": "round-trip"},
            )

            loaded = self.batch.load_prepared_geometry(
                path,
                verify="checksum",
            )

            for name in self.batch._PTGB_ARRAY_NAMES:
                np.testing.assert_array_equal(
                    getattr(loaded, name),
                    getattr(bundle, name),
                )
            self.assertIsInstance(loaded.positions.base, np.memmap)
            self.assertFalse(loaded.positions.flags.writeable)

            torch_ready = self.batch._load_geometry_source(
                path,
                verify_ptgb="header",
            )
            self.assertIsInstance(torch_ready.positions.base, np.memmap)
            self.assertTrue(torch_ready.positions.flags.writeable)

    def test_ptgb_checksum_detects_payload_corruption(self):
        bundle = synthetic_bundle(self.batch)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "corrupt.ptgb"
            self.batch.save_prepared_geometry(bundle, path)
            with path.open("r+b") as stream:
                stream.seek(-1, 2)
                value = stream.read(1)
                stream.seek(-1, 2)
                stream.write(bytes([value[0] ^ 0xFF]))

            self.batch.load_prepared_geometry(path, verify="header")
            with self.assertRaisesRegex(
                self.batch.PTGBFormatError,
                "checksum mismatch",
            ):
                self.batch.load_prepared_geometry(
                    path,
                    verify="checksum",
                )

    def test_cpu_batch_writes_expected_dense_arrays(self):
        bundle = synthetic_bundle(self.batch)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "sample.ptgb"
            output = root / "outputs"
            self.batch.save_prepared_geometry(bundle, source)

            report = self.batch.voxelize_files_batch(
                [source],
                resolution=(8, 8, 8),
                output_dir=output,
                fields=("yarn_id", "material_id", "orientation"),
                device="cpu",
                dtype="float32",
                chunk_voxels=64,
            )

            self.assertEqual(report.succeeded, 1)
            self.assertEqual(report.failed, 0)
            case = output / "sample"
            yarn_id = np.load(case / "yarn_id.npy", mmap_mode="r")
            material_id = np.load(case / "material_id.npy", mmap_mode="r")
            orientation = np.load(case / "orientation.npy", mmap_mode="r")
            self.assertEqual(yarn_id.shape, (8, 8, 8))
            self.assertEqual(yarn_id.dtype, np.int32)
            self.assertEqual(material_id.shape, (8, 8, 8))
            self.assertEqual(material_id.dtype, np.int32)
            self.assertEqual(orientation.shape, (8, 8, 8, 3, 3))
            self.assertEqual(orientation.dtype, np.float32)
            self.assertEqual(int((yarn_id >= 0).sum()), 128)
            np.testing.assert_array_equal(material_id, yarn_id + 1)
            metadata = json.loads(
                (case / "metadata.json").read_text(encoding="utf-8")
            )
            self.assertEqual(metadata["resolution_xyz"], [8, 8, 8])
            self.assertEqual(metadata["grid_shape_zyx"], [8, 8, 8])

            with self.assertRaisesRegex(MemoryError, "81920 bytes"):
                self.batch.voxelize_files_batch(
                    [source],
                    resolution=(8, 8, 8),
                    output_dir=root / "over-budget",
                    fields=("material_id", "orientation"),
                    device="cpu",
                    dtype="float32",
                    batch_size=3,
                    memory_budget_bytes=50_000,
                )

    def test_cpu_batch_writes_stiffness_c21(self):
        from pytexgen.material_fields import isotropic_stiffness_c21

        bundle = synthetic_bundle(self.batch)
        materials = self.batch.MaterialSpec(
            matrix_c21=isotropic_stiffness_c21(3.0, 0.3),
            default_yarn_c21=isotropic_stiffness_c21(30.0, 0.25),
            unit="GPa",
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "material.ptgb"
            output = root / "outputs"
            self.batch.save_prepared_geometry(bundle, source)

            report = self.batch.voxelize_files_batch(
                [source],
                resolution=(8, 8, 8),
                output_dir=output,
                fields=("material_id", "stiffness_c21"),
                materials=materials,
                device="cpu",
                dtype="float64",
                chunk_voxels=64,
            )

            self.assertEqual(report.succeeded, 1)
            material_id = np.load(output / "material" / "material_id.npy")
            stiffness = np.load(output / "material" / "stiffness_c21.npy")
            self.assertEqual(stiffness.shape, (8, 8, 8, 21))
            self.assertEqual(stiffness.dtype, np.float64)
            self.assertEqual(set(np.unique(material_id)), {0, 1})

    def test_prepare_geometry_converts_single_textile_tg3(self):
        import pytexgen as tg

        tg.DeleteTextiles()
        self.addCleanup(tg.DeleteTextiles)
        textile = tg.CTextileWeave2D(2, 2, 1.0, 0.2, True)
        textile.SetYarnWidths(0.8)
        textile.SetYarnHeights(0.1)
        textile.AssignDefaultDomain()
        name = tg.AddTextile(textile)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "plain.tg3"
            prepared = root / "plain.ptgb"
            tg.SaveToXML(str(source), name, tg.OUTPUT_STANDARD)

            with self.assertRaisesRegex(RuntimeError, "global registry"):
                self.batch.prepare_geometry(source, prepared)

            tg.DeleteTextile(name)
            resident = tg.CTextileWeave2D(1, 1, 1.0, 0.2, True)
            resident_name = tg.AddTextile(resident)
            result = self.batch.prepare_geometry(source, prepared)
            loaded = self.batch.load_prepared_geometry(
                result,
                verify="checksum",
            )

            self.assertEqual(result, prepared)
            self.assertEqual(loaded.num_yarns, 4)
            self.assertEqual(loaded.positions.shape[1], 3)
            self.assertEqual(loaded.sections.shape[1], 2)
            self.assertEqual(set(tg.GetTextiles()), {resident_name})


if __name__ == "__main__":
    unittest.main()
