"""Tests for consolidated simulation sample persistence."""

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

try:
    import torch
except ImportError:
    torch = None

import test_simulation_sample as sample_tests


ROOT = Path(__file__).resolve().parent


def load_simulation_io():
    """Load the persistence module after the sample source fixture."""
    path = ROOT / "TexGen" / "simulation_io.py"
    spec = importlib.util.spec_from_file_location(
        "TexGen.simulation_io",
        path,
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class SimulationSamplePersistenceTest(unittest.TestCase):
    def setUp(self):
        fixture = sample_tests.SimulationSampleValidationTest(
            "test_construction_is_zero_copy_and_adopts_voxel_orientation"
        )
        fixture.setUp()
        self.sample = fixture.make_sample()
        self.io = load_simulation_io()

    def assert_sample_equal(self, actual):
        self.assertEqual(actual.voxels.resolution, self.sample.voxels.resolution)
        self.assertEqual(actual.voxels.order, self.sample.voxels.order)
        self.assertEqual(actual.materials.unit, self.sample.materials.unit)
        self.assertEqual(actual.materials.names, self.sample.materials.names)
        self.assertEqual(actual.metadata, self.sample.metadata)
        for name, expected in self.sample.as_dict(copy=False).items():
            with self.subTest(field=name):
                np.testing.assert_array_equal(
                    actual.array(name),
                    expected,
                )

    def test_directory_roundtrip_uses_one_mapped_sparse_topology(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sample"
            self.io.save_simulation_sample(path, self.sample)

            manifest = json.loads((path / "manifest.json").read_text())
            loaded = self.io.load_simulation_sample(path, mmap_mode="r")

            self.assertEqual(
                manifest["schema"],
                "pytexgen.simulation_sample",
            )
            self.assertEqual(manifest["version"], 1)
            self.assertEqual(
                manifest["fields"]["orientation.voxel_indices"],
                manifest["fields"]["stiffness.voxel_indices"],
            )
            self.assertEqual(
                manifest["fields"]["orientation.yarn_ids"],
                manifest["fields"]["stiffness.yarn_ids"],
            )
            self.assertEqual(
                len(list((path / "arrays").glob("*.npy"))),
                len(manifest["arrays"]),
            )
            self.assertIsInstance(loaded.voxels.yarn_id, np.memmap)
            self.assertIsInstance(
                loaded.orientation.orientation1,
                np.memmap,
            )
            self.assertIsInstance(loaded.stiffness.yarn_c21, np.memmap)
            self.assertIs(
                loaded.orientation.voxel_indices,
                loaded.stiffness.voxel_indices,
            )
            self.assertIs(
                loaded.orientation.yarn_ids,
                loaded.stiffness.yarn_ids,
            )
            self.assert_sample_equal(loaded)

    @unittest.skipIf(torch is None, "PyTorch is not installed")
    def test_npz_roundtrip_supports_explicit_torch_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sample.npz"
            self.io.save_simulation_sample(path, self.sample, compressed=True)

            with np.load(path, allow_pickle=False) as archive:
                members = set(archive.files)
                self.assertIn("_manifest_json", members)
                self.assertIn("sparse_voxel_indices", members)
                self.assertNotIn("stiffness_voxel_indices", members)
            loaded = self.io.load_simulation_sample(
                path,
                output="torch",
                device="cpu",
            )

            self.assertEqual(loaded.storage, "torch")
            self.assertEqual(loaded.device, "cpu")
            self.assertIs(
                loaded.orientation.voxel_indices,
                loaded.stiffness.voxel_indices,
            )
            np.testing.assert_array_equal(
                loaded.materials.material_ids.numpy(),
                self.sample.materials.material_ids,
            )

    def test_npz_rejects_memory_mapping(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sample.npz"
            self.io.save_simulation_sample(path, self.sample)

            with self.assertRaisesRegex(ValueError, "mmap_mode"):
                self.io.load_simulation_sample(path, mmap_mode="r")

    def test_loader_rejects_schema_dtype_shape_and_missing_array(self):
        mutations = (
            (
                "schema",
                lambda manifest: manifest.update({"version": 99}),
                "version",
            ),
            (
                "dtype",
                lambda manifest: manifest["arrays"]["material_c21"].update(
                    {"dtype": "float32"}
                ),
                "dtype",
            ),
            (
                "shape",
                lambda manifest: manifest["arrays"]["material_c21"].update(
                    {"shape": [99, 21]}
                ),
                "shape",
            ),
            (
                "missing",
                lambda manifest: (
                    manifest["arrays"]["material_c21"].update(
                        {"location": "arrays/missing.npy"}
                    )
                ),
                "missing",
            ),
        )
        for case, mutate, message in mutations:
            with self.subTest(case=case), tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / "sample"
                self.io.save_simulation_sample(path, self.sample)
                manifest_path = path / "manifest.json"
                manifest = json.loads(manifest_path.read_text())
                mutate(manifest)
                manifest_path.write_text(json.dumps(manifest))

                with self.assertRaisesRegex(ValueError, message):
                    self.io.load_simulation_sample(path)

    def test_save_refuses_to_overwrite_existing_target(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sample"
            path.mkdir()

            with self.assertRaisesRegex(FileExistsError, "exists"):
                self.io.save_simulation_sample(path, self.sample)


if __name__ == "__main__":
    unittest.main(verbosity=2)
