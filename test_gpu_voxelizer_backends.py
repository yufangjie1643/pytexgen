"""Lightweight backend tests for TexGen's Python voxelizer.

These tests avoid building a real CTextile. They patch ``extract_snapshots`` to
return a synthetic straight yarn, then exercise the public voxelizer path for
numpy, adaptive numpy, and torch when torch is installed.
"""

import importlib.util
import inspect
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

    def assert_orientation_storage_api(self):
        self.assertIn(
            "orientation_storage",
            inspect.signature(self.voxelizer.voxelize_textile_data).parameters,
            "voxelize_textile_data does not expose orientation_storage",
        )

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

    def test_progress_callable_wraps_public_path_chunks(self):
        self.patch_extract_snapshots()
        calls = []

        def progress(iterable, total=None, desc=None, unit=None):
            calls.append((total, desc, unit))
            for item in iterable:
                yield item

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "numpy_progress.inp"
            info = self.voxelizer.voxelize_textile(
                FakeTextile(),
                nx=4, ny=4, nz=4,
                out_inp=str(out),
                backend="numpy",
                workers=1,
                chunk_voxels=16,
                verbose=False,
                progress=progress,
            )

        self.assertEqual(info["backend"], "numpy")
        self.assertIn("classify numpy voxels", [call[1] for call in calls])
        self.assertIn("write nodes", [call[1] for call in calls])
        self.assertIn("write elements", [call[1] for call in calls])

    def test_numpy_direct_data_public_path(self):
        self.patch_extract_snapshots()
        data = self.voxelizer.voxelize_textile_data(
            FakeTextile(),
            nx=4, ny=4, nz=4,
            backend="numpy",
            output="numpy",
            include_centers=True,
            workers=1,
            chunk_voxels=16,
            verbose=False,
        )

        self.assertIsInstance(data, self.voxelizer.VoxelGridData)
        self.assertEqual(data.storage, "numpy")
        self.assertEqual(data.yarn_id.shape, (64,))
        self.assertEqual(data.grid.shape, (4, 4, 4))
        self.assertEqual(data.centers.shape, (64, 3))
        self.assertEqual(int(data.occupancy().sum()), 16)
        np.testing.assert_allclose(data.voxel_size, [0.25, 0.25, 0.25])
        materials = data.material_id()
        self.assertEqual(materials.shape, (4, 4, 4))
        self.assertEqual(int(materials.max()), 1)
        self.assertEqual(int(materials.min()), 0)

    def test_numpy_textile_data_uses_flat_provider_bundle(self):
        snap = synthetic_snapshot(self.voxelizer)
        bundle = self.voxelizer.SnapshotBundle.from_snapshots([snap], self.aabb.copy())
        old_extract_bundle = self.voxelizer.extract_snapshot_bundle
        self.voxelizer.extract_snapshot_bundle = lambda _textile: bundle
        self.addCleanup(
            lambda: setattr(self.voxelizer, "extract_snapshot_bundle", old_extract_bundle)
        )

        def fail_to_snapshots():
            raise AssertionError("voxelize_textile_data should keep provider bundle flat")

        bundle.to_snapshots = fail_to_snapshots
        data = self.voxelizer.voxelize_textile_data(
            FakeTextile(),
            nx=4, ny=4, nz=4,
            backend="numpy",
            output="numpy",
            workers=1,
            chunk_voxels=16,
            verbose=False,
            include_orientations=True,
        )

        self.assertEqual(int(data.occupancy().sum()), 16)
        self.assertEqual(data.timings["unpack"], 0.0)
        self.assertEqual(data.orientation1.shape, (4, 4, 4, 3))
        self.assertEqual(data.orientation2.shape, (4, 4, 4, 3))
        yarn_mask = data.occupancy()
        np.testing.assert_allclose(
            data.orientation1[yarn_mask],
            np.broadcast_to([1.0, 0.0, 0.0], data.orientation1[yarn_mask].shape),
        )
        np.testing.assert_allclose(
            data.orientation2[yarn_mask],
            np.broadcast_to([0.0, 0.0, 1.0], data.orientation2[yarn_mask].shape),
        )
        np.testing.assert_allclose(
            data.orientation1[~yarn_mask],
            np.zeros_like(data.orientation1[~yarn_mask]),
        )
        np.testing.assert_allclose(
            data.orientation2[~yarn_mask],
            np.zeros_like(data.orientation2[~yarn_mask]),
        )

    def test_default_backend_is_numpy_and_reports_effective_workers(self):
        self.patch_extract_snapshots()
        data = self.voxelizer.voxelize_textile_data(
            FakeTextile(),
            nx=4, ny=4, nz=4,
            output="numpy",
            workers=4,
            chunk_voxels=8,
            verbose=False,
        )

        self.assertEqual(data.backend, "numpy")
        self.assertEqual(data.storage, "numpy")
        self.assertEqual(data.workers, 4)

        clamped = self.voxelizer.voxelize_textile_data(
            FakeTextile(),
            nx=4, ny=4, nz=4,
            output="numpy",
            workers=4,
            chunk_voxels=1024,
            verbose=False,
        )
        self.assertEqual(clamped.workers, 1)

    def test_numpy_sparse_orientation_matches_dense_yarn_entries(self):
        self.assert_orientation_storage_api()
        self.patch_extract_snapshots()
        dense = self.voxelizer.voxelize_textile_data(
            FakeTextile(),
            nx=4, ny=4, nz=4,
            backend="numpy",
            output="numpy",
            include_orientations=True,
            orientation_storage="dense",
            workers=1,
            chunk_voxels=16,
            verbose=False,
        )
        sparse = self.voxelizer.voxelize_textile_data(
            FakeTextile(),
            nx=4, ny=4, nz=4,
            backend="numpy",
            output="numpy",
            include_orientations=True,
            orientation_storage="sparse",
            workers=1,
            chunk_voxels=16,
            verbose=False,
        )

        indices = np.flatnonzero(dense.yarn_id >= 0)
        self.assertIsNone(sparse.orientation1)
        self.assertIsNone(sparse.orientation2)
        self.assertIsNotNone(sparse.sparse_orientation)
        np.testing.assert_array_equal(
            sparse.sparse_orientation.voxel_indices, indices
        )
        np.testing.assert_array_equal(
            sparse.sparse_orientation.yarn_ids, dense.yarn_id[indices]
        )
        np.testing.assert_allclose(
            sparse.sparse_orientation.orientation1,
            dense.orientation1.reshape(-1, 3)[indices],
        )
        np.testing.assert_allclose(
            sparse.sparse_orientation.orientation2,
            dense.orientation2.reshape(-1, 3)[indices],
        )

    def test_orientation_storage_rejects_unknown_value(self):
        self.assert_orientation_storage_api()
        self.patch_extract_snapshots()
        with self.assertRaisesRegex(ValueError, "orientation_storage"):
            self.voxelizer.voxelize_textile_data(
                FakeTextile(),
                nx=2, ny=2, nz=2,
                orientation_storage="coo",
                verbose=False,
            )

    def test_torch_dense_orientation_is_computed_without_numpy_fallback(self):
        self.assert_orientation_storage_api()
        if self.voxelizer.torch is None:
            self.skipTest("torch is optional")
        self.patch_extract_snapshots()

        data = self.voxelizer.voxelize_textile_data(
            FakeTextile(),
            nx=4, ny=4, nz=4,
            backend="torch",
            device="cpu",
            output="backend",
            include_orientations=True,
            orientation_storage="dense",
            chunk_voxels=16,
            verbose=False,
        )

        self.assertTrue(self.voxelizer._is_torch_tensor(data.orientation1))
        self.assertTrue(self.voxelizer._is_torch_tensor(data.orientation2))
        self.assertEqual(tuple(data.orientation1.shape), (4, 4, 4, 3))
        yarn_mask = data.occupancy()
        expected1 = self.voxelizer.torch.tensor(
            [1.0, 0.0, 0.0], dtype=data.orientation1.dtype
        )
        expected2 = self.voxelizer.torch.tensor(
            [0.0, 0.0, 1.0], dtype=data.orientation2.dtype
        )
        self.voxelizer.torch.testing.assert_close(
            data.orientation1[yarn_mask],
            expected1.expand(int(yarn_mask.sum().item()), 3),
        )
        self.voxelizer.torch.testing.assert_close(
            data.orientation2[yarn_mask],
            expected2.expand(int(yarn_mask.sum().item()), 3),
        )

    def test_torch_sparse_orientation_stays_on_selected_device(self):
        self.assert_orientation_storage_api()
        if self.voxelizer.torch is None:
            self.skipTest("torch is optional")
        self.patch_extract_snapshots()

        data = self.voxelizer.voxelize_textile_data(
            FakeTextile(),
            nx=4, ny=4, nz=4,
            backend="torch",
            device="cpu",
            output="backend",
            include_orientations=True,
            orientation_storage="sparse",
            chunk_voxels=16,
            verbose=False,
        )

        field = data.sparse_orientation
        self.assertIsNone(data.orientation1)
        self.assertIsNone(data.orientation2)
        self.assertEqual(field.storage, "torch")
        self.assertEqual(field.device, "cpu")
        self.assertEqual(field.num_yarn_voxels, 16)
        self.assertEqual(field.orientation1.device.type, "cpu")
        self.assertEqual(field.orientation2.device.type, "cpu")
        self.voxelizer.torch.testing.assert_close(
            field.yarn_ids, data.yarn_id[field.voxel_indices]
        )

    def test_cuda_sparse_orientation_remains_on_gpu(self):
        self.assert_orientation_storage_api()
        if (
            self.voxelizer.torch is None
            or not self.voxelizer.torch.cuda.is_available()
        ):
            self.skipTest("CUDA is optional")
        self.patch_extract_snapshots()

        data = self.voxelizer.voxelize_textile_data(
            FakeTextile(),
            nx=4, ny=4, nz=4,
            backend="torch",
            device="cuda",
            output="backend",
            include_orientations=True,
            orientation_storage="sparse",
            chunk_voxels=16,
            verbose=False,
        )

        self.assertEqual(data.yarn_id.device.type, "cuda")
        self.assertEqual(data.sparse_orientation.orientation1.device.type, "cuda")
        self.assertEqual(data.sparse_orientation.voxel_indices.device.type, "cuda")

    def test_voxel_grid_data_to_matches_torch_style_conversion(self):
        self.patch_extract_snapshots()
        data = self.voxelizer.voxelize_textile_data(
            FakeTextile(),
            nx=4, ny=4, nz=4,
            backend="numpy",
            output="numpy",
            include_centers=True,
            workers=1,
            chunk_voxels=16,
            verbose=False,
        )

        numpy_data = data.to("numpy", dtype=np.float64)
        self.assertEqual(numpy_data.storage, "numpy")
        self.assertEqual(numpy_data.dtype, "float64")
        self.assertEqual(numpy_data.aabb.dtype, np.float64)
        self.assertEqual(numpy_data.centers.dtype, np.float64)
        self.assertEqual(numpy_data.yarn_id.dtype, np.int32)
        np.testing.assert_array_equal(numpy_data.yarn_id, data.yarn_id)
        with self.assertRaisesRegex(ValueError, "floating"):
            data.to("numpy", dtype=np.int32)

        if self.voxelizer.torch is None:
            with self.assertRaisesRegex(ImportError, "Torch backend requested"):
                data.to("torch")
        else:
            torch_data = data.to("torch", device="cpu", dtype="float64")
            self.assertEqual(torch_data.storage, "torch")
            self.assertEqual(torch_data.dtype, "float64")
            self.assertTrue(self.voxelizer._is_torch_tensor(torch_data.yarn_id))
            self.assertEqual(torch_data.aabb.dtype, self.voxelizer.torch.float64)
            self.assertEqual(torch_data.centers.dtype, self.voxelizer.torch.float64)
            numpy_from_torch_dtype = data.to("numpy", dtype=self.voxelizer.torch.float32)
            self.assertEqual(numpy_from_torch_dtype.aabb.dtype, np.float32)
            with self.assertRaisesRegex(ValueError, "floating"):
                data.to("torch", dtype=self.voxelizer.torch.int32)
            roundtrip = torch_data.to("numpy")
            np.testing.assert_array_equal(roundtrip.yarn_id, data.yarn_id)

    def test_voxel_grid_data_npz_roundtrip(self):
        self.patch_extract_snapshots()
        data = self.voxelizer.voxelize_textile_data(
            FakeTextile(),
            nx=4, ny=4, nz=4,
            backend="numpy",
            output="numpy",
            include_centers=True,
            include_orientations=True,
            workers=1,
            chunk_voxels=16,
            verbose=False,
        )

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "voxels.npz"
            data.save_npz(str(path))
            loaded = self.voxelizer.VoxelGridData.load_npz(str(path))

        self.assertEqual(loaded.storage, "numpy")
        self.assertEqual(loaded.resolution, (4, 4, 4))
        self.assertEqual(loaded.centers.shape, (64, 3))
        np.testing.assert_array_equal(loaded.yarn_id, data.yarn_id)
        np.testing.assert_allclose(loaded.aabb, data.aabb)
        np.testing.assert_array_equal(loaded.material_id(), data.material_id())
        np.testing.assert_allclose(loaded.orientation1, data.orientation1)
        np.testing.assert_allclose(loaded.orientation2, data.orientation2)

    def test_voxel_grid_data_npy_dir_roundtrip_and_mmap(self):
        self.patch_extract_snapshots()
        data = self.voxelizer.voxelize_textile_data(
            FakeTextile(),
            nx=4, ny=4, nz=4,
            backend="numpy",
            output="numpy",
            include_centers=True,
            include_orientations=True,
            workers=1,
            chunk_voxels=16,
            verbose=False,
        )

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "voxel_npy"
            data.save_npy_dir(str(path))

            expected = {
                "metadata.json",
                "yarn_id.npy",
                "aabb.npy",
                "centers.npy",
                "orientation1.npy",
                "orientation2.npy",
            }
            self.assertTrue(expected.issubset({item.name for item in path.iterdir()}))

            loaded = self.voxelizer.VoxelGridData.load_npy_dir(str(path))
            mmap_loaded = self.voxelizer.VoxelGridData.load_npy_dir(
                str(path), mmap_mode="r"
            )
            self.assertIsInstance(mmap_loaded.yarn_id, np.memmap)
            self.assertIsInstance(mmap_loaded.orientation1, np.memmap)
            np.testing.assert_array_equal(mmap_loaded.material_id(), data.material_id())
            del mmap_loaded

        self.assertEqual(loaded.storage, "numpy")
        self.assertEqual(loaded.resolution, (4, 4, 4))
        np.testing.assert_array_equal(loaded.yarn_id, data.yarn_id)
        np.testing.assert_allclose(loaded.aabb, data.aabb)
        np.testing.assert_allclose(loaded.centers, data.centers)
        np.testing.assert_allclose(loaded.orientation1, data.orientation1)
        np.testing.assert_allclose(loaded.orientation2, data.orientation2)

        if self.voxelizer.torch is not None:
            with tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / "voxel_npy"
                data.save_npy_dir(str(path))
                torch_loaded = self.voxelizer.VoxelGridData.load_npy_dir(
                    str(path), output="torch", device="cpu", mmap_mode="r"
                )
            self.assertTrue(self.voxelizer._is_torch_tensor(torch_loaded.yarn_id))
            self.assertTrue(self.voxelizer._is_torch_tensor(torch_loaded.orientation1))

    def test_voxel_grid_data_orientation_conversion(self):
        snap = synthetic_snapshot(self.voxelizer)
        data = self.voxelizer.voxelize_snapshots_data(
            [snap],
            self.aabb.copy(),
            nx=4, ny=4, nz=4,
            backend="numpy",
            output="numpy",
            include_orientations=True,
            workers=1,
            chunk_voxels=16,
            verbose=False,
        )

        self.assertEqual(data.orientation1.shape, (4, 4, 4, 3))
        self.assertEqual(data.orientation2.shape, (4, 4, 4, 3))
        numpy_data = data.to("numpy", dtype=np.float64)
        self.assertEqual(numpy_data.orientation1.dtype, np.float64)
        self.assertEqual(numpy_data.orientation2.dtype, np.float64)
        np.testing.assert_allclose(numpy_data.orientation1, data.orientation1)
        np.testing.assert_allclose(numpy_data.orientation2, data.orientation2)

        if self.voxelizer.torch is not None:
            torch_data = data.to("torch", device="cpu", dtype="float64")
            self.assertTrue(self.voxelizer._is_torch_tensor(torch_data.orientation1))
            self.assertTrue(self.voxelizer._is_torch_tensor(torch_data.orientation2))
            self.assertEqual(torch_data.orientation1.dtype, self.voxelizer.torch.float64)
            self.assertEqual(torch_data.orientation2.dtype, self.voxelizer.torch.float64)
            np.testing.assert_allclose(
                torch_data.to("numpy").orientation1,
                data.orientation1,
            )

    def test_voxelize_snapshots_data_and_cache(self):
        snap = synthetic_snapshot(self.voxelizer)
        cached = self.voxelizer.voxelize_snapshots_data(
            [snap],
            self.aabb.copy(),
            nx=4, ny=4, nz=4,
            backend="numpy",
            output="numpy",
            workers=1,
            chunk_voxels=16,
            verbose=False,
        )

        cache = self.voxelizer.VoxelizationCache(
            snapshots=[snap],
            aabb=self.aabb.copy(),
        )
        cached_again = cache.voxelize(
            nx=4, ny=4, nz=4,
            backend="numpy",
            output="numpy",
            workers=1,
            chunk_voxels=16,
            verbose=False,
        )

        self.assertEqual(cached.timings["extract"], 0.0)
        self.assertEqual(int(cached.occupancy().sum()), 16)
        np.testing.assert_array_equal(cached_again.yarn_id, cached.yarn_id)

    def test_snapshot_bundle_roundtrip_and_voxelization(self):
        snap = synthetic_snapshot(self.voxelizer)
        bundle = self.voxelizer.SnapshotBundle.from_snapshots(
            [snap],
            self.aabb.copy(),
        )

        self.assertEqual(bundle.num_yarns, 1)
        np.testing.assert_array_equal(bundle.node_offsets, [0, 2])
        np.testing.assert_array_equal(bundle.section_offsets, [0, 5])
        np.testing.assert_array_equal(bundle.translation_offsets, [0, 1])

        restored = bundle.to_snapshots()
        self.assertEqual(len(restored), 1)
        np.testing.assert_allclose(restored[0].positions, snap.positions)
        np.testing.assert_allclose(restored[0].tangents, snap.tangents)
        np.testing.assert_allclose(restored[0].ups, snap.ups)
        np.testing.assert_allclose(restored[0].sides, snap.sides)
        np.testing.assert_allclose(restored[0].section, snap.section)
        np.testing.assert_allclose(restored[0].translations, snap.translations)

        bundled = self.voxelizer.voxelize_snapshot_bundle_data(
            bundle,
            nx=4, ny=4, nz=4,
            backend="numpy",
            output="numpy",
            include_orientations=True,
            workers=1,
            chunk_voxels=16,
            verbose=False,
        )
        direct = self.voxelizer.voxelize_snapshots_data(
            [snap],
            self.aabb.copy(),
            nx=4, ny=4, nz=4,
            backend="numpy",
            output="numpy",
            include_orientations=True,
            workers=1,
            chunk_voxels=16,
            verbose=False,
        )
        np.testing.assert_array_equal(bundled.yarn_id, direct.yarn_id)
        np.testing.assert_allclose(bundled.orientation1, direct.orientation1)
        np.testing.assert_allclose(bundled.orientation2, direct.orientation2)

    def test_snapshot_bundle_numpy_path_consumes_flat_arrays(self):
        snap = synthetic_snapshot(self.voxelizer)
        bundle = self.voxelizer.SnapshotBundle.from_snapshots(
            [snap],
            self.aabb.copy(),
        )
        direct = self.voxelizer.voxelize_snapshots_data(
            [snap],
            self.aabb.copy(),
            nx=4, ny=4, nz=4,
            backend="numpy",
            output="numpy",
            workers=1,
            chunk_voxels=16,
            verbose=False,
        )

        def fail_to_snapshots():
            raise AssertionError("flat SnapshotBundle path should not unpack to YarnSnapshot")

        bundle.to_snapshots = fail_to_snapshots
        bundled = self.voxelizer.voxelize_snapshot_bundle_data(
            bundle,
            nx=4, ny=4, nz=4,
            backend="numpy",
            output="numpy",
            workers=1,
            chunk_voxels=16,
            verbose=False,
        )

        np.testing.assert_array_equal(bundled.yarn_id, direct.yarn_id)
        self.assertEqual(bundled.timings["unpack"], 0.0)

    def test_extract_snapshot_bundle_uses_provider_mapping(self):
        snap = synthetic_snapshot(self.voxelizer)

        def extract_snapshot_bundle(_textile):
            return {
                "positions": snap.positions,
                "tangents": snap.tangents,
                "ups": snap.ups,
                "sides": snap.sides,
                "node_offsets": np.array([0, 2], dtype=np.int64),
                "sections": snap.section,
                "section_offsets": np.array([0, 5], dtype=np.int64),
                "translations": snap.translations,
                "translation_offsets": np.array([0, 1], dtype=np.int64),
                "aabb": self.aabb.copy(),
            }

        provider = types.SimpleNamespace(extract_snapshot_bundle=extract_snapshot_bundle)
        bundle = self.voxelizer.extract_snapshot_bundle(FakeTextile(), provider=provider)

        self.assertEqual(bundle.num_yarns, 1)
        np.testing.assert_allclose(bundle.aabb, self.aabb)
        np.testing.assert_allclose(bundle.positions, snap.positions)

    def test_fastdata_provider_status_reports_loaded_module(self):
        module_name = "TexGen._fastdata"
        old_module = sys.modules.get(module_name)
        provider = types.ModuleType(module_name)
        provider.extract_snapshot_bundle = lambda _textile: None
        sys.modules[module_name] = provider

        def restore_module():
            if old_module is None:
                sys.modules.pop(module_name, None)
            else:
                sys.modules[module_name] = old_module

        self.addCleanup(restore_module)

        status = self.voxelizer.fastdata_provider_status()

        self.assertTrue(status["available"])
        self.assertEqual(status["module"], module_name)
        self.assertIn(module_name, status["checked"])
        self.assertIsNone(status["error"])

    def test_snapshot_bundle_rejects_invalid_provider_arrays(self):
        snap = synthetic_snapshot(self.voxelizer)
        valid = {
            "positions": snap.positions,
            "tangents": snap.tangents,
            "ups": snap.ups,
            "sides": snap.sides,
            "node_offsets": np.array([0, 2], dtype=np.int64),
            "sections": snap.section,
            "section_offsets": np.array([0, 5], dtype=np.int64),
            "translations": snap.translations,
            "translation_offsets": np.array([0, 1], dtype=np.int64),
            "aabb": self.aabb.copy(),
        }

        cases = [
            ("positions", {"positions": np.zeros((2, 2), dtype=np.float32)}),
            ("tangents", {"tangents": np.zeros((1, 3), dtype=np.float32)}),
            ("node_offsets", {"node_offsets": np.array([1, 2], dtype=np.int64)}),
            ("section_offsets", {"section_offsets": np.array([0, 4], dtype=np.int64)}),
            ("same length", {"translation_offsets": np.array([0, 1, 1], dtype=np.int64)}),
            ("aabb", {"aabb": np.zeros((3, 2), dtype=np.float64)}),
        ]

        for message, overrides in cases:
            with self.subTest(message=message):
                payload = dict(valid)
                payload.update(overrides)
                with self.assertRaisesRegex(ValueError, message):
                    self.voxelizer.SnapshotBundle(**payload)

    def test_voxel_grid_data_to_dlpack_roundtrip_or_missing_error(self):
        self.patch_extract_snapshots()
        data = self.voxelizer.voxelize_textile_data(
            FakeTextile(),
            nx=4, ny=4, nz=4,
            backend="numpy",
            output="numpy",
            workers=1,
            chunk_voxels=16,
            verbose=False,
        )

        if self.voxelizer.torch is None:
            with self.assertRaisesRegex(ImportError, "DLPack"):
                data.to_dlpack("yarn_id")
            return

        torch_mod = self.voxelizer.torch
        yarn_tensor = torch_mod.utils.dlpack.from_dlpack(data.to_dlpack("yarn_id"))
        material_tensor = torch_mod.utils.dlpack.from_dlpack(data.to_dlpack("material_id"))
        occupancy_tensor = torch_mod.utils.dlpack.from_dlpack(data.to_dlpack("occupancy"))

        np.testing.assert_array_equal(yarn_tensor.cpu().numpy(), data.yarn_id)
        np.testing.assert_array_equal(material_tensor.cpu().numpy(), data.material_id())
        np.testing.assert_array_equal(occupancy_tensor.cpu().numpy(), data.occupancy())
        with self.assertRaisesRegex(ValueError, "field"):
            data.to_dlpack("bad_field")

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

    def test_torch_direct_data_keeps_tensor_or_missing_error(self):
        self.patch_extract_snapshots()
        if self.voxelizer.torch is None:
            with self.assertRaisesRegex(ImportError, "Torch backend requested"):
                self.voxelizer.voxelize_textile_data(
                    FakeTextile(),
                    nx=4, ny=4, nz=4,
                    backend="torch",
                    verbose=False,
                )
            return

        data = self.voxelizer.voxelize_textile_data(
            FakeTextile(),
            nx=4, ny=4, nz=4,
            backend="torch",
            device="cpu",
            output="backend",
            include_centers=True,
            chunk_voxels=16,
            verbose=False,
        )

        self.assertEqual(data.storage, "torch")
        self.assertTrue(self.voxelizer._is_torch_tensor(data.yarn_id))
        self.assertTrue(self.voxelizer._is_torch_tensor(data.centers))
        self.assertEqual(tuple(data.grid.shape), (4, 4, 4))
        self.assertEqual(int(data.occupancy().sum().item()), 16)

        numpy_data = data.to_numpy()
        self.assertEqual(numpy_data.storage, "numpy")
        self.assertEqual(numpy_data.grid.shape, (4, 4, 4))
        self.assertEqual(int(numpy_data.occupancy().sum()), 16)

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

    def test_top_level_functions_have_docstrings(self):
        missing = []
        for name, obj in inspect.getmembers(self.voxelizer, inspect.isfunction):
            if obj.__module__ == self.voxelizer.__name__ and not inspect.getdoc(obj):
                missing.append(name)
        self.assertEqual(missing, [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
