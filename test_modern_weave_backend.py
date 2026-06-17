import unittest

import numpy as np


class ModernWeaveApiTest(unittest.TestCase):
    def test_plain_weave_model_exposes_yarn_geometry_and_aabb(self):
        from pytexgen.modern import PlainWeave2D

        model = PlainWeave2D(width=2, height=2, spacing=1.0, thickness=0.2)
        textile = model.to_model()

        self.assertEqual(textile.name, "PlainWeave2D")
        self.assertEqual(len(textile.yarns), 4)
        np.testing.assert_allclose(textile.aabb, [[-0.5, -0.5, -0.01], [1.5, 1.5, 0.21]])
        self.assertEqual(textile.yarns[0].positions.shape, (3, 3))
        self.assertEqual(textile.yarns[0].section.points.shape[1], 2)

    def test_plain_weave_swap_position_matches_texgen_cell_order(self):
        from pytexgen.modern import PlainWeave2D

        model = PlainWeave2D(width=2, height=2, spacing=1.0, thickness=0.2)
        self.assertEqual(model.cell(0, 0), ("y", "x"))
        model.swap_position(0, 0)
        self.assertEqual(model.cell(0, 0), ("x", "y"))
        textile = model.to_model()
        self.assertEqual(len(textile.yarns), 4)
        self.assertLess(textile.yarns[0].positions[0, 2], textile.yarns[2].positions[0, 2])

    def test_numpy_voxelize_model_data_returns_voxel_grid_contract(self):
        from pytexgen.modern import PlainWeave2D, voxelize_model_data

        model = PlainWeave2D(width=2, height=2, spacing=1.0, thickness=0.2)
        data = voxelize_model_data(model, resolution=(4, 4, 2), backend="numpy")

        self.assertEqual(data.resolution, (4, 4, 2))
        self.assertEqual(data.grid.shape, (2, 4, 4))
        self.assertEqual(data.yarn_id.shape, (32,))
        self.assertEqual(data.order, "ix + iy*nx + iz*nx*ny")
        self.assertGreaterEqual(int(data.material_id().max()), 1)

    def test_numpy_auto_workers_matches_serial_output(self):
        from pytexgen.modern import PlainWeave2D, voxelize_model_data

        model = PlainWeave2D(width=2, height=2, spacing=1.0, thickness=0.2)
        serial = voxelize_model_data(
            model,
            resolution=(8, 8, 4),
            backend="numpy",
            workers=1,
        )
        automatic = voxelize_model_data(
            model,
            resolution=(8, 8, 4),
            backend="numpy",
            workers="auto",
        )

        np.testing.assert_array_equal(automatic.yarn_id, serial.yarn_id)

    def test_plain_weave_fast_path_matches_generic_numpy(self):
        from pytexgen.modern import PlainWeave2D, voxelize_model_data

        model = PlainWeave2D(width=2, height=2, spacing=1.0, thickness=0.2)
        fast = voxelize_model_data(
            model,
            resolution=(8, 8, 4),
            backend="numpy",
            workers=2,
        )
        generic = voxelize_model_data(
            model,
            resolution=(8, 8, 4),
            backend="numpy",
            workers=1,
            fast_path=False,
        )

        np.testing.assert_array_equal(fast.yarn_id, generic.yarn_id)
        self.assertEqual(fast.backend, "numpy")
        self.assertEqual(fast.storage, "numpy")

    def test_modern_auto_worker_policy_stays_conservative(self):
        from unittest.mock import patch

        from pytexgen.modern.voxel import _resolve_modern_workers

        with patch("pytexgen.modern.voxel.os.cpu_count", return_value=12):
            self.assertEqual(_resolve_modern_workers("numpy", "auto", (8, 8, 4)), 1)
            self.assertEqual(_resolve_modern_workers("numpy", "auto", (64, 64, 64)), 2)
            self.assertEqual(_resolve_modern_workers("numpy", "auto", (128, 128, 128)), 4)
            self.assertEqual(_resolve_modern_workers("numpy", 12, (64, 64, 64)), 12)
            self.assertIsNone(_resolve_modern_workers("torch", "auto", (64, 64, 64)))

    def test_batch_voxelize_models_uses_process_pool_and_matches_serial(self):
        from pytexgen.modern import PlainWeave2D, voxelize_model_data, voxelize_models_data

        models = [
            PlainWeave2D(width=2, height=2, spacing=1.0, thickness=0.2),
            PlainWeave2D(width=3, height=2, spacing=0.8, thickness=0.18),
            PlainWeave2D(width=2, height=3, spacing=1.1, thickness=0.22),
        ]
        models[1].swap_position(1, 0)
        models[2].swap_position(0, 2)

        serial = [
            voxelize_model_data(model, resolution=(8, 8, 4), backend="numpy", workers=1)
            for model in models
        ]
        batch = voxelize_models_data(
            models,
            resolution=(8, 8, 4),
            backend="numpy",
            workers=2,
            inner_workers=1,
            return_data=True,
            chunksize=1,
        )

        self.assertEqual(len(batch), len(models))
        for expected, actual in zip(serial, batch):
            np.testing.assert_array_equal(actual.yarn_id, expected.yarn_id)
            np.testing.assert_allclose(actual.aabb, expected.aabb)

    def test_batch_voxelize_models_can_write_worker_npz_files(self):
        import tempfile
        from pathlib import Path

        from pytexgen.modern import PlainWeave2D, VoxelBatchFile, voxelize_models_data
        from pytexgen.modern.compat import load_gpu_voxelizer

        models = [
            PlainWeave2D(width=2, height=2, spacing=1.0, thickness=0.2),
            PlainWeave2D(width=2, height=2, spacing=0.9, thickness=0.16),
        ]

        with tempfile.TemporaryDirectory() as tmp:
            results = voxelize_models_data(
                models,
                resolution=(4, 4, 2),
                backend="numpy",
                workers=1,
                binary_dir=tmp,
                file_prefix="case",
                compressed=False,
                return_data=False,
            )

            self.assertEqual(len(results), 2)
            self.assertIsInstance(results[0], VoxelBatchFile)
            self.assertEqual(Path(results[0].path).name, "case_000000.npz")
            self.assertTrue(Path(results[1].path).exists())

            VoxelGridData = load_gpu_voxelizer().VoxelGridData
            loaded = VoxelGridData.load_npz(results[0].path)
            self.assertEqual(loaded.resolution, (4, 4, 2))
            self.assertEqual(int((loaded.yarn_id >= 0).sum()), results[0].occupied)

    def test_torch_backend_matches_numpy_when_available(self):
        import torch  # noqa: F401
        from pytexgen.modern import PlainWeave2D, voxelize_model_data

        model = PlainWeave2D(width=2, height=2, spacing=1.0, thickness=0.2)
        numpy_data = voxelize_model_data(model, resolution=(4, 4, 2), backend="numpy")
        torch_data = voxelize_model_data(
            model,
            resolution=(4, 4, 2),
            backend="torch",
            device="cpu",
        )

        self.assertEqual(torch_data.storage, "torch")
        np.testing.assert_array_equal(torch_data.to_numpy().yarn_id, numpy_data.yarn_id)

    def test_triton_backend_is_reserved_until_kernel_exists(self):
        from pytexgen.modern import PlainWeave2D, voxelize_model_data

        with self.assertRaises(NotImplementedError):
            voxelize_model_data(
                PlainWeave2D(width=2, height=2, spacing=1.0, thickness=0.2),
                resolution=(4, 4, 2),
                backend="triton",
            )

    def test_write_inp_from_voxel_data_uses_legacy_node_and_element_order(self):
        import tempfile
        from pathlib import Path

        from pytexgen.modern import PlainWeave2D, voxelize_model_data, write_inp_from_voxel_data

        data = voxelize_model_data(
            PlainWeave2D(width=2, height=2, spacing=1.0, thickness=0.2),
            resolution=(2, 2, 1),
            backend="numpy",
        )
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "modern.inp"
            write_inp_from_voxel_data(data, out, textile_name="ModernPlain")
            text = out.read_text()

        self.assertIn("*Node", text)
        self.assertIn("*Element, type=C3D8R", text)
        self.assertIn("1, 2, 5, 4, 1, 11, 14, 13, 10", text)

    def test_shallow_cross_auto_binder_positions_match_current_script_rules(self):
        from pytexgen.modern import auto_binder_positions

        positions = auto_binder_positions(
            "straight",
            num_x_yarns=2,
            num_y_yarns=4,
            z_layers=5,
            binder_depth=3,
        )
        self.assertEqual(len(positions), 8)
        self.assertEqual(positions[:4], [(0, 0, 0), (1, 0, 1), (2, 0, 2), (3, 0, 1)])
        self.assertEqual(positions[4:], [(0, 1, 2), (1, 1, 1), (2, 1, 0), (3, 1, 1)])

    def test_shallow_cross_subset_builds_snapshot_compatible_model(self):
        from pytexgen.modern import ShallowCrossLayerToLayer

        model = ShallowCrossLayerToLayer(
            num_x_yarns=2,
            num_y_yarns=4,
            x_spacing=1.4,
            y_spacing=2.2,
            z_layers=5,
            binder_depth=3,
        )
        textile = model.to_model()
        self.assertEqual(textile.name, "ShallowCrossLayerToLayer")
        self.assertGreaterEqual(len(textile.yarns), 6)
        self.assertEqual(textile.aabb.shape, (2, 3))


if __name__ == "__main__":
    unittest.main(verbosity=2)
