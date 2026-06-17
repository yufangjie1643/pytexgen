import unittest

import numpy as np


class ModernWeaveApiTest(unittest.TestCase):
    def test_plain_weave_model_exposes_yarn_geometry_and_aabb(self):
        from pytexgen.modern import PlainWeave2D

        model = PlainWeave2D(width=2, height=2, spacing=1.0, thickness=0.2)
        textile = model.to_model()

        self.assertEqual(textile.name, "PlainWeave2D")
        self.assertEqual(len(textile.yarns), 4)
        np.testing.assert_allclose(textile.aabb, [[0.0, 0.0, 0.0], [2.0, 2.0, 0.2]])
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


if __name__ == "__main__":
    unittest.main(verbosity=2)
