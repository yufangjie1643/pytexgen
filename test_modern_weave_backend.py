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


if __name__ == "__main__":
    unittest.main(verbosity=2)
