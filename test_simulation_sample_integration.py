"""Real TexGen integration coverage for the simulation sample contract."""

import unittest

import numpy as np

import pytexgen as tg
from pytexgen.material_fields import isotropic_stiffness_c21
from pytexgen.simulation_sample import (
    MaterialTable,
    SimulationSample,
    voxelize_textile_simulation_sample,
)


def build_small_plain_weave():
    textile = tg.CTextileWeave2D(2, 2, 1.0, 0.2, False, True)
    for y_index in range(2):
        for x_index in range(2):
            if (x_index + y_index) % 2 == 0:
                textile.SwapPosition(x_index, y_index)
    textile.SetYarnWidths(0.8)
    textile.SetYarnHeights(0.1)
    textile.SetResolution(10)
    textile.AssignDefaultDomain()
    return textile


class SimulationSampleIntegrationTest(unittest.TestCase):
    def test_real_textile_produces_valid_sparse_simulation_sample(self):
        matrix = isotropic_stiffness_c21(3.0, 0.35)
        yarn = isotropic_stiffness_c21(70.0, 0.20)
        materials = MaterialTable(
            c21=np.stack((matrix, yarn)),
            material_ids=np.array([0, 7], dtype=np.int32),
            unit="GPa",
            names=("matrix", "yarn"),
        )

        sample = voxelize_textile_simulation_sample(
            build_small_plain_weave(),
            materials=materials,
            default_yarn_material_id=7,
            metadata={"case": "plain-2x2-integration"},
            nx=6,
            ny=6,
            nz=4,
            backend="numpy",
            workers=1,
            chunk_voxels=64,
            dtype="float64",
            aabb_pruning=True,
            verbose=False,
        )

        self.assertIsInstance(sample, SimulationSample)
        self.assertEqual(sample.voxels.shape, (4, 6, 6))
        self.assertGreater(sample.orientation.num_yarn_voxels, 0)
        self.assertIs(
            sample.orientation.voxel_indices,
            sample.stiffness.voxel_indices,
        )
        self.assertEqual(sample.stiffness.unit, "GPa")
        self.assertEqual(
            set(sample.stiffness.material_ids.tolist()),
            {7},
        )
        np.testing.assert_allclose(
            sample.stiffness.yarn_c21,
            np.broadcast_to(yarn, sample.stiffness.yarn_c21.shape),
            rtol=1e-10,
            atol=1e-12,
        )
        material_grid = sample.array("voxel.material_id", copy=True)
        self.assertEqual(set(np.unique(material_grid).tolist()), {0, 7})
        np.testing.assert_allclose(
            sample.array(
                "stiffness.yarn_c21",
                layout="acdm",
                copy=True,
            ),
            sample.stiffness.to_acdm(batch=True),
            rtol=0.0,
            atol=0.0,
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
