import unittest

import numpy as np


class FastDataExtensionTest(unittest.TestCase):
    def test_core_pointer_snapshot_bundle_uses_compiled_provider(self):
        import pytexgen as tg
        import pytexgen._Core as core
        from pytexgen.gpu_voxelizer import (
            SnapshotBundle,
            extract_snapshot_bundle,
            fastdata_provider_status,
            voxelize_snapshot_bundle_data,
        )

        tg.DeleteTextiles()
        textile = tg.CTextileWeave2D(2, 2, 1.0, 0.2, True)
        textile.SetYarnWidths(0.8)
        textile.SetYarnHeights(0.1)
        textile.AssignDefaultDomain()

        status = fastdata_provider_status()
        self.assertTrue(status["available"], status)
        self.assertIn("extract_from_core_pointer", status["capabilities"])

        direct_mapping = core._fastdata_extract_snapshot_bundle_direct(textile)
        for array_name in (
            "positions", "tangents", "ups", "sides", "node_offsets",
            "sections", "section_offsets", "translations",
            "translation_offsets", "aabb",
        ):
            self.assertTrue(
                direct_mapping[array_name].flags.owndata,
                f"{array_name} should be owned by a direct NumPy ndarray",
            )
        direct_bundle = SnapshotBundle(**direct_mapping)
        self.assertGreater(direct_bundle.num_yarns, 0)

        bundle = extract_snapshot_bundle(textile)

        self.assertIsInstance(bundle, SnapshotBundle)
        self.assertEqual(bundle.num_yarns, direct_bundle.num_yarns)
        self.assertGreater(bundle.num_yarns, 0)
        self.assertEqual(bundle.positions.shape[1], 3)
        self.assertEqual(bundle.sections.shape[1], 2)
        self.assertEqual(bundle.translations.shape[1], 3)
        np.testing.assert_array_equal(bundle.node_offsets[:1], [0])
        np.testing.assert_array_equal(bundle.section_offsets[:1], [0])
        np.testing.assert_array_equal(bundle.translation_offsets[:1], [0])

        data = voxelize_snapshot_bundle_data(
            bundle,
            nx=4, ny=4, nz=4,
            backend="numpy",
            output="numpy",
            workers=1,
            chunk_voxels=16,
            verbose=False,
        )
        self.assertEqual(data.yarn_id.shape, (64,))


if __name__ == "__main__":
    unittest.main(verbosity=2)
