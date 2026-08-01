import tempfile
import unittest
from pathlib import Path

import numpy as np


class FastDataExtensionTest(unittest.TestCase):
    @staticmethod
    def _numeric_csv_rows(path):
        rows = []
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if stripped and stripped[0].isdigit():
                rows.append([part.strip() for part in line.split(",")])
        return rows

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

    def test_exact_voxel_data_matches_legacy_eld_and_ori(self):
        import pytexgen as tg
        from pytexgen.gpu_voxelizer import (
            fastdata_provider_status,
            voxelize_textile_data,
        )

        tg.DeleteTextiles()
        textile = tg.CTextileWeave2D(2, 2, 1.0, 0.2, True)
        textile.SetYarnWidths(0.8)
        textile.SetYarnHeights(0.1)
        textile.AssignDefaultDomain()

        status = fastdata_provider_status()
        self.assertIn(
            "exact_voxelize_from_core_pointer", status["capabilities"]
        )

        resolution = (6, 5, 4)
        exact = voxelize_textile_data(
            textile,
            *resolution,
            classification="exact",
            backend="numpy",
            dtype="float64",
            include_orientations=True,
            orientation_storage="sparse",
            workers=1,
            verbose=False,
        )
        self.assertEqual(exact.classification, "exact")
        self.assertEqual(exact.yarn_id.dtype, np.int32)
        self.assertEqual(exact.yarn_id.shape, (np.prod(resolution),))

        with tempfile.TemporaryDirectory() as directory:
            inp_path = Path(directory) / "legacy.inp"
            voxel_mesh = tg.CRectangularVoxelMesh("CPeriodicBoundaries")
            voxel_mesh.SaveVoxelMesh(
                textile,
                str(inp_path),
                *resolution,
                True,
                True,
                5,
                0,
            )
            element_rows = self._numeric_csv_rows(
                inp_path.with_suffix(".eld")
            )
            orientation_rows = self._numeric_csv_rows(
                inp_path.with_suffix(".ori")
            )

        legacy_yarn_id = np.asarray(
            [int(row[1]) for row in element_rows], dtype=np.int32
        )
        legacy_orientation = np.asarray(
            [[float(value) for value in row[1:7]] for row in orientation_rows],
            dtype=np.float64,
        )
        np.testing.assert_array_equal(exact.yarn_id, legacy_yarn_id)

        occupied = np.flatnonzero(legacy_yarn_id >= 0)
        sparse = exact.sparse_orientation
        self.assertIsNotNone(sparse)
        np.testing.assert_array_equal(sparse.voxel_indices, occupied)
        np.testing.assert_array_equal(sparse.yarn_ids, legacy_yarn_id[occupied])
        np.testing.assert_allclose(
            sparse.orientation1,
            legacy_orientation[occupied, :3],
            rtol=0.0,
            atol=1e-12,
        )

        np.testing.assert_allclose(
            sparse.orientation2,
            legacy_orientation[occupied, 3:],
            rtol=0.0,
            atol=1e-12,
        )

        # A fully orthotropic material is sensitive to swapping local axes 2
        # and 3. Matching its rotated C21 field verifies the direct API uses
        # the same complete material frame as TexGen's Abaqus export.
        from pytexgen.material_fields import (
            orthotropic_stiffness_c21,
            rotate_stiffness_c21,
        )

        local_c21 = orthotropic_stiffness_c21(
            150.0, 12.0, 8.0,
            0.24, 0.19, 0.31,
            5.5, 6.5, 3.5,
        )
        local_field = np.broadcast_to(
            local_c21, (occupied.size, local_c21.size)
        ).copy()
        exact_c21 = rotate_stiffness_c21(
            local_field, sparse.orientation1, sparse.orientation2
        )
        legacy_c21 = rotate_stiffness_c21(
            local_field,
            legacy_orientation[occupied, :3],
            legacy_orientation[occupied, 3:],
        )
        np.testing.assert_allclose(
            exact_c21, legacy_c21, rtol=1e-11, atol=1e-11
        )

        parallel = voxelize_textile_data(
            textile,
            *resolution,
            classification="exact",
            backend="numpy",
            dtype="float64",
            include_orientations=True,
            orientation_storage="sparse",
            workers=4,
            verbose=False,
        )
        parallel_sparse = parallel.sparse_orientation
        self.assertEqual(parallel.workers, 4)
        np.testing.assert_array_equal(parallel.yarn_id, exact.yarn_id)
        np.testing.assert_array_equal(
            parallel_sparse.voxel_indices, sparse.voxel_indices
        )
        np.testing.assert_array_equal(
            parallel_sparse.orientation1, sparse.orientation1
        )
        np.testing.assert_array_equal(
            parallel_sparse.orientation2, sparse.orientation2
        )

    def test_numpy_exact_matches_compiled_exact_ids_and_orientations(self):
        import pytexgen as tg
        from pytexgen.gpu_voxelizer import (
            extract_numpy_exact_geometry,
            voxelize_numpy_exact_geometry_data,
            voxelize_textile_data,
        )

        tg.DeleteTextiles()
        textile = tg.CTextileWeave2D(2, 2, 1.0, 0.2, True)
        textile.SetYarnWidths(0.8)
        textile.SetYarnHeights(0.1)
        textile.AssignDefaultDomain()

        geometry = extract_numpy_exact_geometry(textile)
        self.assertEqual(geometry.num_yarns, textile.GetNumYarns())
        resolution = (9, 8, 7)
        expected = voxelize_textile_data(
            textile,
            *resolution,
            classification="exact",
            backend="numpy",
            dtype="float64",
            include_orientations=True,
            orientation_storage="sparse",
            workers=1,
            verbose=False,
        )
        actual = voxelize_numpy_exact_geometry_data(
            geometry,
            *resolution,
            backend="numpy",
            dtype="float64",
            include_orientations=True,
            orientation_storage="sparse",
            workers=1,
            chunk_voxels=97,
            verbose=False,
        )
        self.assertEqual(actual.classification, "numpy_exact")
        np.testing.assert_array_equal(actual.yarn_id, expected.yarn_id)
        np.testing.assert_array_equal(
            actual.sparse_orientation.voxel_indices,
            expected.sparse_orientation.voxel_indices,
        )
        np.testing.assert_allclose(
            actual.sparse_orientation.orientation1,
            expected.sparse_orientation.orientation1,
            rtol=0.0,
            atol=1e-12,
        )
        np.testing.assert_allclose(
            actual.sparse_orientation.orientation2,
            expected.sparse_orientation.orientation2,
            rtol=0.0,
            atol=1e-12,
        )

        parallel = voxelize_textile_data(
            textile,
            *resolution,
            classification="numpy_exact",
            backend="numpy",
            dtype="float64",
            include_orientations=True,
            orientation_storage="sparse",
            workers=3,
            chunk_voxels=97,
            verbose=False,
        )
        self.assertEqual(parallel.workers, 3)
        np.testing.assert_array_equal(parallel.yarn_id, actual.yarn_id)
        np.testing.assert_array_equal(
            parallel.sparse_orientation.orientation1,
            actual.sparse_orientation.orientation1,
        )
        np.testing.assert_array_equal(
            parallel.sparse_orientation.orientation2,
            actual.sparse_orientation.orientation2,
        )

    def test_numpy_exact_varying_section_orientation_matches_compiled(self):
        import pytexgen as tg
        from pytexgen.gpu_voxelizer import voxelize_textile_data

        textile = tg.CTextile()
        yarn = tg.CYarn()
        yarn.AddNode(tg.CNode(tg.XYZ(-0.5, 0.0, 0.0)))
        yarn.AddNode(tg.CNode(tg.XYZ(0.0, 0.25, 0.1)))
        yarn.AddNode(tg.CNode(tg.XYZ(0.5, 0.0, 0.0)))
        yarn.AssignInterpolation(tg.CInterpolationBezier(False))
        sections = tg.CYarnSectionInterpNode(False, True)
        sections.AddSection(tg.CSectionEllipse(0.25, 0.12))
        sections.AddSection(tg.CSectionEllipse(0.35, 0.18))
        sections.AddSection(tg.CSectionEllipse(0.20, 0.10))
        sections.InsertSection(0, 0.35, tg.CSectionEllipse(0.30, 0.14))
        yarn.AssignSection(sections)
        yarn.SetResolution(30)
        textile.AddYarn(yarn)
        textile.AssignDomain(
            tg.CDomainPlanes(
                tg.XYZ(-0.7, -0.4, -0.3),
                tg.XYZ(0.7, 0.6, 0.3),
            )
        )

        options = dict(
            nx=13,
            ny=11,
            nz=9,
            backend="numpy",
            dtype="float64",
            workers=1,
            verbose=False,
            include_orientations=True,
            orientation_storage="sparse",
        )
        expected = voxelize_textile_data(
            textile, classification="exact", **options
        )
        actual = voxelize_textile_data(
            textile, classification="numpy_exact", **options
        )
        np.testing.assert_array_equal(actual.yarn_id, expected.yarn_id)
        np.testing.assert_allclose(
            actual.sparse_orientation.orientation1,
            expected.sparse_orientation.orientation1,
            rtol=0.0,
            atol=2e-12,
        )
        np.testing.assert_allclose(
            actual.sparse_orientation.orientation2,
            expected.sparse_orientation.orientation2,
            rtol=0.0,
            atol=2e-12,
        )

    def test_numpy_exact_position_interpolated_section_matches_compiled(self):
        import pytexgen as tg
        from pytexgen.gpu_voxelizer import voxelize_textile_data

        textile = tg.CTextile()
        yarn = tg.CYarn()
        yarn.AddNode(tg.CNode(tg.XYZ(-0.5, 0.0, 0.0)))
        yarn.AddNode(tg.CNode(tg.XYZ(0.0, 0.25, 0.1)))
        yarn.AddNode(tg.CNode(tg.XYZ(0.5, 0.0, 0.0)))
        yarn.AssignInterpolation(tg.CInterpolationBezier(False))
        sections = tg.CYarnSectionInterpPosition(True, True)
        sections.AddSection(0.2, tg.CSectionEllipse(0.25, 0.12))
        sections.AddSection(0.6, tg.CSectionEllipse(0.35, 0.18))
        sections.AddSection(0.9, tg.CSectionEllipse(0.20, 0.10))
        yarn.AssignSection(sections)
        yarn.SetResolution(30)
        textile.AddYarn(yarn)
        textile.AssignDomain(
            tg.CDomainPlanes(
                tg.XYZ(-0.7, -0.4, -0.3),
                tg.XYZ(0.7, 0.6, 0.3),
            )
        )

        options = dict(
            nx=13,
            ny=11,
            nz=9,
            backend="numpy",
            dtype="float64",
            workers=1,
            verbose=False,
            include_orientations=True,
            orientation_storage="sparse",
        )
        expected = voxelize_textile_data(
            textile, classification="exact", **options
        )
        actual = voxelize_textile_data(
            textile, classification="numpy_exact", **options
        )
        np.testing.assert_array_equal(actual.yarn_id, expected.yarn_id)
        np.testing.assert_allclose(
            actual.sparse_orientation.orientation1,
            expected.sparse_orientation.orientation1,
            rtol=0.0,
            atol=2e-12,
        )
        np.testing.assert_allclose(
            actual.sparse_orientation.orientation2,
            expected.sparse_orientation.orientation2,
            rtol=0.0,
            atol=2e-12,
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
