"""Tests for the pure Python/torch periodic tetrahedral RVE mesher."""

import importlib.util
import sys
import tempfile
import unittest
from unittest import mock
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent


def load_mesher_module():
    spec = importlib.util.spec_from_file_location(
        "TexGen.torch_periodic_mesher",
        ROOT / "TexGen" / "torch_periodic_mesher.py",
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_weave_script_module():
    spec = importlib.util.spec_from_file_location(
        "script.torch_periodic_weave_rve_tet",
        ROOT / "script" / "torch_periodic_weave_rve_tet.py",
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_octree_voxel_script_module():
    spec = importlib.util.spec_from_file_location(
        "script.torch_octree_weave_rve_voxel",
        ROOT / "script" / "torch_octree_weave_rve_voxel.py",
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class TorchPeriodicMesherTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mesher = load_mesher_module()

    def different_yarn_shared_faces(self, mesh):
        faces = {}
        count = 0
        local_faces = ((0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3))
        elements = mesh.elements.detach().cpu().numpy()
        material_ids = mesh.material_ids.detach().cpu().numpy()
        for eid, tet in enumerate(elements):
            mat = int(material_ids[eid])
            for face in local_faces:
                key = tuple(sorted(int(tet[index]) for index in face))
                if key in faces:
                    other_mat = faces.pop(key)
                    if mat > 0 and other_mat > 0 and mat != other_mat:
                        count += 1
                else:
                    faces[key] = mat
        return count

    def test_periodic_lattice_volume_and_pairs(self):
        mesh = self.mesher.generate_periodic_tet_lattice(
            nx=2,
            ny=2,
            nz=1,
            domain=((0.0, 0.0, 0.0), (2.0, 2.0, 1.0)),
            device="cpu",
        )

        volumes = self.mesher.tet_volumes(mesh.nodes, mesh.elements)
        self.assertEqual(mesh.nodes.shape[1], 3)
        self.assertEqual(mesh.elements.shape[1], 4)
        self.assertEqual(int((volumes <= 1e-14).sum().item()), 0)
        self.assertAlmostEqual(float(volumes.sum().item()), 4.0, places=12)

        pairs = self.mesher.periodic_node_pairs(mesh.nodes, mesh.domain, axes=("x", "y"))
        self.assertEqual(len(pairs["x"]), 6)
        self.assertEqual(len(pairs["y"]), 6)

    def test_topology_coordinate_lattice_is_nonuniform_and_periodic(self):
        domain = ((0.0, 0.0, -0.01), (2.0, 2.0, 0.21))

        xs, ys, zs = self.mesher.plain_weave_topology_coordinates(
            nx=6,
            ny=6,
            nz=3,
            domain=domain,
            gap_size=0.01,
        )
        x_steps = np.diff(xs)
        z_steps = np.diff(zs)

        self.assertAlmostEqual(xs[0], 0.0)
        self.assertAlmostEqual(xs[-1], 2.0)
        self.assertAlmostEqual(ys[0], 0.0)
        self.assertAlmostEqual(ys[-1], 2.0)
        self.assertGreater(len(xs), 7)
        self.assertGreater(len(ys), 7)
        self.assertGreater(len(zs), 4)
        self.assertLess(float(x_steps.min()), float(x_steps.max()))
        self.assertLess(float(z_steps.min()), float(z_steps.max()))
        for value in (0.1, 0.5, 0.9, 1.1, 1.5, 1.9):
            self.assertTrue(bool(np.isclose(xs, value, atol=1e-12).any()))
            self.assertTrue(bool(np.isclose(ys, value, atol=1e-12).any()))
        for value in (0.095, 0.105):
            self.assertTrue(bool(np.isclose(zs, value, atol=1e-12).any()))

        lattice = self.mesher.generate_periodic_tet_lattice_from_coordinates(
            xs,
            ys,
            zs,
            device="cpu",
        )
        volumes = self.mesher.tet_volumes(lattice.nodes, lattice.elements)
        pairs = self.mesher.periodic_node_pairs(lattice.nodes, lattice.domain, axes=("x", "y"))

        self.assertEqual(int((volumes <= 1e-14).sum().item()), 0)
        self.assertAlmostEqual(float(volumes.sum().item()), 0.88, places=10)
        self.assertEqual(len(pairs["x"]), len(ys) * len(zs))
        self.assertEqual(len(pairs["y"]), len(xs) * len(zs))

    def test_plane_cut_preserves_volume_and_splits_materials(self):
        lattice = self.mesher.generate_periodic_tet_lattice(
            nx=1,
            ny=1,
            nz=1,
            domain=((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
            device="cpu",
        )
        phi = lattice.nodes[:, 0] - 0.5
        yarn_ids = np.zeros(lattice.nodes.shape[0], dtype=np.int64)

        cut = self.mesher.cut_tet_mesh_by_levelset(lattice, phi, yarn_ids)
        volumes = self.mesher.tet_volumes(cut.nodes, cut.elements)

        self.assertEqual(int((volumes <= 1e-14).sum().item()), 0)
        self.assertAlmostEqual(float(volumes.sum().item()), 1.0, places=12)
        self.assertGreater(int((cut.material_ids == 0).sum().item()), 0)
        self.assertGreater(int((cut.material_ids == 1).sum().item()), 0)

    def test_plain_weave_rve_exports_inp_and_pbc_pairs(self):
        mesh = self.mesher.build_plain_weave_rve(
            nx=6,
            ny=6,
            nz=3,
            domain=((0.0, 0.0, -0.01), (2.0, 2.0, 0.21)),
            device="cpu",
        )
        volumes = self.mesher.tet_volumes(mesh.nodes, mesh.elements)
        pairs = self.mesher.periodic_node_pairs(mesh.nodes, mesh.domain, axes=("x", "y"))

        bbox_volume = (2.0 * 2.0 * 0.22)
        self.assertEqual(int((volumes <= 1e-14).sum().item()), 0)
        self.assertAlmostEqual(float(volumes.sum().item()), bbox_volume, places=10)
        self.assertGreater(int((mesh.material_ids == 0).sum().item()), 0)
        self.assertGreater(int((mesh.material_ids > 0).sum().item()), 0)
        self.assertGreater(len(pairs["x"]), 0)
        self.assertGreater(len(pairs["y"]), 0)

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "torch_weave.inp"
            csv = Path(tmp) / "torch_weave_pbc_pairs.csv"
            self.mesher.write_abaqus_inp(mesh, out)
            self.mesher.write_pbc_pairs_csv(pairs, csv)

            text = out.read_text(encoding="utf-8")
            self.assertIn("*Element, Type=C3D4", text)
            self.assertIn("*ElSet, ElSet=Matrix", text)
            self.assertIn("*ElSet, ElSet=Yarn0", text)
            self.assertIn("axis,minus_node,plus_node", csv.read_text(encoding="utf-8"))

    def test_gap_size_opens_matrix_gap_at_yarn_crossings(self):
        domain = ((0.0, 0.0, -0.01), (2.0, 2.0, 0.21))
        points = self.mesher.torch.tensor(
            [
                [0.5, 0.5, 0.10],
                [1.5, 0.5, 0.10],
                [0.5, 1.5, 0.10],
                [1.5, 1.5, 0.10],
                [0.2, 0.2, 0.10],
                [0.8, 0.2, 0.10],
                [0.2, 0.8, 0.10],
                [0.8, 0.8, 0.10],
            ],
            dtype=self.mesher.torch.float64,
        )

        phi, _yarn_ids = self.mesher.plain_weave_levelset(points, domain, gap_size=0.01)
        self.assertTrue(bool((phi > 0.0).all().item()))

        mesh = self.mesher.build_plain_weave_rve(
            nx=6,
            ny=6,
            nz=3,
            domain=domain,
            device="cpu",
            gap_size=0.01,
        )
        volumes = self.mesher.tet_volumes(mesh.nodes, mesh.elements)

        self.assertEqual(mesh.metadata["gap_size"], 0.01)
        self.assertEqual(int((volumes <= 1e-14).sum().item()), 0)
        self.assertAlmostEqual(float(volumes.sum().item()), 0.88, places=10)
        self.assertEqual(self.different_yarn_shared_faces(mesh), 0)
        summary = self.mesher.mesh_quality_summary(mesh)
        self.assertEqual(summary["different_yarn_shared_faces"], 0)

    def test_topology_density_plain_weave_rve_preserves_quality_checks(self):
        mesh = self.mesher.build_plain_weave_rve(
            nx=9,
            ny=9,
            nz=4,
            domain=((0.0, 0.0, -0.01), (2.0, 2.0, 0.21)),
            device="cpu",
            gap_size=0.01,
            density_mode="topology",
        )
        summary = self.mesher.mesh_quality_summary(mesh)
        pairs = self.mesher.periodic_node_pairs(mesh.nodes, mesh.domain, axes=("x", "y"))

        self.assertEqual(mesh.metadata["density_mode"], "topology")
        self.assertGreater(mesh.metadata["coordinate_counts"]["x"], 10)
        self.assertEqual(summary["zero_volume_tets_le_1e-14"], 0)
        self.assertAlmostEqual(summary["total_abs_volume"], summary["bbox_volume"], places=10)
        self.assertEqual(summary["different_yarn_shared_faces"], 0)
        self.assertGreater(len(pairs["x"]), 0)
        self.assertGreater(len(pairs["y"]), 0)

    def test_script_accepts_topology_density_arguments(self):
        script = load_weave_script_module()
        argv = [
            "torch_periodic_weave_rve_tet.py",
            "--density-mode",
            "topology",
            "--topology-interface-levels",
            "0",
            "--topology-crossing-levels",
            "2",
            "--topology-gap-levels",
            "1",
        ]

        with mock.patch.object(sys, "argv", argv):
            args = script.parse_args()

        self.assertEqual(args.density_mode, "topology")
        self.assertEqual(args.topology_interface_levels, 0)
        self.assertEqual(args.topology_crossing_levels, 2)
        self.assertEqual(args.topology_gap_levels, 1)

    def test_octree_voxel_rve_exports_hex_inp_and_pbc_pairs(self):
        mesh = self.mesher.build_plain_weave_octree_voxel_rve(
            base_nx=4,
            base_ny=4,
            base_nz=2,
            max_refinement=1,
            domain=((0.0, 0.0, -0.01), (2.0, 2.0, 0.21)),
            device="cpu",
            gap_size=0.01,
        )
        summary = self.mesher.hex_mesh_quality_summary(mesh)
        pairs = self.mesher.periodic_node_pairs(mesh.nodes, mesh.domain, axes=("x", "y"))

        self.assertEqual(mesh.elements.shape[1], 8)
        self.assertEqual(summary["zero_volume_hexes_le_1e-14"], 0)
        self.assertAlmostEqual(summary["total_abs_volume"], 0.88, places=10)
        self.assertEqual(summary["different_yarn_shared_faces"], 0)
        self.assertGreater(summary["material_counts"]["Matrix"], 0)
        self.assertGreater(sum(v for k, v in summary["material_counts"].items() if k.startswith("Yarn")), 0)
        self.assertGreater(mesh.metadata["coordinate_counts"]["x"], 5)
        self.assertGreater(len(pairs["x"]), 0)
        self.assertGreater(len(pairs["y"]), 0)

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "torch_octree_voxel.inp"
            csv = Path(tmp) / "torch_octree_voxel_pbc_pairs.csv"
            self.mesher.write_abaqus_voxel_inp(mesh, out)
            self.mesher.write_pbc_pairs_csv(pairs, csv)

            text = out.read_text(encoding="utf-8")
            self.assertIn("*Element, Type=C3D8R", text)
            self.assertIn("*ElSet, ElSet=Matrix", text)
            self.assertIn("*ElSet, ElSet=Yarn0", text)
            self.assertIn("axis,minus_node,plus_node", csv.read_text(encoding="utf-8"))

    def test_octree_interface_refinement_splits_material_transition_faces(self):
        coarse = self.mesher.build_plain_weave_octree_voxel_rve(
            base_nx=4,
            base_ny=4,
            base_nz=2,
            max_refinement=1,
            domain=((0.0, 0.0, -0.01), (2.0, 2.0, 0.21)),
            device="cpu",
            gap_size=0.01,
            interface_refinement_passes=0,
        )
        refined = self.mesher.build_plain_weave_octree_voxel_rve(
            base_nx=4,
            base_ny=4,
            base_nz=2,
            max_refinement=1,
            domain=((0.0, 0.0, -0.01), (2.0, 2.0, 0.21)),
            device="cpu",
            gap_size=0.01,
            interface_refinement_passes=1,
        )

        coarse_stats = self.mesher.material_interface_face_summary(coarse)
        refined_stats = self.mesher.material_interface_face_summary(refined)

        self.assertGreater(refined.metadata["interface_refinement_added_points"], 0)
        self.assertGreater(refined_stats["faces"], coarse_stats["faces"])
        self.assertLess(refined_stats["max_area"], coarse_stats["max_area"])

    def test_octree_voxel_script_accepts_arguments(self):
        script = load_octree_voxel_script_module()
        argv = [
            "torch_octree_weave_rve_voxel.py",
            "--base-nx",
            "6",
            "--base-ny",
            "5",
            "--base-nz",
            "3",
            "--max-refinement",
            "2",
            "--gap-size",
            "0.01",
            "--interface-refinement-passes",
            "1",
            "--allow-yarn-contact",
        ]

        with mock.patch.object(sys, "argv", argv):
            args = script.parse_args()

        self.assertEqual(args.base_nx, 6)
        self.assertEqual(args.base_ny, 5)
        self.assertEqual(args.base_nz, 3)
        self.assertEqual(args.max_refinement, 2)
        self.assertAlmostEqual(args.gap_size, 0.01)
        self.assertEqual(args.interface_refinement_passes, 1)
        self.assertFalse(args.separate_contacts)


if __name__ == "__main__":
    unittest.main(verbosity=2)
