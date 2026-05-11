"""Regression tests for tetrahedral element generation."""

import math
import os
import tempfile
import unittest
from pathlib import Path

from pytexgen import CMesh, CTetgenMesh, CTextileWeave2D, INP_EXPORT


def build_low_resolution_plain_weave():
    textile = CTextileWeave2D(2, 2, 1.0, 0.2, True)
    textile.SwapPosition(0, 1)
    textile.SwapPosition(1, 0)
    textile.SetYarnWidths(0.8)
    textile.SetYarnHeights(0.1)
    textile.SetResolution(8)
    textile.AssignDefaultDomain()
    return textile


def build_tetgen_smoke_plain_weave():
    textile = CTextileWeave2D(1, 1, 1.0, 0.2, True)
    textile.SetYarnWidths(0.8)
    textile.SetYarnHeights(0.1)
    textile.SetResolution(4)
    textile.AssignDefaultDomain()
    return textile


def read_abaqus_tet_mesh(path):
    nodes = {}
    elements = []
    section = None

    for raw_line in Path(path).read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        lower = line.lower()
        if lower.startswith("*node"):
            section = "node"
            continue
        if lower.startswith("*element"):
            section = "element" if "c3d4" in lower or "c3d10" in lower else None
            continue
        if lower.startswith("*"):
            section = None
            continue

        parts = [part.strip() for part in line.split(",")]
        if section == "node" and len(parts) >= 4:
            nodes[int(parts[0])] = tuple(float(value) for value in parts[1:4])
        elif section == "element" and len(parts) >= 5:
            elements.append((int(parts[0]), tuple(int(value) for value in parts[1:5])))

    return nodes, elements


def signed_tet_volume(a, b, c, d):
    ax, ay, az = a
    bx, by, bz = b
    cx, cy, cz = c
    dx, dy, dz = d
    return (
        (bx - ax) * ((cy - ay) * (dz - az) - (cz - az) * (dy - ay))
        - (by - ay) * ((cx - ax) * (dz - az) - (cz - az) * (dx - ax))
        + (bz - az) * ((cx - ax) * (dy - ay) - (cy - ay) * (dx - ax))
    ) / 6.0


class TetrahedralMeshTest(unittest.TestCase):
    def test_convert_volume_mesh_to_tets_has_no_zero_volume_elements(self):
        textile = build_low_resolution_plain_weave()
        mesh = CMesh()

        textile.AddVolumeToMesh(mesh)
        self.assertGreater(len(list(mesh.GetIndices(CMesh.WEDGE))), 0)

        mesh.ConvertToTetMesh()

        tet_indices = list(mesh.GetIndices(CMesh.TET))
        self.assertGreater(len(tet_indices), 0)
        self.assertEqual(len(tet_indices) % 4, 0)
        self.assertEqual(len(list(mesh.GetIndices(CMesh.WEDGE))), 0)
        self.assertEqual(len(list(mesh.GetIndices(CMesh.HEX))), 0)

        nodes = [
            (mesh.GetNode(i).x, mesh.GetNode(i).y, mesh.GetNode(i).z)
            for i in range(mesh.GetNumNodes())
        ]
        volumes = []
        duplicate_node_tets = []
        zero_volume_tets = []

        for tet_id, start in enumerate(range(0, len(tet_indices), 4)):
            tet = tet_indices[start:start + 4]
            if len(set(tet)) != 4:
                duplicate_node_tets.append((tet_id, tet))
                continue
            volume = abs(signed_tet_volume(*(nodes[i] for i in tet)))
            volumes.append(volume)
            if volume <= 1e-14:
                zero_volume_tets.append((tet_id, tet, volume))

        self.assertFalse(duplicate_node_tets[:5])
        self.assertFalse(zero_volume_tets[:5])
        self.assertGreater(min(volumes), 1e-14)
        self.assertTrue(math.isclose(sum(volumes), mesh.CalculateVolume(), rel_tol=1e-10))

    @unittest.skipUnless(
        os.environ.get("PYTEXGEN_RUN_TETGEN_NATIVE") == "1",
        "set PYTEXGEN_RUN_TETGEN_NATIVE=1 to run the native Tetgen export smoke test",
    )
    def test_tetgen_export_has_no_zero_volume_elements(self):
        textile = build_tetgen_smoke_plain_weave()

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "plain_weave_tet.inp"
            CTetgenMesh(0.0).SaveTetgenMesh(textile, str(out), "p", False, INP_EXPORT)

            self.assertTrue(out.exists())
            nodes, elements = read_abaqus_tet_mesh(out)
            self.assertGreater(len(nodes), 0)
            self.assertGreater(len(elements), 0)

            volumes = []
            duplicate_node_tets = []
            zero_volume_tets = []

            for element_id, tet in elements:
                if len(set(tet)) != 4:
                    duplicate_node_tets.append((element_id, tet))
                    continue
                volume = abs(signed_tet_volume(*(nodes[i] for i in tet)))
                volumes.append(volume)
                if volume <= 1e-14:
                    zero_volume_tets.append((element_id, tet, volume))

            self.assertFalse(duplicate_node_tets[:5])
            self.assertFalse(zero_volume_tets[:5])
            self.assertGreater(min(volumes), 1e-14)


if __name__ == "__main__":
    unittest.main()
