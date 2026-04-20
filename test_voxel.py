"""Minimal smoke test: build a 2D plain weave and export a voxel mesh."""
import time
from pathlib import Path
from pytexgen import (
    CTextileWeave2D,
    CRectangularVoxelMesh,
    AddTextile,
    DeleteTextile,
    SaveToXML,
    OUTPUT_STANDARD,
)


def main():
    out_dir = Path("./test_output")
    out_dir.mkdir(parents=True, exist_ok=True)

    num_x, num_y = 4, 4
    spacing, thickness = 1.0, 0.2

    textile = CTextileWeave2D(num_x, num_y, spacing, thickness)
    for y in range(num_y):
        for x in range(num_x):
            if (x + y) % 2 == 0:
                textile.SwapPosition(x, y)

    textile.SetYarnWidths(0.8)
    textile.SetYarnHeights(0.1)
    textile.AssignDefaultDomain()
    name = AddTextile(textile)

    SaveToXML(str(out_dir / "plain_weave.tg3"), name, OUTPUT_STANDARD)

    vox = CRectangularVoxelMesh("CPeriodicBoundaries")
    resolution = 48
    inp_path = out_dir / f"plain_weave_vox_{resolution}.inp"

    print(f"[*] Voxelising at {resolution}^3 ...")
    t0 = time.perf_counter()
    vox.SaveVoxelMesh(textile, str(inp_path), resolution, resolution, resolution,
                      True, True, 5, 0)
    dt = time.perf_counter() - t0
    print(f"[OK] Voxel mesh saved to {inp_path} ({dt:.2f}s)")

    DeleteTextile(name)


if __name__ == "__main__":
    main()
