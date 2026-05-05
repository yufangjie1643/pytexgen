"""Smoke test for pytexgen.gpu_voxelizer."""
import sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from pytexgen import (
    CShearedTextileWeave2D, AddTextile, DeleteTextile, SaveToXML, OUTPUT_STANDARD,
    CRectangularVoxelMesh,
)
from pytexgen.gpu_voxelizer import voxelize_textile


def build_textile():
    T = CShearedTextileWeave2D(3, 3, 5.0, 2.0, 0.2618, True, True)
    for x, y in [(0, 0), (1, 1), (2, 2)]:
        T.SwapPosition(x, y)
    T.SetXYarnWidths(2.0)
    T.SetYYarnWidths(3.0)
    T.SetYarnHeights(0.8)
    T.AssignDefaultDomain()
    return T


def main():
    out_dir = Path("./_gpu_test"); out_dir.mkdir(exist_ok=True)

    # --- Python numpy path (OpenMP-free) ---
    T1 = build_textile()
    t0 = time.perf_counter()
    info = voxelize_textile(T1, nx=64, ny=64, nz=64,
                            out_inp=str(out_dir / "numpy.inp"),
                            backend="numpy",
                            workers=4)
    print(
        f"[PY] total {time.perf_counter()-t0:.2f}s, "
        f"backend={info['backend']}, workers={info['workers']}, device={info['device']}"
    )

    # --- Reference: TexGen CPU voxelizer ---
    T2 = build_textile()
    name = AddTextile(T2)
    vox = CRectangularVoxelMesh("CPeriodicBoundaries")
    t0 = time.perf_counter()
    vox.SaveVoxelMesh(T2, str(out_dir / "cpu.inp"), 64, 64, 64, True, True, 5, 0)
    print(f"[CPU] total {time.perf_counter()-t0:.2f}s")
    DeleteTextile(name)


if __name__ == "__main__":
    main()
