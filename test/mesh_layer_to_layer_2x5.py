from __future__ import annotations

import argparse
import itertools
import sys
import threading
import time
from contextlib import contextmanager
from pathlib import Path

from pytexgen import (
    CMesher,
    CTexGen,
    DeleteTextiles,
    GetTextile,
    MATERIAL_CONTINUUM,
    NO_BOUNDARY_CONDITIONS,
    ReadFromXML,
)


@contextmanager
def activity(label: str, interval: float = 10.0):
    """Print elapsed-time status while a blocking TexGen C++ call is running."""
    stop = threading.Event()
    started = time.perf_counter()
    spinner = itertools.cycle("|/-\\")

    def report() -> None:
        while not stop.wait(interval):
            elapsed = time.perf_counter() - started
            print(f"[mesh] {next(spinner)} {label} still running, elapsed {elapsed:.0f}s")
            sys.stdout.flush()

    print(f"[mesh] -> {label}")
    sys.stdout.flush()
    thread = threading.Thread(target=report, daemon=True)
    thread.start()
    try:
        yield
    finally:
        stop.set()
        thread.join(timeout=1.0)
        elapsed = time.perf_counter() - started
        print(f"[mesh] <- {label} done in {elapsed:.2f}s")
        sys.stdout.flush()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a TexGen CMesher volume mesh from a .tg3 file."
    )
    parser.add_argument(
        "--input",
        default=r"E:\CMAME\layer_to_layer_2x5.tg3",
        help="Input TexGen .tg3 file.",
    )
    parser.add_argument(
        "--output-dir",
        default=r"E:\yfj_code\pytexgen\test",
        help="Directory for Abaqus .inp/.ori/.eld outputs.",
    )
    parser.add_argument(
        "--seed",
        type=float,
        default=0.5,
        help="TexGen mesher seed length. Use 0.5 for the requested maximum mesh size.",
    )
    parser.add_argument(
        "--merge-tolerance",
        type=float,
        default=0.001,
        help="Layer merge tolerance passed to CMesher.SetMergeTolerance.",
    )
    parser.add_argument(
        "--no-periodic",
        action="store_true",
        help="Disable periodic meshing. Periodic meshing is enabled by default.",
    )
    parser.add_argument(
        "--boundary-conditions",
        choices=("material-continuum", "none"),
        default="material-continuum",
        help=(
            "Boundary condition mode for CMesher. Use 'none' for a faster "
            "unconstrained mesh without periodic boundary equations."
        ),
    )
    parser.add_argument(
        "--quiet-texgen",
        action="store_true",
        help="Disable TexGen C++ detail logs; Python elapsed-time status remains enabled.",
    )
    parser.add_argument(
        "--textile-section-points",
        type=int,
        default=None,
        help=(
            "Optional lower yarn section resolution before meshing. "
            "This changes geometry fidelity; loaded file uses 40."
        ),
    )
    parser.add_argument(
        "--textile-slave-nodes",
        type=int,
        default=None,
        help=(
            "Optional fixed slave-node count per yarn before meshing. "
            "Use only for coarse preview meshes because it changes geometry fidelity."
        ),
    )
    parser.add_argument(
        "--status-interval",
        type=float,
        default=10.0,
        help="Seconds between status messages during blocking TexGen calls.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)

    if not input_path.is_file():
        raise FileNotFoundError(f"Input .tg3 file not found: {input_path}")
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / f"{input_path.stem}_seed{args.seed:g}.inp"

    print(f"[mesh] input:  {input_path}")
    print(f"[mesh] output: {output_path}")
    print(f"[mesh] seed:   {args.seed:g}")
    print(f"[mesh] boundary conditions: {args.boundary_conditions}")

    if args.quiet_texgen:
        CTexGen.GetInstance().SetMessages(False)

    DeleteTextiles()
    with activity("reading .tg3 file", args.status_interval):
        if not ReadFromXML(str(input_path)):
            raise RuntimeError(f"TexGen failed to read XML file: {input_path}")
    textile = GetTextile()

    if args.textile_section_points is not None:
        if args.textile_section_points < 4:
            raise ValueError("--textile-section-points must be >= 4")
        slave_nodes = 0 if args.textile_slave_nodes is None else args.textile_slave_nodes
        if slave_nodes < 0:
            raise ValueError("--textile-slave-nodes must be >= 0")
        label = (
            f"setting textile resolution: section_points={args.textile_section_points}, "
            f"slave_nodes={'auto' if slave_nodes == 0 else slave_nodes}"
        )
        with activity(label, args.status_interval):
            if not textile.SetResolution(args.textile_section_points, slave_nodes):
                raise RuntimeError("textile.SetResolution failed")

    boundary_conditions = (
        NO_BOUNDARY_CONDITIONS
        if args.boundary_conditions == "none"
        else MATERIAL_CONTINUUM
    )
    mesher = CMesher(boundary_conditions)
    mesher.SetPeriodic((not args.no_periodic) and args.boundary_conditions != "none")
    mesher.SetSeed(args.seed)
    mesher.SetMergeTolerance(args.merge_tolerance)

    with activity("creating CMesher volume mesh", args.status_interval):
        if not mesher.CreateMesh(textile):
            raise RuntimeError("CMesher.CreateMesh failed")
    mesh = mesher.GetMesh()
    print(
        "[mesh] created volume mesh: "
        f"{mesh.GetNumElements()} elements, {mesh.GetNumNodes()} nodes"
    )

    with activity("saving Abaqus .inp/.ori/.eld files", args.status_interval):
        mesher.SaveVolumeMeshToABAQUS(str(output_path), textile)
    print(f"[mesh] inp: {output_path}")
    print(f"[mesh] ori: {output_path.with_suffix('.ori')}")
    print(f"[mesh] eld: {output_path.with_suffix('.eld')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
