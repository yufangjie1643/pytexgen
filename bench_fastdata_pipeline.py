"""Phase benchmark for the pytexgen fastdata pipeline.

The synthetic backend benchmark measures only the voxel classification kernel.
This script measures the real TexGen path in separate phases so changes to the
Python-C boundary can be compared against a stable baseline.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, MutableMapping, Optional

import numpy as np


def _default_prepare(context: MutableMapping[str, Any]) -> Any:
    return None


def _default_metadata(result: Any) -> Mapping[str, Any]:
    return {}


@dataclass
class BenchmarkPhase:
    """One benchmark phase with optional untimed preparation."""

    name: str
    run: Callable[[Any, MutableMapping[str, Any]], Any]
    prepare: Callable[[MutableMapping[str, Any]], Any] = _default_prepare
    metadata: Callable[[Any], Mapping[str, Any]] = _default_metadata


@dataclass
class PhaseTiming:
    """Timing result for one benchmark phase."""

    name: str
    seconds: float
    repeat: int
    samples: List[float] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "seconds": self.seconds,
            "repeat": self.repeat,
            "samples": list(self.samples),
            "metadata": dict(self.metadata),
        }


@dataclass
class BenchmarkReport:
    """Completed benchmark report."""

    records: List[PhaseTiming]
    metadata: Dict[str, Any] = field(default_factory=dict)
    context: Dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "metadata": dict(self.metadata),
            "records": [record.as_dict() for record in self.records],
        }

    def format_table(self) -> str:
        rows = ["phase                          best_s   repeat  metadata"]
        rows.append("-" * 72)
        for record in self.records:
            meta = json.dumps(record.metadata, sort_keys=True)
            rows.append(
                f"{record.name:<30} {record.seconds:>7.4f} {record.repeat:>8d}  {meta}"
            )
        return "\n".join(rows)


def run_benchmark_phases(
    phases: Iterable[BenchmarkPhase],
    *,
    repeat: int = 1,
    timer: Callable[[], float] = time.perf_counter,
    metadata: Optional[Mapping[str, Any]] = None,
) -> BenchmarkReport:
    """Run benchmark phases and keep the fastest sample for each phase."""
    if repeat < 1:
        raise ValueError("repeat must be at least 1")

    context: Dict[str, Any] = {}
    records: List[PhaseTiming] = []
    for phase in phases:
        samples: List[float] = []
        best_seconds: Optional[float] = None
        best_result: Any = None
        for _ in range(repeat):
            prepared = phase.prepare(context)
            start = timer()
            result = phase.run(prepared, context)
            seconds = timer() - start
            samples.append(seconds)
            if best_seconds is None or seconds < best_seconds:
                best_seconds = seconds
                best_result = result

        context[phase.name] = best_result
        records.append(
            PhaseTiming(
                name=phase.name,
                seconds=float(best_seconds if best_seconds is not None else 0.0),
                repeat=repeat,
                samples=samples,
                metadata=dict(phase.metadata(best_result)),
            )
        )

    return BenchmarkReport(
        records=records,
        metadata=dict(metadata or {}),
        context=context,
    )


def _make_weave2d(args: argparse.Namespace):
    import pytexgen as tg

    if args.clear_textiles:
        tg.DeleteTextiles()
    textile = tg.CTextileWeave2D(
        args.weave_width,
        args.weave_height,
        args.spacing,
        args.thickness,
        args.refine,
        True,
    )
    textile.SetYarnWidths(args.yarn_width)
    textile.SetYarnHeights(args.yarn_height)
    textile.AssignDefaultDomain()
    return textile


def _build_weave2d(args: argparse.Namespace):
    textile = _make_weave2d(args)
    textile.GetNumYarns()
    return textile


def _bundle_metadata(bundle: Any) -> Dict[str, Any]:
    return {
        "num_yarns": int(bundle.num_yarns),
        "nodes": int(bundle.positions.shape[0]),
        "section_points": int(bundle.sections.shape[0]),
        "translations": int(bundle.translations.shape[0]),
    }


def _voxel_metadata(data: Any) -> Dict[str, Any]:
    yarn_id = data.yarn_id
    if hasattr(yarn_id, "detach"):
        yarn_id = yarn_id.detach().cpu().numpy()
    yarn_id = np.asarray(yarn_id)
    return {
        "resolution": list(data.resolution),
        "occupied": int((yarn_id >= 0).sum()),
        "backend": data.backend,
        "workers": int(data.workers),
    }


def make_real_phases(args: argparse.Namespace) -> List[BenchmarkPhase]:
    import pytexgen._Core as core
    import pytexgen.gpu_voxelizer as gv

    def construct_run(_prepared: Any, _context: MutableMapping[str, Any]):
        return _make_weave2d(args)

    def build_prepare(_context: MutableMapping[str, Any]):
        return _make_weave2d(args)

    def build_run(textile: Any, _context: MutableMapping[str, Any]):
        textile.GetNumYarns()
        return textile

    def built_textile(_context: MutableMapping[str, Any]):
        return _build_weave2d(args)

    def direct_snapshot(textile: Any, _context: MutableMapping[str, Any]):
        mapping = core._fastdata_extract_snapshot_bundle_direct(textile)
        return gv.SnapshotBundle(**mapping)

    def python_snapshot(textile: Any, _context: MutableMapping[str, Any]):
        snapshots, aabb = gv.extract_snapshots(textile)
        return gv.SnapshotBundle.from_snapshots(snapshots, aabb)

    def direct_bundle(_context: MutableMapping[str, Any]):
        textile = _build_weave2d(args)
        mapping = core._fastdata_extract_snapshot_bundle_direct(textile)
        return gv.SnapshotBundle(**mapping)

    def voxelize_bundle(bundle: Any, _context: MutableMapping[str, Any]):
        return gv.voxelize_snapshot_bundle_data(
            bundle,
            nx=args.resolution,
            ny=args.resolution,
            nz=args.resolution,
            backend=args.backend,
            device=args.device,
            dtype=args.dtype,
            chunk_voxels=args.chunk_voxels,
            workers=args.workers,
            verbose=False,
            output="backend",
            aabb_pruning=not args.no_aabb_pruning,
        )

    phases = [
        BenchmarkPhase(
            "construct_assign_domain",
            run=construct_run,
            metadata=lambda textile: {
                "weave_width": args.weave_width,
                "weave_height": args.weave_height,
                "refine": bool(args.refine),
            },
        ),
        BenchmarkPhase(
            "build_refine",
            prepare=build_prepare,
            run=build_run,
            metadata=lambda textile: {"num_yarns": int(textile.GetNumYarns())},
        ),
        BenchmarkPhase(
            "snapshot_direct_core",
            prepare=built_textile,
            run=direct_snapshot,
            metadata=_bundle_metadata,
        ),
    ]
    if not args.skip_python_fallback:
        phases.append(
            BenchmarkPhase(
                "snapshot_python_fallback",
                prepare=built_textile,
                run=python_snapshot,
                metadata=_bundle_metadata,
            )
        )
    if not args.skip_voxel:
        phases.append(
            BenchmarkPhase(
                "voxel_numpy_from_direct",
                prepare=direct_bundle,
                run=voxelize_bundle,
                metadata=_voxel_metadata,
            )
        )
    return phases


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weave-width", type=int, default=2)
    parser.add_argument("--weave-height", type=int, default=2)
    parser.add_argument("--spacing", type=float, default=1.0)
    parser.add_argument("--thickness", type=float, default=0.2)
    parser.add_argument("--yarn-width", type=float, default=0.8)
    parser.add_argument("--yarn-height", type=float, default=0.1)
    parser.add_argument("--refine", action="store_true", help="Enable TexGen weave refinement")
    parser.add_argument("--resolution", type=int, default=16)
    parser.add_argument("--backend", choices=["numpy", "torch", "auto"], default="numpy")
    parser.add_argument("--device", default=None)
    parser.add_argument("--dtype", choices=["float32", "float64"], default="float32")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--chunk-voxels", type=int, default=8192)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--skip-python-fallback", action="store_true")
    parser.add_argument("--skip-voxel", action="store_true")
    parser.add_argument("--no-aabb-pruning", action="store_true")
    parser.add_argument("--no-clear-textiles", dest="clear_textiles", action="store_false")
    parser.set_defaults(clear_textiles=True)
    parser.add_argument("--json-out", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    phases = make_real_phases(args)
    report = run_benchmark_phases(
        phases,
        repeat=args.repeat,
        metadata={
            "weave_width": args.weave_width,
            "weave_height": args.weave_height,
            "resolution": args.resolution,
            "backend": args.backend,
            "dtype": args.dtype,
            "refine": bool(args.refine),
        },
    )
    print(report.format_table())
    if args.json_out is not None:
        args.json_out.write_text(json.dumps(report.as_dict(), indent=2), encoding="utf-8")
        print(f"wrote {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
