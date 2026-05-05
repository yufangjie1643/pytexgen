"""SiC/SiC RVE 并行体素导出基准测试脚本。

默认测试:
    uv run python script/bench_sic_sic_rve_parallel.py

默认会用 4 个并行 worker 生成 64 个单层 RVE，每个网格为 64x64x64。
每个 case 完成后默认删除 .inp/.ori/.eld 等大文件，只保留 build/ 下的
progress.json 和 summary.json 小文件，方便文章其他设备复现实测速度。

常用参数:
    uv run python script/bench_sic_sic_rve_parallel.py --cases 8 --workers 2
    uv run python script/bench_sic_sic_rve_parallel.py --resolution 128 128 128 --cases 1
    uv run python script/bench_sic_sic_rve_parallel.py --keep-output
"""

import argparse
import concurrent.futures
import json
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_DIR = SCRIPT_DIR.parent
MODEL_SCRIPT = SCRIPT_DIR / "sic_sic_shallow_cross_straight.py"


def _normalise_resolution(values):
    if len(values) == 1:
        value = int(values[0])
        return value, value, value
    if len(values) == 3:
        return tuple(int(value) for value in values)
    raise ValueError("--resolution must contain one value or three values")


def _assert_child(path, parent):
    resolved_path = Path(path).resolve()
    resolved_parent = Path(parent).resolve()
    try:
        resolved_path.relative_to(resolved_parent)
    except ValueError as exc:
        raise RuntimeError(f"refusing to delete outside {resolved_parent}: {resolved_path}") from exc
    return resolved_path


def _file_summary(root):
    files = []
    total_bytes = 0
    if not root.exists():
        return files, total_bytes
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        size = path.stat().st_size
        total_bytes += size
        files.append({"name": path.name, "bytes": size})
    return files, total_bytes


def _read_excerpt(path, limit=4000):
    if not path.exists():
        return None
    text = path.read_text(encoding="utf-8", errors="replace")
    return text[:limit]


def _run_case(index, args, resolution, data_root, python_exe):
    case_name = f"case_{index:02d}"
    case_dir = data_root / case_name
    case_dir.mkdir(parents=True, exist_ok=True)

    prefix = f"sic_sic_bench_{resolution[0]}x{resolution[1]}x{resolution[2]}_case{index:02d}"
    params_path = case_dir / "params.json"
    log_path = case_dir / "run.log"

    params = {
        "save_tg3": False,
        "rve_export": {
            "enabled": True,
            "save_dir": str(case_dir),
            "file_prefix": prefix,
            "layer_count": args.layer_count,
            "layers": [args.layer],
            "layers_per_rve": args.layers_per_rve,
            "save_tg3": args.save_tg3,
            "mesh_boundary": args.mesh_boundary,
            "mesh_resolutions": [list(resolution)],
        },
    }
    params_path.write_text(json.dumps(params, ensure_ascii=False, indent=2), encoding="utf-8")

    started = time.perf_counter()
    exit_code = 0
    error = None
    try:
        with log_path.open("w", encoding="utf-8", errors="replace") as log:
            proc = subprocess.run(
                [python_exe, str(MODEL_SCRIPT), f"@{params_path}"],
                cwd=REPO_DIR,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
            )
        exit_code = int(proc.returncode)
    except Exception as exc:  # Keep benchmark summary robust.
        exit_code = 999
        error = str(exc)
        with log_path.open("a", encoding="utf-8", errors="replace") as log:
            log.write(f"\n{error}\n")
    seconds = time.perf_counter() - started

    files, total_bytes = _file_summary(case_dir)
    log_excerpt = _read_excerpt(log_path) if exit_code else None

    if args.cleanup:
        resolved_case = _assert_child(case_dir, data_root)
        shutil.rmtree(resolved_case, ignore_errors=False)

    return {
        "index": index,
        "exit_code": exit_code,
        "seconds": round(seconds, 3),
        "bytes": total_bytes,
        "error": error,
        "log_excerpt": log_excerpt,
        "files": files,
    }


def _write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _summarise(args, resolution, root, data_root, results, wall_seconds, python_exe):
    case_seconds = [float(item["seconds"]) for item in results]
    total_bytes = sum(int(item["bytes"]) for item in results)
    succeeded = sum(1 for item in results if item["exit_code"] == 0)
    failed = len(results) - succeeded
    return {
        "case_count": args.cases,
        "max_parallel": args.workers,
        "resolution": list(resolution),
        "layer": args.layer,
        "layers_per_rve": args.layers_per_rve,
        "wall_seconds": round(wall_seconds, 3),
        "succeeded": succeeded,
        "failed": failed,
        "total_generated_bytes_before_cleanup": total_bytes,
        "total_generated_mb_before_cleanup": round(total_bytes / (1024 * 1024), 2),
        "average_case_seconds": round(sum(case_seconds) / len(case_seconds), 3) if case_seconds else 0.0,
        "min_case_seconds": round(min(case_seconds), 3) if case_seconds else 0.0,
        "max_case_seconds": round(max(case_seconds), 3) if case_seconds else 0.0,
        "cases_per_minute": round((args.cases / wall_seconds) * 60.0, 3) if wall_seconds > 0 else 0.0,
        "cleanup_generated_data": bool(args.cleanup),
        "root": str(root),
        "data_root": str(data_root),
        "python": python_exe,
        "model_script": str(MODEL_SCRIPT),
        "generated_at": datetime.now().astimezone().isoformat(),
        "results": sorted(results, key=lambda item: item["index"]),
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Benchmark parallel SiC/SiC RVE voxel export")
    parser.add_argument("--cases", type=int, default=64, help="number of RVE exports to run (default: 64)")
    parser.add_argument("--workers", type=int, default=4, help="parallel worker count (default: 4)")
    parser.add_argument(
        "--resolution",
        type=int,
        nargs="+",
        default=[64, 64, 64],
        help="one value for cubic mesh or three values nx ny nz (default: 64 64 64)",
    )
    parser.add_argument("--layer", type=int, default=0, help="RVE layer index to export (default: 0)")
    parser.add_argument("--layer-count", type=int, default=5, help="total layer count used for z windows (default: 5)")
    parser.add_argument("--layers-per-rve", type=int, default=1, help="number of layers per RVE window (default: 1)")
    parser.add_argument("--mesh-boundary", default="CPeriodicBoundaries", help="TexGen mesh boundary class name")
    parser.add_argument("--save-tg3", action="store_true", help="also save TG3 files for each case")
    parser.add_argument("--root", default=None, help="benchmark output root (default: build/sic_rve_bench_...)")
    parser.add_argument("--python", default=sys.executable, help="Python executable used to run the model script")
    parser.add_argument("--keep-output", dest="cleanup", action="store_false", help="keep generated case data")
    parser.set_defaults(cleanup=True)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.cases < 1:
        raise ValueError("--cases must be >= 1")
    if args.workers < 1:
        raise ValueError("--workers must be >= 1")

    resolution = _normalise_resolution(args.resolution)
    root = Path(args.root) if args.root else REPO_DIR / "build" / (
        f"sic_rve_bench_{resolution[0]}x{resolution[1]}x{resolution[2]}_{args.workers}parallel"
    )
    if not root.is_absolute():
        root = REPO_DIR / root
    data_root = root / "data"
    progress_path = root / "progress.json"
    summary_path = root / "summary.json"

    if root.exists():
        shutil.rmtree(_assert_child(root, REPO_DIR), ignore_errors=False)
    data_root.mkdir(parents=True, exist_ok=True)

    print(
        "Benchmark: "
        f"cases={args.cases}, workers={args.workers}, resolution={resolution}, cleanup={args.cleanup}"
    )

    started = time.perf_counter()
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [
            executor.submit(_run_case, index, args, resolution, data_root, args.python)
            for index in range(args.cases)
        ]
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            results.append(result)
            elapsed = time.perf_counter() - started
            succeeded = sum(1 for item in results if item["exit_code"] == 0)
            failed = len(results) - succeeded
            progress = {
                "completed": len(results),
                "total": args.cases,
                "succeeded": succeeded,
                "failed": failed,
                "max_parallel": args.workers,
                "elapsed_seconds": round(elapsed, 3),
                "updated_at": datetime.now().astimezone().isoformat(),
            }
            _write_json(progress_path, progress)
            print(
                f"[{len(results):>{len(str(args.cases))}}/{args.cases}] "
                f"case {result['index']:02d}: exit={result['exit_code']} "
                f"time={result['seconds']}s"
            )

    wall_seconds = time.perf_counter() - started
    summary = _summarise(args, resolution, root, data_root, results, wall_seconds, args.python)
    _write_json(summary_path, summary)

    if args.cleanup and data_root.exists():
        shutil.rmtree(_assert_child(data_root, root), ignore_errors=False)

    keys = (
        "case_count",
        "max_parallel",
        "resolution",
        "wall_seconds",
        "succeeded",
        "failed",
        "total_generated_mb_before_cleanup",
        "average_case_seconds",
        "min_case_seconds",
        "max_case_seconds",
        "cases_per_minute",
        "cleanup_generated_data",
    )
    print(json.dumps({key: summary[key] for key in keys}, ensure_ascii=False, indent=2))
    print(f"Summary: {summary_path}")
    return summary


if __name__ == "__main__":
    main()
