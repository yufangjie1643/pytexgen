"""Benchmark model-level multiprocessing for modern weave voxelization."""

from __future__ import annotations

import argparse
import os
import sys
import threading
import time

import psutil

sys.path.insert(0, "src")

from pytexgen.modern import PlainWeave2D, voxelize_models_data


def make_pattern(index: int, width: int, height: int):
    seed = (index + 1) * 0x9E3779B97F4A7C15
    rows = []
    for y in range(height):
        row = []
        for x in range(width):
            bit = ((seed >> ((x + y * width) % 61)) ^ (x * 17 + y * 31 + index)) & 1
            row.append("x" if bit else "y")
        rows.append(tuple(row))
    return tuple(rows)


def make_model(index: int) -> PlainWeave2D:
    width = 3 + (index % 4)
    height = 3 + ((index // 4) % 4)
    spacing = 0.75 + (index % 1000) * 0.0005
    thickness = 0.14 + ((index * 17) % 2000) * 0.00004
    yarn_width = spacing * (0.55 + ((index * 19) % 350) * 0.001)
    yarn_height = thickness * (0.35 + ((index * 23) % 250) * 0.0008)
    model = PlainWeave2D(
        width=width,
        height=height,
        spacing=spacing,
        thickness=thickness,
        yarn_width=yarn_width,
        yarn_height=yarn_height,
    )
    for y, row in enumerate(make_pattern(index, width, height)):
        for x, yarn in enumerate(row):
            if yarn == "x":
                model.swap_position(x, y)
    return model


class CpuMonitor:
    def __init__(self, interval=0.5):
        self.interval = interval
        self.stop = threading.Event()
        self.proc = psutil.Process(os.getpid())
        self.process_cpu = []
        self.global_cpu = []
        self.children = []
        self.prev_times = {}

    def run(self):
        self.proc.cpu_percent(None)
        psutil.cpu_percent(None)
        while not self.stop.wait(self.interval):
            procs = [self.proc] + self.proc.children(recursive=True)
            cpu_seconds = 0.0
            live = 0
            for proc in procs:
                try:
                    times = proc.cpu_times()
                    current = times.user + times.system
                    previous = self.prev_times.get(proc.pid, current)
                    self.prev_times[proc.pid] = current
                    cpu_seconds += max(0.0, current - previous)
                    live += 1
                except psutil.Error:
                    pass
            self.process_cpu.append(cpu_seconds / self.interval * 100.0)
            self.global_cpu.append(psutil.cpu_percent(None))
            self.children.append(max(0, live - 1))

    def __enter__(self):
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.stop.set()
        self.thread.join()

    def summary(self):
        return {
            "samples": len(self.process_cpu),
            "avg_process_cores": (sum(self.process_cpu) / len(self.process_cpu) / 100.0) if self.process_cpu else 0.0,
            "max_process_cores": (max(self.process_cpu) / 100.0) if self.process_cpu else 0.0,
            "avg_global_cpu_percent": sum(self.global_cpu) / len(self.global_cpu) if self.global_cpu else 0.0,
            "max_global_cpu_percent": max(self.global_cpu) if self.global_cpu else 0.0,
            "max_children": max(self.children) if self.children else 0,
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=4000)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--resolution", type=int, default=64)
    parser.add_argument(
        "--return-data",
        action="store_true",
        help="Return full VoxelGridData objects instead of lightweight summaries.",
    )
    args = parser.parse_args()
    count = args.count
    workers = args.workers
    resolution = (args.resolution, args.resolution, args.resolution)
    print(f"python={sys.executable}", flush=True)
    print(f"count={count} workers={workers} resolution={resolution}", flush=True)
    build_start = time.perf_counter()
    models = [make_model(index) for index in range(count)]
    build_s = time.perf_counter() - build_start
    print(f"model_build_parent_s={build_s:.3f}", flush=True)
    start = time.perf_counter()
    with CpuMonitor() as monitor:
        results = voxelize_models_data(
            models,
            resolution=resolution,
            backend="numpy",
            workers=workers,
            inner_workers=1,
            return_data=args.return_data,
            chunksize="auto",
        )
    elapsed = time.perf_counter() - start
    if args.return_data:
        checksum = sum(int(result.yarn_id.astype("int64", copy=False).sum()) for result in results)
        occupied = sum(int((result.yarn_id >= 0).sum()) for result in results)
    else:
        checksum = sum(result.checksum for result in results)
        occupied = sum(result.occupied for result in results)
    print(f"total_s={elapsed:.3f}", flush=True)
    print(f"per_model_ms={elapsed / count * 1000:.3f}", flush=True)
    print(f"models_per_s={count / elapsed:.2f}", flush=True)
    print(f"checksum={checksum} occupied={occupied}", flush=True)
    for key, value in monitor.summary().items():
        print(f"{key}={value:.3f}" if isinstance(value, float) else f"{key}={value}", flush=True)


if __name__ == "__main__":
    main()
