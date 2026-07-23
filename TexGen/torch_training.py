"""Optional PyTorch DataLoader and main-process device prefetch adapters."""

from __future__ import annotations

import multiprocessing
import time
from dataclasses import dataclass
from numbers import Integral
from typing import Any, Optional

from .training_data import (
    RaggedArray,
    SimulationBatch,
    as_torch_batch,
    collate_training_examples,
)


def _import_torch():
    try:
        import torch
    except ImportError as exc:
        raise ImportError(
            'PyTorch is required; install with `pip install "pytexgen[gpu]"`.'
        ) from exc
    return torch


@dataclass(frozen=True)
class _TorchCollator:
    schema: Any
    input_names: tuple
    target_names: tuple

    def __call__(self, examples):
        numpy_batch = collate_training_examples(
            examples,
            self.schema,
            input_names=self.input_names,
            target_names=self.target_names,
        )
        return as_torch_batch(numpy_batch)


def make_simulation_dataloader(
    dataset: Any,
    *,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    pin_memory: bool,
    persistent_workers: Optional[bool] = None,
    prefetch_factor: Optional[int] = None,
    seed: int = 0,
    drop_last: bool = False,
):
    """Build a deterministic DataLoader without initializing CUDA workers."""
    torch = _import_torch()
    if (
        isinstance(batch_size, bool)
        or not isinstance(batch_size, Integral)
        or int(batch_size) <= 0
    ):
        raise ValueError("batch_size must be a positive integer")
    if (
        isinstance(num_workers, bool)
        or not isinstance(num_workers, Integral)
        or int(num_workers) < 0
    ):
        raise ValueError("num_workers must be a non-negative integer")
    if not isinstance(shuffle, bool):
        raise ValueError("shuffle must be Boolean")
    if not isinstance(pin_memory, bool):
        raise ValueError("pin_memory must be Boolean")
    if not isinstance(drop_last, bool):
        raise ValueError("drop_last must be Boolean")
    if isinstance(seed, bool) or not isinstance(seed, Integral):
        raise ValueError("seed must be an integer")
    worker_count = int(num_workers)
    if persistent_workers is None:
        persistent = worker_count > 0
    elif not isinstance(persistent_workers, bool):
        raise ValueError("persistent_workers must be Boolean or None")
    else:
        persistent = persistent_workers
    if worker_count == 0 and persistent:
        raise ValueError(
            "persistent_workers requires num_workers greater than zero"
        )
    if prefetch_factor is not None:
        if (
            isinstance(prefetch_factor, bool)
            or not isinstance(prefetch_factor, Integral)
            or int(prefetch_factor) <= 0
        ):
            raise ValueError(
                "prefetch_factor must be a positive integer or None"
            )
        if worker_count == 0:
            raise ValueError(
                "prefetch_factor requires num_workers greater than zero"
            )
    for attribute in ("schema", "input_names", "target_names"):
        if not hasattr(dataset, attribute):
            raise TypeError(
                "dataset must expose schema and selected field names"
            )

    generator = torch.Generator()
    generator.manual_seed(int(seed))
    kwargs = {
        "dataset": dataset,
        "batch_size": int(batch_size),
        "shuffle": shuffle,
        "num_workers": worker_count,
        "pin_memory": pin_memory,
        "drop_last": drop_last,
        "persistent_workers": persistent,
        "generator": generator,
        "collate_fn": _TorchCollator(
            dataset.schema,
            tuple(dataset.input_names),
            tuple(dataset.target_names),
        ),
    }
    if worker_count > 0:
        kwargs["multiprocessing_context"] = "spawn"
        if prefetch_factor is not None:
            kwargs["prefetch_factor"] = int(prefetch_factor)
    return torch.utils.data.DataLoader(**kwargs)


def _record_batch_stream(
    batch: SimulationBatch,
    stream: Any,
    torch: Any,
) -> int:
    seen = set()
    count = 0

    def record(value):
        nonlocal count
        if isinstance(value, RaggedArray):
            record(value.values)
            record(value.offsets)
            return
        if isinstance(value, dict) or hasattr(value, "items"):
            for child in value.values():
                record(child)
            return
        if not torch.is_tensor(value) or id(value) in seen:
            return
        seen.add(id(value))
        value.record_stream(stream)
        count += 1

    record(batch.inputs)
    record(batch.targets)
    return count


class CudaPrefetcher:
    """Overlap selected pinned-batch H2D transfer with model computation."""

    def __init__(
        self,
        loader: Any,
        *,
        device: Any,
        transform: Any = None,
    ) -> None:
        self.loader = loader
        self.device = device
        self.transform = transform
        self.transferred_bytes = 0
        self.wait_seconds = 0.0
        self.recorded_tensors = 0
        self.stream = None

    def __iter__(self):
        torch = _import_torch()
        device = torch.device(self.device)
        self.transferred_bytes = 0
        self.wait_seconds = 0.0
        self.recorded_tensors = 0
        self.stream = None
        if device.type != "cuda":
            for batch in self.loader:
                result = (
                    self.transform(batch)
                    if self.transform is not None
                    else batch
                )
                if not isinstance(result, SimulationBatch):
                    raise TypeError(
                        "prefetch transform must return SimulationBatch"
                    )
                yield result
            return
        if multiprocessing.current_process().name != "MainProcess":
            raise RuntimeError(
                "CUDA prefetch must run in the main process"
            )

        self.stream = torch.cuda.Stream(device=device)
        iterator = iter(self.loader)

        def preload():
            try:
                cpu_batch = next(iterator)
            except StopIteration:
                return None
            if not isinstance(cpu_batch, SimulationBatch):
                raise TypeError(
                    "loader must yield SimulationBatch values"
                )
            logical_bytes = cpu_batch.nbytes
            with torch.cuda.stream(self.stream):
                batch = cpu_batch.to(
                    device, non_blocking=True
                )
                if self.transform is not None:
                    batch = self.transform(batch)
                    if not isinstance(batch, SimulationBatch):
                        raise TypeError(
                            "prefetch transform must return "
                            "SimulationBatch"
                        )
                event = torch.cuda.Event(blocking=False)
                event.record(self.stream)
            self.transferred_bytes += logical_bytes
            return batch, event

        pending = preload()
        while pending is not None:
            batch, event = pending
            current = torch.cuda.current_stream(device=device)
            wait_start = time.perf_counter()
            current.wait_event(event)
            self.wait_seconds += time.perf_counter() - wait_start
            pending = preload()
            self.recorded_tensors += _record_batch_stream(
                batch, current, torch
            )
            yield batch


__all__ = ["CudaPrefetcher", "make_simulation_dataloader"]
