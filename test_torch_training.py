"""Tests for optional PyTorch loading and CUDA prefetch."""

import builtins
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from test_training_io import publish_dataset

from TexGen.training_data import RaggedArray, SimulationBatch
from TexGen.training_io import SimulationDataset
from TexGen.torch_training import (
    CudaPrefetcher,
    make_simulation_dataloader,
)

try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "PyTorch is not installed")
class TorchDataLoaderTest(unittest.TestCase):
    def make_dataset(self, target):
        publish_dataset(target)
        return SimulationDataset(
            target,
            split="train",
            inputs=(
                "voxel.material_id",
                "orientation.primary",
            ),
            targets=("effective_c21",),
            verify="manifest",
        )

    def test_loader_collates_selected_fields_with_zero_and_two_workers(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "dataset"
            dataset = self.make_dataset(target)
            loaders = [
                make_simulation_dataloader(
                    dataset,
                    batch_size=2,
                    shuffle=False,
                    num_workers=workers,
                    pin_memory=False,
                    seed=11,
                )
                for workers in (0, 2)
            ]

            batches = [next(iter(loader)) for loader in loaders]

            self.assertEqual(
                loaders[1].multiprocessing_context.get_start_method(),
                "spawn",
            )
            for batch in batches:
                self.assertIsInstance(batch, SimulationBatch)
                self.assertEqual(
                    set(batch.inputs),
                    {"voxel.material_id", "orientation.primary"},
                )
                self.assertEqual(
                    set(batch.targets), {"effective_c21"}
                )
                self.assertIsInstance(
                    batch.inputs["voxel.material_id"], torch.Tensor
                )
                self.assertEqual(
                    batch.inputs["voxel.material_id"].dtype,
                    torch.int32,
                )
                self.assertIsInstance(
                    batch.inputs["orientation.primary"],
                    RaggedArray,
                )
                self.assertEqual(batch.sample_ids, ("s0", "s2"))
            torch.testing.assert_close(
                batches[0].inputs["voxel.material_id"],
                batches[1].inputs["voxel.material_id"],
            )
            torch.testing.assert_close(
                batches[0].inputs["orientation.primary"].values,
                batches[1].inputs["orientation.primary"].values,
            )
            torch.testing.assert_close(
                batches[0].targets["effective_c21"],
                batches[1].targets["effective_c21"],
            )

    def test_shuffle_is_reproducible_for_same_seed(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "dataset"
            dataset = self.make_dataset(target)

            def order(seed):
                loader = make_simulation_dataloader(
                    dataset,
                    batch_size=1,
                    shuffle=True,
                    num_workers=0,
                    pin_memory=False,
                    seed=seed,
                )
                return tuple(
                    batch.sample_ids[0] for batch in loader
                )

            self.assertEqual(order(17), order(17))
            self.assertEqual(set(order(17)), {"s0", "s2", "s4"})

    def test_worker_options_are_validated_without_touching_cuda(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "dataset"
            dataset = self.make_dataset(target)
            loader = make_simulation_dataloader(
                dataset,
                batch_size=2,
                shuffle=False,
                num_workers=0,
                pin_memory=False,
                persistent_workers=None,
                prefetch_factor=None,
            )
            self.assertFalse(loader.persistent_workers)
            with self.assertRaisesRegex(ValueError, "prefetch_factor"):
                make_simulation_dataloader(
                    dataset,
                    batch_size=2,
                    shuffle=False,
                    num_workers=0,
                    pin_memory=False,
                    prefetch_factor=2,
                )
            with self.assertRaisesRegex(ValueError, "num_workers"):
                make_simulation_dataloader(
                    dataset,
                    batch_size=2,
                    shuffle=False,
                    num_workers=-1,
                    pin_memory=False,
                )

    def test_torch_import_error_names_gpu_extra(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "dataset"
            dataset = self.make_dataset(target)
            original_import = builtins.__import__

            def reject_torch(name, *args, **kwargs):
                if name == "torch":
                    raise ImportError("injected missing torch")
                return original_import(name, *args, **kwargs)

            with mock.patch(
                "builtins.__import__", side_effect=reject_torch
            ):
                with self.assertRaisesRegex(ImportError, "pytexgen\\[gpu\\]"):
                    make_simulation_dataloader(
                        dataset,
                        batch_size=1,
                        shuffle=False,
                        num_workers=0,
                        pin_memory=False,
                    )


@unittest.skipIf(torch is None, "PyTorch is not installed")
class PrefetcherCpuTest(unittest.TestCase):
    def test_cpu_fallback_preserves_batches_without_transfer_accounting(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "dataset"
            publish_dataset(target)
            dataset = SimulationDataset(
                target,
                split="train",
                inputs=("voxel.material_id",),
                targets=("effective_c21",),
                verify="manifest",
            )
            loader = make_simulation_dataloader(
                dataset,
                batch_size=2,
                shuffle=False,
                num_workers=0,
                pin_memory=False,
            )
            prefetcher = CudaPrefetcher(loader, device="cpu")

            batches = list(prefetcher)

            self.assertEqual(
                tuple(
                    sample_id
                    for batch in batches
                    for sample_id in batch.sample_ids
                ),
                ("s0", "s2", "s4"),
            )
            self.assertTrue(
                all(
                    value.device.type == "cpu"
                    for batch in batches
                    for value in (
                        batch.inputs["voxel.material_id"],
                        batch.targets["effective_c21"],
                    )
                )
            )
            self.assertEqual(prefetcher.transferred_bytes, 0)
            self.assertEqual(prefetcher.recorded_tensors, 0)
            self.assertGreaterEqual(prefetcher.wait_seconds, 0.0)


@unittest.skipUnless(
    torch is not None and torch.cuda.is_available(),
    "CUDA is required for asynchronous prefetch tests",
)
class CudaPrefetcherTest(unittest.TestCase):
    def test_prefetch_matches_sync_transfer_and_accounts_selected_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "dataset"
            publish_dataset(target)
            dataset = SimulationDataset(
                target,
                split="train",
                inputs=(
                    "voxel.material_id",
                    "orientation.primary",
                ),
                targets=("effective_c21",),
                verify="manifest",
            )
            loader = make_simulation_dataloader(
                dataset,
                batch_size=2,
                shuffle=False,
                num_workers=0,
                pin_memory=True,
            )
            cpu_batches = list(loader)
            expected_bytes = sum(batch.nbytes for batch in cpu_batches)
            prefetcher = CudaPrefetcher(loader, device="cuda")

            gpu_batches = list(prefetcher)
            torch.cuda.synchronize()

            self.assertEqual(
                prefetcher.transferred_bytes, expected_bytes
            )
            self.assertGreater(prefetcher.recorded_tensors, 0)
            self.assertIsNotNone(prefetcher.stream)
            self.assertNotEqual(
                prefetcher.stream.cuda_stream,
                torch.cuda.current_stream().cuda_stream,
            )
            for cpu_batch, gpu_batch in zip(
                cpu_batches, gpu_batches
            ):
                self.assertEqual(
                    gpu_batch.inputs["voxel.material_id"].device.type,
                    "cuda",
                )
                torch.testing.assert_close(
                    gpu_batch.inputs["voxel.material_id"].cpu(),
                    cpu_batch.inputs["voxel.material_id"],
                )
                torch.testing.assert_close(
                    gpu_batch.inputs["orientation.primary"].values.cpu(),
                    cpu_batch.inputs["orientation.primary"].values,
                )
                torch.testing.assert_close(
                    gpu_batch.targets["effective_c21"].cpu(),
                    cpu_batch.targets["effective_c21"],
                )

            tensor = gpu_batches[0].targets["effective_c21"]
            shared = torch.utils.dlpack.from_dlpack(
                torch.utils.dlpack.to_dlpack(tensor)
            )
            self.assertEqual(shared.data_ptr(), tensor.data_ptr())

    def test_device_transform_runs_on_prefetch_stream(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "dataset"
            publish_dataset(target)
            dataset = SimulationDataset(
                target,
                split="train",
                inputs=("voxel.material_id",),
                targets=("effective_c21",),
                verify="manifest",
            )
            loader = make_simulation_dataloader(
                dataset,
                batch_size=2,
                shuffle=False,
                num_workers=0,
                pin_memory=True,
            )
            observed_streams = []

            def transform(batch):
                observed_streams.append(
                    torch.cuda.current_stream().cuda_stream
                )
                return batch

            prefetcher = CudaPrefetcher(
                loader, device="cuda", transform=transform
            )
            list(prefetcher)

            self.assertTrue(observed_streams)
            self.assertEqual(
                set(observed_streams),
                {prefetcher.stream.cuda_stream},
            )


if __name__ == "__main__":
    unittest.main()
