"""End-to-end integration coverage for the GPU training data workflow."""

from pathlib import Path
import subprocess
import tempfile
import time
import unittest

import numpy as np

import pytexgen as tg
from pytexgen.acdm_solver import solve_acdm_anisotropic_from_sample
from pytexgen.material_fields import (
    isotropic_stiffness_c21,
    pack_voigt_c21,
)
from pytexgen.simulation_sample import (
    MaterialTable,
    voxelize_textile_simulation_sample,
)

from TexGen.training_data import (
    DatasetQualityPolicy,
    TrainingDatasetSchema,
    TrainingFieldSpec,
    VOXEL_ORDER,
)
from TexGen.training_io import (
    SimulationDataset,
    SimulationDatasetWriter,
)
from TexGen.torch_training import (
    CudaPrefetcher,
    make_simulation_dataloader,
)

try:
    import torch
except ImportError:
    torch = None


REPOSITORY_ROOT = Path(__file__).resolve().parent


def build_real_sample(*, width=0.78):
    textile = tg.CTextileWeave2D(2, 2, 1.0, 0.2, False, True)
    for y_index in range(2):
        for x_index in range(2):
            if (x_index + y_index) % 2 == 0:
                textile.SwapPosition(x_index, y_index)
    textile.SetYarnWidths(width)
    textile.SetYarnHeights(0.1)
    textile.SetResolution(10)
    textile.AssignDefaultDomain()
    materials = MaterialTable(
        c21=np.stack(
            (
                isotropic_stiffness_c21(3.0, 0.35),
                isotropic_stiffness_c21(70.0, 0.20),
            )
        ),
        material_ids=np.array([0, 7], dtype=np.int32),
        unit="GPa",
        names=("matrix", "yarn"),
    )
    return voxelize_textile_simulation_sample(
        textile,
        materials=materials,
        default_yarn_material_id=7,
        metadata={"width": width},
        nx=4,
        ny=4,
        nz=4,
        backend="numpy",
        workers=1,
        chunk_voxels=64,
        dtype="float64",
        aabb_pruning=True,
        verbose=False,
    )


def training_schema(*, shard_size=2):
    return TrainingDatasetSchema(
        inputs=(
            TrainingFieldSpec(
                "voxel.material_id",
                "input",
                "fixed",
                "int32",
                (4, 4, 4),
                semantic="material_id_grid",
            ),
            TrainingFieldSpec(
                "orientation.voxel_indices",
                "input",
                "ragged",
                "int64",
                (),
                semantic="flat_voxel_index",
                ragged_group="yarn_voxels",
            ),
            TrainingFieldSpec(
                "orientation.primary",
                "input",
                "ragged",
                "float64",
                (3,),
                semantic="direction_vector",
                ragged_group="yarn_voxels",
            ),
            TrainingFieldSpec(
                "material.ids",
                "input",
                "ragged",
                "int32",
                (),
                semantic="material_id",
                ragged_group="materials",
            ),
            TrainingFieldSpec(
                "material.c21",
                "input",
                "ragged",
                "float64",
                (21,),
                "GPa",
                "local_engineering_voigt_c21",
                "materials",
            ),
        ),
        targets=(
            TrainingFieldSpec(
                "effective_c21",
                "target",
                "fixed",
                "float64",
                (21,),
                "GPa",
                "engineering_voigt_c21",
            ),
        ),
        grid_shape=(4, 4, 4),
        voxel_order=VOXEL_ORDER,
        shard_size=shard_size,
        statistics_fields=("effective_c21",),
    )


def solver_provenance(*, arithmetic_dtype="float64"):
    return {
        "solver_commit": "integration-fixture",
        "element_formulation": "periodic-c3d8",
        "arithmetic_dtype": arithmetic_dtype,
        "tolerance": 1e-8,
        "maximum_residual": 1e-10,
        "iteration_count": 5,
        "wall_time_seconds": 0.01,
        "target_units": {"effective_c21": "GPa"},
    }


class TrainingPipelinePackagingTest(unittest.TestCase):
    def test_training_modules_are_installed_with_python_package(self):
        cmake = (
            REPOSITORY_ROOT / "Python" / "CMakeLists.txt"
        ).read_text(encoding="utf-8")

        for module in (
            "training_data.py",
            "training_io.py",
            "torch_training.py",
        ):
            with self.subTest(module=module):
                self.assertIn(
                    f'${{CMAKE_CURRENT_SOURCE_DIR}}/../TexGen/{module}',
                    cmake,
                )


class PublicPackageInteroperabilityTest(unittest.TestCase):
    def test_writer_accepts_sample_from_installed_public_namespace(self):
        sample = build_real_sample()
        target_c21 = isotropic_stiffness_c21(12.0, 0.22)

        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "dataset"
            writer = SimulationDatasetWriter.create(
                target,
                schema=training_schema(),
                generation={"source": "real TexGen geometry"},
            )
            writer.append(
                sample,
                targets={"effective_c21": target_c21},
                sample_id="real-plain-weave",
                group_id="geometry-real-plain-weave",
                split="train",
                provenance=solver_provenance(
                    arithmetic_dtype="float32"
                ),
            )
            writer.finalize()

            dataset = SimulationDataset(
                target,
                split="train",
                inputs=("voxel.material_id", "orientation.primary"),
                targets=("effective_c21",),
                verify="sample",
            )
            example = dataset[0]

        self.assertEqual(example.sample_id, "real-plain-weave")
        self.assertEqual(
            example.inputs["voxel.material_id"].shape,
            (4, 4, 4),
        )
        self.assertGreater(
            example.inputs["orientation.primary"].values.shape[0],
            0,
        )
        self.assertEqual(
            example.metadata["provenance"]["arithmetic_dtype"],
            "float32",
        )


@unittest.skipUnless(
    torch is not None and torch.cuda.is_available(),
    "CUDA PyTorch is required for the real training-loop integration test",
)
class RealCudaTrainingLoopTest(unittest.TestCase):
    def test_real_texgen_samples_train_through_two_worker_cuda_pipeline(self):
        labels = (
            isotropic_stiffness_c21(11.0, 0.22),
            isotropic_stiffness_c21(13.0, 0.24),
        )
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "dataset"
            with SimulationDatasetWriter.create(
                target,
                schema=training_schema(shard_size=1),
                generation={
                    "geometry": "TexGen CTextileWeave2D",
                    "resolution": [4, 4, 4],
                },
            ) as writer:
                for index, (width, label) in enumerate(
                    zip((0.58, 0.70), labels)
                ):
                    writer.append(
                        build_real_sample(width=width),
                        targets={"effective_c21": label},
                        sample_id=f"real-{index}",
                        group_id=f"geometry-{index}",
                        split="train",
                        provenance=solver_provenance(
                            arithmetic_dtype="float32"
                        ),
                    )

            dataset = SimulationDataset(
                target,
                split="train",
                inputs=("voxel.material_id",),
                targets=("effective_c21",),
                verify="sample",
            )
            loader = make_simulation_dataloader(
                dataset,
                batch_size=2,
                shuffle=False,
                num_workers=2,
                pin_memory=True,
                persistent_workers=False,
                prefetch_factor=2,
                seed=17,
            )
            prefetcher = CudaPrefetcher(loader, device="cuda")
            model = torch.nn.Sequential(
                torch.nn.Conv3d(1, 4, kernel_size=3, padding=1),
                torch.nn.ReLU(),
                torch.nn.AdaptiveAvgPool3d(1),
                torch.nn.Flatten(),
                torch.nn.Linear(4, 21),
            ).cuda()
            optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)

            batch = next(iter(prefetcher))
            material_ids = batch.inputs["voxel.material_id"]
            target_c21 = batch.targets["effective_c21"]
            prediction = model(
                (material_ids != 0).to(torch.float32).unsqueeze(1)
            )
            loss = torch.nn.functional.mse_loss(
                prediction, target_c21.to(torch.float32)
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            torch.cuda.synchronize()

            dlpack_roundtrip = torch.utils.dlpack.from_dlpack(
                torch.utils.dlpack.to_dlpack(material_ids)
            )

        self.assertTrue(bool(torch.isfinite(loss).item()))
        gradients = [
            parameter.grad
            for parameter in model.parameters()
            if parameter.grad is not None
        ]
        self.assertTrue(gradients)
        self.assertTrue(
            all(bool(torch.isfinite(gradient).all().item())
                for gradient in gradients)
        )
        self.assertEqual(material_ids.device.type, "cuda")
        self.assertEqual(target_c21.device.type, "cuda")
        self.assertGreater(prefetcher.transferred_bytes, 0)
        self.assertGreaterEqual(prefetcher.recorded_tensors, 2)
        self.assertEqual(
            dlpack_roundtrip.data_ptr(),
            material_ids.data_ptr(),
        )


@unittest.skipUnless(
    torch is not None and torch.cuda.is_available(),
    "CUDA PyTorch is required for the Voxel-ACDM integration test",
)
class RealVoxelAcdmLabelTest(unittest.TestCase):
    def test_converged_acdm_label_is_published_with_exact_provenance(self):
        acdm_root = REPOSITORY_ROOT.parent / "Voxel-ACDM"
        if not (acdm_root / "femlib").is_dir():
            self.skipTest("sibling Voxel-ACDM checkout is unavailable")
        commit = subprocess.run(
            ["git", "-C", str(acdm_root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "-C", str(acdm_root), "status", "--porcelain"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
        solver_revision = commit + ("-dirty" if dirty else "")
        cpu_sample = build_real_sample(width=0.70)
        cuda_sample = cpu_sample.to(
            "torch", device="cuda", dtype=torch.float32
        )
        tolerance = 1e-5

        started = time.perf_counter()
        result = solve_acdm_anisotropic_from_sample(
            cuda_sample,
            acdm_root=str(acdm_root),
            dtype="fp32",
            precond="none",
            tol=tolerance,
            max_iter=100,
            element_type="c3d8",
            verbose=False,
        )
        torch.cuda.synchronize()
        wall_time = time.perf_counter() - started
        effective_matrix = result.effective_stiffness_numpy()
        effective_c21 = np.asarray(
            pack_voigt_c21(effective_matrix), dtype=np.float64
        )
        maximum_residual = max(
            float(load_case["rel_res"]) for load_case in result.info
        )
        iteration_count = max(
            int(load_case["iters"]) for load_case in result.info
        )

        self.assertEqual(len(result.info), 6)
        self.assertLessEqual(maximum_residual, tolerance)
        self.assertTrue(
            bool(np.linalg.eigvalsh(effective_matrix).min() > 0.0)
        )
        self.assertEqual(result.timings["dense_input_device"], "cuda:0")

        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "acdm-dataset"
            with SimulationDatasetWriter.create(
                target,
                schema=training_schema(shard_size=1),
                quality=DatasetQualityPolicy(
                    maximum_solver_residual=2e-5
                ),
                generation={
                    "label_solver": "Voxel-ACDM",
                    "grid_shape": [4, 4, 4],
                },
            ) as writer:
                writer.append(
                    cpu_sample,
                    targets={"effective_c21": effective_c21},
                    sample_id="real-acdm-label",
                    group_id="real-acdm-geometry",
                    split="train",
                    provenance={
                        "solver_commit": solver_revision,
                        "element_formulation": "periodic-c3d8",
                        "arithmetic_dtype": "float32",
                        "tolerance": tolerance,
                        "maximum_residual": maximum_residual,
                        "iteration_count": iteration_count,
                        "wall_time_seconds": wall_time,
                        "target_units": {"effective_c21": "GPa"},
                    },
                )
            dataset = SimulationDataset(
                target,
                split="train",
                inputs=("voxel.material_id",),
                targets=("effective_c21",),
                verify="sample",
            )
            stored = dataset[0]

        np.testing.assert_array_equal(
            stored.targets["effective_c21"], effective_c21
        )
        provenance = stored.metadata["provenance"]
        self.assertEqual(provenance["solver_commit"], solver_revision)
        self.assertEqual(provenance["arithmetic_dtype"], "float32")
        self.assertEqual(
            provenance["maximum_residual"], maximum_residual
        )
        self.assertEqual(provenance["iteration_count"], iteration_count)


if __name__ == "__main__":
    unittest.main(verbosity=2)
