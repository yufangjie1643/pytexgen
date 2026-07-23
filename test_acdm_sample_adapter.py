"""Tests for explicit host policy and dense GPU Voxel-ACDM handoff."""

import sys
import types
import unittest
from dataclasses import replace
from unittest import mock

import numpy as np

try:
    import torch
except ImportError:
    torch = None

import test_acdm_solver_adapter as acdm_tests
import test_simulation_sample as sample_tests


class _TrackingCudaPhase:
    def __init__(self):
        self.device = types.SimpleNamespace(type="cuda")
        self.cpu_calls = 0
        self._array = np.array([[[[0, 1], [1, 0]]]], dtype=np.uint8)

    def detach(self):
        return self

    def cpu(self):
        self.cpu_calls += 1
        return types.SimpleNamespace(numpy=lambda: self._array)

    def numel(self):
        return int(self._array.size)

    def element_size(self):
        return int(self._array.dtype.itemsize)


class CompactPhaseHostPolicyTest(unittest.TestCase):
    def setUp(self):
        self.adapter = acdm_tests.load_adapter_module()
        self.voxelizer = sys.modules["TexGen.gpu_voxelizer"]
        self.data = self.voxelizer.VoxelGridData(
            yarn_id=np.array([-1, 0, 1, -1], dtype=np.int32),
            aabb=np.array(
                [[0.0, 0.0, 0.0], [2.0, 2.0, 1.0]],
                dtype=np.float64,
            ),
            resolution=(2, 2, 1),
            backend="numpy",
            device="cpu",
            workers=1,
            dtype="float32",
            timings={},
        )

    def fake_modules(self, *, capable=False):
        calls = {"constructed": 0}

        class FakeCompactSolver:
            SUPPORTS_CUDA_PHASE_IDS = capable

            @classmethod
            def from_E_nu(cls, phase_ids, *args, **kwargs):
                calls["constructed"] += 1
                calls["phase_ids"] = phase_ids
                return cls()

            def compute_effective_stiffness(self, **kwargs):
                return np.eye(6)[None], [["ok"]]

        femlib = types.ModuleType("femlib")
        femlib.extract_engineering_constants = lambda _C: {}
        batched = types.ModuleType("femlib.fem_batched")
        batched.FEMHomogenizerBatchedIsotropicPhases = FakeCompactSolver
        return calls, femlib, batched

    def solve_with_tracking_phase(
        self,
        phase,
        *,
        capable,
        allow_host_phase_pack,
    ):
        calls, femlib, batched = self.fake_modules(capable=capable)
        with mock.patch.object(
            self.adapter,
            "to_acdm_phase_ids",
            return_value=phase,
        ), mock.patch.object(
            self.adapter,
            "import_voxel_acdm",
            return_value=femlib,
        ), mock.patch.dict(
            sys.modules,
            {"femlib": femlib, "femlib.fem_batched": batched},
        ):
            result = self.adapter.solve_acdm_isotropic_from_voxel_data(
                self.data,
                phase_materials={
                    0: {"E": 3.0, "Nu": 0.35},
                    1: {"E": 70.0, "Nu": 0.20},
                },
                precond="none",
                allow_host_phase_pack=allow_host_phase_pack,
            )
        return calls, result

    def test_incapable_solver_rejects_before_host_transfer_or_construction(self):
        phase = _TrackingCudaPhase()
        calls, femlib, batched = self.fake_modules(capable=False)

        with mock.patch.object(
            self.adapter,
            "to_acdm_phase_ids",
            return_value=phase,
        ), mock.patch.object(
            self.adapter,
            "import_voxel_acdm",
            return_value=femlib,
        ), mock.patch.dict(
            sys.modules,
            {"femlib": femlib, "femlib.fem_batched": batched},
        ):
            with self.assertRaisesRegex(RuntimeError, "allow_host_phase_pack"):
                self.adapter.solve_acdm_isotropic_from_voxel_data(
                    self.data,
                    phase_materials={
                        0: {"E": 3.0, "Nu": 0.35},
                        1: {"E": 70.0, "Nu": 0.20},
                    },
                    precond="none",
                )

        self.assertEqual(phase.cpu_calls, 0)
        self.assertEqual(calls["constructed"], 0)

    def test_explicit_host_pack_records_device_and_bytes(self):
        phase = _TrackingCudaPhase()

        calls, result = self.solve_with_tracking_phase(
            phase,
            capable=False,
            allow_host_phase_pack=True,
        )

        self.assertEqual(phase.cpu_calls, 1)
        self.assertIsInstance(calls["phase_ids"], np.ndarray)
        self.assertEqual(result.timings["phase_pack_device"], "cpu")
        self.assertEqual(
            result.timings["phase_pack_bytes"],
            phase.numel() * phase.element_size(),
        )

    def test_capable_solver_receives_original_cuda_phase_object(self):
        phase = _TrackingCudaPhase()

        calls, result = self.solve_with_tracking_phase(
            phase,
            capable=True,
            allow_host_phase_pack=False,
        )

        self.assertEqual(phase.cpu_calls, 0)
        self.assertIs(calls["phase_ids"], phase)
        self.assertIs(result.phase_ids, phase)
        self.assertEqual(result.timings["phase_pack_device"], "cuda")
        self.assertEqual(result.timings["phase_pack_bytes"], 0)


@unittest.skipIf(torch is None, "PyTorch is not installed")
class DenseSampleACDMAdapterTest(unittest.TestCase):
    def setUp(self):
        fixture = sample_tests.SimulationSampleValidationTest(
            "test_construction_is_zero_copy_and_adopts_voxel_orientation"
        )
        fixture.setUp()
        self.fixture = fixture
        self.adapter = acdm_tests.load_adapter_module()

    def fake_solver_module(self, *, supports=True, tensor_output=True):
        calls = {"constructed": 0}

        class FakeDenseSolver:
            SUPPORTS_TORCH_C_VOIGT_FIELDS = supports

            def __init__(
                self,
                C_voigt_fields,
                voxel_size,
                grid_shape,
                **kwargs,
            ):
                calls["constructed"] += 1
                calls["C_voigt_fields"] = C_voigt_fields
                calls["voxel_size"] = voxel_size
                calls["grid_shape"] = grid_shape
                calls["kwargs"] = kwargs

            def compute_effective_stiffness(self, **kwargs):
                calls["solve_kwargs"] = kwargs
                if tensor_output:
                    output = torch.eye(
                        6,
                        dtype=calls["C_voigt_fields"].dtype,
                        device=calls["C_voigt_fields"].device,
                    )[None]
                else:
                    output = np.eye(6, dtype=np.float64)[None]
                calls["output"] = output
                return output, [["ok"]]

            def enable_fft_precond(self, **kwargs):
                calls["precond_kwargs"] = kwargs

        return types.SimpleNamespace(
            FEMHomogenizerBatched=FakeDenseSolver,
        ), calls

    @unittest.skipUnless(
        torch is not None and torch.cuda.is_available(),
        "CUDA is not available",
    )
    def test_dense_adapter_keeps_input_and_output_on_current_cuda_device(self):
        sample = self.fixture.make_sample().to(
            "torch",
            device="cuda",
            dtype=torch.float32,
        )
        solver_module, calls = self.fake_solver_module()

        result = self.adapter.solve_acdm_anisotropic_from_sample(
            sample,
            device=sample.device,
            dtype="fp32",
            precond="none",
            solver_module=solver_module,
        )

        received = calls["C_voigt_fields"]
        self.assertTrue(received.is_cuda)
        self.assertEqual(received.device, sample.stiffness.yarn_c21.device)
        torch.testing.assert_close(
            received,
            sample.stiffness.to_acdm(batch=True),
        )
        self.assertEqual(
            result.C_eff_tensor.data_ptr(),
            calls["output"][0].data_ptr(),
        )
        self.assertIs(result.C_eff, result.C_eff_tensor)
        self.assertEqual(
            result.timings["dense_input_device"],
            str(received.device),
        )
        self.assertEqual(
            result.timings["dense_input_bytes"],
            received.numel() * received.element_size(),
        )

    def test_dense_adapter_rejects_numpy_and_missing_stiffness(self):
        sample = self.fixture.make_sample()
        solver_module, calls = self.fake_solver_module()

        with self.assertRaisesRegex(ValueError, "Torch CUDA"):
            self.adapter.solve_acdm_anisotropic_from_sample(
                sample,
                solver_module=solver_module,
            )
        matrix_only = self.fixture.sample_module.SimulationSample(
            voxels=replace(sample.voxels, sparse_orientation=None),
            materials=sample.materials,
        )
        with self.assertRaisesRegex(ValueError, "stiffness"):
            self.adapter.solve_acdm_anisotropic_from_sample(
                matrix_only,
                solver_module=solver_module,
            )
        self.assertEqual(calls["constructed"], 0)

    @unittest.skipUnless(
        torch is not None and torch.cuda.is_available(),
        "CUDA is not available",
    )
    def test_dense_adapter_rejects_incompatible_solver_before_construction(self):
        sample = self.fixture.make_sample().to(
            "torch",
            device="cuda",
            dtype=torch.float32,
        )
        solver_module, calls = self.fake_solver_module(supports=False)

        with self.assertRaisesRegex(RuntimeError, "Torch C_voigt"):
            self.adapter.solve_acdm_anisotropic_from_sample(
                sample,
                solver_module=solver_module,
            )

        self.assertEqual(calls["constructed"], 0)

    @unittest.skipUnless(
        torch is not None and torch.cuda.is_available(),
        "CUDA is not available",
    )
    def test_effective_stiffness_numpy_is_an_explicit_transfer(self):
        sample = self.fixture.make_sample().to(
            "torch",
            device="cuda",
            dtype=torch.float32,
        )
        solver_module, _ = self.fake_solver_module()
        result = self.adapter.solve_acdm_anisotropic_from_sample(
            sample,
            precond="none",
            solver_module=solver_module,
        )

        numpy_value = result.effective_stiffness_numpy()

        self.assertIsInstance(numpy_value, np.ndarray)
        np.testing.assert_array_equal(numpy_value, np.eye(6))


if __name__ == "__main__":
    unittest.main(verbosity=2)
