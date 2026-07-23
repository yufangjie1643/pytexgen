"""Tests for the checked simulation-sample interoperability benchmark."""

import unittest

import bench_simulation_interop as bench


class SimulationInteropBenchmarkTest(unittest.TestCase):
    @staticmethod
    def passing_metrics():
        return {
            "device": "cuda:0",
            "resolution": 64,
            "num_voxels": 64**3,
            "resident_sparse_bytes": 1024,
            "acdm_dense_bytes": 4096,
            "handoff_host_transfer_bytes": 0,
            "dlpack_pointer_shared": True,
            "phase_equal_numpy": True,
            "acdm_dense_max_abs_error": 5e-7,
            "acdm_dense_tolerance": 2e-6,
            "pytexgen_seconds": 0.1,
            "texgen_cpu_seconds": 0.6,
            "speedup_vs_texgen_cpu": 6.0,
        }

    def test_report_contains_stable_json_contract(self):
        report = bench.build_report(
            self.passing_metrics(),
            dtype="float32",
            min_speedup=5.0,
        )

        required = {
            "schema",
            "device",
            "resolution",
            "num_voxels",
            "resident_sparse_bytes",
            "acdm_dense_bytes",
            "handoff_host_transfer_bytes",
            "dlpack_pointer_shared",
            "phase_equal_numpy",
            "acdm_dense_max_abs_error",
            "pytexgen_seconds",
            "texgen_cpu_seconds",
            "speedup_vs_texgen_cpu",
            "accepted",
        }
        self.assertTrue(required.issubset(report))
        self.assertEqual(
            report["schema"],
            "pytexgen.simulation_interop_benchmark/v1",
        )
        self.assertTrue(report["accepted"])

    def test_gate_rejects_each_correctness_or_transfer_failure(self):
        failures = {
            "dlpack_pointer_shared": False,
            "phase_equal_numpy": False,
            "acdm_dense_max_abs_error": 3e-6,
            "handoff_host_transfer_bytes": 1024,
        }
        for key, value in failures.items():
            with self.subTest(key=key):
                metrics = self.passing_metrics()
                metrics[key] = value
                report = bench.build_report(
                    metrics,
                    dtype="float32",
                    min_speedup=5.0,
                )
                self.assertFalse(report["accepted"])
                self.assertTrue(report["acceptance_failures"])

    def test_gate_rejects_speedup_below_configured_threshold(self):
        metrics = self.passing_metrics()
        metrics["speedup_vs_texgen_cpu"] = 4.99

        report = bench.build_report(
            metrics,
            dtype="float32",
            min_speedup=5.0,
        )

        self.assertFalse(report["accepted"])
        self.assertIn("speedup", " ".join(report["acceptance_failures"]))

    def test_dlpack_handoff_reports_shared_cpu_torch_storage(self):
        try:
            import torch
        except ImportError:
            self.skipTest("PyTorch is not installed")
        fields = {
            "phase": torch.arange(8, dtype=torch.int32),
            "c21": torch.arange(42, dtype=torch.float32).reshape(2, 21),
        }

        result = bench.measure_dlpack_handoff(fields, torch_mod=torch)

        self.assertTrue(result["pointer_shared"])
        self.assertEqual(result["host_transfer_bytes"], 0)
        self.assertEqual(set(result["fields"]), set(fields))
        self.assertTrue(
            all(item["pointer_shared"] for item in result["fields"].values())
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
