"""Tests for GPU material-field benchmark reporting and acceptance gates."""

import unittest

import bench_gpu_material_fields as bench


class MaterialFieldBenchmarkTest(unittest.TestCase):
    @staticmethod
    def acceptance_records(speedup=5.1):
        records = []
        for case in ("plain_2x2", "multi_yarn_8x8"):
            for resolution in (128, 256):
                for mode in ("compute", "practical"):
                    records.append(
                        {
                            "case": case,
                            "resolution": resolution,
                            "mode": mode,
                            "dtype": "float32",
                            "correctness": True,
                            "occupancy_mismatch_fraction": 0.0,
                            "yarn_mismatch_fraction": 0.0,
                            "minimum_axis_dot": 1.0,
                            "stiffness_relative_error": 1e-6,
                            "speedup": speedup,
                        }
                    )
        return records

    def test_summary_uses_median_p90_and_speedup(self):
        result = bench.summarize_timings(
            [5.0, 1.0, 3.0, 2.0, 4.0], cpu_median=20.0
        )

        self.assertEqual(result["median_s"], 3.0)
        self.assertAlmostEqual(result["p90_s"], 4.6)
        self.assertEqual(result["speedup"], 20.0 / 3.0)

    def test_gate_requires_correctness_and_every_large_case(self):
        records = self.acceptance_records(speedup=5.1)
        self.assertTrue(bench.evaluate_acceptance(records)["passed"])

        records[0]["occupancy_mismatch_fraction"] = 0.006
        self.assertFalse(bench.evaluate_acceptance(records)["passed"])

        incomplete = self.acceptance_records(speedup=5.1)[:-1]
        self.assertFalse(bench.evaluate_acceptance(incomplete)["passed"])

    def test_gate_applies_dtype_specific_stiffness_tolerance(self):
        records = self.acceptance_records(speedup=5.1)
        records[0]["stiffness_relative_error"] = 6e-5
        self.assertFalse(bench.evaluate_acceptance(records)["passed"])

        for record in records:
            record["dtype"] = "float64"
            record["stiffness_relative_error"] = 2e-10
        self.assertFalse(bench.evaluate_acceptance(records)["passed"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
