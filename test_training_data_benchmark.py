"""Tests for checked training-data benchmark acceptance logic."""

import json
import math
import unittest

from bench_training_data import evaluate_benchmark


def passing_inputs(**overrides):
    values = {
        "native_samples_per_second": 300.0,
        "npz_samples_per_second": 100.0,
        "synchronous_wait_seconds": 0.4,
        "prefetch_wait_seconds": 0.2,
        "expected_h2d_bytes": 4096,
        "observed_h2d_bytes": 4096,
        "training_loss": 1.25,
        "min_read_speedup": 1.5,
        "min_prefetch_speedup": 1.0,
    }
    values.update(overrides)
    return values


class BenchmarkEvaluationTest(unittest.TestCase):
    def test_passes_good_metrics_and_is_json_serializable(self):
        report = evaluate_benchmark(**passing_inputs())

        self.assertTrue(report["passed"])
        self.assertEqual(report["failed_metrics"], [])
        self.assertEqual(report["read_speedup"], 3.0)
        self.assertEqual(report["prefetch_speedup"], 2.0)
        json.dumps(report, allow_nan=False)

    def test_exact_thresholds_pass(self):
        report = evaluate_benchmark(
            **passing_inputs(
                native_samples_per_second=150.0,
                npz_samples_per_second=100.0,
                synchronous_wait_seconds=0.2,
                prefetch_wait_seconds=0.2,
            )
        )

        self.assertTrue(report["passed"])
        self.assertEqual(report["read_speedup"], 1.5)
        self.assertEqual(report["prefetch_speedup"], 1.0)

    def test_read_regression_fails_independently(self):
        report = evaluate_benchmark(
            **passing_inputs(native_samples_per_second=149.0)
        )

        self.assertFalse(report["passed"])
        self.assertEqual(report["failed_metrics"], ["read_speedup"])

    def test_prefetch_regression_fails_independently(self):
        report = evaluate_benchmark(
            **passing_inputs(prefetch_wait_seconds=0.41)
        )

        self.assertFalse(report["passed"])
        self.assertEqual(report["failed_metrics"], ["prefetch_speedup"])

    def test_h2d_byte_mismatch_fails_independently(self):
        report = evaluate_benchmark(
            **passing_inputs(observed_h2d_bytes=4095)
        )

        self.assertFalse(report["passed"])
        self.assertEqual(report["failed_metrics"], ["h2d_bytes"])

    def test_nonfinite_training_loss_fails_independently(self):
        for loss in (math.nan, math.inf, -math.inf):
            with self.subTest(loss=loss):
                report = evaluate_benchmark(
                    **passing_inputs(training_loss=loss)
                )
                self.assertFalse(report["passed"])
                self.assertEqual(
                    report["failed_metrics"], ["training_loss"]
                )

    def test_rejects_invalid_metrics_and_thresholds(self):
        invalid = (
            ("native_samples_per_second", 0.0),
            ("npz_samples_per_second", -1.0),
            ("synchronous_wait_seconds", -1.0),
            ("prefetch_wait_seconds", -1.0),
            ("expected_h2d_bytes", -1),
            ("observed_h2d_bytes", 1.5),
            ("min_read_speedup", 0.0),
            ("min_prefetch_speedup", 0.0),
        )
        for name, value in invalid:
            with self.subTest(name=name):
                with self.assertRaises((TypeError, ValueError)):
                    evaluate_benchmark(
                        **passing_inputs(**{name: value})
                    )


if __name__ == "__main__":
    unittest.main(verbosity=2)
