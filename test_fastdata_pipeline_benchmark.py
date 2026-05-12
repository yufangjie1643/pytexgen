import unittest

import bench_fastdata_pipeline as bench


class FastDataPipelineBenchmarkTest(unittest.TestCase):
    def test_phase_runner_records_order_and_serializes(self):
        calls = []

        def timer():
            timer.now += 0.5
            return timer.now

        timer.now = 0.0

        phases = [
            bench.BenchmarkPhase(
                "make",
                prepare=lambda context: calls.append("prepare-make") or "seed",
                run=lambda prepared, context: calls.append(("run-make", prepared)) or "textile",
            ),
            bench.BenchmarkPhase(
                "snapshot",
                prepare=lambda context: calls.append(("prepare-snapshot", context["make"])) or context["make"],
                run=lambda prepared, context: calls.append(("run-snapshot", prepared)) or {"nodes": 12},
                metadata=lambda result: {"nodes": result["nodes"]},
            ),
        ]

        report = bench.run_benchmark_phases(
            phases,
            repeat=2,
            timer=timer,
            metadata={"case": "unit"},
        )

        self.assertEqual([record.name for record in report.records], ["make", "snapshot"])
        self.assertEqual(report.metadata["case"], "unit")
        self.assertEqual(report.records[0].repeat, 2)
        self.assertEqual(report.records[1].metadata, {"nodes": 12})
        self.assertEqual(report.context["snapshot"], {"nodes": 12})
        self.assertIn(("prepare-snapshot", "textile"), calls)

        payload = report.as_dict()
        self.assertEqual(payload["records"][0]["name"], "make")
        self.assertEqual(payload["records"][1]["metadata"]["nodes"], 12)
        self.assertIn("snapshot", report.format_table())

    def test_real_phase_list_splits_flat_voxel_work(self):
        args = bench.parse_args(["--resolution", "4", "--skip-python-fallback"])
        names = bench.real_phase_names(args)

        self.assertIn("bundle_numpy_pack", names)
        self.assertIn("voxel_centers", names)
        self.assertIn("voxel_classify_numpy_flat", names)
        self.assertNotIn("voxel_numpy_from_direct", names)


if __name__ == "__main__":
    unittest.main()
