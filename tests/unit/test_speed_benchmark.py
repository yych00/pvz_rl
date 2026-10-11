import unittest

from workbench.gui.speed_benchmark import make_speed_result


class SpeedBenchmarkMathTest(unittest.TestCase):
    def test_rate_is_relative_to_engine_realtime_tick_rate(self):
        result = make_speed_result(
            "animated",
            "Animated native loop",
            ticks=300,
            elapsed_s=0.3,
            baseline_ticks_per_second=100.0,
        )
        self.assertAlmostEqual(result.ticks_per_second, 1000.0)
        self.assertAlmostEqual(result.realtime_multiple, 10.0)

    def test_zero_ticks_stays_zero_rate(self):
        result = make_speed_result(
            "headless",
            "Headless one batch",
            ticks=0,
            elapsed_s=0.02,
            baseline_ticks_per_second=100.0,
        )
        self.assertEqual(result.ticks_per_second, 0.0)
        self.assertEqual(result.realtime_multiple, 0.0)

    def test_continuous_rate_uses_unmodified_engine_baseline(self):
        result = make_speed_result(
            "continuous",
            "Continuous no pause",
            ticks=300,
            elapsed_s=3.0,
            baseline_ticks_per_second=100.0,
        )
        self.assertAlmostEqual(result.ticks_per_second, 100.0)
        self.assertAlmostEqual(result.realtime_multiple, 1.0)


if __name__ == "__main__":
    unittest.main()
