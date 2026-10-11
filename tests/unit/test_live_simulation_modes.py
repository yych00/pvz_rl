import unittest

from workbench.live.early_levels import (
    SIMULATION_ANIMATED,
    SIMULATION_CONTINUOUS,
    SIMULATION_HEADLESS,
    SIMULATION_STEPPED,
    normalize_simulation_mode,
)


class LiveSimulationModeTest(unittest.TestCase):
    def test_supported_modes_and_aliases(self):
        self.assertEqual(normalize_simulation_mode("stepped"), SIMULATION_STEPPED)
        self.assertEqual(normalize_simulation_mode("native"), SIMULATION_ANIMATED)
        self.assertEqual(normalize_simulation_mode("persistent"), SIMULATION_HEADLESS)
        self.assertEqual(normalize_simulation_mode("continuous"), SIMULATION_CONTINUOUS)
        self.assertEqual(normalize_simulation_mode("realtime"), SIMULATION_CONTINUOUS)

    def test_invalid_mode_is_rejected(self):
        with self.assertRaises(ValueError):
            normalize_simulation_mode("turbo")


if __name__ == "__main__":
    unittest.main()
