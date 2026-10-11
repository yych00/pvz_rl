import gc
import tkinter as tk
import unittest

import workbench.main  # noqa: F401 - installs the canonical Workbench patches
from workbench import PvzGymUnitTestGUI


class WorkbenchTickLimitTest(unittest.TestCase):
    def setUp(self):
        self.root = tk.Tk()
        self.root.withdraw()
        self.app = PvzGymUnitTestGUI(self.root)

    def tearDown(self):
        self.root.destroy()
        self.app = self.root = None
        gc.collect()

    def test_tick_budget_input_snapshots_default_custom_and_unlimited_values(self):
        self.assertEqual(self.app.var_training_max_episode_ticks.get(), "120000")
        self.app._parse_common_training_config()
        self.assertEqual(self.app._training_max_episode_ticks, 120_000)
        self.app.var_training_max_episode_ticks.set("12345")
        self.app._parse_common_training_config()
        self.assertEqual(self.app._training_max_episode_ticks, 12345)
        self.app.var_training_max_episode_ticks.set("0")
        self.app._parse_common_training_config()
        self.assertIsNone(self.app._training_max_episode_ticks)
        self.app.var_training_max_episode_ticks.set("-1")
        with self.assertRaisesRegex(ValueError, "Max ticks"):
            self.app._parse_common_training_config()

    def test_active_training_locks_tick_budget_input(self):
        self.app._set_training_controls(True)
        self.assertEqual(str(self.app.ent_training_max_episode_ticks.cget("state")), "disabled")
        self.app._set_training_controls(False)
        self.assertEqual(str(self.app.ent_training_max_episode_ticks.cget("state")), "normal")


if __name__ == "__main__":
    unittest.main()
