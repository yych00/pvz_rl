import unittest
from types import SimpleNamespace
from unittest.mock import Mock

from workbench.main import PvzGymUnitTestGUI


class ForbiddenNativeClient:
    def __getattr__(self, name):
        raise AssertionError(f"Tk daemon entered native client: {name}")


class DaemonNativeContentionTest(unittest.TestCase):
    def test_final_wrapped_daemon_never_enters_native_while_worker_runs(self):
        cases = [{flag: True} for flag in (
            "_live_running", "_level_1_1_live_running", "_training_running",
            "_speed_benchmark_running", "_static_test_running", "_prebattle_advancing",
        )]
        cases.append({"_early_level_live_states": {"1-2": {"running": True}}})
        for state in cases:
            with self.subTest(state=state):
                gui = SimpleNamespace(
                    client=ForbiddenNativeClient(), root=Mock(),
                    refresh_status=Mock(side_effect=AssertionError("native refresh")),
                    _daemon_step=Mock(), **state,
                )
                PvzGymUnitTestGUI._daemon_step(gui)
                gui.root.after.assert_called_once_with(100, gui._daemon_step)
                gui.refresh_status.assert_not_called()

    def test_idle_daemon_still_detects_and_attaches_game(self):
        gui = SimpleNamespace(
            client=Mock(is_loaded=True), root=Mock(), _daemon_step=Mock(),
            refresh_status=Mock(), log=Mock(),
            _early_level_live_states={"1-2": {"running": False}},
        )
        gui.client.is_game_on.return_value = (False, "offline")
        gui.client.attach.return_value = (False, "offline")
        PvzGymUnitTestGUI._daemon_step(gui)
        gui.client.attach.assert_called_once()
        gui.root.after.assert_called_once_with(100, gui._daemon_step)


if __name__ == "__main__":
    unittest.main()
