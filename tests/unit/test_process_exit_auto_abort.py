import threading
import unittest

from workbench.live.early_levels import (
    LEVEL_1_2_SPEC,
    run_early_level_e2e,
    run_early_level_stress,
)
from workbench.live.level_1_1 import (
    run_level_1_1_e2e,
    run_level_1_1_stress,
)


class DummyClient:
    """Mock client 用于模拟游戏进程离线与崩溃场景。"""

    def __init__(self, initially_online=True):
        self._online = initially_online
        self._paused = False
        self.is_loaded = True

    def is_game_on(self):
        return self._online, "online" if self._online else "offline"

    def attach(self):
        return self._online, "ok" if self._online else "not found"

    def get_automation_config(self):
        return {"auto_collect": True, "auto_dialogue": True, "auto_seed": True}

    def set_automation_config(self, *args, **kwargs):
        pass

    def is_paused(self):
        return self._paused

    def pause(self):
        self._paused = True
        return True

    def resume(self):
        self._paused = False
        return True

    def automation_tick(self):
        return 0

    def select_level(self, area, sub):
        return 0, "ok"

    def get_action_mask(self, size=676):
        return 0, [False] * size, size, "ok"

    def advance_ticks(self, ticks):
        if not self._online:
            return 0
        return ticks

    def get_runtime_state(self):
        if not self._online:
            return {
                "process_available": False,
                "game_available": False,
                "board_available": False,
                "phase_name": "OFFLINE",
                "game_status": 0,
            }
        return {
            "process_available": True,
            "game_available": True,
            "board_available": True,
            "phase_name": "BATTLE",
            "game_status": 0,
            "adventure_round": 1,
            "finished_adventure": 0,
            "level_id": 2,
        }

    def get_observation_json(self):
        if not self._online:
            return 0, "process offline", None
        return 1, "ok", {
            "global": {"level_id": 2, "sun": 100, "game_clock": 100},
            "cards": [{"plant_type": 0}],
            "plants": [],
            "zombies": [],
            "sun_drops": [],
        }

    def get_observation(self):
        return self.get_observation_json()


class ProcessExitAutoAbortTest(unittest.TestCase):
    def test_early_level_e2e_aborts_when_offline_at_start(self):
        client = DummyClient(initially_online=False)
        stop_event = threading.Event()

        result = run_early_level_e2e(client, LEVEL_1_2_SPEC, stop_event=stop_event)

        self.assertFalse(result.success)
        self.assertEqual(result.outcome, "ABORTED")
        self.assertTrue(stop_event.is_set())
        self.assertIn("自动终止", result.message)

    def test_early_level_e2e_aborts_when_game_exits_during_battle(self):
        client = DummyClient(initially_online=True)
        stop_event = threading.Event()

        # 在进入战斗轮询时使游戏离线
        orig_runtime = client.get_runtime_state
        call_count = [0]

        def runtime_with_crash():
            call_count[0] += 1
            # 经过准备阶段后，进入战斗主循环的第一次状态查询时退出
            if call_count[0] >= 4:
                client._online = False
            return orig_runtime()

        client.get_runtime_state = runtime_with_crash

        result = run_early_level_e2e(client, LEVEL_1_2_SPEC, stop_event=stop_event)

        self.assertFalse(result.success)
        self.assertEqual(result.outcome, "ABORTED")
        self.assertTrue(stop_event.is_set())
        self.assertIn("异常退出", result.message)

    def test_early_level_e2e_aborts_on_exception_if_game_offline(self):
        client = DummyClient(initially_online=True)
        stop_event = threading.Event()

        def advance_with_disconnect(ticks):
            client._online = False
            raise ConnectionResetError("Game process pipe broken")

        client.advance_ticks = advance_with_disconnect

        result = run_early_level_e2e(client, LEVEL_1_2_SPEC, stop_event=stop_event)

        self.assertFalse(result.success)
        self.assertEqual(result.outcome, "ABORTED")
        self.assertTrue(stop_event.is_set())
        self.assertIn("异常退出", result.message)

    def test_early_level_stress_aborts_immediately_on_crash(self):
        client = DummyClient(initially_online=False)
        stop_event = threading.Event()

        stress_result = run_early_level_stress(
            client,
            LEVEL_1_2_SPEC,
            episodes=5,
            stop_event=stop_event,
        )

        self.assertFalse(stress_result.success)
        # 应在第 1 集离线后直接终止，而不继续跑剩余 4 集
        self.assertEqual(len(stress_result.episode_results), 1)
        self.assertEqual(stress_result.episode_results[0].outcome, "ABORTED")
        self.assertTrue(stop_event.is_set())

    def test_level_1_1_e2e_aborts_when_offline(self):
        client = DummyClient(initially_online=False)
        stop_event = threading.Event()

        result = run_level_1_1_e2e(client, stop_event=stop_event)

        self.assertFalse(result.success)
        self.assertEqual(result.outcome, "ABORTED")
        self.assertTrue(stop_event.is_set())

    def test_level_1_1_stress_aborts_immediately_on_crash(self):
        client = DummyClient(initially_online=False)
        stop_event = threading.Event()

        stress_result = run_level_1_1_stress(
            client,
            episodes=5,
            stop_event=stop_event,
        )

        self.assertFalse(stress_result.success)
        self.assertEqual(len(stress_result.episode_results), 1)
        self.assertEqual(stress_result.episode_results[0].outcome, "ABORTED")
        self.assertTrue(stop_event.is_set())


if __name__ == "__main__":
    unittest.main()
