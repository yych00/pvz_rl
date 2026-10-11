import unittest
from dataclasses import replace

from pvz_gym.env import PvZEnv
from tests.fake_backend import FactClient


class HeadlessFactClient(FactClient):
    def __init__(self):
        super().__init__()
        self.headless_active = False
        self.headless_calls: list[object] = []

    def begin_headless_mode(self) -> bool:
        self.headless_calls.append("begin")
        self.headless_active = True
        return True

    def end_headless_mode(self) -> bool:
        self.headless_calls.append("end")
        self.headless_active = False
        return True

    def step_headless(self, action, ticks):
        if not self.headless_active:
            raise RuntimeError("headless mode is not active")
        self.headless_calls.append(("step_headless", action.type, int(ticks)))
        return super().step(action, ticks)


class DriftingHeadlessFactClient(HeadlessFactClient):
    def step_headless(self, action, ticks):
        if not self.headless_active:
            raise RuntimeError("headless mode is not active")
        self.headless_calls.append(("step_headless", action.type, int(ticks)))
        return FactClient.step(self, action, int(ticks) + 2)


class EarlyTerminalHeadlessFactClient(HeadlessFactClient):
    def step_headless(self, action, ticks):
        if not self.headless_active:
            raise RuntimeError("headless mode is not active")
        self.headless_calls.append(("step_headless", action.type, int(ticks)))
        actual_ticks = int(ticks) if self.advance_calls == 0 else 9
        return FactClient.step(self, action, actual_ticks)


class ContinuousFactClient(FactClient):
    def step_continuous(self, action, target_ticks):
        return super().step(action, int(target_ticks) + 7)


class HeadlessEnvTest(unittest.TestCase):
    def test_reset_rejects_failed_headless_release_before_native_reset(self):
        client = HeadlessFactClient()
        env = PvZEnv(client, simulation_mode="headless")
        env.reset()
        original_end = client.end_headless_mode
        client.end_headless_mode = lambda: False
        try:
            with self.assertRaisesRegex(RuntimeError, "reset was not executed"):
                env.reset()
            self.assertEqual(client.call_order, ["reset"])
            self.assertTrue(env._headless_active)
            self.assertTrue(client.headless_active)
        finally:
            client.end_headless_mode = original_end
            env.close()

    def test_reset_preserves_headless_ownership_when_release_raises(self):
        client = HeadlessFactClient()
        env = PvZEnv(client, simulation_mode="headless")
        env.reset()
        original_end = client.end_headless_mode
        def fail():
            raise RuntimeError("release failed")
        client.end_headless_mode = fail
        try:
            with self.assertRaisesRegex(RuntimeError, "release failed"):
                env.reset()
            self.assertEqual(client.call_order, ["reset"])
            self.assertTrue(env._headless_active)
        finally:
            client.end_headless_mode = original_end
            env.close()

    def test_zombie_truncation_is_info_only_in_all_simulation_modes(self):
        for mode, base in (("stepped", FactClient), ("headless", HeadlessFactClient),
                           ("continuous", ContinuousFactClient)):
            with self.subTest(mode=mode):
                class CrowdedClient(base):
                    def _observation(self):
                        facts = super()._observation()
                        return replace(facts, zombies=facts.zombies * 260)

                env = PvZEnv(CrowdedClient(), simulation_mode=mode)
                try:
                    observation, info = env.reset()
                    self.assertEqual(info["zombies_dropped"], 4)
                    self.assertTrue(info["zombie_truncated"])
                    self.assertNotIn("zombies_dropped", observation)
                    observation, _, _, truncated, info = env.step(0)
                    self.assertFalse(truncated)  # Entity truncation is not episode termination.
                    self.assertEqual(observation["zombie_mask"].sum(), 256)
                    self.assertEqual(info["zombies_dropped"], 4)
                    self.assertTrue(info["zombie_truncated"])
                finally:
                    env.close()

    def test_continuous_mode_never_requests_pause_and_reports_real_elapsed_ticks(self):
        client = ContinuousFactClient()
        env = PvZEnv(
            client,
            level_id=27,
            ticks_per_step=30,
            simulation_mode="continuous",
        )

        _, reset_info = env.reset(seed=7)
        self.assertFalse(client.last_reset_config["deterministic_pause"])
        self.assertEqual(reset_info["simulation_mode"], "continuous")

        _, reward, terminated, truncated, info = env.step(0)
        self.assertEqual(reward, 0.0)
        self.assertFalse(terminated)
        self.assertFalse(truncated)
        self.assertEqual(info["ticks_requested"], 30)
        self.assertEqual(info["ticks_advanced"], 37)
        self.assertEqual(info["game_clock"], 37)

    def test_headless_mode_uses_one_atomic_native_transition(self):
        client = HeadlessFactClient()
        env = PvZEnv(client, level_id=27, ticks_per_step=30, simulation_mode="headless")

        _, info = env.reset(seed=7)
        self.assertEqual(info["simulation_mode"], "headless")
        self.assertTrue(client.headless_active)
        self.assertEqual(client.headless_calls, ["begin"])

        _, reward, terminated, truncated, info = env.step(0)
        self.assertEqual(reward, 0.0)
        self.assertFalse(terminated)
        self.assertFalse(truncated)
        self.assertEqual(info["ticks_advanced"], 30)
        self.assertEqual(info["game_clock"], 30)
        self.assertFalse(info["tick_mismatch"])
        self.assertIsNone(info["tick_warning"])
        self.assertEqual(client.headless_calls[-1], ("step_headless", "noop", 30))
        self.assertEqual(client.call_order, ["reset", "step"])

        _, reward, terminated, truncated, info = env.step(0)
        self.assertEqual(reward, 100.0)
        self.assertTrue(terminated)
        self.assertFalse(truncated)
        self.assertEqual(info["result"], "WON")
        self.assertTrue(info["events"]["won"])
        self.assertFalse(client.headless_active)
        self.assertEqual(client.headless_calls[-1], "end")
        self.assertEqual(client.call_order, ["reset", "step", "step"])

        env.close()
        self.assertTrue(client.closed)

    def test_headless_tick_drift_is_warning_only_and_episode_continues(self):
        client = DriftingHeadlessFactClient()
        env = PvZEnv(client, level_id=27, ticks_per_step=30, simulation_mode="headless")

        env.reset(seed=7)
        with self.assertLogs("pvz_gym.env", level="WARNING") as captured:
            _, reward, terminated, truncated, info = env.step(0)

        self.assertEqual(reward, 0.0)
        self.assertFalse(terminated)
        self.assertFalse(truncated)
        self.assertEqual(info["ticks_requested"], 30)
        self.assertEqual(info["ticks_advanced"], 32)
        self.assertEqual(info["game_clock"], 32)
        self.assertTrue(info["tick_mismatch"])
        self.assertIn("requested=30 actual=32", info["tick_warning"])
        self.assertTrue(any("requested=30 actual=32" in line for line in captured.output))
        self.assertTrue(client.headless_active)
        self.assertEqual(client.headless_calls[-1], ("step_headless", "noop", 30))

    def test_short_terminal_step_does_not_warn_about_drift(self):
        env = PvZEnv(
            EarlyTerminalHeadlessFactClient(),
            ticks_per_step=30,
            simulation_mode="headless",
        )
        try:
            env.reset()
            env.step(0)
            with self.assertNoLogs("pvz_gym.env", level="WARNING"):
                _, reward, terminated, _, info = env.step(0)
            self.assertTrue(terminated)
            self.assertEqual(reward, 100.0)
            self.assertEqual(info["ticks_advanced"], 9)
            self.assertFalse(info["tick_mismatch"])
            self.assertIsNone(info["tick_warning"])
        finally:
            env.close()


if __name__ == "__main__":
    unittest.main()
