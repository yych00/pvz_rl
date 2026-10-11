import threading
import unittest

from eval.random_runner import RandomAgentConfig, run_random_agent
from pvz_gym.env import PvZEnv
from tests.fake_backend import FactClient


class RandomRunnerTest(unittest.TestCase):
    def test_two_clean_terminal_episodes(self):
        env = PvZEnv(FactClient(), ticks_per_step=30)
        events = []
        summary = run_random_agent(
            env,
            RandomAgentConfig(episodes=2, max_steps=10, random_seed=7),
            on_event=lambda event, data: events.append(event),
        )
        self.assertEqual(summary.completed_episodes, 2)
        self.assertEqual(summary.wins, 2)
        self.assertEqual(summary.losses, 0)
        self.assertEqual(summary.integrity_failures, 0)
        self.assertEqual(summary.episode_lengths, [2, 2])
        self.assertEqual(summary.auto_collected, 8)
        self.assertEqual(events.count("episode_end"), 2)

    def test_pre_set_stop_event_stops_without_reset(self):
        env = PvZEnv(FactClient())
        stop = threading.Event()
        stop.set()
        summary = run_random_agent(
            env,
            RandomAgentConfig(episodes=100, max_steps=10),
            stop_event=stop,
        )
        self.assertTrue(summary.stopped)
        self.assertEqual(summary.completed_episodes, 0)

    def test_reset_failure_aborts_instead_of_fake_running_remaining_episodes(self):
        class ResetFailureEnv:
            def __init__(self):
                self.reset_calls = 0

            def reset(self):
                self.reset_calls += 1
                raise RuntimeError("stuck on terminal UI")

        env = ResetFailureEnv()
        summary = run_random_agent(env, RandomAgentConfig(episodes=100, max_steps=10))
        self.assertEqual(env.reset_calls, 1)
        self.assertEqual(summary.reset_failures, 1)
        self.assertEqual(summary.completed_episodes, 0)
        self.assertEqual(summary.integrity_failures, 1)

    def test_legacy_runner_path_is_same_implementation(self):
        from pvz_gym.random_runner import RandomAgentConfig as LegacyConfig
        from pvz_gym.random_runner import run_random_agent as legacy_run

        self.assertIs(LegacyConfig, RandomAgentConfig)
        self.assertIs(legacy_run, run_random_agent)


if __name__ == "__main__":
    unittest.main()
