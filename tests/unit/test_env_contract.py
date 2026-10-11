import unittest

import numpy as np

from pvz_gym.action_space import ACTION_SPACE_SIZE
from pvz_gym.env import PvZEnv
from pvz_gym.observation import observation_is_valid
from tests.fake_backend import FactClient


class GymContractTest(unittest.TestCase):
    def test_reset_step_and_sparse_terminal_reward_use_authoritative_backend(self):
        client = FactClient()
        env = PvZEnv(client, level_id=27, ticks_per_step=30)
        observation, info = env.reset(seed=123)
        self.assertTrue(observation_is_valid(observation))
        self.assertTrue(env.observation_space.contains(observation))
        self.assertEqual(info["game_clock"], 0)
        self.assertTrue(info["rng_seed_applied"])
        self.assertEqual(client.last_reset_config["seed"], 123)
        self.assertEqual(env.action_space.n, ACTION_SPACE_SIZE)

        observation, reward, terminated, truncated, info = env.step(0)
        self.assertEqual(reward, 0.0)
        self.assertFalse(terminated)
        self.assertFalse(truncated)
        self.assertEqual(info["game_clock"], 30)
        self.assertEqual(info["ticks_requested"], 30)
        self.assertEqual(info["ticks_advanced"], 30)
        self.assertTrue(info["action_result"]["success"])
        self.assertEqual(info["automation_collected"], 2)
        self.assertEqual(client.call_order, ["reset", "step"])

        _, reward, terminated, truncated, info = env.step(0)
        self.assertEqual(reward, 100.0)
        self.assertTrue(terminated)
        self.assertFalse(truncated)
        self.assertEqual(info["result"], "WON")
        self.assertTrue(info["events"]["won"])
        self.assertEqual(client.call_order, ["reset", "step", "step"])

    def test_action_mask_and_optional_time_limit(self):
        env = PvZEnv(FactClient(), max_episode_steps=1)
        env.reset()
        mask = env.get_action_mask()
        self.assertEqual(len(mask), env.action_space.n)
        self.assertTrue(mask[0])
        *_, truncated, _ = env.step(0)
        self.assertTrue(truncated)

    def test_terminal_fallback_is_native_step_fact_not_python_retry(self):
        env = PvZEnv(FactClient(terminal_observation_unavailable=True))
        initial, _ = env.reset()
        env.step(0)
        observation, reward, terminated, truncated, info = env.step(0)
        self.assertTrue(terminated)
        self.assertFalse(truncated)
        self.assertEqual(reward, 100.0)
        self.assertEqual(info["result"], "WON")
        self.assertTrue(info["terminal_observation_fallback"])
        self.assertEqual(set(observation), set(initial))
        self.assertTrue(observation_is_valid(observation))

    def test_close_closes_owned_session_backend(self):
        client = FactClient()
        env = PvZEnv(client)
        env.close()
        self.assertTrue(client.closed)


if __name__ == "__main__":
    unittest.main()
