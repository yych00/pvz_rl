"""CI smoke checks for stable public behavior, independent of training settings."""

import unittest

from pvz_gym import ACTION_SPACE_SIZE, PvZEnv, decode_action, encode_action
from tests.fake_backend import FactClient


class BasicSmokeTest(unittest.TestCase):
    def test_action_encoding_round_trips(self):
        for action_id in (0, 1, ACTION_SPACE_SIZE - 1):
            with self.subTest(action_id=action_id):
                self.assertEqual(encode_action(decode_action(action_id)), action_id)

    def test_environment_reset_and_step(self):
        backend = FactClient()
        env = PvZEnv(backend)
        try:
            observation, info = env.reset()
            self.assertTrue(env.observation_space.contains(observation))
            self.assertIsInstance(info, dict)
            self.assertTrue(env.action_space.contains(0))

            observation, reward, terminated, truncated, info = env.step(0)
            self.assertTrue(env.observation_space.contains(observation))
            self.assertIsInstance(reward, (int, float))
            self.assertIsInstance(terminated, bool)
            self.assertIsInstance(truncated, bool)
            self.assertIsInstance(info, dict)
        finally:
            env.close()
        self.assertTrue(backend.closed)


if __name__ == "__main__":
    unittest.main()
