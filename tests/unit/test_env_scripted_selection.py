import unittest

from pvz_gym.env import PvZEnv
from pvz_gym.types import GameStatus
from tests.unit.test_episode_runner import SameEpisodeBackend


class ScriptedSelectionEnvTests(unittest.TestCase):
    def test_adaptive_shuffle_handoff_and_episode_identity(self):
        backend = SameEpisodeBackend()
        env = PvZEnv(backend, seed_selection_mode="adaptive_shuffled", level_id=4, playthrough=2)
        _, info = env.reset(seed=42)
        self.assertEqual(info["seed_selection"]["mode"], "adaptive_shuffled")
        self.assertEqual((backend.reset_calls, backend.wait_calls), (1, 1))
        self.assertEqual((info["level_id"], info["playthrough"], info["episode_seed"]), (4, 2, 42))
        self.assertTrue(info["rng_seed_applied"])

    def test_custom_same_episode_handoff_preserves_applied_rng(self):
        backend = SameEpisodeBackend()
        env = PvZEnv(backend, seed_selection_mode="custom", seed_types=(0,), level_id=4)
        _, info = env.reset(seed=123)
        self.assertEqual(backend.reset_calls, 1)
        self.assertEqual(backend.wait_calls, 1)
        self.assertTrue(backend.confirmed)
        self.assertEqual(backend.selected, {0})
        self.assertEqual(info["seed_selection"]["actual"], [0])
        self.assertTrue(info["rng_seed_applied"])
        self.assertEqual(info["result"], GameStatus.PLAYING.name)

    def test_adaptive_uses_game_chooser_and_no_second_reset(self):
        backend = SameEpisodeBackend()
        env = PvZEnv(backend, seed_selection_mode="adaptive", level_id=4)
        _, info = env.reset()
        self.assertEqual(backend.selected, {1})
        self.assertEqual((backend.reset_calls, backend.wait_calls), (1, 1))
        self.assertIn("预览不可用", info["seed_selection"]["warnings"][0])

    def test_shuffled_training_same_episode_and_reported_mode(self):
        backend = SameEpisodeBackend()
        env = PvZEnv(backend, seed_selection_mode="fixed_shuffled", level_id=4)
        _, info = env.reset(seed=42)
        self.assertEqual(info["seed_selection"]["mode"], "fixed_shuffled")
        self.assertEqual(backend.selected, {1})
        self.assertEqual((backend.reset_calls, backend.wait_calls), (1, 1))
        self.assertTrue(info["rng_seed_applied"])

    def test_custom_requires_cards(self):
        with self.assertRaisesRegex(ValueError, "requires"):
            PvZEnv(SameEpisodeBackend(), seed_selection_mode="custom")

    def test_fixed_training_same_episode_and_reported_mode(self):
        backend = SameEpisodeBackend()
        env = PvZEnv(backend, seed_selection_mode="fixed", level_id=4)
        _, info = env.reset(seed=42)
        self.assertEqual(info["seed_selection"]["mode"], "fixed")
        self.assertEqual(backend.selected, {1})
        self.assertEqual((backend.reset_calls, backend.wait_calls), (1, 1))
        self.assertTrue(info["rng_seed_applied"])


if __name__ == "__main__":
    unittest.main()
