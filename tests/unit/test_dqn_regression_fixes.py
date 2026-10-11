import tempfile
import unittest
from pathlib import Path

import numpy as np

from pvz_gym.observation import empty_observation
from train.dqn.config import DoubleDQNConfig
from train.dqn.controller import DoubleDQNTrainerController
from train.dqn.networks import SpatialQNetwork, torch
from train.dqn.replay import ReplayBuffer, remember_selected_action_mask


class DQNRegressionFixesTest(unittest.TestCase):
    def test_replay_persists_current_and_next_action_masks(self):
        observation = empty_observation()
        replay = ReplayBuffer(capacity=2, action_dim=5, seed=7)
        current_mask = np.array([True, False, True, False, False], dtype=np.bool_)
        next_mask = np.array([True, True, False, False, False], dtype=np.bool_)
        remember_selected_action_mask(current_mask)
        replay.add(observation, 2, 1.0, observation, False, next_mask, ticks_advanced=100)

        batch = replay.sample(1)
        np.testing.assert_array_equal(batch.next_action_masks[0], next_mask)

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "paired.replay.npz"
            replay.save(path, env_step=12, episode=3)
            restored, metadata = ReplayBuffer.load(path, action_dim=5, capacity=2, seed=7)
            restored_batch = restored.sample(1)
            np.testing.assert_array_equal(restored_batch.next_action_masks[0], next_mask)
            self.assertEqual(metadata["env_step"], 12)
            self.assertEqual(metadata["episode"], 3)

    @unittest.skipIf(torch is None, "PyTorch is required for masked Dueling tests")
    def test_dueling_center_ignores_illegal_actions(self):
        scores = torch.tensor([[1.0, 3.0, 1000.0]])
        mask = torch.tensor([[True, True, False]])
        centered = SpatialQNetwork._center(scores, mask)
        self.assertEqual(float(centered[0, 0]), -1.0)
        self.assertEqual(float(centered[0, 1]), 1.0)
        self.assertEqual(float(centered[0, 2]), 998.0)
        self.assertEqual(float(centered[0, :2].mean()), 0.0)

    def test_non_divisible_tiered_eval_cadences_are_not_lost(self):
        controller = DoubleDQNTrainerController(
            DoubleDQNConfig(reward_scheme="staged"),
            fast_eval_every_episodes=50,
            full_eval_every_episodes=120,
        )
        self.assertEqual(controller.fast_eval_every_episodes, 50)
        self.assertEqual(controller.full_eval_every_episodes, 120)

        controller._summary.episodes = 50
        self.assertFalse(controller._is_full_eval_episode())
        controller._summary.episodes = 110
        self.assertFalse(controller._is_full_eval_episode())
        controller._summary.episodes = 120
        self.assertTrue(controller._is_full_eval_episode())
        controller._summary.episodes = 600
        self.assertTrue(controller._is_full_eval_episode())

    def test_latest_sync_copies_the_matching_paired_replay(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            source = run_dir / "checkpoint-000000100.pt"
            source.write_bytes(b"model-100")
            controller = DoubleDQNTrainerController(DoubleDQNConfig(reward_scheme="staged"), run_dir=run_dir)

            latest = controller._sync_latest_checkpoint(source)

            self.assertEqual(latest.read_bytes(), b"model-100")
            self.assertEqual(controller._replay_path(), run_dir / "replay.npz")


if __name__ == "__main__":
    unittest.main()
