from __future__ import annotations

import unittest
from dataclasses import replace

from train.dqn.config import DoubleDQNConfig
from train.dqn.reward import REWARD_SCHEME_NAMES, training_reward


class RewardSchemeTests(unittest.TestCase):
    def test_named_schemes_are_registered(self):
        self.assertEqual(REWARD_SCHEME_NAMES, ("sparse", "balanced", "dense_early", "transition_dense", "staged"))

    def test_transition_dense_is_the_new_default(self):
        config = DoubleDQNConfig()
        self.assertEqual(config.reward_scheme, "transition_dense")

    def test_balanced_scheme_calculation(self):
        config = replace(DoubleDQNConfig(), reward_scheme="balanced")
        events = {
            "mowers_used": 1,
            "zombie_threat_delta": 0.5,
            "terminal_plant_score": 0.5,
            "terminal_sun_score": 0.5,
        }
        expected = 1.0 - 0.20 + 0.10 + 0.15 + 0.05
        self.assertAlmostEqual(training_reward(100.0, events, config), expected)

    def test_balanced_clamps_bounded_diagnostics(self):
        config = replace(DoubleDQNConfig(), reward_scheme="balanced")
        events = {
            "zombie_threat_delta": 9.0,
            "terminal_plant_score": 9.0,
            "terminal_sun_score": 9.0,
        }
        self.assertAlmostEqual(training_reward(0.0, events, config), 0.60)

    def test_sparse_preserves_existing_behavior_when_selected(self):
        config = replace(DoubleDQNConfig(), reward_scheme="sparse")
        self.assertAlmostEqual(training_reward(100.0, {"mowers_used": 1}, config), 0.9)
        self.assertAlmostEqual(training_reward(-100.0, {"mowers_used": 1}, config), -1.1)

    def test_dense_early_uses_registered_event_weights(self):
        config = replace(DoubleDQNConfig(), reward_scheme="dense_early")
        events = {
            "mowers_used": 1,
            "plants_lost": 2,
            "zombies_killed": 3,
            "waves_advanced": 1,
            "sun_collected": 25,
            "sun_spent": 500,
        }
        expected = 1.0 - 0.20 - 0.02 + 0.009 + 0.03 + 0.0025
        self.assertAlmostEqual(training_reward(100.0, events, config), expected)

    def test_dense_early_does_not_reward_spending_or_actions_directly(self):
        config = replace(DoubleDQNConfig(), reward_scheme="dense_early")
        self.assertAlmostEqual(
            training_reward(0.0, {"sun_spent": 1000}, config),
            0.0,
        )

    def test_unknown_scheme_is_rejected(self):
        with self.assertRaises(ValueError):
            replace(DoubleDQNConfig(), reward_scheme="unknown")


if __name__ == "__main__":
    unittest.main()
