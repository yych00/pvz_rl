from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import asdict, replace
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np

from pvz_gym.action_space import (
    ACTION_SPACE_SIZE, PLANT_BASE, SHOVEL_NORMAL_BASE, SHOVEL_PUMPKIN_BASE,
    COB_BASE, INTERACT_BASE,
)
from pvz_gym.observation import empty_observation
from train.dqn import agent as agent_module
from train.dqn.config import DoubleDQNConfig
from train.dqn.exploration import (
    legal_neighbor_actions, q_guided_fraction, sample_balanced_legal_action,
    sample_neighbor_action, sample_q_guided_action, parse_exploration_ratio,
)
from train.dqn.run_state import load_run_config


class ExplorationModesTest(unittest.TestCase):
    def test_ratio_controls_mixture_and_validates_config(self):
        self.assertEqual(DoubleDQNConfig(reward_scheme="staged").exploration_ratio, (50, 50))
        for mode in ("q_guided_mixed", "q_guided_neighbor"):
            self.assertEqual(q_guided_fraction(mode, (50, 50)), 0.5)
            self.assertEqual(q_guided_fraction(mode, (80, 20)), 0.8)
            self.assertEqual(q_guided_fraction(mode, (100, 0)), 1)
            self.assertEqual(q_guided_fraction(mode, (0, 100)), 0)
            self.assertEqual(q_guided_fraction(mode, (1e308, 1e308)), 0.5)
        self.assertEqual(q_guided_fraction("epsilon_random", (80, 20)), 0)
        for raw in ("", "50", "1:2:3", "0:0", "-1:50", "nan:1", "inf:1", "x:50"):
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                parse_exploration_ratio(raw)
        self.assertEqual(parse_exploration_ratio(" 50 : 50 "), (50, 50))
        self.assertEqual(parse_exploration_ratio("0:100"), (0, 100))
        for ratio in ((0, 0), (-1, 50), (float("nan"), 1), (float("inf"), 1), (50,)):
            with self.subTest(ratio=ratio), self.assertRaises(ValueError):
                DoubleDQNConfig(reward_scheme="staged", exploration_ratio=ratio)
        with self.assertRaises(ValueError):
            DoubleDQNConfig(reward_scheme="staged", exploration_mode="unknown")

    def test_q_guidance_masks_illegal_and_preserves_reward_scale(self):
        mask = np.zeros(ACTION_SPACE_SIZE, dtype=bool)
        mask[[0, 1, 2]] = True
        q = np.full(ACTION_SPACE_SIZE, 1e20)
        q[:3] = [-2, 0, 3]
        rng = np.random.default_rng(5)
        actions = [sample_q_guided_action(q, mask, rng) for _ in range(2000)]
        self.assertTrue(set(actions) <= {0, 1, 2})
        self.assertGreater(actions.count(2), 3 * actions.count(1))
        original_rng = np.random.default_rng(6)
        scaled_rng = np.random.default_rng(6)
        for _ in range(100):
            self.assertEqual(sample_q_guided_action(q, mask, original_rng),
                             sample_q_guided_action(q * 0.001 + 70, mask, scaled_rng))
        mask[:] = False
        with self.assertRaises(RuntimeError):
            sample_q_guided_action(q, mask, rng)

    def test_q_guided_shovel_uses_configured_family_weight(self):
        mask = np.zeros(ACTION_SPACE_SIZE, dtype=bool)
        q = np.zeros(ACTION_SPACE_SIZE)
        weights = DoubleDQNConfig(reward_scheme="staged").exploration_type_weights
        rng = Mock()
        rng.choice.return_value = 0
        for shovel in (SHOVEL_NORMAL_BASE, SHOVEL_PUMPKIN_BASE):
            mask[:] = False
            mask[[0, shovel]] = True
            for shovel_q in (0, 100):
                with self.subTest(shovel=shovel, q=shovel_q):
                    q[shovel] = shovel_q
                    sample_q_guided_action(q, mask, rng, weights)
                    np.testing.assert_allclose(rng.choice.call_args.kwargs["p"], [2 / 2.1, .1 / 2.1])
            sample_q_guided_action(q, mask, rng, (2, 4, 0, .1, .1))
            np.testing.assert_array_equal(rng.choice.call_args.kwargs["p"], [1, 0])
            sample_q_guided_action(q, mask, rng, (2, 4, 2, .1, .1))
            np.testing.assert_allclose(rng.choice.call_args.kwargs["p"], [.5, .5])

    def test_weights_map_all_families_and_preserve_within_family_q_preference(self):
        actions = [0, PLANT_BASE, PLANT_BASE + 1, SHOVEL_NORMAL_BASE,
                   SHOVEL_PUMPKIN_BASE, COB_BASE, INTERACT_BASE]
        mask = np.zeros(ACTION_SPACE_SIZE, dtype=bool)
        mask[actions] = True
        q = np.zeros(ACTION_SPACE_SIZE)
        weights = (1, 2, 3, 4, 5)
        rng = Mock()
        rng.choice.return_value = 0
        sample_q_guided_action(q, mask, rng, weights)
        expected = np.array([1, 1, 1, 1.5, 1.5, 4, 5], dtype=float)
        np.testing.assert_allclose(rng.choice.call_args.kwargs["p"], expected / expected.sum())
        q[PLANT_BASE + 1] = 2
        sample_q_guided_action(q, mask, rng, weights)
        probabilities = rng.choice.call_args.kwargs["p"]
        self.assertAlmostEqual(probabilities[2] / probabilities[1], np.exp(2))
        self.assertAlmostEqual(probabilities[1:3].sum(), 2 / 15)
        self.assertAlmostEqual(probabilities[3:5].sum(), 3 / 15)
        # Multiplying every weight by the same factor leaves probabilities unchanged.
        sample_q_guided_action(q, mask, rng, np.array(weights) * 1e-200)
        np.testing.assert_allclose(rng.choice.call_args.kwargs["p"], probabilities)

    def test_family_probabilities_ignore_action_counts_and_cross_family_q(self):
        mask = np.ones(ACTION_SPACE_SIZE, dtype=bool)
        q = np.linspace(-100, 100, ACTION_SPACE_SIZE)
        rng = Mock()
        rng.choice.return_value = 0
        ranges = ((0, PLANT_BASE), (PLANT_BASE, SHOVEL_NORMAL_BASE),
                  (SHOVEL_NORMAL_BASE, COB_BASE), (COB_BASE, INTERACT_BASE),
                  (INTERACT_BASE, ACTION_SPACE_SIZE))
        for weights in ((2, 4, .1, .1, .1), (0, 0, 0, 0, 0), (1e308,) * 5):
            with self.subTest(weights=weights):
                sample_q_guided_action(q, mask, rng, weights)
                probabilities = rng.choice.call_args.kwargs["p"]
                expected = np.array(weights, dtype=float)
                if expected.max() > 0:
                    expected /= expected.max()
                else:
                    expected.fill(1)
                expected /= expected.sum()
                np.testing.assert_allclose([probabilities[start:end].sum() for start, end in ranges], expected)

    def test_zero_and_invalid_priors_keep_sampling_legal_and_numerically_stable(self):
        mask = np.zeros(ACTION_SPACE_SIZE, dtype=bool)
        mask[[0, PLANT_BASE]] = True
        q = np.zeros(ACTION_SPACE_SIZE)
        q[PLANT_BASE] = 1e30
        rng = Mock()
        rng.choice.return_value = 0
        sample_q_guided_action(q, mask, rng, (1e-300, 0, 0, 0, 0))
        np.testing.assert_array_equal(rng.choice.call_args.kwargs["p"], [1, 0])
        # No positive available weights falls back to equal legal-family probabilities.
        sample_q_guided_action(q, mask, rng, (0, 0, 0, 0, 1))
        np.testing.assert_allclose(rng.choice.call_args.kwargs["p"], [.5, .5])
        for weights in ((1, 2), (1, 1, -1, 1, 1), (float("nan"), 1, 1, 1, 1),
                        (1, 1, float("inf"), 1, 1)):
            with self.subTest(weights=weights), self.assertRaises(ValueError):
                sample_q_guided_action(q, mask, rng, weights)

    def test_eight_neighbors_preserve_card_and_action_subtype_without_row_wrap(self):
        mask = np.ones(ACTION_SPACE_SIZE, dtype=bool)
        # 仅种植动作支持 8 邻域扩散
        for start in (PLANT_BASE, PLANT_BASE + 9 * 54):
            with self.subTest(start=start):
                anchor = start + 2 * 9 + 4
                expected = {start + r * 9 + c for r in (1, 2, 3) for c in (3, 4, 5)} - {anchor}
                self.assertEqual(set(legal_neighbor_actions(anchor, mask)), expected)
                self.assertEqual(set(legal_neighbor_actions(start, mask)),
                                 {start + 1, start + 9, start + 10})
                self.assertEqual(set(legal_neighbor_actions(start + 53, mask)),
                                 {start + 52, start + 43, start + 44})
                sparse = np.zeros(ACTION_SPACE_SIZE, dtype=bool)
                sparse[anchor + 1] = True
                np.testing.assert_array_equal(legal_neighbor_actions(anchor, sparse), [anchor + 1])
                sparse[:] = False
                self.assertEqual(legal_neighbor_actions(anchor, sparse).size, 0)
        # 铲子及非种植动作禁用近格探索，返回空数组
        for non_plant in (0, SHOVEL_NORMAL_BASE, SHOVEL_PUMPKIN_BASE, COB_BASE, INTERACT_BASE):
            self.assertEqual(legal_neighbor_actions(non_plant, mask).size, 0)

    def test_sample_neighbor_action_respects_q_values(self):
        neighbors = [10, 11, 12]
        q_values = np.zeros(ACTION_SPACE_SIZE)
        q_values[10] = -5.0
        q_values[11] = 0.0
        q_values[12] = 5.0
        rng = Mock()
        rng.choice.side_effect = lambda actions, p=None: actions[int(np.argmax(p))]
        chosen = sample_neighbor_action(neighbors, q_values, rng)
        self.assertEqual(chosen, 12)
        probabilities = rng.choice.call_args.kwargs["p"]
        self.assertGreater(probabilities[2], probabilities[1])
        self.assertGreater(probabilities[1], probabilities[0])

    def test_run_config_round_trips_exploration_settings(self):
        config = DoubleDQNConfig(reward_scheme="staged", exploration_mode="q_guided_neighbor", exploration_ratio=(30, 70))
        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp, "config.json").write_text(json.dumps(asdict(config)), encoding="utf-8")
            loaded = load_run_config(tmp)
        self.assertEqual(loaded, config)


@unittest.skipIf(agent_module.torch is None, "PyTorch required")
class AgentExplorationTest(unittest.TestCase):
    def agent(self, mode, ratio=(50, 50), epsilon=0.5):
        config = DoubleDQNConfig(reward_scheme="staged", hidden_sizes=(16, 8), device="cpu",
                                 exploration_mode=mode, exploration_ratio=ratio,
                                 epsilon_start=epsilon, epsilon_end=epsilon, seed=42)
        agent = agent_module.DoubleDQNAgent(ACTION_SPACE_SIZE, config)
        q = agent_module.torch.zeros(1, ACTION_SPACE_SIZE)
        q[0, 0] = 5
        agent.online = Mock(return_value=q)
        agent._observation_tensors = Mock(return_value={})
        return agent

    def test_branch_frequencies_and_ratio(self):
        mask = np.zeros(ACTION_SPACE_SIZE, dtype=bool)
        mask[[0, 10, 11, 40]] = True
        # Distinct sentinel actions expose branch choices without conflating
        # greedy outcomes with Q-guided samples that also select the best Q.
        cases = (("q_guided_mixed", (50, 50), {0: .5, 11: .25, 40: .25}),
                 ("epsilon_random", (50, 50), {0: .5, 40: .5}),
                 ("q_guided_neighbor", (50, 50), {0: .5, 11: .25, 10: .25}),
                 ("q_guided_mixed", (80, 20), {0: .5, 11: .4, 40: .1}),
                 ("q_guided_neighbor", (80, 20), {0: .5, 11: .4, 10: .1}))
        with patch.object(agent_module, "sample_q_guided_action", return_value=11), \
             patch.object(agent_module, "sample_balanced_legal_action", return_value=40), \
             patch.object(agent_module, "legal_neighbor_actions", return_value=np.array([10])):
            for mode, ratio, probabilities in cases:
                with self.subTest(mode=mode, ratio=ratio):
                    agent = self.agent(mode, ratio)
                    actions = [agent.select_action({}, mask, env_step=0) for _ in range(6000)]
                    self.assertEqual(set(actions), set(probabilities))
                    for action, expected in probabilities.items():
                        self.assertAlmostEqual(actions.count(action) / len(actions), expected, delta=.02)

    def test_random_bypasses_network_and_greedy_bypasses_rng(self):
        mask = np.zeros(ACTION_SPACE_SIZE, dtype=bool)
        mask[[0, PLANT_BASE]] = True
        agent = self.agent("epsilon_random", epsilon=1)
        self.assertIn(agent.select_action({}, mask, env_step=0), (0, PLANT_BASE))
        agent.online.assert_not_called()
        for mode in ("q_guided_mixed", "epsilon_random", "q_guided_neighbor"):
            agent = self.agent(mode)
            before = json.dumps(agent._rng.bit_generator.state)
            self.assertEqual(agent.select_action({}, mask, env_step=0, explore=False), 0)
            self.assertEqual(json.dumps(agent._rng.bit_generator.state), before)

    def test_only_neighbor_branch_excludes_shovel_family_and_actions(self):
        mask = np.zeros(ACTION_SPACE_SIZE, dtype=bool)
        mask[[0, SHOVEL_NORMAL_BASE, SHOVEL_NORMAL_BASE + 1]] = True
        for mode, ratio in (("q_guided_mixed", (100, 0)), ("q_guided_neighbor", (100, 0)),
                            ("q_guided_neighbor", (0, 100))):
            with self.subTest(mode=mode, ratio=ratio):
                agent = self.agent(mode, ratio, epsilon=1)
                with patch.object(agent_module, "sample_q_guided_action", return_value=0) as guided:
                    self.assertEqual(agent.select_action({}, mask, env_step=0), 0)
                expected_weights = list(agent.config.exploration_type_weights)
                expected_mask = mask.copy()
                if mode == "q_guided_neighbor" and ratio == (0, 100):
                    expected_weights[2] = 0
                    expected_mask[SHOVEL_NORMAL_BASE:COB_BASE] = False
                self.assertEqual(tuple(guided.call_args.args[3]), tuple(expected_weights))
                np.testing.assert_array_equal(guided.call_args.args[1], expected_mask)
                np.testing.assert_array_equal(mask[[0, SHOVEL_NORMAL_BASE, SHOVEL_NORMAL_BASE + 1]], True)

    def test_neighbor_family_proportions_and_zero_weight_fallback_never_shovel(self):
        mask = np.zeros(ACTION_SPACE_SIZE, dtype=bool)
        mask[[0, PLANT_BASE + 10, PLANT_BASE + 11]] = True
        mask[SHOVEL_NORMAL_BASE:COB_BASE] = True
        agent = self.agent("q_guided_neighbor", ratio=(0, 100), epsilon=1)
        actions = [agent.select_action({}, mask, env_step=0) for _ in range(1500)]
        self.assertTrue(set(actions) <= {0, PLANT_BASE + 10, PLANT_BASE + 11})
        self.assertAlmostEqual(actions.count(0) / len(actions), 2 / 6, delta=.04)
        # Even if only shovel has configured weight, the neighbor fallback excludes it.
        agent.config = replace(agent.config, exploration_type_weights=(0, 0, 1, 0, 0))
        actions = [agent.select_action({}, mask, env_step=0) for _ in range(1500)]
        self.assertTrue(set(actions) <= {0, PLANT_BASE + 10, PLANT_BASE + 11})
        self.assertAlmostEqual(actions.count(0) / len(actions), .5, delta=.04)

    def test_neighbor_falls_back_to_anchor(self):
        for anchor in (0, PLANT_BASE + 10, COB_BASE, INTERACT_BASE):
            mask = np.zeros(ACTION_SPACE_SIZE, dtype=bool)
            mask[anchor] = True
            agent = self.agent("q_guided_neighbor", ratio=(0, 100), epsilon=1)
            self.assertEqual(agent.select_action({}, mask, env_step=0), anchor)

    def test_checkpoint_restores_mode_ratio_and_action_sequence(self):
        observation = empty_observation()
        mask = np.zeros(ACTION_SPACE_SIZE, dtype=bool)
        mask[:18] = True
        for mode in ("q_guided_mixed", "epsilon_random", "q_guided_neighbor"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as tmp:
                config = DoubleDQNConfig(reward_scheme="staged", hidden_sizes=(16, 8), device="cpu",
                                         exploration_mode=mode, exploration_ratio=(80, 20))
                original = agent_module.DoubleDQNAgent(ACTION_SPACE_SIZE, config)
                for _ in range(7):
                    original.select_action(observation, mask, env_step=100)
                path = original.save(Path(tmp, "exploration.pt"), env_step=100, episode=1)
                restored, loaded, _ = agent_module.DoubleDQNAgent.from_checkpoint(path, device="cpu")
                self.assertEqual(loaded.exploration_mode, mode)
                self.assertEqual(loaded.exploration_ratio, (80, 20))
                expected = [original.select_action(observation, mask, env_step=100) for _ in range(25)]
                actual = [restored.select_action(observation, mask, env_step=100) for _ in range(25)]
                self.assertEqual(actual, expected)


if __name__ == "__main__":
    unittest.main()
