from __future__ import annotations

import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np

from pvz_gym.action_space import ACTION_SPACE_SIZE, PLANT_BASE, SHOVEL_NORMAL_BASE
from pvz_gym.observation import empty_observation, encode_observation
from train.dqn.agent import DoubleDQNAgent
from train.dqn.config import DoubleDQNConfig
from train.dqn.networks import SpatialQNetwork, torch
from train.dqn.optimized_controller import DoubleDQNTrainerController
from train.dqn.replay import ReplayBuffer


@unittest.skipIf(torch is None, "PyTorch required for Q initialization")
class InitialQTest(unittest.TestCase):
    def config(self, **kwargs):
        return DoubleDQNConfig(reward_scheme="staged", hidden_sizes=(16, 8), device="cpu", n_step=1,
                               replay_capacity=8, batch_size=2, **kwargs)

    def populated_observation(self):
        return encode_observation({
            "global_state": {"grid_rows": 5, "sun": 200},
            "cards": [{"slot": 0, "plant_type": 0, "cost": 100, "ready": True}],
            "plants": [{"row": 2, "col": 4, "layer": 1, "type": 2, "hp": 100, "max_hp": 100}],
            "zombies": [{"row": 2, "x": 440, "type": 2, "body_hp": 100}],
        })

    def q(self, network, observation):
        tensors = {name: torch.as_tensor(value).unsqueeze(0) for name, value in observation.items()}
        with torch.no_grad():
            return network(tensors)[0]

    def test_all_initial_actions_are_zero_independent_of_weights_states_and_seeds(self):
        for weights in ((2, 4, .01, .1, .1), (0, 1, 8, 0, 2),
                        (1e300, 2e300, 0, 3e300, 4e300)):
            for seed in (0, 7):
                with self.subTest(weights=weights, seed=seed):
                    agent = DoubleDQNAgent(ACTION_SPACE_SIZE,
                                           self.config(exploration_type_weights=weights, seed=seed))
                    expected = torch.zeros(ACTION_SPACE_SIZE)
                    for observation in (empty_observation(), self.populated_observation()):
                        torch.testing.assert_close(self.q(agent.online, observation), expected, rtol=0, atol=0)
                        torch.testing.assert_close(self.q(agent.target, observation), expected, rtol=0, atol=0)
                    self.assertFalse(agent.target.training)
                    self.assertGreater(int(torch.count_nonzero(agent.online.cell_encoder[0].weight)), 0)

    def test_opening_greedy_breaks_zero_q_ties_by_lowest_legal_id(self):
        agent = DoubleDQNAgent(ACTION_SPACE_SIZE, self.config())
        mask = np.zeros(ACTION_SPACE_SIZE, dtype=bool)
        mask[[0, SHOVEL_NORMAL_BASE]] = True
        self.assertEqual(agent.select_action(empty_observation(), mask, env_step=0, explore=False), 0)
        mask[PLANT_BASE + 23] = True
        self.assertEqual(agent.select_action(empty_observation(), mask, env_step=0, explore=False), 0)
        # Exploration weights do not create a greedy preference for shovel.
        custom = DoubleDQNAgent(ACTION_SPACE_SIZE, self.config(exploration_type_weights=(0, 1, 8, 0, 2)))
        self.assertEqual(custom.select_action(empty_observation(), mask, env_step=0, explore=False), 0)
        mask[0] = False
        self.assertEqual(custom.select_action(empty_observation(), mask, env_step=0, explore=False), PLANT_BASE + 23)

    def test_real_td_updates_can_change_initial_q_and_learn_spatial_scores(self):
        agent = DoubleDQNAgent(ACTION_SPACE_SIZE, self.config())
        observation = self.populated_observation()
        initial = self.q(agent.online, observation).clone()
        replay = ReplayBuffer(8, ACTION_SPACE_SIZE)
        mask = np.zeros(ACTION_SPACE_SIZE, dtype=bool)
        mask[0] = True
        for _ in range(2):
            replay.add(observation, PLANT_BASE + 2 * 9 + 4, -1, observation, True, mask,
                       ticks_advanced=100)
        for _ in range(3):
            self.assertTrue(np.isfinite(agent.learn(replay.sample(2))))
        self.assertEqual(agent.update_count, 3)
        self.assertGreater(float((self.q(agent.online, observation) - initial).abs().max()), 1e-5)
        self.assertGreater(int(torch.count_nonzero(agent.online.action_score[-1].weight)), 0)
        self.assertGreater(int(torch.count_nonzero(agent.online.cell_encoder[0].weight.grad)), 0)

    def test_resume_and_weights_only_preserve_saved_heads_with_changed_priors(self):
        config = self.config()
        source = DoubleDQNAgent(ACTION_SPACE_SIZE, config)
        with torch.no_grad():
            source.online.action_score[-1].bias.add_(7)
            source.online.action_family_embedding.weight.add_(torch.arange(5).view(5, 1) * 3)
            source.target.value_head.bias.add_(9)
        source.update_count = 12
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = source.save(Path(tmp, "source.pt"), env_step=42, episode=3)
            current = replace(config, exploration_type_weights=(1, 1, 50, 1, 1))
            for mode in ("resume", "weights_only"):
                with self.subTest(mode=mode):
                    controller = DoubleDQNTrainerController(current, run_dir=Path(tmp),
                                                           checkpoint_path=checkpoint,
                                                           checkpoint_load_mode=mode)
                    restored, _, _, _ = controller._build_agent(ACTION_SPACE_SIZE)
                    self.assertEqual(restored.config.exploration_type_weights, current.exploration_type_weights)
                    for name, value in source.online.state_dict().items():
                        torch.testing.assert_close(restored.online.state_dict()[name], value, rtol=0, atol=0)
                    expected_target = source.target if mode == "resume" else source.online
                    for name, value in expected_target.state_dict().items():
                        torch.testing.assert_close(restored.target.state_dict()[name], value, rtol=0, atol=0)
                    self.assertEqual(restored.update_count, 12 if mode == "resume" else 0)

    def test_zero_initialization_preserves_encoder_parameters(self):
        network = SpatialQNetwork(hidden_sizes=(16, 8))
        before = network.cell_encoder[0].weight.detach().clone()
        network.initialize_zero_q()
        torch.testing.assert_close(network.cell_encoder[0].weight, before, rtol=0, atol=0)
        torch.testing.assert_close(self.q(network, self.populated_observation()),
                                   torch.zeros(ACTION_SPACE_SIZE), rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
