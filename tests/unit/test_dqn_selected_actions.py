from dataclasses import replace
import copy
import unittest
from unittest.mock import patch

import numpy as np
import torch

from pvz_gym.action_space import ACTION_SPACE_SIZE
from pvz_gym.observation import empty_observation
from train.dqn.agent import DoubleDQNAgent
from train.dqn.config import DoubleDQNConfig
from train.dqn.networks import SpatialQNetwork
from train.dqn.replay import ReplayBatch, ReplayBuffer


def active_observation():
    obs = empty_observation()
    obs['global'][7] = 1
    obs['scene'][0] = 3
    obs['cards'][:, 0] = 1
    obs['card_type'][:] = np.arange(1, 11)
    obs['zombie_mask'][:2] = 1
    obs['zombie_type'][:2] = [2, 3]
    obs['zombie_row'][:2] = [1, 6]
    obs['zombies'][:2, :2] = [[.5, .25], [.8, .7]]
    obs['plants'][0, 2, 2, 0] = 1
    obs['plant_type'][0, 2, 2] = 1
    return obs


class SelectedActionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def test_all_757_ids_and_batched_queries_equal_full(self):
        torch.manual_seed(71)
        net = SpatialQNetwork(hidden_sizes=(16, 8))
        obs = {key: torch.as_tensor(value).unsqueeze(0) for key, value in active_observation().items()}
        with torch.no_grad():
            full = net(obs)
            selected = net.forward_selected_actions(obs, torch.arange(ACTION_SPACE_SIZE).unsqueeze(0))
        torch.testing.assert_close(selected, full, rtol=2e-5, atol=2e-6)
        for invalid in (torch.tensor([-1]), torch.tensor([ACTION_SPACE_SIZE]), torch.tensor([1.0]),
                        torch.tensor([[1], [2]])):
            with self.assertRaises(ValueError):
                net.forward_selected_actions(obs, invalid)

    def test_selected_backward_matches_full_gather_for_every_family(self):
        torch.manual_seed(91)
        net = SpatialQNetwork(hidden_sizes=(16, 8))
        reference = copy.deepcopy(net)
        actions = torch.tensor([0, 1, 540, 541, 595, 649, 703, 756])
        obs = {key: torch.as_tensor(np.stack([value] * len(actions)))
               for key, value in active_observation().items()}
        expected = reference(obs).gather(1, actions[:, None]).squeeze(1)
        actual = net.forward_selected_actions(obs, actions)
        torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)
        expected.square().mean().backward()
        actual.square().mean().backward()
        for (name, parameter), (_, other) in zip(net.named_parameters(), reference.named_parameters()):
            with self.subTest(parameter=name):
                if other.grad is None:
                    self.assertIsNone(parameter.grad)
                else:
                    torch.testing.assert_close(parameter.grad, other.grad, rtol=3e-4, atol=3e-6)
        for reader in net.action_global_attention:
            self.assertGreater(float(reader.q_proj.weight.grad.abs().sum()), 0)
        self.assertGreater(float(net.action_local_attention.q_proj.weight.grad.abs().sum()), 0)

    def test_td_loss_per_error_and_update_match_reference(self):
        config = DoubleDQNConfig(device='cpu', hidden_sizes=(16, 8), n_step=1,
                                 batch_size=8, replay_capacity=16, reward_scheme='sparse',
                                 selected_action_forward=False)
        full = DoubleDQNAgent(ACTION_SPACE_SIZE, config)
        with torch.no_grad():
            full.online.action_score[-1].weight.normal_(0, .02)
        full.target.load_state_dict(full.online.state_dict())
        selected = copy.deepcopy(full)
        selected.config = replace(config, selected_action_forward=True)
        replay = ReplayBuffer(16, ACTION_SPACE_SIZE)
        obs = active_observation()
        for i, action in enumerate([0, 1, 540, 541, 595, 649, 703, 756]):
            mask = np.zeros(ACTION_SPACE_SIZE, dtype=bool)
            mask[[action, (action + 17) % ACTION_SPACE_SIZE]] = True
            if i == 6: mask[::7] = True
            if i == 7: mask[:] = False
            replay.add(obs, action, float(i)/10, obs, i == 7, mask, ticks_advanced=150)
        batch = replay.sample(8, n_step=1)
        batch = replace(batch, weights=np.linspace(.3, 1, 8, dtype=np.float32))
        a, errors_a = full.learn(batch, return_td_errors=True)
        b, errors_b = selected.learn(batch, return_td_errors=True)
        self.assertAlmostEqual(a, b, places=5)
        np.testing.assert_allclose(errors_a, errors_b, rtol=2e-5, atol=2e-6)
        for name, value in full.online.state_dict().items():
            torch.testing.assert_close(value, selected.online.state_dict()[name], rtol=3e-4, atol=3e-6)

    def test_sparse_next_actions_preserve_ties_padding_and_noop_fallback(self):
        config = DoubleDQNConfig(device='cpu', hidden_sizes=(16, 8), reward_scheme='sparse',
                                 selected_action_forward=True)
        agent = DoubleDQNAgent(ACTION_SPACE_SIZE, config)
        # Initialization gives equal Q scores: the first legal ID must win,
        # including rows whose padding ID (NOOP) is itself illegal.
        obs = {key: np.stack([value] * 4) for key, value in active_observation().items()}
        masks = np.zeros((4, ACTION_SPACE_SIZE), dtype=bool)
        masks[0, [12, 35]] = True
        masks[1, 756] = True
        masks[3, [5, 25, 500]] = True
        batch = ReplayBatch(obs, np.zeros(4, dtype=np.int64), np.zeros(4, dtype=np.float32),
                            obs, np.array([False, False, True, False]), masks,
                            ticks_advanced=np.full(4, 150, dtype=np.int64))
        with patch.object(agent.online, 'forward', side_effect=AssertionError('dense forward used')), \
             patch.object(agent.online, 'forward_selected_actions', wraps=agent.online.forward_selected_actions) as online, \
             patch.object(agent.target, 'forward_selected_actions', wraps=agent.target.forward_selected_actions) as target:
            loss = agent._one_step_loss(batch)
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(tuple(online.call_args_list[1].args[1].shape), (4, 3))
        np.testing.assert_array_equal(target.call_args.args[1].numpy(), [12, 756, 0, 5])
        self.assertFalse(masks[2].any())  # replay mask was not mutated
        dense_batch = replace(batch, next_action_masks=np.ones_like(masks))
        with patch.object(agent.online, 'forward', wraps=agent.online.forward) as dense, \
             patch.object(agent.target, 'forward_selected_actions', wraps=agent.target.forward_selected_actions) as target:
            agent._one_step_loss(dense_batch)
        self.assertEqual(dense.call_count, 1)
        np.testing.assert_array_equal(target.call_args.args[1].numpy(), np.zeros(4, dtype=np.int64))

    def test_visualization_reuses_one_batch_forward_and_preserves_exploration_rng(self):
        config = DoubleDQNConfig(device='cpu', hidden_sizes=(16, 8), reward_scheme='sparse',
                                 exploration_mode='epsilon_random', epsilon_start=1, epsilon_end=1)
        reference = DoubleDQNAgent(ACTION_SPACE_SIZE, config)
        visualized = DoubleDQNAgent(ACTION_SPACE_SIZE, config)
        obs = active_observation()
        masks = [np.ones(ACTION_SPACE_SIZE, dtype=bool)] * 3
        expected = reference.select_actions([obs]*3, masks, env_step=0)
        with patch.object(visualized.online, 'forward', wraps=visualized.online.forward) as forward:
            actual = visualized.select_actions([obs]*3, masks, env_step=0, visualize=True)
        self.assertEqual(actual, expected)
        self.assertEqual(forward.call_count, 1)
        self.assertEqual(len(visualized.last_visualizations), 3)
        self.assertEqual([x['selected_action'] for x in visualized.last_visualizations], actual)
        self.assertEqual(visualized._rng.bit_generator.state, reference._rng.bit_generator.state)


if __name__ == '__main__':
    unittest.main()
