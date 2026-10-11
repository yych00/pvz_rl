import tempfile
import unittest
from pathlib import Path

import numpy as np

from pvz_gym.action_space import ACTION_SPACE_SIZE, MAX_CARD_SLOTS, MAX_ROWS, COLS, PLANT_ACTIONS
from pvz_gym.observation import encode_observation
from train.dqn.agent import DoubleDQNAgent
from train.dqn.config import DoubleDQNConfig
from train.dqn.networks import NETWORK_SPATIAL, SpatialQNetwork, torch


@unittest.skipIf(torch is None, "PyTorch required")
class CardOrderInvarianceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def setUp(self):
        torch.manual_seed(41)

    def observations(self, *, boss=False, reward=False):
        cards = [
            {"slot": slot, "plant_type": kind, "cost": cost,
             "cooldown_duration": 1000, "cooldown_remaining": cooldown,
             "ready": cooldown == 0, "is_imitater": imitation}
            for slot, kind, cost, cooldown, imitation in (
                (0, 1, 50, 0, False), (2, 0, 100, 300, False),
                (4, 2, 150, 800, False), (7, 0, 100, 0, True),
            )
        ]
        raw = {"global_state": {"grid_rows": 5, "sun": 175}, "cards": cards,
               "plants": [{"row": 2, "col": 1, "type": 1, "hp": 100, "max_hp": 100}],
               "zombies": [{"row": 2, "x": 400, "type": 2, "body_hp": 200}]}
        first = encode_observation(raw, boss_projectiles=boss)
        second = encode_observation({**raw, "cards": []}, boss_projectiles=boss)
        obs = {name: torch.as_tensor(np.stack((value, second[name])))
               for name, value in first.items()}
        if reward:
            obs["reward_context"] = torch.tensor([[.1, .2, .3], [.3, .1, .2]])
        return obs

    def permute(self, obs):
        permutation = torch.stack([torch.randperm(MAX_CARD_SLOTS) for _ in range(2)])
        result = dict(obs)
        result["card_type"] = obs["card_type"].gather(1, permutation)
        result["cards"] = obs["cards"].gather(1, permutation[..., None].expand_as(obs["cards"]))
        return result, permutation

    def action_indices(self, permutation):
        cells = MAX_ROWS * COLS
        plants = 1 + permutation[..., None] * cells + torch.arange(cells)
        tail = torch.arange(1 + PLANT_ACTIONS, ACTION_SPACE_SIZE).expand(2, -1)
        return torch.cat((torch.zeros(2, 1, dtype=torch.long), plants.flatten(1), tail), dim=1)

    def test_full_q_and_masked_greedy_action_follow_cards_not_slots(self):
        for boss, reward in ((False, False), (True, False), (False, True), (True, True)):
            with self.subTest(boss=boss, reward=reward):
                network = SpatialQNetwork(hidden_sizes=(32, 16), boss_projectiles=boss,
                                          reward_context=reward).eval()
                self.assertFalse(hasattr(network, "card_slot_embedding"))
                obs = self.observations(boss=boss, reward=reward)
                with torch.no_grad():
                    baseline = network(obs)
                    for _ in range(3):
                        swapped, permutation = self.permute(obs)
                        action_indices = self.action_indices(permutation)
                        actual = network(swapped)
                        expected = baseline.gather(1, action_indices)
                        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
                        mask = torch.zeros_like(baseline, dtype=torch.bool)
                        mask[:, 0] = True
                        # Distinct legal target cells for each available card.
                        for slot in (0, 2, 4, 7):
                            mask[0, 1 + slot * (MAX_ROWS * COLS) + slot] = True
                        swapped_mask = mask.gather(1, action_indices)
                        before = baseline.masked_fill(~mask, -torch.inf).argmax(1)
                        after = actual.masked_fill(~swapped_mask, -torch.inf).argmax(1)
                        mapped = action_indices.gather(1, after[:, None]).squeeze(1)
                        torch.testing.assert_close(mapped, before)

    def test_selected_action_td_gradients_are_invariant(self):
        network = SpatialQNetwork(hidden_sizes=(32, 16), reward_context=True).train()
        obs = self.observations(boss=True, reward=True)
        swapped, permutation = self.permute(obs)
        inverse = permutation.argsort(1)
        cells = MAX_ROWS * COLS
        actions = torch.tensor([[1, 1 + 2 * cells + 8, 0, 1 + PLANT_ACTIONS + 7],
                                [1 + 7 * cells + 3, 0, 1 + PLANT_ACTIONS + 61, ACTION_SPACE_SIZE - 1]])
        moved = actions.clone()
        plant = (actions >= 1) & (actions <= PLANT_ACTIONS)
        slots = ((actions - 1) // cells).clamp(0, MAX_CARD_SLOTS - 1)
        moved[plant] = (1 + inverse.gather(1, slots) * cells + (actions - 1) % cells)[plant]
        expected = network.forward_selected_actions(obs, actions)
        actual = network.forward_selected_actions(swapped, moved)
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
        with torch.no_grad():
            torch.testing.assert_close(actual, network(swapped).gather(1, moved), rtol=1e-5, atol=1e-6)
        targets = torch.tensor([[1., -.5, .2, 2.], [-1., .3, -.2, .7]])
        params = tuple(network.parameters())
        first = torch.autograd.grad(torch.nn.functional.smooth_l1_loss(expected, targets), params)
        second = torch.autograd.grad(torch.nn.functional.smooth_l1_loss(actual, targets), params)
        for (name, _), before, after in zip(network.named_parameters(), first, second):
            with self.subTest(parameter=name):
                torch.testing.assert_close(after, before, rtol=2e-4, atol=2e-6)

    def test_v9_checkpoints_resume_but_v7_and_v8_are_rejected(self):
        config = DoubleDQNConfig(hidden_sizes=(32, 16), device="cpu", boss_projectiles=True)
        agent = DoubleDQNAgent(ACTION_SPACE_SIZE, config)
        with tempfile.TemporaryDirectory() as tmp:
            path = agent.save(Path(tmp) / "model.pt", env_step=10, episode=2)
            loaded, _, _ = DoubleDQNAgent.from_checkpoint(path, device="cpu")
            self.assertEqual(loaded.network_type, NETWORK_SPATIAL)
            obs = self.observations(boss=True, reward=True)
            with torch.no_grad():
                torch.testing.assert_close(agent.online(obs), loaded.online(obs), rtol=0, atol=0)
            payload = torch.load(path, weights_only=False)
            for old_type in ("entity_action_global_local_attention_v7",
                             "entity_action_global_local_attention_card_set_v8"):
                payload["network_type"] = old_type
                torch.save(payload, path)
                for weights_only in (False, True):
                    with self.subTest(network_type=old_type, weights_only=weights_only):
                        with self.assertRaisesRegex(ValueError, "unsupported"):
                            DoubleDQNAgent.from_checkpoint(path, device="cpu", weights_only=weights_only)

    @unittest.skipUnless(torch is not None and torch.cuda.is_available(), "CUDA required")
    def test_cuda_full_q_and_backward_preserve_card_permutation(self):
        network = SpatialQNetwork(reward_context=True, boss_projectiles=True).cuda().train()
        obs = self.observations(boss=True, reward=True)
        swapped, permutation = self.permute(obs)
        indices = self.action_indices(permutation).cuda()
        obs = {name: value.cuda() for name, value in obs.items()}
        swapped = {name: value.cuda() for name, value in swapped.items()}
        original_q = network(obs)
        swapped_q = network(swapped)
        torch.testing.assert_close(swapped_q, original_q.gather(1, indices), rtol=2e-5, atol=2e-6)
        (swapped_q - .5).square().mean().backward()
        self.assertTrue(all(p.grad is not None and bool(torch.isfinite(p.grad).all())
                            for p in network.parameters()))


if __name__ == "__main__":
    unittest.main()
