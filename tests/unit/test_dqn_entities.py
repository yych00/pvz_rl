import unittest

import numpy as np

from pvz_gym.action_space import ACTION_SPACE_SIZE, PLANT_BASE, PLANT_ACTIONS
from pvz_gym.observation import MAX_ZOMBIES, encode_observation
from train.dqn.networks import SpatialQNetwork, torch


ENTITY_FIELDS = (
    "zombies",
    "zombie_mask",
    "zombie_type",
    "zombie_phase",
    "zombie_height",
    "zombie_row",
    "zombie_target_row",
    "zombie_target_col",
)


@unittest.skipIf(torch is None, "PyTorch is required for entity attention")
class EntityAttentionTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.network = SpatialQNetwork(hidden_sizes=(32, 16)).eval()

    def observation(self, count):
        raw = {"zombies": [{"type": i % 26, "phase": i % 90, "height": i % 10,
                             "row": i % 6, "x": 200 + i, "y": 100,
                             "body_hp": 100 + i, "target_row": i % 6,
                             "target_col": i % 9, "phase_counter": 10 + i}
                            for i in range(count)]}
        return {name: torch.as_tensor(value).unsqueeze(0)
                for name, value in encode_observation(raw).items()}

    def test_zombie_grid_aggregates_count_and_total_hp(self):
        raw = {
            "zombies": [
                {"row": 2, "x": 200, "body_hp": 100, "helm_hp": 50},
                {"row": 2, "x": 220, "body_hp": 200, "shield_hp": 100},
                {"row": 2, "x": 900, "body_hp": 400},
            ]
        }
        observation = encode_observation(raw)
        # x=200/220 -> col 2. Count reference is 4, HP reference is 20000.
        self.assertAlmostEqual(
            float(observation["zombie_grid"][2, 2, 0]),
            float(np.log1p(2.0) / np.log1p(4.0)),
        )
        self.assertAlmostEqual(
            float(observation["zombie_grid"][2, 2, 1]),
            float(np.log1p(450.0) / np.log1p(20000.0)),
        )
        # Off-board spawn x is clamped into the last board column.
        self.assertGreater(float(observation["zombie_grid"][2, 8, 0]), 0.0)

    def test_permuting_entities_and_padding_does_not_change_q(self):
        obs = self.observation(7)
        permutation = torch.randperm(MAX_ZOMBIES)
        shuffled = {
            name: value[:, permutation] if name in ENTITY_FIELDS else value
            for name, value in obs.items()
        }
        with torch.no_grad():
            torch.testing.assert_close(self.network(obs), self.network(shuffled), rtol=1e-5, atol=1e-6)

    def test_padding_garbage_and_empty_boards_do_not_affect_q(self):
        for count in (0, 3):
            with self.subTest(count=count):
                obs = self.observation(count)
                dirty = {name: value.clone() for name, value in obs.items()}
                dirty["zombies"][:, count:] = float("nan")
                for name in ("zombie_type", "zombie_phase", "zombie_height", "zombie_row",
                             "zombie_target_row", "zombie_target_col"):
                    dirty[name][:, count:] = 99999
                with torch.no_grad():
                    q = self.network(obs)
                    self.assertTrue(torch.isfinite(q).all())
                    torch.testing.assert_close(q, self.network(dirty), rtol=0, atol=0)

    def test_all_256_entities_reach_attention_and_receive_gradients(self):
        obs = self.observation(256)
        obs["zombies"].requires_grad_()
        zombies, _mask = self.network._zombie_entities(obs)
        self.assertEqual(tuple(zombies.shape), (1, 256, 128))
        q = self.network(obs)
        self.assertEqual(tuple(q.shape), (1, ACTION_SPACE_SIZE))
        # Global action readers cover every valid zombie on the board.
        q[:, PLANT_BASE:PLANT_BASE + PLANT_ACTIONS].sum().backward()
        gradients = obs["zombies"].grad
        self.assertTrue(torch.isfinite(gradients).all())
        self.assertTrue((gradients.abs().sum(dim=-1) > 0).all())
        for name, parameter in self.network.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(torch.isfinite(parameter.grad).all(), name)

    def test_masked_slots_have_zero_gradient(self):
        obs = self.observation(2)
        obs["zombies"].requires_grad_()
        self.network(obs)[:, PLANT_BASE].sum().backward()
        self.assertGreater(float(obs["zombies"].grad[:, :2].abs().sum()), 0)
        self.assertEqual(float(obs["zombies"].grad[:, 2:].abs().sum()), 0)

    def test_duplicate_entity_is_not_erased_by_a_mean(self):
        obs = self.observation(1)
        duplicate = {name: value.clone() for name, value in obs.items()}
        for name in ENTITY_FIELDS:
            duplicate[name][:, 1] = duplicate[name][:, 0]
        with torch.no_grad():
            self.assertGreater(float((self.network(obs) - self.network(duplicate)).abs().max()), 1e-6)


if __name__ == "__main__":
    unittest.main()
