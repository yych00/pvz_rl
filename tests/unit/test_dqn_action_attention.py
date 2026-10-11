import tempfile
import unittest
from pathlib import Path

import numpy as np

from pvz_gym.action_space import (
    ACTION_SPACE_SIZE, COB_BASE, COLS, INTERACT_BASE, PLANT_BASE,
    SHOVEL_NORMAL_BASE, SHOVEL_PUMPKIN_BASE, decode_action, encode_action,
)
from pvz_gym.observation import encode_observation
from train.dqn.agent import DoubleDQNAgent
from train.dqn.config import DoubleDQNConfig
from train.dqn.networks import NETWORK_LOCAL_V6, NETWORK_SPATIAL, SpatialQNetwork, torch
from train.dqn.optimized_controller import DoubleDQNTrainerController
from train.dqn.replay import ReplayBuffer


@unittest.skipIf(torch is None, "PyTorch required for action attention")
class ActionAttentionTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.network = SpatialQNetwork(hidden_sizes=(32, 16)).eval()
        self.cell = 2 * COLS + 4
        self.actions = [encode_action(card_slot=0, row=2, col=4),
                        SHOVEL_NORMAL_BASE + self.cell, SHOVEL_PUMPKIN_BASE + self.cell,
                        COB_BASE + self.cell, INTERACT_BASE + self.cell]

    def observation(self, **raw):
        return {name: torch.as_tensor(value).unsqueeze(0)
                for name, value in encode_observation(raw).items()}

    def test_every_query_matches_the_public_action_layout(self):
        for action_id in range(ACTION_SPACE_SIZE):
            action = decode_action(action_id)
            family = {"noop": 0, "plant": 1, "shovel": 2, "cob_fire": 3, "interact_cell": 4}[action.type]
            self.assertEqual(int(self.network.action_families_index[action_id]), family)
            if family:
                row = action.target_row if action.type == "cob_fire" else action.row
                col = action.target_col if action.type == "cob_fire" else action.col
                self.assertEqual(int(self.network.action_target_cells[action_id]), row * COLS + col)

    def test_imitater_identity_reaches_plant_query_and_q_score(self):
        card = {"slot": 0, "plant_type": 14, "cost": 75, "ready": True}
        regular = self.observation(cards=[card])
        imitation = self.observation(cards=[{**card, "is_imitater": True}])
        action = self.actions[0]
        with torch.no_grad():
            first = self.network._action_branches(regular)[0][:, action]
            second = self.network._action_branches(imitation)[0][:, action]
            self.assertGreater(float((first - second).abs().max()), 1e-6)
            self.assertGreater(float((self.network(regular)[:, action]
                                      - self.network(imitation)[:, action]).abs().max()), 1e-6)

    def test_remote_entities_reach_every_family_only_through_global_branch(self):
        base = self.observation()
        for raw in (
            {"plants": [{"row": 0, "col": 5, "layer": 1, "type": 2, "hp": 100, "max_hp": 100}]},
            {"zombies": [{"row": 0, "x": 440, "type": 23, "body_hp": 5000}]},
        ):
            remote = self.observation(**raw)
            with torch.no_grad():
                base_branches = self.network._action_branches(base)
                remote_branches = self.network._action_branches(remote)
                for action in self.actions:
                    torch.testing.assert_close(base_branches[0][:, action], remote_branches[0][:, action],
                                               rtol=0, atol=0)
                    torch.testing.assert_close(base_branches[2][:, action], remote_branches[2][:, action],
                                               rtol=0, atol=0)
                    self.assertGreater(float((base_branches[1][:, action]
                                              - remote_branches[1][:, action]).abs().max()), 1e-6)
                    self.assertGreater(float((self.network(base)[:, action]
                                              - self.network(remote)[:, action]).abs()), 1e-6)

    def test_local_reader_receives_only_visible_entity_gradients(self):
        observation = self.observation(zombies=[
            {"row": 1, "x": 440, "body_hp": 100},
            {"row": 0, "x": 440, "body_hp": 100},
        ])
        for action in self.actions:
            with self.subTest(action=action):
                observation["zombies"] = observation["zombies"].detach().requires_grad_()
                observation["zombie_grid"] = observation["zombie_grid"].detach().requires_grad_()
                self.network._action_branches(observation)[2][:, action].square().sum().backward()
                gradient = observation["zombies"].grad.abs().sum(-1)[0]
                self.assertGreater(float(gradient[0]), 0)
                self.assertEqual(float(gradient[1:].sum()), 0)
                grid_gradient = observation["zombie_grid"].grad.abs().sum(-1).flatten()
                visibility = self.network.cell_cell_visibility[self.cell]
                self.assertGreater(float(grid_gradient[visibility].sum()), 0)
                self.assertEqual(float(grid_gradient[~visibility].sum()), 0)

    def test_cards_families_layers_and_targets_condition_both_readers(self):
        observation = self.observation(cards=[
            {"slot": 0, "plant_type": 0, "cost": 100, "ready": True},
            {"slot": 1, "plant_type": 2, "cost": 150, "ready": True},
        ], zombies=[{"row": 2, "x": 440, "body_hp": 100}])
        queries, global_features, local_features, _ = self.network._action_branches(observation)
        candidates = self.actions + [encode_action(card_slot=1, row=2, col=4),
                                     encode_action(card_slot=0, row=2, col=5)]
        for features in (queries, global_features, local_features):
            for action in candidates[1:]:
                self.assertGreater(float((features[:, candidates[0]] - features[:, action]).abs().max()), 1e-6)

    def test_global_reader_sees_other_cards_and_plant_positions(self):
        base = self.observation(global_state={"grid_rows": 5})
        variants = [self.observation(global_state={"grid_rows": 5}, cards=[
            {"slot": 1, "plant_type": 2, "cost": 150, "ready": True}])]
        for row, col in ((0, 0), (4, 8)):
            variants.append(self.observation(global_state={"grid_rows": 5}, plants=[
                {"row": row, "col": col, "layer": 1, "type": 2, "hp": 100, "max_hp": 100}]))
        with torch.no_grad():
            baseline = self.network(base)
            predictions = [self.network(obs) for obs in variants]
            for prediction in predictions:
                for action in [0] + self.actions:
                    self.assertGreater(float((prediction[:, action] - baseline[:, action]).abs()), 1e-6)
            self.assertGreater(float((predictions[1] - predictions[2]).abs().max()), 1e-6)

    def test_padding_and_inactive_rows_cannot_leak_even_with_nan_garbage(self):
        base = self.observation(global_state={"grid_rows": 5})
        garbage = {name: value.clone() for name, value in base.items()}
        for name in ("plants", "grid", "zombie_grid"):
            if name == "plants":
                garbage[name][:, :, 5] = float("nan")
            else:
                garbage[name][:, 5] = float("nan")
        garbage["plant_type"][:, :, 5] = 99999
        garbage["plant_state"][:, :, 5] = 99999
        garbage["grid_terrain"][:, 5] = 99999
        garbage["mower_state"][:, 5] = 99999
        for name in ("cards", "dropped_cards"):
            garbage[name][..., 1:] = float("nan")
        for name in ("card_type", "dropped_card_type", "zombie_type", "zombie_phase",
                     "zombie_height", "zombie_target_row", "zombie_target_col", "zombie_row"):
            garbage[name].fill_(99999)
        garbage["zombies"].fill_(float("nan"))
        with torch.no_grad():
            torch.testing.assert_close(self.network(base), self.network(garbage), rtol=0, atol=0)
            queries, global_features, local_features, _ = self.network._action_branches(
                self.observation(global_state={"grid_rows": 0}))
            for features in (queries, global_features, local_features):
                self.assertTrue(bool(torch.isfinite(features).all()))
            self.assertEqual(float(local_features[:, 0].abs().sum()), 0)

    def test_zombie_overflow_remains_visible_through_board_tokens(self):
        zombies = [{"row": 4, "x": 700, "body_hp": 100} for _ in range(256)]
        base = self.observation(zombies=zombies)
        overflow = self.observation(zombies=zombies + [{"row": 1, "x": 440, "body_hp": 1000}])
        torch.testing.assert_close(base["zombies"], overflow["zombies"], rtol=0, atol=0)
        with torch.no_grad():
            for action in self.actions:
                self.assertGreater(float((self.network(base)[:, action]
                                          - self.network(overflow)[:, action]).abs()), 1e-6)

    def test_real_td_training_reaches_both_readers_for_each_target_family(self):
        config = DoubleDQNConfig(reward_scheme="staged", hidden_sizes=(32, 16), replay_capacity=8,
                                 batch_size=2, n_step=1, device="cpu")
        observation = encode_observation({"global_state": {"grid_rows": 5}, "cards": [
            {"slot": 0, "plant_type": 0, "cost": 100, "ready": True}], "zombies": [
            {"row": 1, "x": 440, "body_hp": 100}, {"row": 0, "x": 440, "body_hp": 100}]})
        mask = np.zeros(ACTION_SPACE_SIZE, dtype=bool)
        mask[0] = True
        for action in [0] + self.actions:
            with self.subTest(action=action):
                agent = DoubleDQNAgent(ACTION_SPACE_SIZE, config)
                replay = ReplayBuffer(8, ACTION_SPACE_SIZE)
                for _ in range(2):
                    replay.add(observation, action, 1, observation, True, mask, ticks_advanced=100)
                for _ in range(3):
                    self.assertTrue(np.isfinite(agent.learn(replay.sample(2))))
                readers = list(agent.online.action_global_attention)
                if action != 0:
                    readers.append(agent.online.action_local_attention)
                for reader in readers:
                    for projection in (reader.q_proj, reader.k_proj, reader.v_proj):
                        self.assertGreater(float(projection.weight.grad.abs().sum()), 0)


@unittest.skipIf(torch is None, "PyTorch required for attention migration")
class AttentionMigrationTest(unittest.TestCase):
    def test_v6_is_rejected_in_all_load_modes(self):
        config = DoubleDQNConfig(reward_scheme="staged", hidden_sizes=(32, 16),
                                 replay_capacity=8, batch_size=2, n_step=1, device="cpu")
        with self.assertRaisesRegex(ValueError, "unsupported"):
            DoubleDQNAgent(ACTION_SPACE_SIZE, config, network_type=NETWORK_LOCAL_V6)
        source = DoubleDQNAgent(ACTION_SPACE_SIZE, config)
        with tempfile.TemporaryDirectory() as tmp:
            path = source.save(Path(tmp) / "model.pt", env_step=30, episode=3)
            payload = torch.load(path, weights_only=False)
            payload["network_type"] = NETWORK_LOCAL_V6
            torch.save(payload, path)
            for weights_only in (False, True):
                with self.assertRaisesRegex(ValueError, "unsupported"):
                    DoubleDQNAgent.from_checkpoint(path, device="cpu", weights_only=weights_only)


if __name__ == "__main__":
    unittest.main()
