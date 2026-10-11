import tempfile
import unittest
from pathlib import Path

import numpy as np

from pvz_gym.action_space import ACTION_SPACE_SIZE, COLS, MAX_ROWS, encode_action
from pvz_gym.observation import encode_observation
from train.dqn.agent import DoubleDQNAgent
from train.dqn.config import DoubleDQNConfig
from train.dqn.networks import NETWORK_LOCAL_V6, LegacyLocalSpatialQNetwork, SpatialQNetwork, torch
from train.dqn.replay import ReplayBuffer


@unittest.skipIf(torch is None, "PyTorch is required for local attention")
class LocalAttentionTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.network = LegacyLocalSpatialQNetwork(hidden_sizes=(32, 16)).eval()

    def observation(self, **raw):
        return {name: torch.as_tensor(value).unsqueeze(0)
                for name, value in encode_observation(raw).items()}

    def test_visibility_is_same_row_plus_eight_neighbours(self):
        visible = self.network.cell_cell_visibility
        for row in range(MAX_ROWS):
            for col in range(COLS):
                expected = {
                    r * COLS + c for r in range(MAX_ROWS) for c in range(COLS)
                    if r == row or (abs(r - row) <= 1 and abs(c - col) <= 1)
                }
                actual = set(visible[row * COLS + col].nonzero().flatten().tolist())
                self.assertEqual(actual, expected)
        self.assertEqual(int(visible[2 * COLS + 4].sum()), 15)
        self.assertEqual(int(visible[0].sum()), 11)
        # All three layers use the same set of visible board cells.
        for layer in range(3):
            torch.testing.assert_close(
                self.network.cell_plant_visibility[:, layer * 54:(layer + 1) * 54], visible,
            )

    def test_zombies_use_native_lawn_columns_and_active_rows(self):
        obs = self.observation(
            global_state={"grid_rows": 5},
            zombies=[
                {"row": 2, "x": 900},  # same row, off-board -> edge column
                {"row": 1, "x": 100},  # col 0 with LAWN_XMIN=40
                {"row": 1, "x": 319},  # col 3
                {"row": 1, "x": 321},  # col 3
                {"row": 1, "x": 480},  # col 5
                {"row": 0, "x": 360},  # two rows away
                {"row": 5, "x": 360},  # inactive sixth row
                {"row": -1, "x": 360},
            ],
        )
        visible = self.network._cell_zombie_visibility(obs, obs["zombie_mask"].bool())
        self.assertEqual(visible[0, 2 * COLS + 4, :8].tolist(),
                         [True, False, True, True, True, False, False, False])
        self.assertTrue(bool(visible[0, 2 * COLS, 1]))  # neighbour col 0
        self.assertTrue(bool(visible[0, 2 * COLS + 2, 2]))  # neighbour col 3
        self.assertFalse(bool(visible[0, 5 * COLS, :-1].any()))
        self.assertTrue(bool(visible[..., -1].all()))  # empty-entity token

    def test_entity_columns_match_zombie_grid_boundaries(self):
        for x, expected in ((39, 0), (40, 0), (119, 0), (120, 1), (199, 1), (200, 2)):
            with self.subTest(x=x):
                obs = self.observation(
                    global_state={"grid_rows": 5},
                    zombies=[{"row": 1, "x": x, "body_hp": 100}],
                )
                projected = int(self.network._normalized_x_to_grid_col(obs["zombies"][..., 0])[0, 0])
                grid_col = int(obs["zombie_grid"][0, 1, :, 0].argmax())
                self.assertEqual(projected, expected)
                self.assertEqual(projected, grid_col)

    def test_boss_projectiles_share_native_lawn_column_boundary(self):
        network = SpatialQNetwork(hidden_sizes=(32, 16), boss_projectiles=True).eval()
        query = 1 * COLS + 2
        for x, expected_visible in ((119, False), (120, True)):
            with self.subTest(x=x):
                obs = self.observation(
                    global_state={"grid_rows": 5},
                    boss_projectiles=[{"valid": True, "type": 0, "row": 0, "x": x, "y": 100}],
                )
                active = network._active_cells(obs)
                _, mask, visible = network._boss_projectile_entities(obs, active)
                self.assertTrue(bool(mask[0, 0]))
                self.assertEqual(bool(visible[0, query, 0]), expected_visible)

    def test_invisible_entities_do_not_change_local_tokens_or_q(self):
        # (0,5) is visible to neighbour (1,5), but not target (2,4).
        # This catches two-hop propagation as well as global entity pooling.
        base = self.observation()
        remote = self.observation(
            plants=[{"row": 0, "col": 5, "layer": 1, "type": 2,
                     "hp": 1000, "max_hp": 1000}],
            zombies=[{"row": 0, "x": 440, "type": 23, "body_hp": 5000}],
        )
        target = 2 * COLS + 4
        action = encode_action(card_slot=0, row=2, col=4)
        with torch.no_grad():
            torch.testing.assert_close(self.network._encode(base)[1][:, target],
                                       self.network._encode(remote)[1][:, target], rtol=0, atol=0)
            torch.testing.assert_close(self.network(base)[:, action], self.network(remote)[:, action],
                                       rtol=0, atol=0)
            self.assertGreater(float((self.network(base) - self.network(remote)).abs().max()), 1e-6)

    def test_visible_plants_and_zombies_change_local_q(self):
        base = self.observation()
        action = encode_action(card_slot=0, row=2, col=4)
        nearby = (
            self.observation(plants=[{"row": 1, "col": 5, "layer": 1,
                                      "type": 2, "hp": 100, "max_hp": 100}]),
            self.observation(zombies=[{"row": 1, "x": 440, "type": 2, "body_hp": 100}]),
            self.observation(zombies=[{"row": 2, "x": 40, "type": 2, "body_hp": 100}]),
        )
        with torch.no_grad():
            for obs in nearby:
                self.assertGreater(float((self.network(base)[:, action] -
                                          self.network(obs)[:, action]).abs()), 1e-6)

    def test_zombie_grid_reaches_network_without_individual_slots(self):
        base = self.observation()
        action = encode_action(card_slot=0, row=2, col=4)
        local = {name: value.clone() for name, value in base.items()}
        local["zombie_grid"][0, 1, 5] = torch.tensor([0.8, 1.2])
        remote = {name: value.clone() for name, value in base.items()}
        remote["zombie_grid"][0, 0, 5] = torch.tensor([0.8, 1.2])
        with torch.no_grad():
            self.assertGreater(float((self.network(base)[:, action] -
                                      self.network(local)[:, action]).abs()), 1e-6)
            torch.testing.assert_close(self.network(base)[:, action], self.network(remote)[:, action],
                                       rtol=0, atol=0)
        local["zombie_grid"].requires_grad_()
        self.network(local)[:, action].sum().backward()
        gradient = local["zombie_grid"].grad[0].abs().sum(-1).flatten()
        self.assertGreater(float(gradient[1 * COLS + 5]), 0)
        self.assertEqual(float(gradient[~self.network.cell_cell_visibility[2 * COLS + 4]].sum()), 0)

    def test_only_visible_zombies_receive_candidate_gradient(self):
        obs = self.observation(zombies=[
            {"row": 1, "x": 440, "body_hp": 100},
            {"row": 0, "x": 440, "body_hp": 100},
        ])
        obs["zombies"].requires_grad_()
        action = encode_action(card_slot=0, row=2, col=4)
        self.network(obs)[:, action].sum().backward()
        gradients = obs["zombies"].grad.abs().sum(-1)
        self.assertGreater(float(gradients[0, 0]), 0)
        self.assertEqual(float(gradients[0, 1:].sum()), 0)
        self.assertIsNone(self.network.cell_zombie_attention.relative_bias)
        self.assertFalse(hasattr(self.network, "_cell_zombie_relation"))

    def test_zombies_beyond_individual_capacity_still_reach_grid_input(self):
        zombies = [{"row": 4, "x": 700, "body_hp": 100} for _ in range(256)]
        base = self.observation(zombies=zombies)
        overflow = self.observation(zombies=zombies + [{"row": 1, "x": 440, "body_hp": 1000}])
        torch.testing.assert_close(base["zombies"], overflow["zombies"], rtol=0, atol=0)
        self.assertEqual(int(overflow["zombie_mask"].sum()), 256)
        action = encode_action(card_slot=0, row=2, col=4)
        with torch.no_grad():
            self.assertGreater(float((self.network(base)[:, action] -
                                      self.network(overflow)[:, action]).abs()), 1e-6)

    def test_inactive_rows_do_not_leak_via_grid_or_plant_keys(self):
        base = self.observation(global_state={"grid_rows": 5})
        inactive = self.observation(
            global_state={"grid_rows": 5},
            plants=[{"row": 5, "col": 4, "layer": 1, "type": 2, "hp": 100, "max_hp": 100}],
            zombies=[{"row": 5, "x": 360, "body_hp": 100}],
        )
        action = encode_action(card_slot=0, row=4, col=4)
        with torch.no_grad():
            torch.testing.assert_close(self.network(base)[:, action], self.network(inactive)[:, action],
                                       rtol=0, atol=0)

    def test_noop_changes_with_remote_plants_zombies_and_layout(self):
        base = self.observation(global_state={"grid_rows": 5})
        changed = (
            self.observation(global_state={"grid_rows": 5}, plants=[
                {"row": 0, "col": 0, "layer": 1, "type": 2, "hp": 100, "max_hp": 100}]),
            self.observation(global_state={"grid_rows": 5}, plants=[
                {"row": 4, "col": 8, "layer": 1, "type": 2, "hp": 100, "max_hp": 100}]),
            self.observation(global_state={"grid_rows": 5}, zombies=[
                {"row": 4, "x": 700, "type": 2, "body_hp": 100}]),
        )
        with torch.no_grad():
            base_q = self.network(base)[:, 0]
            changed_q = [self.network(obs)[:, 0] for obs in changed]
            for q in changed_q:
                self.assertGreater(float((q - base_q).abs()), 1e-6)
            # Moving the same plant must also be distinguishable.
            self.assertGreater(float((changed_q[0] - changed_q[1]).abs()), 1e-6)

    def test_noop_receives_gradient_from_every_active_cell(self):
        for rows in (5, 6):
            with self.subTest(rows=rows):
                obs = self.observation(global_state={"grid_rows": rows})
                obs["zombie_grid"].requires_grad_()
                self.network(obs)[:, 0].sum().backward()
                gradient = obs["zombie_grid"].grad[0].abs().sum(-1)
                self.assertTrue(bool((gradient[:rows] > 0).all()))
                self.assertEqual(float(gradient[rows:].sum()), 0)

    def test_noop_ignores_inactive_rows_and_empty_board_is_finite(self):
        base = self.observation(global_state={"grid_rows": 5})
        inactive = self.observation(global_state={"grid_rows": 5}, plants=[
            {"row": 5, "col": 8, "layer": 1, "type": 2, "hp": 100, "max_hp": 100}],
            zombies=[{"row": 5, "x": 700, "body_hp": 100}])
        inactive["grid"][0, 5] = 1
        inactive["zombie_grid"][0, 5] = 1
        inactive["mower_state"][0, 5] = 1
        with torch.no_grad():
            torch.testing.assert_close(self.network(base)[:, 0], self.network(inactive)[:, 0],
                                       rtol=0, atol=0)
            empty = self.observation(global_state={"grid_rows": 0})
            self.assertTrue(bool(torch.isfinite(self.network(empty)).all()))



@unittest.skipIf(torch is None, "PyTorch is required for checkpoint validation")
class LocalCheckpointTest(unittest.TestCase):
    def config(self):
        return DoubleDQNConfig(reward_scheme="staged", hidden_sizes=(32, 16), replay_capacity=8, batch_size=2, device="cpu")

    def test_old_architectures_are_rejected_in_both_load_modes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            agent = DoubleDQNAgent(ACTION_SPACE_SIZE, self.config())
            path = agent.save(root / "source.pt", env_step=90, episode=3)
            payload = torch.load(path, weights_only=False)
            for network_type in ("entity_cell_local_attention_v5", "entity_cell_relative_attention_v4",
                                 "entity_cell_relative_attention_event_heads_v1"):
                payload["network_type"] = network_type
                torch.save(payload, path)
                for weights_only in (False, True):
                    with self.subTest(network_type=network_type, weights_only=weights_only):
                        with self.assertRaisesRegex(ValueError, "network type.*unsupported"):
                            DoubleDQNAgent.from_checkpoint(path, device="cpu", weights_only=weights_only)

    def test_current_checkpoint_round_trips_optimizer_and_predictions(self):
        with tempfile.TemporaryDirectory() as tmp:
            agent = DoubleDQNAgent(ACTION_SPACE_SIZE, self.config())
            obs = agent._observation_tensors(encode_observation({}), batched=False)
            (agent.online(obs) - 1).square().mean().backward()
            agent.optimizer.step()
            agent.update_count = 1
            self.assertGreater(float(agent.online.action_score[-1].weight.abs().sum()), 0)
            path = agent.save(Path(tmp) / "current.pt", env_step=10, episode=2)
            restored, _, _ = DoubleDQNAgent.from_checkpoint(path, device="cpu")
            self.assertEqual(restored.update_count, 1)
            self.assertEqual(len(restored.optimizer.state), len(agent.optimizer.state))
            with torch.no_grad():
                torch.testing.assert_close(restored.online(obs), agent.online(obs), rtol=0, atol=0)
            # Missing weights in a current checkpoint must not be silently randomized.
            payload = torch.load(path, weights_only=False)
            payload["online"].pop("cell_encoder.0.weight")
            torch.save(payload, path)
            with self.assertRaisesRegex(RuntimeError, "Missing key"):
                DoubleDQNAgent.from_checkpoint(path, device="cpu")


if __name__ == "__main__":
    unittest.main()
