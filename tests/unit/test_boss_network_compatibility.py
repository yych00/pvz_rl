"""Boss entity attention and legacy checkpoint/replay input boundaries."""
from dataclasses import replace
import tempfile
import json
from pathlib import Path
import unittest

import numpy as np
import torch

from pvz_gym.action_space import ACTION_SPACE_SIZE
from pvz_gym.observation import encode_observation, LEGACY_OBSERVATION_SPEC
from train.dqn.agent import DoubleDQNAgent
from train.dqn.config import DoubleDQNConfig
from train.dqn.run_state import load_run_config
from train.dqn.credit import EpisodeEventCredit
from train.dqn.features import normalize_observation
from train.dqn.networks import SpatialQNetwork
from train.dqn.replay import ReplayBuffer


def observation():
    return encode_observation({"global_state": {"grid_rows": 5, "scene": 4},
        "boss_projectiles": [
            {"type": 0, "row": 1, "x": 610, "y": 240, "valid": True},
            {"type": 1, "row": 4, "x": 820, "y": 440, "valid": True}]})


class BossNetworkCompatibilityTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def config(self, **kwargs):
        return DoubleDQNConfig(device="cpu", hidden_sizes=(16, 8), batch_size=2,
                              replay_capacity=8, n_step=1, reward_scheme="sparse", **kwargs)

    def test_real_entities_reach_attention_and_selected_matches_full(self):
        net = SpatialQNetwork(hidden_sizes=(16, 8), boss_projectiles=True)
        obs = {key: torch.as_tensor(value).unsqueeze(0) for key, value in observation().items()}
        selected = net.forward_selected_actions(obs, torch.arange(ACTION_SPACE_SIZE).unsqueeze(0))
        torch.testing.assert_close(selected, net(obs), rtol=2e-5, atol=2e-6)
        selected.square().mean().backward()
        self.assertGreater(float(net.dynamic_entity_class_embedding.weight.grad[1].abs().sum()), 0)
        self.assertGreater(float(net.type_embedding.weight.grad[-2:].abs().sum()), 0)

    def test_padding_and_inactive_rows_cannot_change_q(self):
        net = SpatialQNetwork(hidden_sizes=(16, 8), boss_projectiles=True)
        base = observation()
        bad = {key: value.copy() for key, value in base.items()}
        bad["boss_projectiles"][2:] = np.nan
        bad["boss_projectile_type"][2:] = 9999
        bad["boss_projectile_row"][2:] = -888
        bad["boss_projectile_mask"][2] = 1
        bad["boss_projectile_type"][2] = 1
        bad["boss_projectile_row"][2] = 6  # Inactive sixth row on a five-row roof.
        tensors = lambda value: {key: torch.as_tensor(array).unsqueeze(0) for key, array in value.items()}
        with torch.no_grad():
            clean_q, bad_q = net(tensors(base)), net(tensors(bad))
        self.assertTrue(bool(torch.isfinite(bad_q).all()))
        torch.testing.assert_close(clean_q, bad_q, rtol=0, atol=0)

    def test_disabled_input_and_legacy_projection_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "required"):
            self.config(boss_projectiles=False)
        with self.assertRaisesRegex(ValueError, "required"):
            normalize_observation(observation(), boss_projectiles=False)

    def test_legacy_dense_context_uses_legacy_shapes(self):
        replay = ReplayBuffer(8, ACTION_SPACE_SIZE, boss_projectiles=False)
        replay.set_reward_context(True)
        self.assertEqual(set(replay.observation_spec), set(LEGACY_OBSERVATION_SPEC) | {"reward_context"})
        self.assertNotIn("boss_projectiles", replay._observations)

    def test_legacy_checkpoint_is_rejected_in_all_load_modes(self):
        source = DoubleDQNAgent(ACTION_SPACE_SIZE, self.config())
        with tempfile.TemporaryDirectory() as tmp:
            path = source.save(Path(tmp) / "current.pt", env_step=33, episode=2)
            original = torch.load(path, weights_only=False)
            for saved_flag in (None, False):
                payload = dict(original)
                payload["config"] = dict(original["config"])
                if saved_flag is None:
                    payload["config"].pop("boss_projectiles")
                else:
                    payload["config"]["boss_projectiles"] = saved_flag
                torch.save(payload, path)
                (Path(tmp) / "config.json").write_text(json.dumps(payload["config"]), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "legacy models"):
                    load_run_config(tmp)
                for weights_only in (False, True):
                    with self.assertRaisesRegex(ValueError, "legacy models"):
                        DoubleDQNAgent.from_checkpoint(path, device="cpu", weights_only=weights_only)
            torch.save(original, path)
            agent, config, _ = DoubleDQNAgent.from_checkpoint(path, device="cpu")
            self.assertTrue(config.boss_projectiles)
            self.assertEqual(config.replay_scheme, "per")



if __name__ == "__main__":
    unittest.main()
