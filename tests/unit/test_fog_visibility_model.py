from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from pvz_gym.action_space import ACTION_SPACE_SIZE
from pvz_gym.observation import OBSERVATION_SCHEMA, encode_observation
from train.dqn.agent import DoubleDQNAgent
from train.dqn.config import DoubleDQNConfig
from train.dqn.networks import SpatialQNetwork
from train.dqn.replay import ReplayBuffer
from tests.unit.test_fog_visibility import fog_snapshot


OLD_SCHEMA = "pvz_observation_entities_256_zombie_grid_imitater_boss_projectiles_v4"


class FogModelContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def test_v4_fields_reach_network_and_replay_roundtrip(self):
        obs = encode_observation(fog_snapshot())
        replay = ReplayBuffer(capacity=1, action_dim=ACTION_SPACE_SIZE)
        replay.add(obs, 0, 0, obs, False, np.ones(ACTION_SPACE_SIZE, dtype=bool), ticks_advanced=150)
        with tempfile.TemporaryDirectory() as tmp:
            restored, _ = ReplayBuffer.load(replay.save(Path(tmp) / "fog.npz"))
            batch = restored.sample(1).observations
        self.assertEqual(obs["grid"].shape, (6, 9, 9))
        self.assertIn("boss_nonspatial", OBSERVATION_SCHEMA)
        for key in obs:
            np.testing.assert_array_equal(batch[key][0], obs[key])
        tensors = {key: torch.as_tensor(value).unsqueeze(0) for key, value in obs.items()}
        tensors["grid"].requires_grad_(True)
        net = SpatialQNetwork(hidden_sizes=(16, 8))
        q = net(tensors)
        self.assertEqual(q.shape, (1, ACTION_SPACE_SIZE))
        q.square().mean().backward()
        self.assertTrue(torch.isfinite(tensors["grid"].grad).all())

    def test_v4_checkpoint_loads_for_resume_and_weights_only(self):
        config = DoubleDQNConfig(device="cpu", hidden_sizes=(16, 8), batch_size=2,
                                 replay_capacity=8, reward_scheme="sparse")
        agent = DoubleDQNAgent(ACTION_SPACE_SIZE, config)
        with tempfile.TemporaryDirectory() as tmp:
            path = agent.save(Path(tmp) / "old.pt", env_step=10, episode=1)
            saved = torch.load(path, map_location="cpu", weights_only=False)
            saved["observation_schema"] = OLD_SCHEMA
            torch.save(saved, path)
            for weights_only in (False, True):
                with self.subTest(weights_only=weights_only):
                    loaded, _, payload = DoubleDQNAgent.from_checkpoint(path, device="cpu", weights_only=weights_only)
                    self.assertEqual(payload["observation_migration"]["source"], OLD_SCHEMA)
                    for key, value in agent.online.state_dict().items():
                        torch.testing.assert_close(loaded.online.state_dict()[key], value)
            saved["observation_schema"] = OLD_SCHEMA.replace("boss_projectiles_v4", "boss_projectiles_fog_visibility_v5")
            torch.save(saved, path)
            with self.assertRaisesRegex(ValueError, "observation schema"):
                DoubleDQNAgent.from_checkpoint(path, device="cpu")

    def test_v4_and_unknown_replay_schemas_are_rejected(self):
        replay = ReplayBuffer(capacity=1, action_dim=1)
        obs = encode_observation(fog_snapshot())
        replay.add(obs, 0, 0, obs, False, [True], ticks_advanced=150)
        with tempfile.TemporaryDirectory() as tmp:
            path = replay.save(Path(tmp) / "old.npz")
            with np.load(path, allow_pickle=False) as data:
                saved = {key: data[key] for key in data.files}
            saved["observation_schema"] = np.asarray(OLD_SCHEMA)
            np.savez(path, **saved)
            with self.assertRaisesRegex(ValueError, "observation schema"):
                ReplayBuffer.load(path)
            saved["observation_schema"] = np.asarray(OLD_SCHEMA.replace("boss_projectiles_v4", "boss_projectiles_fog_visibility_v5"))
            np.savez(path, **saved)
            with self.assertRaisesRegex(ValueError, "observation schema"):
                ReplayBuffer.load(path)
