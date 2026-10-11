from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import asdict, replace
from pathlib import Path
from unittest.mock import patch

import numpy as np

from pvz_gym.action_space import ACTION_SPACE_SIZE
from pvz_gym.observation import empty_observation
from train.common import EnvConfig
from train.dqn import agent as agent_module
from train.dqn.agent import DoubleDQNAgent
from train.dqn.config import DoubleDQNConfig, convert_gamma, saved_discount_settings
from train.dqn.credit import EpisodeEventCredit
from train.dqn.optimized_controller import DoubleDQNTrainerController
from train.dqn.replay import ReplayBatch, ReplayBuffer
from train.dqn.run_state import load_run_config
from train.dqn import train as train_cli


class TickCreditTest(unittest.TestCase):
    def setUp(self):
        self.config = DoubleDQNConfig(gamma=0.5, gamma_reference_ticks=150, reward_scheme="sparse")
        self.observation = empty_observation()
        self.mask = np.zeros(ACTION_SPACE_SIZE, dtype=np.bool_)
        self.mask[0] = True

    def trajectory(self, durations, *, capacity=8, mower_at=None):
        replay = ReplayBuffer(capacity, ACTION_SPACE_SIZE)
        credit = EpisodeEventCredit(replay, self.config)
        for index, ticks in enumerate(durations):
            terminal = index == len(durations) - 1
            credit.add(replay, self.observation, 0, 100 if terminal else 0,
                       self.observation, terminal, False, self.mask,
                       {"result": "WON" if terminal else "PLAYING",
                        "ticks_advanced": ticks, "ticks_requested": 1,
                        "events": {"mowers_used": int(index == mower_at)}})
        return replay, credit




    def test_one_step_mode_also_stores_actual_ticks_and_mower_time(self):
        config = replace(self.config, credit_assignment="one_step")
        replay = ReplayBuffer(2, ACTION_SPACE_SIZE)
        credit = EpisodeEventCredit(replay, config)
        credit.add(replay, self.observation, 0, 0, self.observation, False, False, self.mask,
                   {"ticks_advanced": 225, "events": {"mowers_used": 1}})
        self.assertEqual(credit.first_mower_tick, 225)
        np.testing.assert_array_equal(replay.sample(1).ticks_advanced, [225])

    def test_missing_or_corrupt_saved_ticks_cannot_silently_resume(self):
        replay, _ = self.trajectory([75, 225, 30])
        with tempfile.TemporaryDirectory() as tmp:
            path = replay.save(Path(tmp) / "replay.npz")
            with np.load(path) as payload:
                data = {name: payload[name] for name in payload.files}
            legacy = {name: value for name, value in data.items() if name != "ticks_advanced"}
            np.savez(path, **legacy)
            with self.assertRaisesRegex(ValueError, "legacy replay"):
                ReplayBuffer.load(path)
            for invalid in (np.array([-1, 225, 30]), np.array([75.5, 225, 30]), np.array([75, 225])):
                data["ticks_advanced"] = invalid
                np.savez(path, **data)
                with self.assertRaisesRegex(ValueError, "tick durations"):
                    ReplayBuffer.load(path)

    def test_old_config_anchors_gamma_to_saved_interval_and_new_config_keeps_unit(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.json"
            path.write_text(json.dumps({"env": {"ticks_per_step": 30}, "gamma": 0.9,
                                        "boss_projectiles": True}), encoding="utf-8")
            config = load_run_config(tmp)
            self.assertEqual(config.gamma_reference_ticks, 100)
            self.assertAlmostEqual(config.gamma, 0.9**(100/30))
            changed = replace(config, env=replace(config.env, ticks_per_step=300))
            self.assertEqual(changed.gamma_reference_ticks, 100)
            path.write_text(json.dumps({"env": {"ticks_per_step": 30}, "gamma_reference_ticks": 150,
                                        "boss_projectiles": True}), encoding="utf-8")
            normalized = load_run_config(tmp)
            self.assertEqual(normalized.gamma_reference_ticks, 100)
            self.assertAlmostEqual(normalized.gamma, 0.9995**(100/150))

    def test_reference_unit_must_be_a_positive_integer(self):
        for invalid in (0, -1, 1.5, True):
            with self.assertRaisesRegex(ValueError, "gamma_reference_ticks"):
                replace(self.config, gamma_reference_ticks=invalid)

    def test_normalization_is_idempotent_and_preserves_legacy_discount_rate(self):
        old = {"gamma": 0.9995, "env": {"ticks_per_step": 150}, "boss_projectiles": True,
               "exploration_type_weights": [2, 4, 0.01, 0.05]}
        normalized = saved_discount_settings(old, 150)
        self.assertAlmostEqual(normalized["gamma"], 0.9996666388827142)
        self.assertAlmostEqual(normalized["gamma"]**300, 0.9995**200, places=12)
        for _ in range(5):
            self.assertEqual(saved_discount_settings(normalized, 30), normalized)
        self.assertEqual(convert_gamma(1.0, 30), 1.0)
        with self.assertRaisesRegex(ValueError, "unsupported discount_schema_version"):
            saved_discount_settings({"discount_schema_version": 99})
        with self.assertRaisesRegex(ValueError, "missing gamma_reference_ticks"):
            saved_discount_settings({"discount_schema_version": 1, "gamma": 0.999})
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.json"
            path.write_text(json.dumps(old), encoding="utf-8")
            config = load_run_config(tmp)
            self.assertEqual(config.exploration_type_weights, (2, 4, 0.01, 0.05, 0.0))
            path.write_text(json.dumps(asdict(replace(config, env=EnvConfig(ticks_per_step=30)))), encoding="utf-8")
            self.assertEqual(load_run_config(tmp).gamma, config.gamma)







    def test_cli_changes_interval_independently_and_exposes_discount_settings(self):
        for argv, expected in ((["train", "--ticks-per-step", "30"], (30, 0.999, 100)),
                               (["train", "--ticks-per-step", "300", "--gamma", "0.9",
                                 "--gamma-reference-ticks", "75"], (300, 0.9**(100/75), 100))):
            with patch("sys.argv", argv), patch.object(train_cli, "train") as launch:
                train_cli.main()
            config = launch.call_args.args[0]
            self.assertEqual((config.env.ticks_per_step, config.gamma_reference_ticks), (expected[0], expected[2]))
            self.assertAlmostEqual(config.gamma, expected[1])


@unittest.skipIf(agent_module.torch is None, "PyTorch is required")
class TickTDTest(unittest.TestCase):
    def batch(self):
        observation = empty_observation()
        observations = {name: np.stack([value] * 4) for name, value in observation.items()}
        mask = np.zeros((4, ACTION_SPACE_SIZE), dtype=np.bool_)
        mask[:, 0] = True
        return ReplayBatch(observations, np.zeros(4, dtype=np.int64), np.zeros(4, dtype=np.float32),
                           observations, np.array([False, False, False, True]), mask,
                           np.array([0, 75, 300, 1], dtype=np.int64))

    def test_td_uses_sample_durations_and_terminal_mask(self):
        torch = agent_module.torch

        class ConstantQ(torch.nn.Module):
            def __init__(self, value):
                super().__init__()
                self.value = torch.nn.Parameter(torch.tensor(float(value)))

            def forward(self, observation):
                return self.value.expand(observation["global"].shape[0], ACTION_SPACE_SIZE)

        for ticks_per_step in (1, 600):
            config = DoubleDQNConfig(reward_scheme="staged", env=EnvConfig(ticks_per_step=ticks_per_step), gamma=0.5,
                                    gamma_reference_ticks=150,
                                    hidden_sizes=(16, 8), device="cpu")
            agent = DoubleDQNAgent(ACTION_SPACE_SIZE, config)
            agent.online, agent.target = ConstantQ(0), ConstantQ(2)
            batch = self.batch()
            expected_targets = torch.tensor([2, 2 * 0.5**0.5, 0.5, 0])
            expected = torch.nn.functional.smooth_l1_loss(torch.zeros(4), expected_targets)
            loss = agent._one_step_loss(batch)
            self.assertAlmostEqual(float(loss), float(expected), places=6)
            agent.config = replace(config, env=EnvConfig(ticks_per_step=600))
            np.testing.assert_allclose(agent._transition_discounts(batch).numpy(), [1, 0.5**0.5, 0.25, 0.5**(1/150)])

    def test_checkpoint_migration_and_explicit_unit_round_trip(self):
        torch = agent_module.torch
        config = DoubleDQNConfig(reward_scheme="staged", env=EnvConfig(ticks_per_step=30), gamma_reference_ticks=150,
                                credit_assignment="one_step", hidden_sizes=(16, 8), device="cpu")
        agent = DoubleDQNAgent(ACTION_SPACE_SIZE, config)
        with tempfile.TemporaryDirectory() as tmp:
            path = agent.save(Path(tmp) / "checkpoint.pt", env_step=1, episode=0)
            loaded, current, _ = DoubleDQNAgent.from_checkpoint(path, device="cpu")
            self.assertEqual(current.gamma_reference_ticks, 100)
            self.assertAlmostEqual(current.gamma, config.gamma**(100/150))
            for name, value in agent.online.state_dict().items():
                self.assertTrue(torch.equal(loaded.online.state_dict()[name], value))
            payload = torch.load(path, weights_only=False)
            payload["config"].pop("gamma_reference_ticks")
            payload["config"].pop("discount_schema_version")
            payload["config"]["exploration_type_weights"] = [2, 4, 0.01, 0.05]
            torch.save(payload, path)
            _, migrated, _ = DoubleDQNAgent.from_checkpoint(path, device="cpu")
            self.assertEqual(migrated.gamma_reference_ticks, 100)
            self.assertAlmostEqual(migrated.gamma, config.gamma**(100/30))
            self.assertEqual(migrated.exploration_type_weights, (2, 4, 0.01, 0.05, 0.0))
            with patch("sys.argv", ["train", "--resume", str(path), "--ticks-per-step", "300"]), \
                 patch.object(train_cli, "train") as launch:
                train_cli.main()
            resumed = launch.call_args.args[0]
            self.assertEqual((resumed.env.ticks_per_step, resumed.gamma_reference_ticks), (300, 100))
            self.assertAlmostEqual(resumed.gamma, config.gamma**(100/30))
            with patch("sys.argv", ["train", "--resume", str(path), "--gamma", "0.999"]), \
                 patch.object(train_cli, "train") as launch:
                train_cli.main()
            self.assertEqual(launch.call_args.args[0].gamma, 0.999)


if __name__ == "__main__":
    unittest.main()
