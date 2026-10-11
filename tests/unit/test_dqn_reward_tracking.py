import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr
from dataclasses import fields
from pathlib import Path
from unittest.mock import patch

import numpy as np

from pvz_gym.action_space import ACTION_SPACE_SIZE
from pvz_gym.observation import empty_observation
from train.dqn import agent as agent_module
from train.dqn import train as train_cli
from train.dqn.agent import DoubleDQNAgent
from train.dqn.config import DoubleDQNConfig
from train.dqn.credit import EpisodeEventCredit
from train.dqn.optimized_controller import DoubleDQNTrainerController
from train.dqn.replay import ReplayBatch, ReplayBuffer
from train.dqn.reward import training_reward


class OrdinaryRewardTrackingTest(unittest.TestCase):
    def test_retired_options_are_rejected_by_config_cli_and_replay(self):
        with self.assertRaisesRegex(ValueError, "credit_assignment"):
            DoubleDQNConfig(reward_scheme="staged", credit_assignment="event_heads")
        with self.assertRaisesRegex(TypeError, "event_credit_weight"):
            DoubleDQNConfig(reward_scheme="staged", event_credit_weight=0.2)
        with self.assertRaisesRegex(ValueError, "credit_assignment"):
            ReplayBuffer(8, ACTION_SPACE_SIZE, credit_assignment="event_heads")
        for flags in (("--credit-assignment", "event_heads"), ("--event-credit-weight", "0.2")):
            with self.subTest(flags=flags), patch("sys.argv", ["train", *flags]), redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    train_cli._parse_args()

    def test_every_mower_penalty_stays_on_its_transition(self):
        for scheme in ("sparse", "balanced", "dense_early"):
            with self.subTest(scheme=scheme):
                config = DoubleDQNConfig(reward_scheme=scheme)
                replay = ReplayBuffer(8, ACTION_SPACE_SIZE)
                tracker = EpisodeEventCredit(replay, config)
                observation = empty_observation()
                mask = np.ones(ACTION_SPACE_SIZE, dtype=np.bool_)
                expected = []
                for index, (raw, mowers, ticks) in enumerate(((0, 0, 75), (0, 2, 225), (0, 1, 30), (100, 0, 0))):
                    events = {"mowers_used": mowers}
                    expected.append(training_reward(raw, events, config))
                    tracker.add(replay, observation, 0, raw, observation, index == 3, False, mask,
                                {"events": events, "ticks_advanced": ticks})
                    # Later events must not rewrite rewards on earlier rows.
                    np.testing.assert_allclose(replay._rewards[:index + 1], expected)
                self.assertEqual((tracker.first_mower_step, tracker.first_mower_tick, tracker.ticks), (2, 300, 330))
                penalty = -0.1 if scheme == "sparse" else -0.2
                self.assertAlmostEqual(expected[1], 2 * penalty)
                self.assertAlmostEqual(expected[2], penalty)
                np.testing.assert_array_equal(replay._ticks_advanced[:4], [75, 225, 30, 0])

    def test_reset_seals_n_step_boundary_and_restarts_mower_diagnostics(self):
        config = DoubleDQNConfig(reward_scheme="balanced")
        replay = ReplayBuffer(8, ACTION_SPACE_SIZE)
        observation = empty_observation()
        mask = np.ones(ACTION_SPACE_SIZE, dtype=np.bool_)
        first = EpisodeEventCredit(replay, config)
        first.add(replay, observation, 0, 0, observation, False, False, mask,
                  {"events": {"mowers_used": 1}, "ticks_advanced": 225})
        second = EpisodeEventCredit(replay, config)
        self.assertTrue(bool(replay._episode_ends[0]))
        self.assertFalse(bool(replay._dones[0]))  # interruption retains bootstrap
        self.assertFalse(second.first_mower_seen)
        self.assertIsNone(second.first_mower_tick)
        second.add(replay, observation, 0, 0, observation, False, False, mask,
                   {"events": {"mowers_used": 1}, "ticks_advanced": 30})
        self.assertEqual(second.first_mower_tick, 30)
        self.assertEqual(first.first_mower_tick, 225)

    def test_replay_contains_ordinary_td_fields_and_rejects_event_artifacts(self):
        self.assertEqual({field.name for field in fields(ReplayBatch)}, {
            "observations", "actions", "rewards", "next_observations", "dones",
            "next_action_masks", "ticks_advanced", "indices", "weights",
        })
        replay = ReplayBuffer(8, ACTION_SPACE_SIZE)
        with tempfile.TemporaryDirectory() as tmp:
            path = replay.save(Path(tmp) / "replay.npz")
            with np.load(path) as payload:
                data = {name: payload[name] for name in payload.files}
            self.assertFalse(any(name.startswith("credit__") for name in data))
            data["credit_assignment"] = np.asarray("event_heads")
            np.savez(path, **data)
            with self.assertRaisesRegex(ValueError, "credit_assignment"):
                ReplayBuffer.load(path, credit_assignment="one_step")
            data["credit_assignment"] = np.asarray("one_step")
            data["credit__event_targets"] = np.zeros((0, 2))
            np.savez(path, **data)
            with self.assertRaisesRegex(ValueError, "event-credit replay"):
                ReplayBuffer.load(path)


@unittest.skipIf(agent_module.torch is None, "PyTorch is required")
class OrdinaryTDIntegrationTest(unittest.TestCase):
    def test_controller_trains_and_persists_rewards_ticks_and_mower_diagnostics(self):
        class FakeEnv:
            def __init__(self, *args, **kwargs):
                self.action_space = type("ActionSpace", (), {"n": ACTION_SPACE_SIZE})()
                self.mask = np.zeros(ACTION_SPACE_SIZE, dtype=np.bool_)
                self.mask[0] = True

            def reset(self, **kwargs):
                self.index = 0
                return empty_observation(), {"action_mask": self.mask}

            def step(self, action):
                self.index += 1
                won = self.index == 3
                return empty_observation(), 100.0 if won else 0.0, won, False, {
                    "action_mask": self.mask, "result": "WON" if won else "PLAYING",
                    "action_result": {"success": True},
                    "ticks_advanced": (75, 225, 30)[self.index - 1],
                    "episode_ticks": (75, 300, 330)[self.index - 1],
                    "events": {"mowers_used": (2, 1, 0)[self.index - 1]},
                }

            def close(self):
                pass

        config = DoubleDQNConfig(training_architecture="legacy", hidden_sizes=(16, 8), replay_capacity=8, batch_size=2,
                                 total_steps=5, learning_starts=0, train_every=1,
                                 checkpoint_every=0, autosave_every=0, device="cpu", reward_scheme="balanced")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            controller = DoubleDQNTrainerController(config, run_dir=root,
                fast_eval_every_episodes=0, full_eval_every_episodes=0)
            with patch("train.dqn.base_controller.PvZEnv", FakeEnv), patch("train.dqn.base_controller.PvZClient"):
                summary = controller.run()
            self.assertEqual((summary.env_steps, summary.wins, summary.mowers_used), (5, 1, 6))
            self.assertEqual(controller._agent.update_count, 3)
            replay, _ = ReplayBuffer.load(root / "replay.npz")
            np.testing.assert_allclose(replay._rewards[:5], [-0.4, -0.2, 1, -0.4, -0.2])
            np.testing.assert_array_equal(replay._ticks_advanced[:5], [75, 225, 30, 75, 225])
            records = [json.loads(line) for line in (root / "episodes.jsonl").read_text().splitlines()]
            self.assertEqual((records[0]["first_mower_step"], records[0]["first_mower_tick"]), (1, 75))
            self.assertEqual(records[0]["episode_ticks"], 330)
            self.assertAlmostEqual(records[0]["training_episode_reward"], 0.4)
            loaded, saved_config, _ = DoubleDQNAgent.from_checkpoint(controller.latest_checkpoint, device="cpu")
            self.assertEqual((loaded.credit_assignment, saved_config.credit_assignment), ("one_step", "one_step"))
            self.assertEqual(loaded.update_count, 3)

    def test_checkpoint_cannot_silently_normalize_retired_config(self):
        torch = agent_module.torch
        config = DoubleDQNConfig(reward_scheme="staged", hidden_sizes=(16, 8), device="cpu")
        with tempfile.TemporaryDirectory() as tmp:
            agent = DoubleDQNAgent(ACTION_SPACE_SIZE, config)
            path = agent.save(Path(tmp) / "invalid.pt", env_step=4, episode=1)
            payload = torch.load(path, weights_only=False)
            payload["config"]["credit_assignment"] = "event_heads"
            torch.save(payload, path)
            for weights_only in (False, True):
                with self.assertRaisesRegex(ValueError, "credit_assignment"):
                    DoubleDQNAgent.from_checkpoint(path, device="cpu", weights_only=weights_only)
