from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

try:
    import torch
except ImportError:
    torch = None

from pvz_gym.action_space import ACTION_SPACE_SIZE
from pvz_gym.observation import empty_observation
from train.dqn import train as train_cli
from train.dqn.agent import DoubleDQNAgent
from train.dqn.config import DoubleDQNConfig
from train.dqn.credit import EpisodeEventCredit
from train.dqn.optimized_controller import DoubleDQNTrainerController
from train.dqn.replay import ReplayBuffer
from train.dqn.run_state import load_run_config


class NStepReturnTest(unittest.TestCase):
    def add(self, replay, value, reward, ticks, *, done=False, end=False):
        observation, next_observation = empty_observation(), empty_observation()
        observation['global'][0], next_observation['global'][0] = value, value + 1
        mask = np.zeros(replay.action_dim, dtype=np.bool_)
        mask[value % replay.action_dim] = True
        replay.add(observation, value % replay.action_dim, reward, next_observation, done, mask,
                   ticks_advanced=ticks, episode_end=end)

    def at(self, replay, offsets, n, gamma=0.5):
        indices = np.asarray(offsets, dtype=np.int64)
        with patch.object(replay, '_rng', SimpleNamespace(integers=lambda *a, **k: indices.copy())):
            return replay.sample(len(indices), n_step=n, gamma=gamma, gamma_reference_ticks=100)

    def trajectory(self, capacity=8):
        replay = ReplayBuffer(capacity, 2)
        for index, (reward, ticks) in enumerate(zip([1, 2, 4, 8, 16], [50, 150, 0, 100, 25])):
            self.add(replay, index, reward, ticks)
        return replay

    def test_variable_ticks_three_five_and_one_step_returns(self):
        replay = self.trajectory()
        for n in (1, 3, 5):
            batch = self.at(replay, [0], n)
            rewards, durations = [1, 2, 4, 8, 16], [50, 150, 0, 100, 25]
            expected = sum(rewards[k] * 0.5**(sum(durations[:k]) / 100) for k in range(n))
            self.assertAlmostEqual(float(batch.rewards[0]), expected, places=6)
            self.assertEqual(int(batch.ticks_advanced[0]), sum(durations[:n]))
            self.assertEqual(float(batch.next_observations['global'][0, 0]), n)
            self.assertEqual(int(np.argmax(batch.next_action_masks[0])), (n - 1) % 2)
            self.assertFalse(batch.dones[0])
        np.testing.assert_array_equal(replay._rewards[:5], [1, 2, 4, 8, 16])

    def test_terminal_flush_stops_before_next_episode_and_masks_bootstrap(self):
        replay = ReplayBuffer(8, 2)
        self.add(replay, 0, 1, 100)
        self.add(replay, 1, 10, 50, done=True)
        self.add(replay, 2, 999, 100)
        batch = self.at(replay, [0, 1], 5)
        np.testing.assert_allclose(batch.rewards, [6, 10])
        np.testing.assert_array_equal(batch.ticks_advanced, [150, 50])
        np.testing.assert_array_equal(batch.dones, [True, True])
        np.testing.assert_array_equal(batch.next_observations['global'][:, 0], [2, 2])
        self.assertEqual(replay._eligible_count(5), 2)

    def test_truncation_flush_stops_before_reset_but_keeps_bootstrap(self):
        replay = ReplayBuffer(8, 2)
        self.add(replay, 0, 1, 100)
        self.add(replay, 1, 10, 50, end=True)
        self.add(replay, 2, 999, 100)
        batch = self.at(replay, [0, 1], 5)
        np.testing.assert_allclose(batch.rewards, [6, 10])
        np.testing.assert_array_equal(batch.dones, [False, False])
        np.testing.assert_array_equal(batch.next_observations['global'][:, 0], [2, 2])

    def test_unfinished_tail_waits_until_full_horizon_or_boundary(self):
        replay = ReplayBuffer(8, 2)
        self.add(replay, 0, 1, 100)
        self.add(replay, 1, 2, 100)
        self.assertFalse(replay.can_sample(3))
        with self.assertRaisesRegex(ValueError, 'complete n-step'):
            replay.sample(1, n_step=3)
        self.add(replay, 2, 4, 100)
        self.assertEqual(replay._eligible_count(3), 1)
        replay.end_episode()
        self.assertEqual(replay._eligible_count(3), 3)
        np.testing.assert_allclose(self.at(replay, [0, 1, 2], 3).rewards, [3, 4, 4])

    def test_ring_wrap_resize_and_resume_do_not_change_raw_returns(self):
        replay = ReplayBuffer(5, 2)
        for index in range(8):
            self.add(replay, index, index + 1, 100)
        batch = self.at(replay, [0], 3)
        self.assertEqual(float(batch.observations['global'][0, 0]), 3)
        self.assertEqual(float(batch.rewards[0]), 4 + 0.5 * 5 + 0.25 * 6)
        with tempfile.TemporaryDirectory() as tmp:
            path = replay.save(Path(tmp) / 'replay.npz')
            restored, _ = ReplayBuffer.load(path, capacity=4)
        np.testing.assert_array_equal(restored._rewards[:4], [5, 6, 7, 8])
        self.assertEqual(float(self.at(restored, [0], 3).rewards[0]), 5 + 0.5 * 6 + 0.25 * 7)
        self.assertTrue(restored.supports_n_step)

    def test_changing_n_and_gamma_reuses_raw_rewards(self):
        replay = self.trajectory()
        old = float(self.at(replay, [0], 3).rewards[0])
        changed = float(self.at(replay, [0], 5, gamma=1).rewards[0])
        self.assertEqual(changed, 31)
        self.assertEqual(float(self.at(replay, [0], 3).rewards[0]), old)
        np.testing.assert_array_equal(replay._rewards[:5], [1, 2, 4, 8, 16])

    def test_random_wrapped_trajectories_match_reference_for_every_eligible_start(self):
        rng = np.random.default_rng(17)
        for _ in range(20):
            replay = ReplayBuffer(7, 2)
            rewards = rng.normal(size=14)
            durations = rng.integers(0, 301, size=14)
            ends = rng.random(14) < 0.3
            terminal = ends & (rng.random(14) < 0.5)
            for index in range(14):
                self.add(replay, index, rewards[index], int(durations[index]),
                         end=bool(ends[index]), done=bool(terminal[index]))
            for n in (1, 3, 5):
                eligible = [start for start in range(7, 14)
                            if start + n <= 14 or ends[start:14].any()]
                self.assertEqual(replay._eligible_count(n), len(eligible))
                if not eligible:
                    continue
                batch = self.at(replay, list(range(len(eligible))), n)
                for row, start in enumerate(eligible):
                    expected, elapsed = 0.0, 0
                    for index in range(start, min(start + n, 14)):
                        expected += float(np.float32(rewards[index])) * 0.5**(elapsed / 100)
                        elapsed += int(durations[index])
                        if ends[index]:
                            break
                    self.assertAlmostEqual(float(batch.rewards[row]), expected, places=6)
                    self.assertEqual(int(batch.ticks_advanced[row]), elapsed)
                    self.assertEqual(bool(batch.dones[row]), bool(terminal[index]))
                    self.assertEqual(float(batch.next_observations['global'][row, 0]), index + 1)

    def test_unknown_or_partial_return_metadata_cannot_resume_silently(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self.trajectory().save(Path(tmp) / 'replay.npz')
            with np.load(path) as data:
                original = {key: data[key] for key in data.files}
            for bad_version in (2, 1.5, True):
                np.savez(path, **{**original, 'raw_return_schema_version': np.asarray(bad_version)})
                with self.assertRaisesRegex(ValueError, 'raw return'):
                    ReplayBuffer.load(path)
            partial = {key: value for key, value in original.items() if key != 'episode_ends'}
            np.savez(path, **partial)
            with self.assertRaisesRegex(ValueError, 'incomplete raw return'):
                ReplayBuffer.load(path)


    def test_tracker_records_truncation_and_seals_interrupted_resume_episode(self):
        # This test covers replay boundaries, independent of staged accounting.
        config = DoubleDQNConfig(credit_assignment='one_step', reward_scheme='sparse')
        replay = ReplayBuffer(8, 2)
        observation = empty_observation()
        credit = EpisodeEventCredit(replay, config)
        credit.add(replay, observation, 0, 0, observation, False, True, [True, False], {'ticks_advanced': 150})
        self.assertTrue(replay._episode_ends[0])
        self.assertFalse(replay._dones[0])
        self.add(replay, 1, 1, 100)
        EpisodeEventCredit(replay, config)
        self.assertTrue(replay._episode_ends[1])
        self.assertFalse(replay._dones[1])

    def test_old_replay_only_allows_one_step_even_after_resaving(self):
        replay = self.trajectory()
        with tempfile.TemporaryDirectory() as tmp:
            path = replay.save(Path(tmp) / 'replay.npz')
            with np.load(path) as data:
                old = {key: data[key] for key in data.files
                       if key not in ('episode_ends', 'raw_return_schema_version', 'supports_n_step')}
            np.savez(path, **old)
            restored, _ = ReplayBuffer.load(path)
            self.assertFalse(restored.supports_n_step)
            self.assertEqual(len(restored.sample(1).rewards), 1)
            with self.assertRaisesRegex(ValueError, 'legacy replay'):
                restored.sample(1, n_step=3)
            restored.save(path)
            again, _ = ReplayBuffer.load(path)
            self.assertFalse(again.supports_n_step)

    def test_controller_reuses_new_replay_on_n_change_and_skips_legacy_for_n3(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / 'checkpoint.pt'
            config = DoubleDQNConfig(reward_scheme="staged", credit_assignment='one_step', replay_capacity=8, batch_size=2)
            self.trajectory().save(root / 'replay.npz', env_step=5, episode=0)
            for n in (1, 3, 5):
                controller = DoubleDQNTrainerController(replace(config, n_step=n),
                                                       run_dir=root, checkpoint_path=path)
                restored = controller._restore_replay(action_dim=2, start_step=5, start_episode=0)
                self.assertEqual(len(restored), 5)
                self.assertTrue(controller.summary.replay_restored)
            with np.load(root / 'replay.npz') as data:
                old = {key: data[key] for key in data.files
                       if key not in ('episode_ends', 'raw_return_schema_version', 'supports_n_step')}
            np.savez(root / 'replay.npz', **old)
            controller = DoubleDQNTrainerController(config, run_dir=root, checkpoint_path=path)
            self.assertEqual(len(controller._restore_replay(action_dim=2, start_step=5, start_episode=0)), 0)
            self.assertFalse(controller.summary.replay_restored)

    def test_config_cli_defaults_validation_and_legacy_resume(self):
        self.assertEqual(DoubleDQNConfig(reward_scheme="staged").n_step, 3)
        for invalid in (0, -1, 1.5, True, 9):
            with self.assertRaisesRegex(ValueError, 'n_step'):
                DoubleDQNConfig(reward_scheme="staged", n_step=invalid, replay_capacity=8, batch_size=2)
        with patch('sys.argv', ['train', '--n-step', '5']), patch.object(train_cli, 'train') as launch:
            train_cli.main()
        self.assertEqual(launch.call_args.args[0].n_step, 5)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            values = asdict(DoubleDQNConfig(reward_scheme="staged"))
            values.pop('n_step')
            (root / 'config.json').write_text(json.dumps(values), encoding='utf-8')
            (root / 'latest.pt').touch()
            self.assertEqual(load_run_config(root).n_step, 1)
            for arguments, expected in (([], 1), (['--n-step', '3'], 3), (['--n-step', '5'], 5)):
                with patch('sys.argv', ['train', '--resume-run', str(root), *arguments]), \
                     patch.object(train_cli, 'train') as launch:
                    train_cli.main()
                self.assertEqual(launch.call_args.args[0].n_step, expected)

    @unittest.skipIf(torch is None, "PyTorch required")
    def test_checkpoint_n_change_keeps_network_and_optimizer_state(self):
        import torch
        config = DoubleDQNConfig(reward_scheme="staged", credit_assignment='one_step', hidden_sizes=(16, 8), device='cpu')
        original = DoubleDQNAgent(ACTION_SPACE_SIZE, config)
        seed_replay = ReplayBuffer(3, ACTION_SPACE_SIZE)
        self.add(seed_replay, 0, 1, 100, done=True)
        original.learn(seed_replay.sample(1))
        optimizer_before = original.optimizer.state_dict()
        original.update_count = 17
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoint = original.save(root / 'checkpoint.pt', env_step=5, episode=1)
            _, saved_config, _ = DoubleDQNAgent.from_checkpoint(checkpoint, device='cpu')
            self.assertEqual(saved_config.n_step, 3)
            for n in (1, 5):
                controller = DoubleDQNTrainerController(replace(config, n_step=n), run_dir=root,
                                                       checkpoint_path=checkpoint)
                agent, step, episode, _ = controller._build_agent(ACTION_SPACE_SIZE)
                self.assertEqual((agent.config.n_step, agent.update_count, step, episode), (n, 17, 5, 1))
                optimizer_after = agent.optimizer.state_dict()
                self.assertEqual(optimizer_before['param_groups'], optimizer_after['param_groups'])
                for parameter, state in optimizer_before['state'].items():
                    for key, value in state.items():
                        self.assertTrue(torch.equal(value, optimizer_after['state'][parameter][key]))
                for name, tensor in original.online.state_dict().items():
                    self.assertTrue(torch.equal(tensor, agent.online.state_dict()[name]))
            payload = torch.load(checkpoint, weights_only=False)
            payload['config'].pop('n_step')
            torch.save(payload, checkpoint)
            _, legacy_config, _ = DoubleDQNAgent.from_checkpoint(checkpoint, device='cpu')
            self.assertEqual(legacy_config.n_step, 1)


if __name__ == '__main__':
    unittest.main()
