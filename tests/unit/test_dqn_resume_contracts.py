from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from pvz_gym.action_space import ACTION_SPACE_SIZE
from pvz_gym.observation import empty_observation
from pvz_gym.types import Observation
from train.dqn.agent import DoubleDQNAgent
from train.dqn.base_controller import DoubleDQNTrainerController as BaseController
from train.dqn.config import DoubleDQNConfig
from train.dqn.checkpoint_contract import declare_checkpoint
from train.dqn.credit import EpisodeEventCredit
from train.dqn.optimized_controller import DoubleDQNTrainerController
from train.dqn.replay import ReplayBuffer
from train.dqn.reward import effective_reward_settings

try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, 'checkpoint contracts require PyTorch')
class ResumeContractsTest(unittest.TestCase):
    def config(self, **kwargs):
        kwargs.setdefault("reward_scheme", "staged")
        return DoubleDQNConfig( hidden_sizes=(16, 8), device='cpu', n_step=1,
                              replay_capacity=8, batch_size=2, **kwargs)

    def source(self, root, config, *, legacy=False):
        agent = DoubleDQNAgent(ACTION_SPACE_SIZE, config)
        agent.update_count = 17
        checkpoint = agent.save(root / 'source.pt', env_step=3, episode=1)
        replay = ReplayBuffer(8, ACTION_SPACE_SIZE, credit_assignment=config.credit_assignment)
        raw_observation = Observation(global_state={
            'sun': 0, 'game_mode': 0, 'total_waves': 1,
            'spawn_plan_available': True, 'zombie_spawn_plan': [0] + [-1] * 49,
        })
        credit = EpisodeEventCredit(replay, config, {'raw_observation': raw_observation})
        observation = empty_observation()
        mask = np.zeros(ACTION_SPACE_SIZE, dtype=np.bool_)
        mask[0] = True
        for index in range(3):
            credit.add(replay, observation, 0, 100 if index == 2 else 0, observation,
                       index == 2, False, mask,
                       {'raw_observation': raw_observation, 'ticks_advanced': 100,
                        'result': 'WON' if index == 2 else 'PLAYING',
                        'events': {'mowers_used': int(index == 0)}})
        replay.save(root / 'replay.npz', env_step=3, episode=1,
                    gamma=config.gamma, gamma_reference_ticks=config.gamma_reference_ticks)
        payload = torch.load(checkpoint, map_location='cpu', weights_only=False)
        payload['replay_file'] = 'replay.npz'
        torch.save(declare_checkpoint(payload, replay.checkpoint_payload()), checkpoint)
        if legacy:
            with np.load(root / 'replay.npz') as data:
                old = {name: data[name] for name in data.files if name not in (
                    'reward_settings_json', 'discount_gamma', 'gamma_reference_ticks',
                )}
            np.savez(root / 'replay.npz', **old)
        return agent, checkpoint

    def restore(self, root, checkpoint, config, cls=DoubleDQNTrainerController):
        events = []
        controller = cls(config, run_dir=root, checkpoint_path=checkpoint,
                         on_event=lambda name, data: events.append((name, data)))
        agent, step, episode, _ = controller._build_agent(ACTION_SPACE_SIZE)
        replay = controller._restore_replay(action_dim=ACTION_SPACE_SIZE,
                                             start_step=step, start_episode=episode)
        return controller, agent, replay, events

    def test_reward_change_rebuilds_replay_but_keeps_network_and_progress(self):
        for mode in ('one_step',):
            for legacy in (False, True):
                with self.subTest(mode=mode, legacy=legacy), tempfile.TemporaryDirectory() as tmp:
                    root = Path(tmp)
                    old = self.config(credit_assignment=mode, reward_scheme='sparse')
                    original, checkpoint = self.source(root, old, legacy=legacy)
                    for config in (replace(old, reward_scheme='balanced'),
                                   replace(old, mower_penalty=0.2), replace(old, reward_scale=0.02)):
                        controller, agent, replay, events = self.restore(root, checkpoint, config)
                        self.assertEqual(len(replay), 0)
                        self.assertFalse(controller.summary.replay_restored)
                        self.assertEqual(agent.update_count, 17)
                        self.assertTrue(any(name == 'replay_restore_skipped' and 'reward settings changed' in data['reason']
                                            for name, data in events))
                        for name, value in original.online.state_dict().items():
                            self.assertTrue(torch.equal(value, agent.online.state_dict()[name]))
                    # The original snapshot must remain usable and unchanged.
                    _, _, replay, _ = self.restore(root, checkpoint, old)
                    self.assertEqual(len(replay), 3)
                    self.assertAlmostEqual(float(replay._rewards[0]), -0.1)

    def test_discount_horizon_loss_weight_and_inactive_reward_scalars_reuse_replay(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            old = self.config(reward_scheme='balanced')
            _, checkpoint = self.source(root, old)
            current = replace(old, gamma=0.9, n_step=3,
                              reward_scale=99, mower_penalty=99)
            controller, _, replay, _ = self.restore(root, checkpoint, current)
            self.assertTrue(controller.summary.replay_restored)
            self.assertEqual(len(replay), 3)
            self.assertEqual(effective_reward_settings(old), effective_reward_settings(current))
            np.testing.assert_allclose(replay._rewards[:3], [-0.2, 0, 1])
            with patch.object(replay, '_rng', SimpleNamespace(integers=lambda *a, **k: np.array([0]),
                                                               uniform=lambda *a, **k: np.array([0.0]))):
                batch = replay.sample(1, n_step=current.n_step, gamma=current.gamma,
                                      gamma_reference_ticks=current.gamma_reference_ticks)
            np.testing.assert_allclose(batch.rewards, [-0.2 + 0.9**2])
            np.testing.assert_array_equal(batch.ticks_advanced, [300])
            self.assertTrue(bool(batch.dones[0]))

    def test_both_save_paths_preserve_discount_metadata_and_raw_transitions(self):
        for cls in (BaseController, DoubleDQNTrainerController):
            with self.subTest(cls=cls), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                config = self.config(gamma=0.9)
                _, checkpoint = self.source(root, config, legacy=True)
                controller, agent, replay, _ = self.restore(root, checkpoint, config, cls)
                self.assertTrue(controller.summary.replay_restored)
                self.assertEqual(len(replay), 3)
                np.testing.assert_allclose(replay._rewards[:3], [-0.2, 0, 1])
                fields = ('_rewards', '_ticks_advanced', '_episode_ends', '_next_action_masks')
                before = {name: getattr(replay, name)[:3].copy() for name in fields}
                controller._agent, controller._replay = agent, replay
                controller.summary.env_steps, controller.summary.episodes = 3, 1
                saved = controller.save_checkpoint(root / 'manual.pt')
                _, metadata = ReplayBuffer.load(root / 'replay.npz')
                self.assertEqual(metadata['discount_gamma'], 0.9)
                self.assertEqual(metadata['gamma_reference_ticks'], 100)
                self.assertEqual(metadata['reward_settings'], effective_reward_settings(config))
                again, _, restored, events = self.restore(root, saved, config, cls)
                self.assertTrue(again.summary.replay_restored)
                for name in fields:
                    np.testing.assert_array_equal(getattr(restored, name)[:3], before[name])
                self.assertTrue(restored.supports_n_step)
                self.assertTrue(any(name == 'replay_restored' for name, _ in events))

    def test_reward_metadata_survives_direct_resave_and_rejects_future_version(self):
        for scheme in ('balanced', 'staged'):
            with self.subTest(scheme=scheme), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                self.source(root, self.config(reward_scheme=scheme))
                replay, metadata = ReplayBuffer.load(root / 'replay.npz')
                replay.save(root / 'copy.npz')
                _, copied = ReplayBuffer.load(root / 'copy.npz')
                self.assertEqual(copied['reward_settings'], metadata['reward_settings'])
                with np.load(root / 'copy.npz') as data:
                    invalid = {name: data[name] for name in data.files}
                # Versions 1 and 2 are supported; keep the payload otherwise valid.
                invalid['reward_settings_json'] = np.asarray(json.dumps({
                    **metadata['reward_settings'], 'version': 3,
                }))
                np.savez(root / 'copy.npz', **invalid)
                with self.assertRaisesRegex(ValueError, 'reward settings'):
                    ReplayBuffer.load(root / 'copy.npz')

    def test_tracker_cannot_mix_reward_settings_inside_one_replay(self):
        replay = ReplayBuffer(8, ACTION_SPACE_SIZE)
        config = self.config(credit_assignment='one_step', reward_scheme='sparse')
        EpisodeEventCredit(replay, config)
        with self.assertRaisesRegex(ValueError, 'reward settings changed'):
            EpisodeEventCredit(replay, replace(config, reward_scheme='balanced'))

    def test_resume_restores_agent_episode_and_torch_random_sequences(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = self.config()
            controller = DoubleDQNTrainerController(config, run_dir=root)
            controller._agent = DoubleDQNAgent(ACTION_SPACE_SIZE, config)
            controller._agent._rng.random(11)
            controller._episode_rng.integers(0, 2**32, size=7, dtype=np.uint64)
            controller.summary.env_steps, controller.summary.episodes = 99, 7
            checkpoint = controller.save_checkpoint(root / 'latest.pt')
            expected_agent = controller._agent._rng.random(5)
            expected_episode = controller._episode_rng.integers(0, 2**32, size=5, dtype=np.uint64)
            expected_torch = torch.rand(5)
            resumed = DoubleDQNTrainerController(config, run_dir=root, checkpoint_path=checkpoint)
            agent, step, episode, _ = resumed._build_agent(ACTION_SPACE_SIZE)
            self.assertEqual((step, episode), (99, 7))
            np.testing.assert_array_equal(agent._rng.random(5), expected_agent)
            np.testing.assert_array_equal(resumed._episode_rng.integers(0, 2**32, size=5, dtype=np.uint64), expected_episode)
            self.assertTrue(torch.equal(torch.rand(5), expected_torch))

    @unittest.skipUnless(torch is not None and torch.cuda.is_available(), 'CUDA RNG test requires GPU')
    def test_cuda_random_sequence_survives_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = replace(self.config(), device='cuda')
            agent = DoubleDQNAgent(ACTION_SPACE_SIZE, config)
            torch.rand(7, device='cuda')
            checkpoint = agent.save(Path(tmp) / 'cuda.pt', env_step=9, episode=1)
            expected = torch.rand(5, device='cuda').cpu()
            resumed, _, _ = DoubleDQNAgent.from_checkpoint(checkpoint, device='cuda')
            self.assertEqual(resumed.device.type, 'cuda')
            self.assertTrue(torch.equal(torch.rand(5, device='cuda').cpu(), expected))

    def test_weights_only_resets_agent_episode_and_torch_random_sequences(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = self.config(seed=19)
            original = DoubleDQNAgent(ACTION_SPACE_SIZE, config)
            original._rng.random(11)
            checkpoint = original.save(root / 'source.pt', env_step=99, episode=7,
                                       episode_rng_state=np.random.default_rng(999).bit_generator.state)
            current = replace(config, seed=23)
            controller = DoubleDQNTrainerController(current, run_dir=root, checkpoint_path=checkpoint,
                                                   checkpoint_load_mode='weights_only')
            agent, step, episode, _ = controller._build_agent(ACTION_SPACE_SIZE)
            self.assertEqual((step, episode, agent.update_count), (0, 0, 0))
            np.testing.assert_array_equal(agent._rng.random(5), np.random.default_rng(23).random(5))
            np.testing.assert_array_equal(controller._episode_rng.random(5), np.random.default_rng(23).random(5))
            expected = torch.rand(5, generator=torch.Generator().manual_seed(23))
            self.assertTrue(torch.equal(torch.rand(5), expected))

    def test_legacy_rng_fallback_is_deterministic_and_does_not_restart_seed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = self.config()
            _, checkpoint = self.source(root, config)
            payload = torch.load(checkpoint, weights_only=False)
            for key in ('agent_rng_state', 'torch_rng_state', 'cuda_rng_states', 'episode_rng_state'):
                payload.pop(key, None)
            torch.save(payload, checkpoint)
            first = DoubleDQNTrainerController(config, run_dir=root, checkpoint_path=checkpoint)
            agent1, _, _, _ = first._build_agent(ACTION_SPACE_SIZE)
            second = DoubleDQNTrainerController(config, run_dir=root, checkpoint_path=checkpoint)
            agent2, _, _, _ = second._build_agent(ACTION_SPACE_SIZE)
            draws1, draws2 = agent1._rng.random(5), agent2._rng.random(5)
            np.testing.assert_array_equal(draws1, draws2)
            self.assertFalse(np.array_equal(draws1, np.random.default_rng(config.seed).random(5)))
            np.testing.assert_array_equal(first._episode_rng.random(5), second._episode_rng.random(5))
            self.assertNotEqual(first._episode_rng.bit_generator.state, np.random.default_rng(config.seed).bit_generator.state)


    def test_resume_automatically_loads_latest_replay_on_step_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = self.config()
            _, checkpoint = self.source(root, config)
            # 制造 step 差异（例如 checkpoint step 3，replay 保存为 step 5）
            with np.load(root / 'replay.npz') as data:
                replay_data = {name: data[name] for name in data.files}
            replay_data['env_step'] = np.asarray(5)
            replay_data['episode'] = np.asarray(2)
            np.savez_compressed(root / 'stopped-replay-000000005-test.npz', **replay_data)

            # 在 resume 模式下恢复
            controller = DoubleDQNTrainerController(config, run_dir=root, checkpoint_path=checkpoint)
            agent, step, episode, _ = controller._build_agent(ACTION_SPACE_SIZE)
            replay = controller._restore_replay(action_dim=ACTION_SPACE_SIZE,
                                                start_step=step, start_episode=episode)
            self.assertTrue(controller.summary.replay_restored)
            self.assertEqual(len(replay), 3)

    def test_resume_loads_latest_replay_even_if_checkpoint_is_model_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = self.config()
            _, checkpoint = self.source(root, config)
            # 将 checkpoint 修改为 model_only
            payload = torch.load(checkpoint, weights_only=False)
            payload['checkpoint_kind'] = 'model_only'
            payload.pop('replay_file', None)
            torch.save(payload, checkpoint)

            controller = DoubleDQNTrainerController(config, run_dir=root, checkpoint_path=checkpoint)
            agent, step, episode, _ = controller._build_agent(ACTION_SPACE_SIZE)
            replay = controller._restore_replay(action_dim=ACTION_SPACE_SIZE,
                                                start_step=step, start_episode=episode)
            self.assertTrue(controller.summary.replay_restored)
            self.assertEqual(len(replay), 3)


if __name__ == '__main__':
    unittest.main()
