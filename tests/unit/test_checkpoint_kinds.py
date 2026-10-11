from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from pvz_gym.action_space import ACTION_SPACE_SIZE
from pvz_gym.observation import empty_observation
from train.dqn.agent import DoubleDQNAgent
from train.dqn.config import DoubleDQNConfig
from train.dqn.optimized_controller import DoubleDQNTrainerController
from train.dqn.replay import ReplayBuffer
from train.dqn.reward import effective_reward_settings


class CheckpointKindTest(unittest.TestCase):
    def source(self, root):
        config = DoubleDQNConfig(device='cpu', hidden_sizes=(16, 8), n_step=1,
                                 reward_scheme='sparse', replay_capacity=8, batch_size=2)
        controller = DoubleDQNTrainerController(config, run_dir=root, eval_every_episodes=0)
        controller._agent = DoubleDQNAgent(ACTION_SPACE_SIZE, config)
        controller._agent.update_count = 19
        controller._replay = ReplayBuffer(8, ACTION_SPACE_SIZE)
        controller._replay.reward_settings = effective_reward_settings(config)
        obs = empty_observation()
        for _ in range(3):
            controller._replay.add(obs, 0, .2, obs, False, [True]*ACTION_SPACE_SIZE, ticks_advanced=150)
        controller.summary.env_steps = 3
        controller.summary.episodes = 1
        full = controller.save_checkpoint(root/'full.pt')
        model = controller.save_checkpoint(root/'latest.pt')
        controller._flush_checkpoint_writer(close=True)
        return config, full, model

    def restore(self, config, root, checkpoint, mode):
        events = []
        controller = DoubleDQNTrainerController(config, run_dir=root, checkpoint_path=checkpoint,
                                               checkpoint_load_mode=mode, eval_every_episodes=0,
                                               on_event=lambda name, data: events.append((name, data)))
        controller._agent, step, episode, _ = controller._build_agent(ACTION_SPACE_SIZE)
        replay = controller._restore_replay(action_dim=ACTION_SPACE_SIZE, start_step=step, start_episode=episode)
        return controller, replay, events

    def test_full_and_model_metadata_and_explicit_restore_modes(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            config, full, model = self.source(root)
            full_data = torch.load(full, weights_only=False)
            model_data = torch.load(model, weights_only=False)
            self.assertEqual(full_data['checkpoint_kind'], 'full')
            self.assertTrue(full_data['strict_resume_supported'])
            self.assertEqual(model_data['checkpoint_kind'], 'model_only')
            self.assertFalse(model_data['strict_resume_supported'])
            self.assertNotIn('replay_file', model_data)
            self.assertTrue((root/'latest-model.pt').is_file())
            restored, replay, _ = self.restore(config, root, full, 'strict_resume')
            self.assertEqual(len(replay), 3)
            self.assertTrue(restored.summary.replay_restored)
            model_resume, replay, events = self.restore(config, root, full, 'model_resume')
            self.assertEqual(len(replay), 0)
            self.assertEqual(model_resume._agent.update_count, 19)
            self.assertTrue(any(name == 'replay_restore_skipped' for name, _ in events))
            with self.assertRaisesRegex(ValueError, 'model-only'):
                self.restore(config, root, model, 'strict_resume')
            ordinary_resume, replay, events = self.restore(config, root, model, 'resume')
            self.assertEqual(len(replay), 3)
            self.assertTrue(ordinary_resume.summary.replay_restored)
            self.assertTrue(any(name == 'replay_restored' for name, _ in events))
            _, replay, events = self.restore(config, root, model, 'model_resume')
            self.assertEqual(len(replay), 0)
            self.assertTrue(any(data.get('warmup_restarted') for _, data in events))

    def test_missing_paired_replay_rejects_substitute_with_identical_counters(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            config, full, _ = self.source(root)
            payload = torch.load(full, weights_only=False)
            paired = root / payload['replay_file']
            replacement = root / 'stopped-replay-000000003-substitute.npz'
            replacement.write_bytes(paired.read_bytes())
            paired.unlink()
            with patch.object(ReplayBuffer, 'load', wraps=ReplayBuffer.load) as load:
                with self.assertRaisesRegex(ValueError, 'strict_resume.*missing'):
                    self.restore(config, root, full, 'strict_resume')
                load.assert_not_called()
            ordinary, replay, events = self.restore(config, root, full, 'resume')
            self.assertTrue(ordinary.summary.replay_restored)
            self.assertEqual(len(replay), 3)
            self.assertTrue(any(name == 'replay_restored' and data['path'] == str(replacement)
                                for name, data in events))

    def test_model_only_ordinary_resume_without_sidecars_restarts_warmup(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            config, _, model = self.source(root)
            for sidecar in root.glob('*.npz'):
                sidecar.unlink()
            controller, replay, events = self.restore(config, root, model, 'resume')
            self.assertEqual(len(replay), 0)
            self.assertFalse(controller.summary.replay_restored)
            self.assertEqual(controller._agent.update_count, 19)
            self.assertTrue(any(name == 'replay_restore_skipped' and data.get('warmup_restarted')
                                for name, data in events))

    def test_strict_missing_corrupt_or_stale_pair_does_not_start_empty_replay(self):
        for damage in ('missing', 'corrupt', 'stale'):
            with self.subTest(damage=damage), tempfile.TemporaryDirectory() as folder:
                root = Path(folder)
                config, full, _ = self.source(root)
                saved = torch.load(full, weights_only=False)
                sidecar = root/saved['replay_file']
                if damage == 'missing': sidecar.unlink()
                elif damage == 'corrupt': sidecar.write_bytes(b'broken snapshot')
                else:
                    replay, _ = ReplayBuffer.load(sidecar)
                    replay.save(sidecar, env_step=2, episode=1)
                with self.assertRaisesRegex(ValueError, 'strict_resume'):
                    self.restore(config, root, full, 'strict_resume')

    def test_strict_reward_change_rejects_and_weights_only_resets(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            config, full, _ = self.source(root)
            with self.assertRaisesRegex(ValueError, 'reward settings changed'):
                self.restore(replace(config, mower_penalty=.9), root, full, 'strict_resume')
            transferred, replay, _ = self.restore(config, root, full, 'weights_only')
            self.assertEqual(len(replay), 0)
            self.assertEqual(transferred._agent.update_count, 0)

    def test_strict_does_not_replace_missing_or_broken_training_states(self):
        for key in ('optimizer', 'target', 'agent_rng_state', 'torch_rng_state', 'episode_rng_state'):
            with self.subTest(key=key), tempfile.TemporaryDirectory() as folder:
                root = Path(folder)
                config, full, _ = self.source(root)
                payload = torch.load(full, weights_only=False)
                payload.pop(key)
                torch.save(payload, full)
                with self.assertRaisesRegex(ValueError, 'strict_resume'):
                    self.restore(config, root, full, 'strict_resume')
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            config, full, _ = self.source(root)
            payload = torch.load(full, weights_only=False)
            payload['optimizer']['param_groups'][0]['params'].pop()
            torch.save(payload, full)
            with self.assertRaisesRegex(ValueError, 'strict_resume.*optimizer'):
                self.restore(config, root, full, 'strict_resume')

    def test_strict_requires_replay_rng_and_full_alias_tracks_every_full_save(self):
        for damage in ('missing', 'invalid'):
            with self.subTest(damage=damage), tempfile.TemporaryDirectory() as folder:
                root = Path(folder)
                config, full, _ = self.source(root)
                payload = torch.load(full, weights_only=False)
                path = root / payload['replay_file']
                with np.load(path) as archive:
                    replay = {name: archive[name] for name in archive.files}
                if damage == 'missing':
                    replay.pop('rng_state_json')
                else:
                    replay['rng_state_json'] = np.asarray('not-json')
                np.savez_compressed(path, **replay)
                with self.assertRaisesRegex(ValueError, 'strict_resume.*(Replay RNG|rng_state_json)'):
                    self.restore(config, root, full, 'strict_resume')
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            config, full, _ = self.source(root)
            controller, replay, _ = self.restore(config, root, full, 'strict_resume')
            controller._replay = replay
            controller.summary.env_steps = 4
            controller.save_checkpoint(root / 'checkpoint-000000004.pt')
            controller._flush_checkpoint_writer(close=True)
            latest = torch.load(root / 'latest-full.pt', weights_only=False)
            self.assertEqual((latest['env_step'], latest['checkpoint_kind']), (4, 'full'))
            from train.dqn.run_state import resolve_resume_checkpoint
            controller.summary.env_steps = 5
            controller.save_checkpoint(root / 'latest.pt')
            controller._flush_checkpoint_writer(close=True)
            selected = resolve_resume_checkpoint(root, mode='strict_resume')
            self.assertEqual(torch.load(selected, weights_only=False)['env_step'], 4)

    def test_publish_failure_cannot_overwrite_a_previous_matching_sidecar(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            config, full, _ = self.source(root)
            previous_model = full.read_bytes()
            payload = torch.load(full, weights_only=False)
            previous_sidecar = root / payload['replay_file']
            previous_bytes = previous_sidecar.read_bytes()
            controller, replay, _ = self.restore(config, root, full, 'strict_resume')
            controller._replay = replay
            controller.summary.env_steps, controller.summary.episodes = 3, 1
            replay.sample(2)  # Same counters, different replay RNG snapshot.
            with patch('train.dqn.agent._atomic_torch_save', side_effect=OSError('publish failed')):
                with self.assertRaisesRegex(OSError, 'publish failed'):
                    controller.save_checkpoint(full)
            controller._flush_checkpoint_writer(close=True)
            self.assertEqual(full.read_bytes(), previous_model)
            self.assertEqual(previous_sidecar.read_bytes(), previous_bytes)
            again, _, _ = self.restore(config, root, full, 'strict_resume')
            self.assertTrue(again.summary.replay_restored)

    def test_strict_run_selection_skips_corrupt_and_incomplete_models(self):
        from train.dqn.run_state import resolve_resume_checkpoint
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            _, full, _ = self.source(root)
            (root / 'stopped.pt').write_bytes(b'broken checkpoint')
            self.assertEqual(torch.load(resolve_resume_checkpoint(root, mode='strict_resume'), weights_only=False)['env_step'], 3)
            incomplete = torch.load(full, weights_only=False)
            incomplete.pop('update_count')
            torch.save(incomplete, root / 'stopped.pt')
            self.assertNotEqual(resolve_resume_checkpoint(root, mode='strict_resume').name, 'stopped.pt')


if __name__ == '__main__': unittest.main()
