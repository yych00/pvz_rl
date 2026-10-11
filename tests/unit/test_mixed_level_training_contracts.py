"""Episode identity survives level changes and missing terminal metadata."""

import json
import queue
import tempfile
import threading
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from pvz_gym.observation import empty_observation
from train.common import EnvConfig
from train.curriculum import episode_level_metadata
from train.dqn.base_controller import DoubleDQNTrainerController
from train.dqn.config import DoubleDQNConfig
from train.dqn.parallel_controller import run_parallel
from train.runtime import TrainingRuntimeConfig
from train.sampler import _worker
from tests.unit.test_dqn_parallel_evaluation import MemorySampler


POOL = ((41, 1), (49, 2))


class MixedMemorySampler(MemorySampler):
    def facts(self, i, done=False):
        info = super().facts(i, done)
        if self.steps[i] == 0:
            level_id, playthrough = POOL[(self.episodes[i] - 1 + i) % len(POOL)]
            info.update(level_id=level_id, playthrough=playthrough,
                        episode_seed=1000 + self.episodes[i],
                        seed_selection={"mode": "adaptive_shuffled", "actual": [33, 1, 34]})
        return info


class SerialEnvironment:
    def __init__(self, *args, **kwargs):
        from pvz_gym.action_space import ACTION_SPACE_SIZE
        self.level_id = kwargs['level_id']
        self.playthrough = kwargs['playthrough']
        self.action_space = type('ActionSpace', (), {'n': ACTION_SPACE_SIZE})()
        self.mask = [True] + [False] * (ACTION_SPACE_SIZE - 1)

    def reset(self, *, seed):
        return empty_observation(), dict(action_mask=self.mask, episode_seed=seed)

    def step(self, action):
        return empty_observation(), 100., True, False, dict(
            action_mask=self.mask, result='WON', events={}, ticks_advanced=100,
            episode_ticks=100, action_result={'success': True})

    def close(self):
        pass


class MixedLevelTrainingContracts(unittest.TestCase):
    def config(self, **updates):
        config = DoubleDQNConfig(
            env=EnvConfig(level_id=41, level_pool=POOL), reward_scheme='sparse',
            hidden_sizes=(16, 8), device='cpu', total_steps=8,
            replay_capacity=16, batch_size=2, learning_starts=16, n_step=1,
            autosave_every=0, checkpoint_every=0, training_architecture='legacy',
            runtime=TrainingRuntimeConfig(num_envs=2))
        return replace(config, **updates)

    def test_parallel_log_and_step_keep_reset_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            events = []
            config = self.config()
            controller = DoubleDQNTrainerController(config, run_dir=directory,
                eval_every_episodes=0, on_event=lambda name, data: events.append((name, data)))
            run_parallel(controller, sampler_factory=MixedMemorySampler)
            records = [json.loads(line) for line in Path(directory, 'episodes.jsonl').read_text().splitlines()]
            self.assertGreaterEqual(len(records), 2)
            for record in records:
                i, episode = record['environment_id'], record['environment_episode']
                self.assertEqual((record['level_id'], record['playthrough']), POOL[(episode - 1 + i) % 2])
                self.assertEqual(record['episode_seed'], 1000 + episode)
                self.assertEqual(record['seed_selection']['actual'], [33, 1, 34])
            steps = [data for name, data in events if name == 'step']
            self.assertTrue(steps)
            self.assertTrue(all((data['level_id'], data['playthrough']) in POOL for data in steps))

    def test_serial_log_keeps_level_after_each_reset(self):
        config = self.config(total_steps=8, runtime=TrainingRuntimeConfig(num_envs=1))
        with tempfile.TemporaryDirectory() as directory:
            events = []
            controller = DoubleDQNTrainerController(config, run_dir=directory, eval_every_episodes=0,
                on_event=lambda name, data: events.append((name, data)))
            with patch('train.dqn.base_controller.PvZEnv', SerialEnvironment), \
                 patch('train.dqn.base_controller.PvZClient'):
                controller.run()
            records = [json.loads(line) for line in Path(directory, 'episodes.jsonl').read_text().splitlines()]
            self.assertEqual(len(records), 8)
            self.assertEqual({(r['level_id'], r['playthrough']) for r in records}, set(POOL))
            steps = [data for name, data in events if name == 'step']
            self.assertEqual([(s['level_id'], s['playthrough']) for s in steps],
                             [(r['level_id'], r['playthrough']) for r in records])

    def test_worker_reset_reports_sampled_level_and_explicit_seed(self):
        inbox, outbox, stop = queue.Queue(), queue.Queue(), threading.Event()
        for _ in range(5):
            inbox.put(('reset', 123))
        inbox.put(('close', None))
        config = self.config()
        _worker(lambda: SerialEnvironment(level_id=41, playthrough=1), (),
                config, 0, inbox, outbox, stop, initial_seed=123)
        infos = []
        while not outbox.empty():
            kind, _, payload, _ = outbox.get()
            self.assertEqual(kind, 'ready')
            infos.append(payload[2])
        self.assertEqual(len(infos), 6)
        # An explicit evaluation seed must continue to allow random level draws.
        self.assertEqual({(i['level_id'], i['playthrough']) for i in infos}, set(POOL))
        for info in infos:
            self.assertIn((info['level_id'], info['playthrough']), POOL)
            self.assertEqual(info['episode_seed'], 123)

    def test_reset_metadata_is_not_a_reference_to_environment(self):
        config = self.config()
        env = SerialEnvironment(level_id=49, playthrough=2)
        metadata = episode_level_metadata({}, config.env, env)
        env.level_id, env.playthrough = 41, 1
        self.assertEqual((metadata['level_id'], metadata['playthrough']), (49, 2))


if __name__ == '__main__':
    unittest.main()
