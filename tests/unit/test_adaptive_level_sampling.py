import json
import tempfile
import threading
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from train.adaptive_curriculum import AdaptiveCurriculum
from train.common import EnvConfig
from train.dqn.config import DoubleDQNConfig
from train.dqn.evaluation_plan import evaluation_plan, protocol_fields, protocol_matches
from train.dqn.parallel_controller import _evaluate_parallel
from train.runtime import TrainingRuntimeConfig
from pvz_gym.action_space import ACTION_SPACE_SIZE
from pvz_gym.profiling import TimingRecorder
from tests.unit.test_dqn_parallel_evaluation import MemorySampler


class LevelSampler(MemorySampler):
    def __init__(self, config):
        super().__init__(config)
        self.levels = {}

    def reset(self, i, seed=None, *, level=None):
        self.levels[i] = level or (41, 1)
        return super().reset(i, seed)

    def facts(self, i, done=False):
        return dict(super().facts(i, done), level_id=self.levels[i][0],
                    playthrough=self.levels[i][1])


def config(**kwargs):
    return DoubleDQNConfig(env=EnvConfig(level_pool=((41, 1), (48, 2))),
        reward_scheme='sparse', adaptive_level_sampling=True,
        runtime=TrainingRuntimeConfig(num_envs=4), **kwargs)


class AdaptiveLevelTests(unittest.TestCase):
    def test_window_bounds_and_excluded_results(self):
        curriculum = AdaptiveCurriculum(config())
        for _ in range(80):
            curriculum.observe(dict(level_id=41, playthrough=1, result='WON', episode_steps=100))
            curriculum.observe(dict(level_id=48, playthrough=2, result='LOST', episode_steps=300))
        self.assertEqual([len(x) for x in curriculum.history.values()], [40, 40])
        self.assertGreater(curriculum.probabilities[1], curriculum.probabilities[0])
        self.assertAlmostEqual(sum(curriculum.probabilities), 1)
        before = curriculum.snapshot()
        for result, truncated in [('UNKNOWN', False), ('LOST', True)]:
            curriculum.observe(dict(level_id=48, playthrough=2, result=result, truncated=truncated))
        curriculum.observe(dict(level_id=48, playthrough=1, result='LOST'))
        self.assertEqual(before, curriculum.snapshot())
        rows = curriculum.rows()
        self.assertAlmostEqual(rows[1]['step_share'], .75)
        rng = np.random.default_rng(7)
        samples = [curriculum.sample(rng) for _ in range(10000)]
        self.assertGreater(samples.count((48, 2)), samples.count((41, 1)))

    def test_hard_level_gets_majority_and_mastered_levels_rehearse(self):
        pool = tuple((i, 1) for i in range(1, 9))
        cfg = replace(config(), env=EnvConfig(level_pool=pool))
        c = AdaptiveCurriculum(cfg)
        for _ in range(40):
            for i in range(1, 9):
                # One 10% level, one 85% level, and six mastered levels.
                wins = 4 if i == 1 else 34 if i == 2 else 40
                c.observe(dict(level_id=i, playthrough=1,
                               result='WON' if _ < wins else 'LOST'))
        self.assertGreater(c.probabilities[0], .6)
        self.assertGreater(c.probabilities[1], c.probabilities[2])
        self.assertGreater(c.probabilities[2], .1 / len(pool))
        self.assertAlmostEqual(sum(c.probabilities), 1)

    def test_legacy_checkpoint_recomputes_probabilities(self):
        c = AdaptiveCurriculum(config())
        for _ in range(40):
            c.observe(dict(level_id=41, playthrough=1, result='WON'))
            c.observe(dict(level_id=48, playthrough=2, result='LOST'))
        old = c.snapshot()
        old['version'] = 1
        old['probabilities'] = [.5, .5]
        restored = AdaptiveCurriculum(config())
        self.assertTrue(restored.restore(old))
        self.assertGreater(restored.probabilities[1], .5)

    def test_squared_failure_weights_and_half_half_update(self):
        curriculum = AdaptiveCurriculum(config(level_sampling_update_every=2))
        curriculum.observe(dict(level_id=41, playthrough=1, result='WON'))
        curriculum.observe(dict(level_id=48, playthrough=2, result='LOST'))
        # Weights: 0.05 and 1.05; target mixes 5% uniform.
        target = .05 / 2 + .95 * .05 / 1.10
        expected = .5 * .5 + .5 * target
        self.assertAlmostEqual(curriculum.probabilities[0], expected)
        self.assertAlmostEqual(sum(curriculum.probabilities), 1)

    def test_recovery_shifts_sampling_back_and_roundtrip(self):
        c = AdaptiveCurriculum(config())
        for _ in range(100):
            c.observe(dict(level_id=41, playthrough=1, result='WON'))
            c.observe(dict(level_id=48, playthrough=2, result='LOST'))
        before = c.probabilities[1]
        restored = AdaptiveCurriculum(config())
        self.assertTrue(restored.restore(json.loads(json.dumps(c.snapshot()))))
        self.assertEqual(c.snapshot(), restored.snapshot())
        for _ in range(100):
            for pair in c.pool:
                c.observe(dict(level_id=pair[0], playthrough=pair[1], result='WON'))
                restored.observe(dict(level_id=pair[0], playthrough=pair[1], result='WON'))
        self.assertLess(c.probabilities[1], before)
        self.assertEqual(c.snapshot(), restored.snapshot())
        disabled = AdaptiveCurriculum(replace(config(), adaptive_level_sampling=False))
        disabled.restore(c.snapshot())
        self.assertEqual(disabled.probabilities, [.5, .5])

    def test_evaluation_plan_and_protocol_separation(self):
        c = SimpleNamespace(config=config(), eval_seed_start=10001,
                            eval_max_episode_steps=None)
        tasks = evaluation_plan(c, 20)
        self.assertEqual(len(tasks), 8)
        self.assertEqual([pair for _, pair in tasks].count((48, 2)), 4)
        record = dict(protocol_fields(c, 8), max_episode_steps=None,
                      max_episode_ticks=c.config.env.max_episode_ticks)
        self.assertTrue(protocol_matches(c, record, 20))
        c.config = replace(c.config, evaluation_per_level=5)
        self.assertFalse(protocol_matches(c, record, 20))
        c.config = replace(c.config, evaluation_mode='random', evaluation_random_episodes=23)
        self.assertEqual(len(evaluation_plan(c, 20)), 23)
        self.assertFalse(protocol_matches(c, record, 20))

    def test_parallel_per_level_and_retry_keep_playthrough(self):
        cfg = config()
        events = []
        c = SimpleNamespace(config=cfg, eval_seed_start=10001, eval_seed_count=20,
            eval_max_episode_steps=None, _summary=SimpleNamespace(episodes=0),
            _runtime_pause_hook=None, _agent_lock=threading.RLock(),
            pause_event=threading.Event(), stop_event=threading.Event(),
            _agent=SimpleNamespace(update_count=0, select_actions=lambda states, masks, **kw: [0]*len(states)),
            _emit=lambda name, **data: events.append((name, data)), _wait_if_paused=lambda: True)
        sampler = LevelSampler(cfg)
        tasks = [(100+i, pair) for i, pair in enumerate([(41, 1), (48, 2)] * 4)]
        sampler.fail_seed = 101
        with patch('train.dqn.evaluation_plan.evaluation_plan', return_value=tasks):
            record = _evaluate_parallel(c, sampler, ACTION_SPACE_SIZE, env_step=0,
                                        timing=TimingRecorder(False))
        self.assertTrue(record['complete'])
        self.assertEqual(record['retries'], 1)
        self.assertEqual(record['episodes'], 8)
        self.assertEqual({(r['level_id'], r['playthrough']): r['episodes']
                          for r in record['levels']}, {(41, 1): 4, (48, 2): 4})
        self.assertTrue(all(n != 'episode_end' for n, _ in events))

    def test_controller_restores_checkpoint_or_older_episode_log(self):
        from train.dqn.controller import DoubleDQNTrainerController
        from train.dqn.agent import DoubleDQNAgent
        with tempfile.TemporaryDirectory() as folder:
            cfg = replace(config(), hidden_sizes=(8, 4), device='cpu')
            controller = DoubleDQNTrainerController(cfg, run_dir=folder)
            controller._agent = DoubleDQNAgent(ACTION_SPACE_SIZE, cfg)
            controller._restore_curriculum({}, 0)
            records = [dict(episode=i+1, level_id=48 if i%2 else 41,
                            playthrough=2 if i%2 else 1, result='LOST' if i%2 else 'WON',
                            episode_steps=10) for i in range(80)]
            for record in records:
                controller._record_curriculum_episode(record)
            payload = controller._agent.checkpoint_payload(env_step=800, episode=80)
            before = controller._curriculum.snapshot()
            controller._restore_curriculum(payload, 80)
            self.assertEqual(before, controller._curriculum.snapshot())
            (Path(folder)/'episodes.jsonl').write_text('\n'.join(json.dumps(r) for r in records))
            controller._restore_curriculum({}, 80)
            self.assertEqual(before, controller._curriculum.snapshot())
            controller._restore_curriculum(payload, 0)
            self.assertEqual(controller._curriculum.valid_episodes, 0)

    def test_config_validation(self):
        for kwargs in [dict(evaluation_per_level=0), dict(level_sampling_uniform_mix=float('nan')),
                       dict(level_sampling_window=0), dict(level_sampling_max_weight=.5)]:
            with self.assertRaises(ValueError):
                config(**kwargs)

    def test_resume_recomputes_probabilities_for_changed_settings(self):
        curriculum = AdaptiveCurriculum(config())
        for _ in range(60):
            curriculum.observe(dict(level_id=41, playthrough=1, result='WON'))
            curriculum.observe(dict(level_id=48, playthrough=2, result='LOST'))
        self.assertGreater(curriculum.probabilities[1], .5)
        resumed = AdaptiveCurriculum(replace(config(), level_sampling_uniform_mix=1.0))
        self.assertTrue(resumed.restore(curriculum.snapshot()))
        self.assertEqual(resumed.probabilities, [.5, .5])
        self.assertEqual(resumed.valid_episodes, curriculum.valid_episodes)
        resumed.restore(curriculum.snapshot())
        self.assertEqual(len(resumed.history[(41, 1)]), 40)

    def test_corrupt_state_does_not_mutate_history(self):
        curriculum = AdaptiveCurriculum(config())
        before = curriculum.snapshot()
        for state in [[], dict(before, history=['bad', []]),
                      dict(before, probabilities=['bad', .5]),
                      dict(before, counts=[[0], [0, 0]]), dict(before, valid_episodes=-1)]:
            self.assertFalse(curriculum.restore(state))
            self.assertEqual(before, curriculum.snapshot())

    def test_large_finite_weights_still_normalize(self):
        curriculum = AdaptiveCurriculum(config(level_sampling_max_weight=1e308))
        for _ in range(60):
            for pair in curriculum.pool:
                curriculum.observe(dict(level_id=pair[0], playthrough=pair[1], result='LOST'))
        self.assertAlmostEqual(sum(curriculum.probabilities), 1)
        curriculum.sample(np.random.default_rng(0))

    def test_remote_evaluation_updates_worker_rng_after_reset_and_step(self):
        from collections import deque
        from train.dqn.parallel_controller import RemoteEvaluationEnv
        cfg = config()
        sampler = LevelSampler(cfg)
        def receive(timeout=0):
            message = LevelSampler.receive(sampler, timeout)
            if message is None:
                return None
            kind, env_id, payload = message
            return kind, env_id, (*payload[:-1], {'evaluation_rng': kind})
        sampler.receive = receive
        controller = SimpleNamespace(config=cfg, stop_event=threading.Event(),
                                     _agent=SimpleNamespace(update_count=0))
        env = RemoteEvaluationEnv(sampler, 0, deque(), controller)
        env.level_id, env.playthrough = 48, 2
        env.reset(seed=1)
        self.assertEqual(sampler.states['0']['rng_state'], {'evaluation_rng': 'ready'})
        env.step(0)
        self.assertEqual(sampler.states['0']['rng_state'], {'evaluation_rng': 'transition'})


if __name__ == '__main__':
    unittest.main()
