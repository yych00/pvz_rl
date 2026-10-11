from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from pvz_gym.action_space import ACTION_SPACE_SIZE
from pvz_gym.observation import empty_observation
from pvz_gym.types import Observation
from train.dqn.actor_learner import AsyncActorLearner
from train.dqn.agent import DoubleDQNAgent
from train.dqn.config import DoubleDQNConfig
from train.dqn.base_controller import DoubleDQNTrainerController
from train.dqn.parallel_controller import run_parallel
from train.dqn.optimized_controller import DoubleDQNTrainerController as OptimizedController
from train.dqn.replay import ReplayBuffer
from train.runtime import TrainingRuntimeConfig
from tests.unit.test_parallel_training import EvaluationFailureSampler


class ActorLearnerTest(unittest.TestCase):
    def controller(self, **updates):
        config = DoubleDQNConfig(device="cpu", hidden_sizes=(8, 4), reward_scheme="sparse",
            replay_capacity=4, batch_size=2, learning_starts=2, n_step=1,
            policy_sync_interval=2, max_pending_updates=2,
            runtime=TrainingRuntimeConfig(max_policy_lag=2))
        config = replace(config, **updates)
        c = SimpleNamespace(config=config, _agent=DoubleDQNAgent(ACTION_SPACE_SIZE, config),
            _replay=ReplayBuffer(4, ACTION_SPACE_SIZE, scheme=config.replay_scheme),
            _agent_lock=threading.RLock(), _replay_lock=threading.RLock(),
            stop_event=threading.Event(), pause_event=threading.Event(),
            _summary=SimpleNamespace(env_steps=4, last_loss=None))
        c.pause_event.set()
        self.obs = empty_observation()
        self.mask = np.ones(ACTION_SPACE_SIZE, dtype=bool)
        for _ in range(4):
            c._replay.add(self.obs, 0, 1., self.obs, True, self.mask, ticks_advanced=100)
        return c

    def wait_for(self, predicate):
        deadline = time.monotonic() + 5
        while not predicate():
            if time.monotonic() >= deadline:
                self.fail("background learner did not reach expected state")
            time.sleep(.005)

    def fake_learn(self, c):
        def update(batch):
            with torch.no_grad():
                next(c._agent.online.parameters()).add_(1)
            c._agent.update_count += 1
            c._agent.last_td_errors = np.full(c.config.batch_size, 99., dtype=np.float32)
            return 1.
        return update

    def test_inference_runs_while_training_holds_learner_lock_and_weights_are_isolated(self):
        c = self.controller()
        rng = torch.get_rng_state().clone()
        learner = AsyncActorLearner(c)
        torch.testing.assert_close(torch.get_rng_state(), rng)
        initial = next(learner.actor.online.parameters()).detach().clone()
        self.assertNotEqual(next(c._agent.online.parameters()).data_ptr(),
                            next(learner.actor.online.parameters()).data_ptr())
        started, release = threading.Event(), threading.Event()
        def blocked(batch):
            started.set()
            if not release.wait(5): raise TimeoutError("test learner gate timed out")
            return self.fake_learn(c)(batch)
        with patch.object(c._agent, "learn", side_effect=blocked):
            learner.start()
            try:
                learner.grant(1)
                self.assertTrue(started.wait(5))
                self.assertEqual(learner.actor.select_actions([self.obs], [self.mask],
                    env_step=0, explore=False), [0])
                torch.testing.assert_close(next(learner.actor.online.parameters()), initial)
                self.assertEqual(learner.status()["pending_learning_updates"], 1)
                release.set()
                learner.drain()
                torch.testing.assert_close(next(learner.actor.online.parameters()), initial)
                learner.sync_actor()
                self.assertEqual(learner.actor.update_count, 1)
                torch.testing.assert_close(next(learner.actor.online.parameters()), initial + 1)
            finally:
                release.set()
                learner.close()

    def test_slow_inflight_policy_bounds_updates_and_pending_credits_are_not_dropped(self):
        c = self.controller()
        learner = AsyncActorLearner(c)
        with patch.object(c._agent, "learn", side_effect=self.fake_learn(c)):
            learner.start()
            try:
                learner.set_policy_floor({0: (None, 0)})
                learner.grant(5)
                self.wait_for(lambda: learner.status()["learner_policy_version"] == 2)
                learner.sync_actor()
                learner.set_policy_floor({0: (None, 0)})
                self.assertTrue(learner.backpressured())
                with learner.quiesce():
                    state = learner.checkpoint_state()
                    self.assertEqual(state["pending_learning_updates"], 3)
                    self.assertEqual(c._agent.update_count, 2)
                learner.set_policy_floor({})
                self.wait_for(lambda: c._agent.update_count == 4)
                learner.drain()
                self.assertEqual(c._agent.update_count, 5)
                self.assertEqual(learner.status()["pending_learning_updates"], 0)
            finally:
                learner.close()

    def test_per_writeback_ignores_slots_overwritten_during_learning(self):
        c = self.controller(replay_scheme="per")
        started, release = threading.Event(), threading.Event()
        def blocked(batch):
            started.set()
            if not release.wait(5): raise TimeoutError("test learner gate timed out")
            return self.fake_learn(c)(batch)
        learner = AsyncActorLearner(c)
        with patch.object(c._agent, "learn", side_effect=blocked):
            learner.start()
            try:
                learner.grant(1)
                self.assertTrue(started.wait(5))
                with c._replay_lock:
                    for _ in range(4):
                        c._replay.add(self.obs, 0, 2., self.obs, True, self.mask, ticks_advanced=100)
                release.set()
                learner.drain()
                np.testing.assert_array_equal(c._replay._priorities, np.ones(4))
            finally:
                release.set()
                learner.close()

    def test_pause_and_stop_keep_unspent_budget_and_resume_actor_snapshot(self):
        c = self.controller()
        learner = AsyncActorLearner(c)
        learner.start()
        try:
            with learner.quiesce():
                learner.grant(3)
                state = learner.checkpoint_state()
                self.assertEqual(state["pending_learning_updates"], 3)
                c.stop_event.set()
            learner.close()

            self.assertFalse(learner.thread.is_alive())
            c.stop_event.clear()
            restored = AsyncActorLearner(c, restored=state)
            with patch.object(c._agent, "learn", side_effect=self.fake_learn(c)):
                restored.start()
                try:
                    restored.drain()
                    self.assertEqual(c._agent.update_count, 3)
                finally:
                    restored.close()
        finally:
            learner.close()

    def test_resume_with_tighter_lag_starts_actor_at_current_learner_version(self):
        c = self.controller(runtime=TrainingRuntimeConfig(max_policy_lag=1))
        old_policy = {key: value.detach().clone() for key, value in c._agent.online.state_dict().items()}
        with torch.no_grad():
            next(c._agent.online.parameters()).add_(1)
        c._agent.update_count = 4
        learner = AsyncActorLearner(c, restored={'actor_policy': old_policy, 'actor_policy_version': 0})
        self.assertEqual(learner.actor.update_count, 4)
        torch.testing.assert_close(next(learner.actor.online.parameters()), next(c._agent.online.parameters()))

    def test_background_failure_reaches_controller_and_reclaims_sampler_and_thread(self):
        with tempfile.TemporaryDirectory() as directory:
            config = replace(self.controller().config, total_steps=4, train_every=1,
                autosave_every=0, checkpoint_every=0,
                runtime=TrainingRuntimeConfig(num_envs=2, asynchronous=True,
                    inference_batch_size=2, inference_max_wait_ms=0))
            controller = DoubleDQNTrainerController(config, run_dir=directory, eval_every_episodes=0)
            sampler = EvaluationFailureSampler(config)
            sampler.close = unittest.mock.Mock()
            with patch.object(DoubleDQNAgent, "learn", side_effect=ValueError("intentional learner failure")):
                with self.assertRaisesRegex(RuntimeError, "background training failed"):
                    run_parallel(controller, sampler_factory=lambda *a, **kw: sampler)
            self.assertEqual(controller.state, "FAILED")
            sampler.close.assert_called()
            self.assertFalse(any(t.name == "pvz-dqn-learner" for t in threading.enumerate()))
            self.assertFalse(Path(directory, "final.pt").exists())

    def test_controller_keeps_sampling_during_a_blocked_update_and_finishes_exact_budget(self):
        started, release = threading.Event(), threading.Event()
        original_learn = DoubleDQNAgent.learn
        def blocked(agent, batch):
            started.set()
            if not release.wait(5): raise TimeoutError("sampling failed to progress during learning")
            return original_learn(agent, batch)
        with tempfile.TemporaryDirectory() as directory:
            config = replace(self.controller().config, total_steps=6, train_every=1,
                max_pending_updates=128, autosave_every=0, checkpoint_every=0,
                runtime=TrainingRuntimeConfig(num_envs=2, asynchronous=True,
                    inference_batch_size=2, inference_max_wait_ms=0))
            def on_event(name, data):
                if name == 'step' and data['env_step'] == 4:
                    self.assertTrue(started.is_set())
                    release.set()
            controller = DoubleDQNTrainerController(config, run_dir=directory,
                eval_every_episodes=0, on_event=on_event)
            sampler = EvaluationFailureSampler(config)
            original_receive = sampler.receive
            def receive(timeout=.05):
                if controller.summary.env_steps >= 2:
                    self.assertTrue(started.wait(5))
                return original_receive(timeout)
            sampler.receive = receive
            try:
                with patch.object(DoubleDQNAgent, 'learn', blocked):
                    summary = run_parallel(controller, sampler_factory=lambda *a, **kw: sampler)
                self.assertEqual(summary.env_steps, 6)
                self.assertEqual(controller._agent.update_count, 5)
                payload = torch.load(Path(directory, 'final.pt'), weights_only=False)
                self.assertEqual(payload['sampling_state']['pending_learning_updates'], 0)
                self.assertEqual(payload['sampling_state']['actor_policy_version'], 5)
                self.assertTrue(Path(directory, 'learner_timing.json').is_file())
                timing = json.loads(Path(directory, 'learner_timing.json').read_text(encoding='utf-8'))
                for name in ('learn', 'learn_backward_wall', 'learn_data_prepare_wall',
                             'learn_optimizer_wall', 'agent_lock_wait'):
                    self.assertEqual(timing[name]['count'], 5)
                # Overwritten sample slots are intentionally excluded from PER.
                self.assertGreater(timing['per_priority_update']['count'], 0)
                self.assertLessEqual(timing['per_priority_update']['count'], 5)
                self.assertIsNone(controller._agent.learner_profiler)
            finally:
                release.set()

    def test_defaults_validation_and_legacy_checkpoint_config(self):
        self.assertEqual(DoubleDQNConfig().training_architecture, "actor_learner")
        for updates in ({"training_architecture": "unknown"}, {"policy_sync_interval": 0},
                        {"max_pending_updates": True}, {"runtime": TrainingRuntimeConfig(max_policy_lag=0)}):
            with self.subTest(updates=updates), self.assertRaises(ValueError):
                replace(DoubleDQNConfig(), **updates)
        with tempfile.TemporaryDirectory() as directory:
            c = self.controller()
            payload = c._agent.checkpoint_payload(env_step=4, episode=0)
            payload["config"].pop("training_architecture")
            path = Path(directory, "old.pt")
            torch.save(payload, path)
            _, config, _ = DoubleDQNAgent.from_checkpoint(path, device="cpu")
            self.assertEqual(config.training_architecture, "legacy")

    def test_cli_selects_default_and_legacy_architectures_with_sync_controls(self):
        from train.dqn.train import main
        cases = (([], 'actor_learner', 32, 128),
                 (['--training-architecture', 'legacy', '--max-policy-lag', '0'], 'legacy', 32, 128),
                 (['--training-architecture', 'actor_learner', '--policy-sync-interval', '8',
                   '--max-pending-updates', '24'], 'actor_learner', 8, 24))
        for args, architecture, interval, capacity in cases:
            with self.subTest(args=args), patch('sys.argv', ['train'] + args), \
                 patch('pvz_gym.env_file.get_game_executable', return_value=None), \
                 patch('train.dqn.train.train') as train:
                main()
                config = train.call_args.args[0]
                self.assertEqual(config.training_architecture, architecture)
                self.assertEqual(config.policy_sync_interval, interval)
                self.assertEqual(config.max_pending_updates, capacity)

    def test_async_reward_schemes_and_required_observation_contract(self):
        class RewardSampler(EvaluationFailureSampler):
            def facts(self, done=False):
                return dict(super().facts(done), raw_observation=Observation(global_state={
                    'sun': 50, 'spawn_plan_available': True, 'total_waves': 1,
                    'zombie_spawn_plan': [0] + [-1] * 49}, zombie_health=()))
        for scheme in ('sparse', 'staged', 'transition_dense'):
            for boss_projectiles in (True,):
                with self.subTest(scheme=scheme, boss_projectiles=boss_projectiles), \
                     tempfile.TemporaryDirectory() as directory:
                    config = replace(self.controller().config, reward_scheme=scheme,
                        boss_projectiles=boss_projectiles, total_steps=6, train_every=1,
                        autosave_every=0, checkpoint_every=0,
                        runtime=TrainingRuntimeConfig(num_envs=2, asynchronous=True,
                            inference_batch_size=2, inference_max_wait_ms=0))
                    controller = DoubleDQNTrainerController(config, run_dir=directory, eval_every_episodes=0)
                    sampler = RewardSampler(config)
                    summary = run_parallel(controller, sampler_factory=lambda *a, **kw: sampler)
                    self.assertEqual(summary.env_steps, 6)
                    self.assertEqual(controller._agent.update_count, 5)
                    self.assertEqual('reward_context' in controller._replay.observation_spec,
                                     scheme == 'transition_dense')

    def test_model_only_resume_resets_old_debt_so_empty_replay_can_warm_up(self):
        with tempfile.TemporaryDirectory() as directory:
            config = replace(self.controller().config, total_steps=8, train_every=1,
                autosave_every=0, checkpoint_every=0,
                runtime=TrainingRuntimeConfig(num_envs=2, asynchronous=True,
                    inference_batch_size=2, inference_max_wait_ms=0))
            agent = DoubleDQNAgent(ACTION_SPACE_SIZE, config)
            agent.sampling_state = {'pending_learning_updates': 10}
            path = agent.save(Path(directory, 'source.pt'), env_step=4, episode=0)
            events = []
            controller = OptimizedController(config, run_dir=directory, checkpoint_path=path,
                checkpoint_load_mode='model_resume', eval_every_episodes=0,
                on_event=lambda name, data: events.append((name, data)))
            sampler = EvaluationFailureSampler(config)
            summary = run_parallel(controller, sampler_factory=lambda *a, **kw: sampler)
            self.assertEqual(summary.env_steps, 8)
            self.assertEqual(controller._agent.update_count, 3)
            self.assertEqual(len(controller._replay), 4)
            self.assertTrue(any(name == 'learning_budget_reset' and data['discarded_updates'] == 10
                                for name, data in events))

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
    def test_cuda_training_and_actor_sync_use_independent_model_storage(self):
        c = self.controller(device='cuda')
        learner = AsyncActorLearner(c)
        learner.start()
        try:
            learner.grant(4)
            deadline = time.monotonic() + 15
            while learner.status()['pending_learning_updates']:
                if time.monotonic() > deadline:
                    self.fail('CUDA actor/learner did not complete updates')
                actions = learner.actor.select_actions([self.obs], [self.mask],
                    env_step=4, explore=False)
                self.assertTrue(self.mask[actions[0]])
                learner.sync_actor()
                learner.set_policy_floor({})
            learner.drain()
            learner.sync_actor()
            self.assertEqual(c._agent.update_count, 4)
            self.assertEqual(learner.actor.update_count, 4)
            self.assertTrue(np.isfinite(c._summary.last_loss))
            for actor, trained in zip(learner.actor.online.parameters(), c._agent.online.parameters()):
                self.assertNotEqual(actor.data_ptr(), trained.data_ptr())
                torch.testing.assert_close(actor, trained)
        finally:
            learner.close()

        stats = learner.timing.snapshot()
        self.assertEqual(stats['learn']['count'], 4)
        self.assertEqual(stats['learn_backward_wall']['count'], 4)
        self.assertEqual(stats['learn_backward_cuda']['count'], 1)
        self.assertIsNone(c._agent.learner_profiler)


if __name__ == "__main__":
    unittest.main()
