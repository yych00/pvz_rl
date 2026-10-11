from __future__ import annotations
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch
import tempfile
import threading
import time
import queue
import json
import unittest
import numpy as np
import torch
from pvz_gym.observation import empty_observation
from pvz_gym.action_space import ACTION_SPACE_SIZE
from train.common import EnvConfig
from train.runtime import TrainingRuntimeConfig, PolicyReply, SampleIdentity
from train.sampler import ProcessSampler
from train.dqn.agent import DoubleDQNAgent
from train.dqn.config import DoubleDQNConfig
from train.dqn.base_controller import DoubleDQNTrainerController
from train.dqn.parallel_controller import run_parallel
from train.dqn.replay import ReplayBuffer


class FakeEnvironment:
    """Spawnable environment fixture, not a game simulator."""
    def __init__(self, environment_id, failure_marker=None): self.i=environment_id; self.marker=failure_marker; self.steps=0
    def facts(self, done=False):
        obs = empty_observation(); obs['global'][0] = self.i*100+self.steps
        return obs, dict(action_mask=(True,)+(False,)*(ACTION_SPACE_SIZE-1),
            ticks_advanced=100, ticks_requested=100, game_clock=self.steps*100,
            episode_ticks=self.steps*100, result="WON" if done else "PLAYING",
            events={}, action_result={"success":True})
    def reset(self, *, seed): self.steps=0; return self.facts()
    def step(self, action):
        if self.marker and not Path(self.marker).exists():
            Path(self.marker).touch(); raise RuntimeError("one intentional worker failure")
        assert action == 0
        self.steps += 1
        obs, info = self.facts(self.steps==3)
        return obs, float(self.i*10+self.steps), self.steps==3, False, info
    def close(self): pass


def fake_environment(i, marker=None): return FakeEnvironment(i, marker)


class EvaluationFailureSampler(ProcessSampler):
    """Deterministic workers using the production failure and recovery logic."""
    def __init__(self, config, *, failure_environment_id=1, **kwargs):
        self.config = config; self.runtime = config.runtime
        self.output = queue.Queue()
        self.workers, self.inputs, self.outstanding = {}, {}, {}
        self.restarts = {i: 0 for i in range(config.runtime.num_envs)}
        self.generations = {i: -1 for i in self.restarts}
        self._failed_workers = set()
        self.states = {}; self.episodes = {}; self.dead = set()
        self.failure_environment_id = failure_environment_id
        self.injected = False; self.delayed_ready = None; self.empty_polls = 0

    def _send(self, kind, i, payload):
        self.output.put((kind, i, payload, self.generations[i]))

    def _start(self, i, restored=None):
        self.generations[i] += 1; self._failed_workers.discard(i); self.dead.discard(i)
        self.workers[i] = Mock(is_alive=lambda: i not in self.dead)
        self.inputs[i] = Mock()
        self.episodes[i] = int((restored or {}).get('episode', 0))
        self.reset(i)

    def facts(self, done=False):
        return dict(action_mask=(True,) + (False,) * (ACTION_SPACE_SIZE - 1),
                    result='WON' if done else 'PLAYING', events={},
                    action_result={'success': True}, ticks_advanced=100, episode_ticks=100)

    def reset(self, i, seed=None):
        self.episodes[i] += 1
        self.outstanding[i] = time.monotonic()
        payload = (SampleIdentity(i, self.episodes[i], 0), empty_observation(), self.facts(), None)
        if seed is not None and not self.injected:
            self.injected = True
            peer = self.failure_environment_id
            self.dead.add(peer)
            self._send('error', peer, 'one failure during evaluation')
            if peer == i: return
            self.delayed_ready = (i, payload)
            return
        self._send('ready', i, payload)

    def receive(self, timeout=0.05):
        message = super().receive(0)
        if message is None and self.delayed_ready is not None:
            self.empty_polls += 1
            if self.empty_polls == 3:
                self._send('ready', *self.delayed_ready)
                self.delayed_ready = None
        return message

    def dispatch(self, reply):
        i = reply.identity.environment_id
        self.outstanding[i] = time.monotonic()
        self._send('transition', i, (reply.identity, reply.action, reply.policy_version,
                   empty_observation(), 100., True, False, self.facts(True), None))

    def close(self): pass


class ParallelTrainingTest(unittest.TestCase):
    def test_failure_during_evaluation_recovers_once_and_training_finishes(self):
        for failing_environment in (0, 1):
            with self.subTest(failing_environment=failing_environment), tempfile.TemporaryDirectory() as directory:
                config = self.config(total_steps=6, learning_starts=32, autosave_every=0,
                    checkpoint_every=0, runtime=replace(self.config().runtime,
                        inference_batch_size=1, inference_max_wait_ms=0))
                events = []
                controller = DoubleDQNTrainerController(config, run_dir=directory,
                    eval_every_episodes=1, eval_seed_count=1,
                    on_event=lambda name, data: events.append((name, data)))
                sampler = EvaluationFailureSampler(config, failure_environment_id=failing_environment)
                summary = run_parallel(controller, sampler_factory=lambda *args, **kwargs: sampler)
                self.assertEqual(controller.state, 'FINISHED')
                self.assertEqual(summary.env_steps, 6)
                self.assertEqual(sampler.restarts[failing_environment], 1)
                recoveries = [data for name, data in events if name in ('worker_recovery', 'evaluation_retry')]
                self.assertEqual(len(recoveries), 1)
                self.assertEqual(recoveries[0]['environment_id'], failing_environment)
                self.assertTrue(Path(directory, 'final.pt').is_file())
                self.assertEqual(len(controller._replay), 6)
                if failing_environment == 1:
                    self.assertEqual(sampler.empty_polls, 3)
                    # The peer's completed transition was buffered before its
                    # error; recovery must retain that episode's training data.
                    self.assertTrue(any(name == 'episode_end'
                        and data['environment_id'] == 1 and data['environment_episode'] == 1
                        for name, data in events))

    def test_dead_or_timed_out_generation_reports_error_once(self):
        for dead in (True, False):
            with self.subTest(dead=dead):
                sampler = object.__new__(ProcessSampler)
                sampler.output = queue.Queue()
                sampler.generations = {0: 0}; sampler._failed_workers = set()
                sampler.workers = {0: SimpleNamespace(is_alive=lambda: not dead, exitcode=7 if dead else None)}
                sampler.runtime = TrainingRuntimeConfig(worker_timeout_s=1)
                sampler.outstanding = {0: time.monotonic() - 2}
                message = sampler.receive(0)
                self.assertEqual(message[:2], ('error', 0))
                self.assertIn('exited (exitcode=7)' if dead else 'command timed out', message[2])
                if not dead: self.assertIn('worker_timeout_s=1', message[2])
                self.assertIsNone(sampler.receive(0))
                sampler.output.put(('error', 0, 'late duplicate', 0))
                self.assertIsNone(sampler.receive(0))

    def test_recovery_record_is_appended_without_ui_callback(self):
        with tempfile.TemporaryDirectory() as directory:
            controller = DoubleDQNTrainerController(self.config(), run_dir=directory)
            for attempt in (1, 2):
                controller._emit('worker_recovery', environment_id=0, attempt=attempt,
                                 error='TimeoutError: 原生调用超时\nfull traceback')
            records = [json.loads(line) for line in
                       Path(directory, 'runtime_events.jsonl').read_text(encoding='utf-8').splitlines()]
            self.assertEqual([record['attempt'] for record in records], [1, 2])
            self.assertEqual(records[0]['error'], 'TimeoutError: 原生调用超时\nfull traceback')
            self.assertIn('timestamp', records[0])

    def config(self, **updates):
        return replace(DoubleDQNConfig(env=EnvConfig(ticks_per_step=100), device="cpu", hidden_sizes=(16,8),
            reward_scheme="sparse", exploration_mode="epsilon_random", epsilon_start=1, epsilon_end=1,
            total_steps=17, replay_capacity=32, batch_size=2, learning_starts=2, n_step=2,
            train_every=2, gradient_steps=2, autosave_every=8, checkpoint_every=12,
            runtime=TrainingRuntimeConfig(num_envs=2, asynchronous=True, inference_batch_size=2,
                worker_timeout_s=30, max_policy_lag=2, ui_update_interval_ms=0)), **updates)
    def run_fixture(self, config, directory, *, checkpoint=None, marker=None, stop_after=None):
        events=[]
        def on_event(event, data):
            events.append((event, data))
            if event == 'step' and stop_after is not None and data['env_step'] >= stop_after:
                controller.stop()
        controller=DoubleDQNTrainerController(config, run_dir=directory, checkpoint_path=checkpoint,
            eval_every_episodes=0, on_event=on_event)
        def sampler_factory(config, **kwargs):
            return ProcessSampler(config, factory=fake_environment, factory_args=[(0,), (1,marker)])
        summary=run_parallel(controller, sampler_factory=sampler_factory)
        return controller, summary, events

    def test_stop_during_ingest_does_not_dispatch_another_policy_batch(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(ProcessSampler, 'dispatch', autospec=True,
                              side_effect=ProcessSampler.dispatch) as dispatch:
                controller, summary, events = self.run_fixture(self.config(total_steps=100),
                    directory, stop_after=3)
            self.assertTrue(summary.stopped)
            self.assertEqual(summary.env_steps, 3)
            # At most the other worker's existing command can still be in flight.
            self.assertLessEqual(dispatch.call_count, 4)
            self.assertEqual(controller.state, 'STOPPED')
            checkpoint = torch.load(Path(directory, 'stopped.pt'), weights_only=False)
            replay, metadata = ReplayBuffer.load(Path(directory, checkpoint['replay_file']))
            self.assertEqual(metadata['env_step'], 3)
            self.assertEqual(len(replay), 3)

    def test_async_no_overshoot_update_ratio_stream_isolation_and_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            controller, summary, events=self.run_fixture(self.config(n_step=1), directory)
            self.assertEqual(summary.env_steps, 17)
            self.assertEqual(controller._agent.update_count, 16)
            replay=controller._replay
            self.assertEqual(set(replay._stream_ids[:len(replay)]), {0,1})
            for slot in range(len(replay)):
                successor=replay._slot_for_id(int(replay._next_ids[slot]))
                if successor is not None: self.assertEqual(replay._stream_ids[slot],replay._stream_ids[successor])
            self.assertTrue(Path(directory,'final.pt').is_file())
            checkpoint=Path(directory,'checkpoint-000000012.pt')
            model=torch.load(checkpoint,weights_only=False)
            saved,meta=ReplayBuffer.load(Path(directory,model['replay_file']))
            self.assertEqual(meta['env_step'],12); self.assertEqual(len(saved),12)
            resumed, summary, resumed_events=self.run_fixture(self.config(total_steps=23,n_step=1),directory,
                checkpoint=Path(directory,'final.pt'))
            self.assertEqual(summary.env_steps,23); self.assertTrue(summary.replay_restored)
            loaded = [data for event, data in resumed_events if event == 'checkpoint_loaded']
            self.assertEqual(len(loaded), 1)
            self.assertEqual(loaded[0]['env_step'], 17)
            self.assertEqual(loaded[0]['episode'], controller._summary.episodes)
            self.assertTrue(loaded[0]['replay_restored'])
            self.assertEqual(loaded[0]['replay_size'], 17)
            self.assertFalse(resumed._agent.sampling_state['exact_game_resume'])
            self.assertEqual(set(resumed._agent.sampling_state['workers']),{'0','1'})

    def test_sync_and_single_worker_recovery_keep_other_stream_running(self):
        with tempfile.TemporaryDirectory() as directory:
            marker=str(Path(directory,'failed-once'))
            controller, summary, events=self.run_fixture(self.config(total_steps=10,
                training_architecture="legacy",
                runtime=replace(self.config().runtime,asynchronous=False)),directory,marker=marker)
            recoveries=[data for name,data in events if name=='worker_recovery']
            self.assertEqual(len(recoveries),1); self.assertEqual(recoveries[0]['environment_id'],1)
            records = [json.loads(line) for line in
                       Path(directory, 'runtime_events.jsonl').read_text(encoding='utf-8').splitlines()]
            self.assertEqual(records[0]['event'], 'worker_recovery')
            self.assertIn('one intentional worker failure', records[0]['error'])
            self.assertEqual(summary.env_steps,10)
            self.assertEqual(len(controller._replay),10)

    def test_n_step_interleaving_wrap_and_checkpoint_do_not_join_environments(self):
        replay=ReplayBuffer(5,2)
        obs=empty_observation()
        for episode in range(2):
            for step in range(3):
                for stream in range(2):
                    replay.add(obs,0,10**stream,obs,step==2,[True,False],ticks_advanced=100,stream_id=stream)
        with patch.object(replay,'_rng',SimpleNamespace(integers=lambda *a,**k:np.array([0,1]))):
            batch=replay.sample(2,n_step=3,gamma=1,gamma_reference_ticks=100)
        np.testing.assert_array_equal(batch.rewards,[30,2])
        with tempfile.TemporaryDirectory() as directory:
            path=replay.save(Path(directory,'replay.npz')); restored,_=ReplayBuffer.load(path)
        with patch.object(restored,'_rng',SimpleNamespace(integers=lambda *a,**k:np.array([0,1]))):
            again=restored.sample(2,n_step=3,gamma=1,gamma_reference_ticks=100)
        np.testing.assert_array_equal(again.rewards,batch.rewards)

    def test_one_batch_forward_masks_actions_for_each_environment(self):
        agent=DoubleDQNAgent(ACTION_SPACE_SIZE,self.config())
        observations=[empty_observation() for _ in range(3)]
        masks=np.zeros((3,ACTION_SPACE_SIZE),dtype=bool)
        masks[np.arange(3),[0,15,600]]=True
        with patch.object(agent.online,'forward',wraps=agent.online.forward) as forward:
            self.assertEqual(agent.select_actions(observations,masks,env_step=0,explore=False),[0,15,600])
            self.assertEqual(forward.call_count,1)

    def test_async_checkpoint_owns_snapshots_before_training_mutates_them(self):
        with tempfile.TemporaryDirectory() as directory:
            owner = threading.get_ident()
            notifications = []
            controller=DoubleDQNTrainerController(self.config(),run_dir=directory,eval_every_episodes=0,
                on_event=lambda event, data: notifications.append((event, threading.get_ident())))
            controller._agent=DoubleDQNAgent(ACTION_SPACE_SIZE,self.config())
            controller._replay=ReplayBuffer(4,ACTION_SPACE_SIZE)
            obs=empty_observation()
            controller._replay.add(obs,0,3,obs,False,[True]*ACTION_SPACE_SIZE,ticks_advanced=100)
            controller._summary.env_steps=1
            original=next(controller._agent.online.parameters()).detach().clone()
            controller._save_scheduled_checkpoint(Path(directory,'latest.pt'))
            with torch.no_grad(): next(controller._agent.online.parameters()).add_(100)
            controller._replay._rewards[0]=999
            controller._checkpoint_writer.pending.result()
            self.assertEqual(notifications, [])
            controller._flush_checkpoint_writer(close=True)
            self.assertEqual([event for event, _ in notifications], ['checkpoint', 'replay_checkpoint'])
            self.assertTrue(all(thread_id == owner for _, thread_id in notifications))
            model=torch.load(Path(directory,'latest.pt'),weights_only=False)
            torch.testing.assert_close(next(iter(model['online'].values())),original)
            saved,meta=ReplayBuffer.load(Path(directory,model['replay_file']))
            self.assertEqual(float(saved._rewards[0]),3); self.assertEqual(meta['env_step'],1)
