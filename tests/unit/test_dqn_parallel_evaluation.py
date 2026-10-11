"""Evaluation barrier, message ownership and fixed policy contracts."""
from collections import deque
from dataclasses import replace
from pathlib import Path
import tempfile
import threading
import time
import unittest

import torch

from pvz_gym.action_space import ACTION_SPACE_SIZE
from train.common import EnvConfig
from train.runtime import TrainingRuntimeConfig, SampleIdentity, PolicyReply
from train.dqn.config import DoubleDQNConfig
from train.dqn.controller import DoubleDQNTrainerController
from train.dqn.parallel_controller import run_parallel
from train.sampler import ProcessSampler
from pvz_gym.observation import empty_observation


class SeedEnvironment:
    def reset(self, *, seed):
        return empty_observation(), {'seed': seed}
    def close(self): pass
    def step(self, action):
        return empty_observation(), 0., True, False, {}
    def pause(self):
        raise AssertionError('terminal environment cannot confirm native pause')
    def resume(self):
        raise AssertionError('terminal environment has no native pause to release')


def seed_environment(*args): return SeedEnvironment()


class MemorySampler:
    def __init__(self, config, **kwargs):
        self.config = config
        self.states = {}
        self.outstanding = {}
        self.restarts = {i: 0 for i in range(config.runtime.num_envs)}
        self.episodes = dict(self.restarts)
        self.steps = dict(self.restarts)
        self.seeds = {}
        self.messages = deque()
        self.commands = []
        self.paused = False
        self.fail_seed = None
        self.failed = False
        self.bad_transition = False

    def start(self, restored):
        for i in self.episodes: self.reset(i)

    def facts(self, i, done=False):
        return dict(action_mask=[True]+[False]*(ACTION_SPACE_SIZE-1),
                    result='WON' if done else 'PLAYING', events={},
                    action_result={'success': True}, ticks_advanced=100,
                    episode_ticks=self.steps[i]*100)

    def reset(self, i, seed=None):
        assert i not in self.outstanding, 'reset before draining prior result'
        self.seeds[i] = seed
        self.steps[i] = 0
        self.episodes[i] += 1
        self.outstanding[i] = True
        self.commands.append(('reset', i, seed))
        self.messages.append(('ready', i, (SampleIdentity(i,self.episodes[i],0),
            empty_observation(), self.facts(i), None)))

    def receive(self, timeout=0):
        if not self.messages: return None
        msg = self.messages.popleft()
        self.outstanding.pop(msg[1], None)
        return msg

    def dispatch(self, reply):
        assert not self.paused
        i = reply.identity.environment_id
        self.outstanding[i] = True
        self.steps[i] += 1
        self.commands.append(('step', i, self.seeds[i]))
        if self.seeds[i] == self.fail_seed and self.fail_seed is not None and not self.failed:
            self.failed = True
            self.messages.append(('error', i, 'intentional evaluation failure'))
            return
        done = self.steps[i] >= i+2
        identity = reply.identity
        if self.bad_transition and self.seeds[i] is not None:
            identity = replace(identity, episode_id=identity.episode_id+10)
        self.messages.append(('transition', i, (identity,reply.action,reply.policy_version,
            empty_observation(),1.,done,False,self.facts(i,done),None)))

    def restart(self, i):
        if self.restarts[i] >= self.config.runtime.restart_limit: raise RuntimeError('restart limit')
        self.restarts[i] += 1
        self.reset(i)

    def pause(self, *, on_message, stop_event):
        self.paused = True
        while self.messages: on_message(self.receive())
        return []

    def resume(self, **kwargs):
        self.paused = False
        return []

    def close(self): pass


class ParallelEvaluationTest(unittest.TestCase):
    def run_case(self, directory, *, count=8, architecture='legacy', callback=None,
                 sampler_setup=None, **controller_kwargs):
        config=DoubleDQNConfig(env=EnvConfig(ticks_per_step=100), total_steps=42,
            device='cpu', hidden_sizes=(8,4), replay_capacity=64, batch_size=2,
            learning_starts=2, n_step=2, train_every=2, gradient_steps=1,
            reward_scheme='sparse', exploration_mode='epsilon_random',
            training_architecture=architecture, autosave_every=0, checkpoint_every=0,
            runtime=TrainingRuntimeConfig(num_envs=4, asynchronous=True,
                inference_batch_size=4, max_policy_lag=64, ui_update_interval_ms=0))
        events=[]
        def event(name,data):
            events.append((name,data))
            if callback: callback(c,name,data)
        c=DoubleDQNTrainerController(config,run_dir=directory,on_event=event,
            **(controller_kwargs or dict(eval_every_episodes=1,eval_seed_start=100,eval_seed_count=count)))
        sampler=MemorySampler(config)
        if sampler_setup: sampler_setup(sampler)
        summary=run_parallel(c,sampler_factory=lambda *a,**k:sampler)
        return c,summary,events,sampler

    def test_all_workers_finish_training_then_share_seeds_without_replay_pollution(self):
        with tempfile.TemporaryDirectory() as directory:
            c,summary,events,sampler=self.run_case(directory)
            rounds=[d for n,d in events if n=='evaluation_end']
            self.assertTrue(rounds)
            self.assertTrue(all(d['complete'] and d['episodes']==8 for d in rounds))
            self.assertEqual(summary.env_steps,42)
            self.assertEqual(len(c._replay),42)
            self.assertFalse(any(n=='episode_aborted' for n,d in events))
            first_start=next(i for i,(n,d) in enumerate(events) if n=='evaluation_start')
            self.assertEqual({d['environment_id'] for n,d in events[:first_start] if n=='episode_end'},set(range(4)))
            first_end=next(i for i,(n,d) in enumerate(events) if n=='evaluation_end')
            games=[d for n,d in events[first_start:first_end] if n=='evaluation_episode']
            self.assertEqual(sorted(d['seed'] for d in games),list(range(100,108)))
            self.assertEqual({d['environment_id'] for d in games},set(range(4)))
            self.assertEqual(len({d['policy_version'] for d in games}),1)
            self.assertTrue(all(d['wall_time_ms']>=0 for d in games))

    def test_one_seed_uses_one_worker_and_recovers_same_seed(self):
        with tempfile.TemporaryDirectory() as directory:
            c,summary,events,sampler=self.run_case(directory,count=1,
                sampler_setup=lambda s:setattr(s,'fail_seed',100))
            retries=[d for n,d in events if n=='evaluation_retry']
            self.assertEqual(len(retries),1)
            self.assertEqual(retries[0]['seed'],100)
            self.assertTrue(all(d['complete'] for n,d in events if n=='evaluation_end'))
            self.assertEqual(len(c._replay),summary.env_steps)

    def test_pause_during_evaluation_drains_into_evaluation_and_continue(self):
        resumed=[]
        def callback(c,name,data):
            if name=='evaluation_episode' and not resumed:
                resumed.append(True)
                c.pause()
                def resume():
                    deadline=time.monotonic()+5
                    while c.state!='PAUSED' and time.monotonic()<deadline: time.sleep(.005)
                    c.resume()
                threading.Thread(target=resume,daemon=True).start()
        with tempfile.TemporaryDirectory() as directory:
            c,summary,events,sampler=self.run_case(directory,callback=callback)
            self.assertTrue(any(n=='state' and d.get('state')=='PAUSED' for n,d in events))
            self.assertEqual(c.state,'FINISHED')
            self.assertEqual(len(c._replay),42)
            self.assertTrue(all(d['complete'] for n,d in events if n=='evaluation_end'))

    def test_stop_during_evaluation_does_not_save_partial_best(self):
        def callback(c,name,data):
            if name=='evaluation_episode': c.stop()
        with tempfile.TemporaryDirectory() as directory:
            c,summary,events,sampler=self.run_case(directory,callback=callback)
            self.assertEqual(c.state,'STOPPED')
            self.assertFalse(Path(directory,'best.pt').exists())
            self.assertEqual(len(c._replay),summary.env_steps)
            self.assertFalse(next(d for n,d in events if n=='evaluation_end')['complete'])

    def test_actor_learner_best_is_the_evaluated_version(self):
        versions=[]
        def callback(c,name,data):
            if name=='evaluation_start':
                versions.append((data['policy_version'],
                    {k:v.detach().clone() for k,v in c._agent.online.state_dict().items()}))
            if name=='best_checkpoint':
                model=torch.load(data['path'],weights_only=False)
                self.assertEqual(model['update_count'],versions[-1][0])
                for k,v in versions[-1][1].items(): torch.testing.assert_close(model['online'][k],v)
        with tempfile.TemporaryDirectory() as directory:
            c,summary,events,sampler=self.run_case(directory,architecture='actor_learner',callback=callback)
            self.assertTrue(versions)
            self.assertTrue(Path(directory,'best.pt').exists())

    def test_tiered_protocol_is_captured_before_peer_episodes_finish(self):
        with tempfile.TemporaryDirectory() as directory:
            c,summary,events,sampler=self.run_case(directory,fast_eval_every_episodes=1,
                fast_eval_seed_count=1,full_eval_every_episodes=10,full_eval_seed_count=8)
            rounds=[d for n,d in events if n=='evaluation_end']
            self.assertTrue(rounds)
            self.assertEqual(rounds[0]['kind'],'fast')
            self.assertEqual(rounds[0]['requested_episodes'],1)

    def test_full_evaluation_takes_priority_while_workers_finish(self):
        with tempfile.TemporaryDirectory() as directory:
            c,summary,events,sampler=self.run_case(directory,fast_eval_every_episodes=1,
                fast_eval_seed_count=1,full_eval_every_episodes=4,full_eval_seed_count=8)
            rounds=[d for n,d in events if n=='evaluation_end']
            self.assertEqual(rounds[0]['kind'],'full')
            self.assertEqual(rounds[0]['requested_episodes'],8)

    def test_stale_transition_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(RuntimeError,'mismatched evaluation'):
                self.run_case(directory,sampler_setup=lambda s:setattr(s,'bad_transition',True))

    def test_worker_recycling_preserves_explicit_reset_seed(self):
        config=DoubleDQNConfig(runtime=TrainingRuntimeConfig(num_envs=1,
            max_episodes_per_worker=1, worker_timeout_s=30))
        sampler=ProcessSampler(config,factory=seed_environment,factory_args=[()])
        def receive_ready():
            deadline=time.monotonic()+30
            while time.monotonic()<deadline:
                message=sampler.receive(.1)
                if message is not None:
                    self.assertEqual(message[0],'ready')
                    return message
            self.fail('no reset acknowledgement')
        try:
            sampler.start()
            receive_ready()
            for seed in (100,101):
                sampler.reset(0,seed=seed)
                message=receive_ready()
                self.assertEqual(message[2][2]['seed'],seed)
            sampler.dispatch(PolicyReply(message[2][0],0,0))
            deadline=time.monotonic()+30
            transition=None
            while transition is None and time.monotonic()<deadline:
                transition=sampler.receive(.1)
            self.assertIsNotNone(transition)
            self.assertEqual(transition[0],'transition')
            ack=sampler.pause()
            self.assertTrue(ack[0]['terminal_idle'])
            self.assertTrue(ack[0]['paused'])
            ack=sampler.resume()
            self.assertFalse(ack[0]['paused'])
        finally:
            sampler.close()
