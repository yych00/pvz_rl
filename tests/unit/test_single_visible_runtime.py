"""Single-worker visualization and controls with a spawnable offline fixture."""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from pvz_gym.action_space import ACTION_SPACE_SIZE
from pvz_gym.observation import empty_observation
from train.common import EnvConfig
from train.dqn.base_controller import DoubleDQNTrainerController
from train.dqn.config import DoubleDQNConfig
from train.dqn.parallel_controller import run_parallel
from train.runtime import TrainingRuntimeConfig
from train.sampler import ProcessSampler


class VisibleFixture:
    def __init__(self, process_id, discard_step=0):
        self.runtime_process_id = process_id
        self.runtime_instance_id = "offline-slot-0"
        self.simulation_mode = "continuous"
        self.steps = 0
        self.discard_step = discard_step
        self.max_episode_ticks = None
        self.session = SimpleNamespace(backend=SimpleNamespace())
    def facts(self, done=False):
        state = empty_observation()
        state['global'][0] = self.steps
        return state, dict(action_mask=(True,) + (False,) * (ACTION_SPACE_SIZE - 1),
            ticks_advanced=150, ticks_requested=150, game_clock=self.steps * 150,
            episode_ticks=self.steps * 150, result="WON" if done else "PLAYING",
            events={}, action_result={"success": True},
            replay_eligible=not (self.discard_step and self.steps == self.discard_step),
            observation_quality="failed" if self.discard_step and self.steps == self.discard_step else "fresh")
    def reset(self, *, seed):
        self.steps = 0
        return self.facts()
    def step(self, action):
        self.steps += 1
        state, info = self.facts(self.steps == 4)
        return state, 1.0, self.steps == 4, False, info
    def pause(self):
        pass
    def resume(self):
        pass
    def close(self):
        pass


def visible_fixture(process_id=123, discard_step=0):
    return VisibleFixture(process_id, discard_step)


class SingleVisibleRuntimeTest(unittest.TestCase):
    def config(self, runtime):
        return DoubleDQNConfig(env=EnvConfig(simulation_mode="continuous", ticks_per_step=150),
            runtime=runtime, device="cpu", hidden_sizes=(16, 8),
            reward_scheme="sparse", total_steps=8, replay_capacity=16, batch_size=2,
            learning_starts=2, n_step=1, train_every=2, gradient_steps=1,
            epsilon_start=1, epsilon_end=1, exploration_mode="epsilon_random",
            autosave_every=0, checkpoint_every=0)

    def test_executable_and_pid_routes_both_sample_learn_and_publish(self):
        for binding in ({"game_executable": "offline-fixture.exe"}, {"process_ids": (123,)}):
            with self.subTest(binding=binding), tempfile.TemporaryDirectory() as directory:
                runtime = TrainingRuntimeConfig(num_envs=1, visualize_q=True,
                    ui_update_interval_ms=0, asynchronous_checkpoints=False, **binding)
                events = []
                controller = DoubleDQNTrainerController(self.config(runtime), run_dir=directory,
                    eval_every_episodes=0, on_event=lambda name, data: events.append((name, data)))
                factory = lambda config, **kwargs: ProcessSampler(config, factory=visible_fixture,
                    factory_args=[(123,)])
                summary = run_parallel(controller, sampler_factory=factory)
                frames = [data for name, data in events if name == 'visualization']
                self.assertEqual(summary.env_steps, 8)
                self.assertEqual(len(controller._replay), 8)
                self.assertGreater(controller._agent.update_count, 0)
                self.assertTrue(frames)
                self.assertTrue(all(frame['process_id'] == 123 for frame in frames))
                self.assertTrue(all(frame['instance_id'] == 'offline-slot-0' for frame in frames))
                self.assertTrue(all('selected_action' in frame and 'policy_version' in frame for frame in frames))
                self.assertTrue(Path(directory, 'final.pt').is_file())

    def test_pause_reset_continue_has_ack_and_preserves_learner(self):
        with tempfile.TemporaryDirectory() as directory:
            runtime = TrainingRuntimeConfig(num_envs=1, process_ids=(123,), ui_update_interval_ms=0,
                                            asynchronous_checkpoints=False)
            events, kept = [], []
            requested = False
            def on_event(name, data):
                nonlocal requested
                events.append((name, data))
                if name == 'step' and not requested:
                    requested = True
                    kept.extend([controller._agent, controller._replay])
                    controller.pause()
                if name == 'state' and data['state'] == 'PAUSED':
                    self.assertTrue(data['acknowledged'])
                    self.assertTrue(data['control_ack'])
                    if data.get('operation') == 'reset':
                        self.assertIs(controller._agent, kept[0])
                        self.assertIs(controller._replay, kept[1])
                        controller.resume()
                    else:
                        controller.reset_game()
            controller = DoubleDQNTrainerController(self.config(runtime), run_dir=directory,
                                                    eval_every_episodes=0, on_event=on_event)
            summary = run_parallel(controller, sampler_factory=lambda config, **kwargs:
                ProcessSampler(config, factory=visible_fixture, factory_args=[(123,)]))
            self.assertEqual(summary.env_steps, 8)
            self.assertEqual(controller.state, 'FINISHED')
            self.assertEqual(len([1 for name, _ in events if name == 'episode_aborted']), 1)
            states = [data['state'] for name, data in events if name == 'state']
            self.assertEqual(states, ['RUNNING', 'PAUSING', 'PAUSED', 'RESETTING', 'PAUSED', 'RESUMING', 'RUNNING'])

    def test_bad_observation_breaks_replay_chain_and_does_not_bank_updates(self):
        with tempfile.TemporaryDirectory() as directory:
            runtime = TrainingRuntimeConfig(num_envs=1, process_ids=(123,), ui_update_interval_ms=0,
                                            asynchronous_checkpoints=False)
            config = replace(self.config(runtime), total_steps=4, train_every=1)
            events = []
            controller = DoubleDQNTrainerController(config, run_dir=directory, eval_every_episodes=0,
                on_event=lambda name, data: events.append((name, data)))
            run_parallel(controller, sampler_factory=lambda config, **kwargs:
                ProcessSampler(config, factory=visible_fixture, factory_args=[(123, 2)]))
            self.assertEqual(len(controller._replay), 2)
            self.assertEqual(controller._agent.update_count, 1)
            self.assertEqual(len([1 for name, _ in events if name == 'transition_discarded']), 2)
            self.assertEqual(controller.summary.episode_lengths, [4])
            self.assertEqual(int(controller._replay._next_ids[0]), -1)

    def test_loaded_model_visualization_reuses_greedy_forward_and_keeps_scores(self):
        from train.dqn.agent import DoubleDQNAgent
        from train.dqn.evaluate import evaluate_checkpoint
        runtime = TrainingRuntimeConfig(num_envs=1, process_ids=(123,), visualize_q=True)
        config = self.config(runtime)
        agent = DoubleDQNAgent(ACTION_SPACE_SIZE, config)
        events = []
        with patch('train.dqn.evaluate.DoubleDQNAgent.from_checkpoint', return_value=(agent, config, {})), \
                patch('train.sampler.native_environment', return_value=(VisibleFixture(123), None)), \
                patch.object(agent.online, 'forward', wraps=agent.online.forward) as forward:
            visible = evaluate_checkpoint('offline.pt', seed_count=1,
                visualize_q=True, on_event=lambda name, data: events.append((name, data)))
        self.assertEqual(forward.call_count, visible.total_steps)
        frames = [data for name, data in events if name == 'visualization']
        self.assertTrue(frames)
        self.assertTrue(all(frame['process_id'] == 123 and frame['episode_id'] == 1 for frame in frames))
        self.assertTrue(all(frame['selected_action'] == 0 for frame in frames))
        with patch('train.dqn.evaluate.DoubleDQNAgent.from_checkpoint', return_value=(agent, config, {})), \
                patch('train.sampler.native_environment', return_value=(VisibleFixture(123), None)):
            baseline = evaluate_checkpoint('offline.pt', seed_count=1, visualize_q=False)
        self.assertEqual((visible.wins, visible.total_score, visible.total_steps, visible.ticks_advanced_sum),
                         (baseline.wins, baseline.total_score, baseline.total_steps, baseline.ticks_advanced_sum))


if __name__ == '__main__':
    unittest.main()
