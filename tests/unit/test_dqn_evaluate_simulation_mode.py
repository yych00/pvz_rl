import threading
import unittest
from dataclasses import replace
from unittest.mock import Mock, patch

from pvz_gym.env_file import resolve_game_paths
from train.common import EnvConfig
from train.dqn.config import DoubleDQNConfig
from train.dqn.evaluate import evaluate_checkpoint


class EvaluateSimulationModeTest(unittest.TestCase):
    def test_each_evaluation_decision_clears_menus_with_selected_clock_mode(self):
        from tests.unit.test_dqn_evaluation_cadence import SingleStepEnv, RecordingAgent
        from types import SimpleNamespace
        from train.runtime import TrainingRuntimeConfig
        for owned in (False, True):
            for mode in ("continuous", "stepped", "headless", "turbo"):
                with self.subTest(owned=owned, mode=mode):
                    config = DoubleDQNConfig(reward_scheme="balanced", runtime=TrainingRuntimeConfig(
                        game_executable="saved.exe" if owned else None))
                    client = Mock()
                    env = SingleStepEnv()
                    env.session = SimpleNamespace(backend=client)
                    events = []
                    client.dismiss_battle_dialogs.side_effect = lambda **_: events.append("dismiss")
                    original_step = env.step
                    def step(action):
                        events.append("step")
                        return original_step(action)
                    env.step = step
                    with (
                        patch("train.dqn.evaluate.DoubleDQNAgent.from_checkpoint",
                              return_value=(RecordingAgent(), config, {})),
                        patch("train.dqn.evaluate.PvZClient", return_value=client),
                        patch("train.dqn.evaluate.PvZEnv", return_value=env),
                        patch("train.sampler.native_environment", return_value=(env, Mock())),
                    ):
                        summary = evaluate_checkpoint("saved.pt", seed_count=2, simulation_mode=mode)
                    self.assertEqual(summary.wins, 2)
                    self.assertEqual(events, ["dismiss", "step", "dismiss", "step"])
                    self.assertEqual(client.dismiss_battle_dialogs.call_count, 2)
                    client.dismiss_battle_dialogs.assert_called_with(keep_paused=mode != "continuous")

    def test_saved_deck_is_inherited_and_explicit_deck_overrides_it(self):
        from types import SimpleNamespace
        from train.runtime import TrainingRuntimeConfig
        stopped = threading.Event(); stopped.set()
        for owned in (False, True):
            config = DoubleDQNConfig(runtime=TrainingRuntimeConfig(
                game_executable='saved.exe' if owned else None))
            env = Mock(); env.session = SimpleNamespace(backend=Mock())
            with (
                patch('train.dqn.evaluate.DoubleDQNAgent.from_checkpoint',
                      return_value=(object(), config, {'seed_types': (0, 1, 2)})),
                patch('train.dqn.evaluate.PvZClient'),
                patch('train.dqn.evaluate.PvZEnv', return_value=env) as constructor,
                patch('train.sampler.native_environment', return_value=(env, Mock())) as factory,
            ):
                for explicit, expected in ((None, (0, 1, 2)), ((3, 4), (3, 4))):
                    with self.subTest(owned=owned, explicit=explicit):
                        evaluate_checkpoint('saved.pt', seed_count=1, stop_event=stopped,
                                            seed_types=explicit)
                        actual = factory.call_args.args[4] if owned else constructor.call_args.kwargs['seed_types']
                        self.assertEqual(actual, expected)

    def test_staged_evaluation_settles_rewards_and_reports_breakdown(self):
        from pvz_gym.action_space import ACTION_SPACE_SIZE
        from pvz_gym.observation import empty_observation
        from tests.unit.test_dqn_staged_reward import raw, info
        from train.dqn.credit import EpisodeEventCredit
        from train.dqn.replay import ReplayBuffer
        config = DoubleDQNConfig(reward_scheme='staged')
        mask = [True] + [False] * (ACTION_SPACE_SIZE - 1)
        initial = dict(info(raw(), 0), action_mask=mask)
        steps = [(100, 1000, False, 'PLAYING', 1),
                 (100, 100, True, 'WON', 0)]
        for terminal in (True, False):
            with self.subTest(terminal=terminal):
                events = []
                env = Mock(); env.reset.return_value = (empty_observation(), initial)
                facts = [dict(info(raw(sun), ticks, result=result if terminal or not done else 'PLAYING', mowers=mowers), action_mask=mask)
                         for sun, ticks, done, result, mowers in steps]
                env.step.side_effect = [(empty_observation(), 100. if done and terminal else 0.,
                                         done and terminal, done and not terminal, values)
                                        for (_, _, done, _, _), values in zip(steps, facts)]
                agent = Mock(); agent.select_action.return_value = 0
                with patch('train.dqn.evaluate.DoubleDQNAgent.from_checkpoint', return_value=(agent, config, {})), \
                     patch('train.dqn.evaluate.PvZClient'), \
                     patch('train.dqn.evaluate.PvZEnv', return_value=env) as constructor:
                    summary = evaluate_checkpoint('saved.pt', seed_count=1,
                                                  on_event=lambda name, data: events.append((name, data)))
                replay = ReplayBuffer(4, ACTION_SPACE_SIZE)
                credit = EpisodeEventCredit(replay, config, initial)
                expected = sum(credit.add(replay, empty_observation(), 0, 0., empty_observation(),
                                         done and terminal, done and not terminal, mask, dict(values))
                               for (_, _, done, _, _), values in zip(steps, facts))
                self.assertAlmostEqual(summary.total_training_score, expected)
                self.assertAlmostEqual(expected, .9 if terminal else -.1)
                episode = next(data for name, data in events if name == 'episode_end')
                self.assertAlmostEqual(episode['training_score'], expected)
                self.assertEqual(episode['reward_breakdown']['terminal_reward'], 1. if terminal else 0.)
                self.assertEqual(constructor.call_args.kwargs['reward_stage_ticks'], 1000)
                env.close.assert_called_once()

    def test_evaluation_publishes_without_instantiating_desktop_renderer(self):
        try:
            from .test_dqn_evaluation_cadence import SingleStepEnv, RecordingAgent
        except ImportError:
            from test_dqn_evaluation_cadence import SingleStepEnv, RecordingAgent
        config = DoubleDQNConfig(reward_scheme="balanced",
                                 env=EnvConfig(simulation_mode="continuous"))
        env = SingleStepEnv()
        env.close = Mock()
        overlay = Mock()
        overlay.is_enabled.side_effect = RuntimeError("overlay unavailable")
        with (
            patch("train.dqn.evaluate.DoubleDQNAgent.from_checkpoint", return_value=(RecordingAgent(), config, {})),
            patch("train.dqn.evaluate.PvZClient"),
            patch("train.dqn.evaluate.PvZEnv", return_value=env),
            patch("workbench.gui.q_heatmap_overlay.get_q_heatmap_overlay", return_value=overlay) as lookup,
        ):
            result = evaluate_checkpoint("saved.pt", seed_count=1)
        self.assertEqual(result.wins, 1)
        env.close.assert_called_once()
        lookup.assert_not_called()

    def test_environment_construction_failure_closes_client(self):
        config = DoubleDQNConfig()
        with (
            patch("train.dqn.evaluate.DoubleDQNAgent.from_checkpoint", return_value=(object(), config, {})),
            patch("train.dqn.evaluate.PvZClient") as client,
            patch("train.dqn.evaluate.PvZEnv", side_effect=ValueError("invalid level")),
        ):
            with self.assertRaisesRegex(ValueError, "invalid level"):
                evaluate_checkpoint("saved.pt", level_id=51, seed_count=1)
            client.return_value.close.assert_called_once()

    def test_saved_multi_instance_config_evaluates_one_owned_instance(self):
        from types import SimpleNamespace
        from unittest.mock import Mock
        from train.runtime import TrainingRuntimeConfig
        config = DoubleDQNConfig(runtime=TrainingRuntimeConfig(num_envs=4,game_executable='saved.exe'))
        stopped=threading.Event(); stopped.set()
        env=Mock(); env.session=SimpleNamespace(backend=Mock())
        instance=Mock()
        with (
            patch('train.dqn.evaluate.DoubleDQNAgent.from_checkpoint',return_value=(object(),config,{})),
            patch('train.sampler.native_environment',return_value=(env,instance)) as factory,
        ):
            evaluate_checkpoint('saved.pt',seed_count=1,stop_event=stopped,simulation_mode='turbo',max_episode_steps=3)
        used, runtime, environment_id, _, _ = factory.call_args.args
        self.assertEqual(runtime.num_envs,1)
        self.assertEqual(runtime.game_executable,'saved.exe')
        self.assertEqual(runtime.instance_root, str(resolve_game_paths('saved.exe').instance_root.parent / 'evaluation'))
        self.assertEqual(used.env.simulation_mode,'turbo')
        self.assertEqual(factory.call_args.kwargs['max_episode_steps'],3)
        env.close.assert_called_once(); instance.close.assert_called_once()

        # Evaluation shares the runtime directory; changing checkpoints does not create
        # another game copy inside each run directory.
        custom_config = replace(config, runtime=replace(config.runtime, instance_root='game/runtime/custom'))
        with (
            patch('train.dqn.evaluate.DoubleDQNAgent.from_checkpoint', return_value=(object(),custom_config,{})),
            patch('train.sampler.native_environment', return_value=(env,instance)) as factory,
        ):
            for checkpoint in ('runs/first/saved.pt', 'runs/second/saved.pt'):
                evaluate_checkpoint(checkpoint, seed_count=1, stop_event=stopped)
        roots = {call.args[1].instance_root for call in factory.call_args_list}
        self.assertEqual(roots, {str(resolve_game_paths(instance_root='game/runtime/custom').instance_root.parent / 'evaluation')})

    def test_override_uses_selected_mode_and_default_uses_checkpoint_mode(self):
        config = DoubleDQNConfig(reward_scheme="staged", env=EnvConfig(simulation_mode="headless"))
        stopped = threading.Event()
        stopped.set()

        with (
            patch("train.dqn.evaluate.DoubleDQNAgent.from_checkpoint", return_value=(object(), config, {})),
            patch("train.dqn.evaluate.PvZClient"),
            patch("train.dqn.evaluate.PvZEnv") as environment,
        ):
            evaluate_checkpoint("saved.pt", seed_count=1, stop_event=stopped,
                                simulation_mode="continuous", playthrough=2,
                                ticks_per_step=30)
            self.assertEqual(environment.call_args.kwargs["simulation_mode"], "continuous")
            self.assertEqual(environment.call_args.kwargs["playthrough"], 2)
            self.assertEqual(environment.call_args.kwargs["ticks_per_step"], 30)

            evaluate_checkpoint("saved.pt", seed_count=1, stop_event=stopped)
            self.assertEqual(environment.call_args.kwargs["simulation_mode"], "headless")
            self.assertEqual(environment.call_args.kwargs["playthrough"], config.env.playthrough)
            self.assertEqual(environment.call_args.kwargs["ticks_per_step"], config.env.ticks_per_step)


if __name__ == "__main__":
    unittest.main()
