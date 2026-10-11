"""Episode budgets use authoritative clock deltas, independent of step size."""

import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from pvz_gym.env import PvZEnv
from pvz_gym.types import GameStatus, StepEvents
from tests.fake_backend import FactClient
from train.common import EnvConfig


class ClockClient(FactClient):
    def __init__(self, ticks=(20,), initial_clock=0, terminal_after=None,
                 terminal_status=GameStatus.WON):
        super().__init__()
        self.actual_ticks = ticks
        self.initial_clock = initial_clock
        self.terminal_after = terminal_after
        self.terminal_status = terminal_status
        self.headless_active = False

    def _observation(self):
        observation = super()._observation()
        return replace(
            observation,
            global_state={
                **observation.global_state, 'game_mode': 0, 'spawn_plan_available': True,
                'zombie_spawn_plan': ([0] + [-1] * 49) * observation.global_state['total_waves'],
            },
            cards=tuple({**card, 'source': 1} for card in observation.cards),
            zombie_health=tuple({**zombie, 'body_hp': zombie['hp'], 'from_wave': 0,
                                 'is_dead_or_dying': False, 'phase': 0}
                                for zombie in observation.zombies),
        )

    def reset(self, config):
        facts = super().reset(config)
        self.clock = self.initial_clock
        return replace(facts, observation=self._observation(), game_clock=self.clock)

    def step(self, action, requested_ticks):
        actual = self.actual_ticks[min(self.advance_calls, len(self.actual_ticks) - 1)]
        facts = super().step(action, actual)
        self.status = (self.terminal_status if self.advance_calls == self.terminal_after
                       else GameStatus.PLAYING)
        return replace(
            facts, observation=self._observation(), status=self.status,
            events=StepEvents(won=self.status == GameStatus.WON,
                              lost=self.status == GameStatus.LOST),
            action_mask=self.get_action_mask(),
        )

    def step_continuous(self, action, requested_ticks):
        return self.step(action, requested_ticks)

    def step_headless(self, action, requested_ticks):
        if not self.headless_active:
            raise RuntimeError("headless is inactive")
        return self.step(action, requested_ticks)

    def begin_headless_mode(self):
        self.headless_active = True
        return True

    def end_headless_mode(self):
        self.headless_active = False
        return True


class EpisodeTickLimitTest(unittest.TestCase):
    def test_actual_deltas_accumulate_to_limit_in_all_modes(self):
        for mode in ("stepped", "headless", "continuous"):
            with self.subTest(mode=mode):
                client = ClockClient(ticks=(20, 39, 1))
                env = PvZEnv(client, ticks_per_step=20, max_episode_ticks=60,
                             simulation_mode=mode)
                try:
                    _, info = env.reset()
                    self.assertEqual(info["episode_ticks"], 0)
                    for elapsed in (20, 59, 60):
                        _, reward, terminated, truncated, info = env.step(0)
                        self.assertEqual(info["episode_ticks"], elapsed)
                        self.assertEqual(info["max_episode_ticks"], 60)
                        self.assertFalse(terminated)
                        self.assertEqual(truncated, elapsed == 60)
                        self.assertEqual(reward, 0.0)
                    self.assertEqual(info["result"], "PLAYING")
                    self.assertEqual(info["truncation_reason"], "max_episode_ticks")
                    self.assertEqual(client.call_order, ["reset", "step", "step", "step"])
                    if mode == "headless":
                        self.assertFalse(client.headless_active)
                finally:
                    env.close()

    def test_requested_ticks_do_not_spend_unadvanced_budget(self):
        env = PvZEnv(ClockClient(ticks=(10,)), ticks_per_step=100, max_episode_ticks=25)
        env.reset()
        try:
            for elapsed in (10, 20, 30):
                _, _, _, truncated, info = env.step(0)
                self.assertEqual(info["ticks_requested"], 100)
                self.assertEqual(info["episode_ticks"], elapsed)
                self.assertEqual(truncated, elapsed == 30)
        finally:
            env.close()

    def test_continuous_clock_jump_spends_actual_game_time(self):
        env = PvZEnv(ClockClient(ticks=(80,)), ticks_per_step=30,
                     max_episode_ticks=60, simulation_mode="continuous")
        env.reset()
        try:
            _, _, terminated, truncated, info = env.step(0)
            self.assertFalse(terminated)
            self.assertTrue(truncated)
            self.assertEqual(info["episode_ticks"], 80)
        finally:
            env.close()

    def test_reset_excludes_prebattle_clock_and_starts_fresh_budget(self):
        env = PvZEnv(ClockClient(initial_clock=50_000), max_episode_ticks=30)
        try:
            for _ in range(2):
                _, info = env.reset()
                self.assertEqual(info["game_clock"], 50_000)
                self.assertEqual(info["window_start_game_clock"], 50_000)
                self.assertEqual(info["episode_ticks"], 0)
                _, _, _, truncated, info = env.step(0)
                self.assertFalse(truncated)
                self.assertEqual(info["episode_ticks"], 20)
        finally:
            env.close()

    def test_clock_reset_rebases_without_erasing_observed_window_ticks(self):
        env = PvZEnv(ClockClient(ticks=(20, -50_000, 20), initial_clock=50_000),
                     max_episode_ticks=35)
        try:
            env.reset()
            _, _, _, truncated, info = env.step(0)
            self.assertFalse(truncated)
            self.assertEqual(info["episode_ticks"], 20)
            self.assertFalse(info["tick_clock_rebased"])
            _, _, _, truncated, info = env.step(0)
            self.assertFalse(truncated)
            self.assertEqual(info["episode_ticks"], 20)
            self.assertTrue(info["tick_clock_rebased"])
            _, _, _, truncated, info = env.step(0)
            self.assertTrue(truncated)
            self.assertEqual(info["episode_ticks"], 40)
        finally:
            env.close()

    def test_native_win_or_loss_takes_precedence_at_tick_boundary(self):
        for mode in ("stepped", "headless", "continuous"):
            for status in (GameStatus.WON, GameStatus.LOST):
                with self.subTest(mode=mode, status=status):
                    env = PvZEnv(ClockClient(terminal_after=1, terminal_status=status),
                                 max_episode_ticks=20, simulation_mode=mode)
                    try:
                        env.reset()
                        _, _, terminated, truncated, info = env.step(0)
                        self.assertTrue(terminated)
                        self.assertFalse(truncated)
                        self.assertIsNone(info["truncation_reason"])
                        self.assertEqual(info["result"], status.name)
                    finally:
                        env.close()

    def test_none_disables_tick_budget_and_legacy_step_limit_is_optional(self):
        for step_limit in (None, 1):
            env = PvZEnv(ClockClient(ticks=(400_000,)), max_episode_ticks=None,
                         max_episode_steps=step_limit)
            try:
                env.reset()
                _, _, terminated, truncated, info = env.step(0)
                self.assertFalse(terminated)
                self.assertEqual(truncated, step_limit == 1)
                self.assertEqual(info["truncation_reason"],
                                 "max_episode_steps" if step_limit else None)
            finally:
                env.close()

    def test_environment_and_training_share_default_and_validate_tick_limits(self):
        env = PvZEnv(ClockClient())
        try:
            self.assertEqual(env.max_episode_ticks, 120_000)
            self.assertEqual(EnvConfig().max_episode_ticks, 120_000)
            self.assertIsNone(EnvConfig(max_episode_ticks=None).max_episode_ticks)
        finally:
            env.close()
        for invalid in (0, -1, 2.5, True):
            for factory in (PvZEnv, EnvConfig):
                with self.subTest(factory=factory, invalid=invalid):
                    with self.assertRaisesRegex(ValueError, "max_episode_ticks"):
                        factory(max_episode_ticks=invalid)

    def test_training_cli_tick_limit_override_and_disable(self):
        from train.dqn import train as cli

        for value, expected in ((1234, 1234), (0, None)):
            with self.subTest(value=value):
                with patch("sys.argv", ["train", "--max-episode-ticks", str(value)]), \
                        patch.object(cli, "train") as run:
                    cli.main()
                self.assertEqual(run.call_args.args[0].env.max_episode_ticks, expected)

    def test_tick_truncation_keeps_td_bootstrap(self):
        from pvz_gym.action_space import ACTION_SPACE_SIZE
        from pvz_gym.observation import empty_observation
        from train.dqn.config import DoubleDQNConfig
        from train.dqn.credit import EpisodeEventCredit
        from train.dqn.replay import ReplayBuffer

        for scheme in ('sparse', 'balanced', 'dense_early', 'staged'):
            with self.subTest(scheme=scheme):
                config = DoubleDQNConfig(reward_scheme=scheme)
                replay = ReplayBuffer(2, ACTION_SPACE_SIZE)
                raw_observation = ClockClient()._observation()
                credit = EpisodeEventCredit(replay, config, {'raw_observation': raw_observation})
                observation = empty_observation()
                credit.add(replay, observation, 0, 0.0, observation, False, True,
                           (True,) * ACTION_SPACE_SIZE,
                           {'raw_observation': raw_observation, 'ticks_advanced': 20,
                            'result': 'PLAYING', 'truncation_reason': 'max_episode_ticks'})
                batch = replay.sample(1)
                self.assertFalse(batch.dones[0])
                self.assertTrue(replay._episode_ends[0])
                self.assertEqual(batch.ticks_advanced[0], 20)

    def test_dqn_resets_tick_limited_episode_and_records_budget(self):
        from train.dqn import agent as agent_module
        if agent_module.torch is None:
            self.skipTest("PyTorch is required")
        from train.dqn.config import DoubleDQNConfig
        from train.dqn.optimized_controller import DoubleDQNTrainerController
        from train.dqn.replay import ReplayBuffer

        config = DoubleDQNConfig(
            training_architecture="legacy",
            reward_scheme="staged",
            env=EnvConfig(ticks_per_step=20, max_episode_ticks=40),
            total_steps=3, replay_capacity=8, batch_size=2, learning_starts=100,
            checkpoint_every=0, autosave_every=0, device="cpu",
        )
        client = ClockClient()
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            controller = DoubleDQNTrainerController(
                config, run_dir=run_dir, fast_eval_every_episodes=0,
                full_eval_every_episodes=0,
            )
            with patch("train.dqn.base_controller.PvZClient", return_value=client):
                summary = controller.run()
            self.assertEqual((summary.episodes, summary.wins, summary.losses), (1, 0, 0))
            self.assertEqual(client.call_order.count("reset"), 2)
            records = [json.loads(line) for line in
                       (run_dir / "episodes.jsonl").read_text().splitlines()]
            self.assertEqual(records[0]["episode_ticks"], 40)
            self.assertEqual(records[0]["max_episode_ticks"], 40)
            self.assertTrue(records[0]["truncated"])
            self.assertEqual(records[0]["truncation_reason"], "max_episode_ticks")
            self.assertEqual(records[0]["reward_breakdown"]["reward_stages"], 1)
            self.assertEqual(records[0]["reward_breakdown"]["terminal_reward"], 0)
            replay, _ = ReplayBuffer.load(run_dir / 'replay.npz')
            self.assertEqual(len(replay), 3)
            self.assertEqual(replay._dones[:3].tolist(), [False, False, False])
            self.assertEqual(replay._episode_ends[:3].tolist(), [False, True, False])
            self.assertEqual(json.loads((run_dir / "config.json").read_text())
                             ["env"]["max_episode_ticks"], 40)


if __name__ == "__main__":
    unittest.main()
