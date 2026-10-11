from __future__ import annotations

import unittest
import json
import tempfile
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np

from pvz_gym.action_space import ACTION_SPACE_SIZE
from pvz_gym.observation import empty_observation
from pvz_gym.types import Observation
from pvz_gym.env import PvZEnv
from pvz_gym.types import GameStatus, StepEvents
from tests.fake_backend import FactClient
from train.dqn.config import DoubleDQNConfig
from train.dqn.credit import EpisodeEventCredit
from train.dqn.replay import ReplayBuffer
from train.dqn.reward import effective_reward_settings, training_reward
from train.dqn.staged_reward import StagedReward, planned_health_budget, resource_value


def zombie(key=1, hp=270, *, kind=0, dead=False, phase=0, maximum=270, **extra):
    return {"id": key, "type": kind, "body_hp": hp, "max_hp": maximum,
            "helm_hp": 0, "helm_max_hp": 0, "shield_hp": 0, "shield_max_hp": 0,
            "flying_hp": 0, "flying_max_hp": 0, "from_wave": 0,
            "is_dead_or_dying": dead, "phase": phase, **extra}


def raw(sun=0, *, plan=(0,) * 10, cards=(), plants=(), drops=(), health=(), **global_extra):
    schedule = list(plan) + [-1] * (50 - len(plan))
    return Observation(global_state={"sun": sun, "game_mode": 0, "total_waves": 1,
                                      "spawn_plan_available": True, "zombie_spawn_plan": schedule,
                                      **global_extra}, cards=cards, plants=plants,
                       dropped_cards=drops, zombie_health=health,
                       zombies=tuple(z for z in health if not z.get("is_dead_or_dying")))


def info(board, ticks=1000, *, mowers=0, killed=0, result="PLAYING", fallback=False):
    return {"raw_observation": board, "ticks_advanced": ticks, "result": result,
            "terminal_observation_fallback": fallback,
            "events": {"mowers_used": mowers, "zombies_killed": killed, "lost": result == "LOST"}}


class StagedMetricsTest(unittest.TestCase):
    def test_imitater_keeps_target_value_through_transformation_and_special_nuts_use_definition(self):
        before = raw(plants=({"type": 48, "imitater_type": 2},))
        after = raw(plants=({"type": 2},))
        self.assertEqual(resource_value(before), 150)
        self.assertEqual(resource_value(before), resource_value(after))
        self.assertEqual(resource_value(raw(cards=({"source": 2, "plant_type": 49, "cost": 0},))), 0)
        with self.assertRaises(ValueError):
            resource_value(raw(plants=({"type": 48},)))

    def test_free_packets_use_original_prices_and_drop_aliases_are_deduplicated(self):
        board = raw(50, cards=(
            {"slot": 0, "plant_type": 2, "cost": 0, "source": 1},
            {"slot": 1, "plant_type": 2, "cost": 0, "source": 2},
            {"slot": 2, "plant_type": 2, "cost": 0, "source": 3, "source_id": 9},
        ), drops=({"id": 9, "type": 2, "coin_type": 16},))
        self.assertEqual(resource_value(board), 500)

    def test_card_to_plant_and_sun_to_plant_preserve_inventory_value(self):
        conveyor = raw(cards=({"plant_type": 0, "cost": 0, "source": 2},))
        planted = raw(plants=({"type": 0, "hp": 1, "max_hp": 300},))
        self.assertEqual(resource_value(conveyor), resource_value(planted))
        self.assertEqual(resource_value(raw(100)), resource_value(planted))

    def test_upgrades_include_consumed_bases(self):
        before = raw(500, plants=({"type": 34}, {"type": 34}))
        after = raw(plants=({"type": 47},))
        self.assertEqual(resource_value(before), resource_value(after))

    def test_fixed_slots_cooldown_and_plant_injury_do_not_change_value(self):
        before = raw(cards=({"plant_type": 0, "source": 1, "cooldown_remaining": 0},),
                     plants=({"type": 0, "hp": 300},))
        after = raw(cards=({"plant_type": 0, "source": 1, "cooldown_remaining": 750},),
                    plants=({"type": 0, "hp": 1},))
        self.assertEqual(resource_value(before), resource_value(after))

    def test_unknown_price_and_missing_provenance_do_not_silently_become_zero(self):
        with self.assertRaisesRegex(ValueError, "original sun price"):
            resource_value(raw(plants=({"type": 99},)))
        with self.assertRaisesRegex(ValueError, "source metadata"):
            resource_value(raw(cards=({"plant_type": 0, "cost": 0},)))

    def test_budget_comes_from_all_planned_waves_and_independent_hp_pools(self):
        board = raw(plan=(0, 2, 4, 6, 16, 13))
        self.assertEqual(planned_health_budget(board), 270 + 640 + 1370 + 1370 + 290 + 1380)
        schedule = [0] + [-1] * 49 + [4] + [-1] * 49
        board = raw(total_waves=2, zombie_spawn_plan=schedule)
        self.assertEqual(planned_health_budget(board), 1640)

    def test_missing_or_malformed_plan_is_rejected(self):
        for board in (raw(spawn_plan_available=False), raw(zombie_spawn_plan=[0]), raw(plan=())):
            with self.assertRaises(ValueError):
                planned_health_budget(board)


class StagedAccountingTest(unittest.TestCase):
    def test_stage_cap_is_once_per_stage_not_once_per_transition(self):
        tracker = StagedReward(info(raw(), 0))
        for step in range(1, 10):
            self.assertEqual(tracker.step(0, info(raw(step * 50), 100),
                                         terminated=False, truncated=False), 0)
        self.assertAlmostEqual(tracker.step(0, info(raw(500), 100),
                                           terminated=False, truncated=False), 0.1)
        self.assertAlmostEqual(tracker.step(0, info(raw(1000)),
                                           terminated=False, truncated=False), 0.1)
        self.assertEqual(tracker.stages, 2)

    def test_stage_lifetime_cap_stops_infinite_accumulation(self):
        tracker = StagedReward(info(raw(), 0))
        # 连续 10 个阶段，每个阶段资产增加 500，均达到单阶段上限 0.10，累计达到 1.00
        for i in range(1, 11):
            reward = tracker.step(0, info(raw(i * 500)), terminated=False, truncated=False)
            self.assertAlmostEqual(reward, 0.1)
        self.assertAlmostEqual(tracker.accumulated_stage_reward, 1.0)
        self.assertEqual(tracker.stages, 10)

        # 超过 1.00 封顶后，即便继续拖时间增加阳光产出，阶段辅助奖励也强制归零
        reward_11 = tracker.step(0, info(raw(11 * 500)), terminated=False, truncated=False)
        self.assertEqual(reward_11, 0.0)
        self.assertAlmostEqual(tracker.accumulated_stage_reward, 1.0)
        self.assertEqual(tracker.last_breakdown["stage_reward"], 0.0)

        reward_12 = tracker.step(0, info(raw(12 * 500)), terminated=False, truncated=False)
        self.assertEqual(reward_12, 0.0)
        self.assertAlmostEqual(tracker.accumulated_stage_reward, 1.0)
        self.assertEqual(tracker.stages, 12)

    def test_mower_is_immediate_and_keeps_penalty_outside_the_cap(self):
        tracker = StagedReward(info(raw(), 0))
        self.assertAlmostEqual(tracker.step(0, info(raw(2000), 500, mowers=1),
                                           terminated=False, truncated=False), -0.2)
        self.assertAlmostEqual(tracker.step(0, info(raw(2000), 500),
                                           terminated=False, truncated=False), 0.1)
        self.assertAlmostEqual(tracker.step(0, info(raw(4000), mowers=2),
                                           terminated=False, truncated=False), -0.3)

    def test_negative_stage_rewards_are_not_clipped(self):
        tracker = StagedReward(info(raw(2000), 0))
        self.assertAlmostEqual(tracker.step(0, info(raw(500), mowers=1),
                                           terminated=False, truncated=False), -1.7)

    def test_damage_uses_initial_pools_and_does_not_reward_spawning(self):
        tracker = StagedReward(info(raw(), 0))
        healthy = zombie(helm_hp=370, helm_max_hp=370)
        tracker.step(0, info(raw(health=(healthy,)), 500), terminated=False, truncated=False)
        self.assertEqual(tracker.damage, 0)
        damaged = {**healthy, "helm_hp": 100}
        reward = tracker.step(0, info(raw(health=(damaged,)), 500), terminated=False, truncated=False)
        self.assertAlmostEqual(reward, 0.5 * 270 / 2700)
        self.assertEqual(tracker.damage, 270)

    def test_newly_observed_injured_zombie_and_death_record_are_counted_once(self):
        tracker = StagedReward(info(raw(), 0))
        tracker.step(0, info(raw(health=(zombie(hp=100),)), 500), terminated=False, truncated=False)
        self.assertEqual(tracker.damage, 170)
        tracker.step(0, info(raw(health=(zombie(hp=80, dead=True, phase=1),)), 500),
                     terminated=False, truncated=False)
        self.assertEqual(tracker.damage, 270)
        tracker.step(0, info(raw(health=(zombie(hp=80, dead=True, phase=1),))),
                     terminated=False, truncated=False)
        self.assertEqual(tracker.damage, 270)

    def test_healing_cannot_reward_the_same_health_twice(self):
        tracker = StagedReward(info(raw(health=(zombie(),)), 0))
        for hp in (100, 200, 100):
            tracker.step(0, info(raw(health=(zombie(hp=hp),)), 100), terminated=False, truncated=False)
        self.assertEqual(tracker.damage, 170)

    def test_instant_kill_requires_native_confirmation_and_escapes_are_not_damage(self):
        tracker = StagedReward(info(raw(health=(zombie(), zombie(2, kind=20, maximum=450, hp=450))), 0))
        tracker.step(0, info(raw(), killed=2), terminated=False, truncated=False)
        self.assertEqual(tracker.damage, 270)

    def test_lost_cleanup_and_terminal_fallback_do_not_manufacture_damage_or_assets(self):
        tracker = StagedReward(info(raw(100, health=(zombie(),)), 0))
        reward = tracker.step(-100, info(raw(), 200, killed=1, result="LOST", fallback=True),
                              terminated=True, truncated=False)
        self.assertEqual(reward, -1)
        self.assertEqual(tracker.damage, 0)
        tracker = StagedReward(info(raw(100, health=(zombie(),)), 0))
        tracker.step(-100, info(raw(100, health=(zombie(dead=True, phase=1),)), 200, result="LOST"),
                     terminated=True, truncated=False)
        self.assertEqual(tracker.damage, 0)

    def test_terminal_settles_partial_stage_and_damage_bonus_once(self):
        tracker = StagedReward(info(raw(health=(zombie(),)), 0))
        board = raw(200, health=(zombie(hp=0, dead=True, phase=1),))
        reward = tracker.step(100, info(board, 200, result="WON"), terminated=True, truncated=False)
        self.assertAlmostEqual(reward, 0.1 + 1 + 0.5 * 0.1)
        with self.assertRaisesRegex(ValueError, "finalized"):
            tracker.step(100, info(board, 0, result="WON"), terminated=True, truncated=False)

    def test_truncation_settles_partial_stage_without_terminal_bonus(self):
        tracker = StagedReward(info(raw(), 0))
        reward = tracker.step(0, info(raw(200), 200), terminated=False, truncated=True)
        self.assertEqual(reward, 0.1)
        self.assertEqual(tracker.last_breakdown["terminal_reward"], 0)

    def test_large_continuous_jump_does_not_invent_multiple_stage_payouts(self):
        tracker = StagedReward(info(raw(), 0))
        reward = tracker.step(0, info(raw(5000), 5500), terminated=False, truncated=False)
        self.assertEqual(reward, 0.1)
        self.assertEqual(tracker.stages, 1)

    def test_replay_reward_contract_and_truncation_bootstrap(self):
        config = replace(DoubleDQNConfig(reward_scheme="staged"), reward_scheme="staged")
        replay = ReplayBuffer(8, ACTION_SPACE_SIZE)
        tracker = EpisodeEventCredit(replay, config, info(raw(), 0))
        observation = empty_observation()
        mask = np.ones(ACTION_SPACE_SIZE, dtype=np.bool_)
        row_info = info(raw(500), 200, mowers=1)
        reward = tracker.add(replay, observation, 0, 0, observation, False, True, mask, row_info)
        self.assertAlmostEqual(reward, -0.1)
        self.assertAlmostEqual(float(replay._rewards[0]), reward)
        self.assertFalse(bool(replay._dones[0]))
        self.assertTrue(bool(replay._episode_ends[0]))
        self.assertTrue(row_info["reward_breakdown"]["stage_settled"])
        self.assertEqual(replay.reward_settings, effective_reward_settings(config))
        with self.assertRaisesRegex(ValueError, "per-episode"):
            training_reward(0, {}, config)
        with self.assertRaisesRegex(ValueError, "rebuild replay"):
            EpisodeEventCredit(replay, replace(config, reward_scheme="balanced"))


class StagedIntegrationTest(unittest.TestCase):
    def test_environment_reaches_stage_boundary_and_keeps_raw_entities(self):
        class PlayingClient(FactClient):
            def step(self, action, ticks):
                facts = super().step(action, ticks)
                self.status = GameStatus.PLAYING
                return replace(facts, observation=self._observation(), status=self.status,
                               events=StepEvents(), action_mask=self.get_action_mask())

            def begin_headless_mode(self):
                return True

            def end_headless_mode(self):
                return True

            def step_headless(self, action, ticks):
                return self.step(action, ticks)

            def step_continuous(self, action, ticks):
                return self.step(action, ticks + 7)

        for mode in ("stepped", "headless", "continuous"):
            with self.subTest(mode=mode):
                client = PlayingClient()
                env = PvZEnv(client, simulation_mode=mode, ticks_per_step=150, reward_stage_ticks=1000)
                try:
                    _, reset_info = env.reset()
                    self.assertIsInstance(reset_info["raw_observation"], Observation)
                    requests = []
                    for _ in range(8):
                        _, _, _, _, step_info = env.step(0)
                        requests.append(step_info["ticks_requested"])
                        self.assertIsInstance(step_info["raw_observation"], Observation)
                    self.assertEqual(requests[:6], [150] * 6)
                    self.assertEqual(requests[6], 58 if mode == "continuous" else 100)
                    self.assertEqual(requests[7], 150)
                finally:
                    env.close()

    def test_native_serialization_roundtrip_preserves_unprojected_reward_facts(self):
        import subprocess
        from pvz_gym.bridge.native import _parse_observation_data
        executable = Path(__file__).resolve().parents[2] / "build" / "pvz_gym_zombie_facts_test.exe"
        if not executable.is_file():
            self.skipTest("native fact fixture has not been built")
        board = _parse_observation_data(json.loads(subprocess.check_output([str(executable), "--json"], text=True)))
        self.assertTrue(board.global_state["spawn_plan_available"])
        self.assertEqual(board.global_state["zombie_spawn_plan"], [0, 2, -1])
        self.assertEqual(board.cards[0]["source_id"], 17)
        self.assertEqual(board.plants[0]["imitater_type"], 2)
        self.assertEqual(board.zombie_health[0]["shield_max_hp"], 1100)
        self.assertEqual(board.zombie_health[0]["from_wave"], 8)

    def test_controller_learns_and_saves_the_new_rewards_and_stage_diagnostics(self):
        from train.dqn import agent as agent_module
        from train.dqn.optimized_controller import DoubleDQNTrainerController
        if agent_module.torch is None:
            self.skipTest("PyTorch is required")

        class FakeEnv:
            def __init__(self, *args, **kwargs):
                self.action_space = type("ActionSpace", (), {"n": ACTION_SPACE_SIZE})()
                self.mask = np.zeros(ACTION_SPACE_SIZE, dtype=np.bool_)
                self.mask[0] = True
                self.asserted_stage_ticks = kwargs["reward_stage_ticks"]

            def reset(self, **kwargs):
                self.index = 0
                return empty_observation(), {"action_mask": self.mask, **info(raw(), 0)}

            def step(self, action):
                self.index += 1
                won = self.index == 3
                row_info = info(raw(500 * self.index), (500, 500, 200)[self.index - 1],
                                mowers=int(self.index == 1), result="WON" if won else "PLAYING")
                row_info.update(action_mask=self.mask, action_result={"success": True},
                                episode_ticks=(500, 1000, 1200)[self.index - 1])
                return empty_observation(), 100.0 if won else 0.0, won, False, row_info

            def close(self):
                pass

        config = DoubleDQNConfig(training_architecture="legacy", reward_scheme="staged", hidden_sizes=(16, 8), replay_capacity=8,
                                 batch_size=2, total_steps=5, learning_starts=0, train_every=1,
                                 checkpoint_every=0, autosave_every=0, device="cpu")
        emitted = []
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            controller = DoubleDQNTrainerController(config, run_dir=root,
                fast_eval_every_episodes=0, full_eval_every_episodes=0,
                on_event=lambda event, payload: emitted.append((event, payload)))
            with patch("train.dqn.base_controller.PvZEnv", FakeEnv), patch("train.dqn.base_controller.PvZClient"):
                summary = controller.run()
            self.assertEqual((summary.env_steps, summary.wins), (5, 1))
            self.assertEqual(controller._agent.update_count, 3)
            mower_step = next(payload for event, payload in emitted
                              if event == 'step' and payload['env_step'] == 1)
            self.assertEqual(mower_step['reward_scheme'], 'staged')
            self.assertEqual(mower_step['mower_reward'], -0.2)
            self.assertEqual(mower_step['mower_reward'], mower_step['reward_breakdown']['mower_reward'])
            replay, _ = ReplayBuffer.load(root / "replay.npz")
            np.testing.assert_allclose(replay._rewards[:5], [-0.2, 0.1, 1.1, -0.2, 0.1])
            self.assertEqual(replay.reward_settings, effective_reward_settings(config))
            records = [json.loads(line) for line in (root / "episodes.jsonl").read_text().splitlines()]
            self.assertAlmostEqual(records[0]["training_episode_reward"], 1.0)
            self.assertEqual(records[0]["reward_breakdown"]["reward_stages"], 2)
            self.assertTrue(any(event == "step" and payload["env_step"] == 2 for event, payload in emitted))
            resumed = DoubleDQNTrainerController(config, run_dir=root,
                checkpoint_path=controller.latest_checkpoint)
            _, start_step, start_episode, _ = resumed._build_agent(ACTION_SPACE_SIZE)
            restored = resumed._restore_replay(action_dim=ACTION_SPACE_SIZE,
                start_step=start_step, start_episode=start_episode)
            self.assertTrue(resumed.summary.replay_restored)
            self.assertEqual(len(restored), 5)
            changed = DoubleDQNTrainerController(replace(config, reward_scheme="balanced"),
                run_dir=root, checkpoint_path=controller.latest_checkpoint)
            _, start_step, start_episode, _ = changed._build_agent(ACTION_SPACE_SIZE)
            discarded = changed._restore_replay(action_dim=ACTION_SPACE_SIZE,
                start_step=start_step, start_episode=start_episode)
            self.assertEqual(len(discarded), 0)
            self.assertFalse(changed.summary.replay_restored)
            # Evaluation uses the same staged account, including a partial
            # stage flush when evaluation's own step budget ends the rollout.
            controller.eval_seed_count = 1
            controller.eval_max_episode_steps = 2
            evaluated = controller._evaluate_current_agent(
                FakeEnv(reward_stage_ticks=1000), ACTION_SPACE_SIZE, env_step=5)
            self.assertTrue(evaluated["complete"])
            self.assertEqual(evaluated["mean_score"], 0)
            self.assertAlmostEqual(evaluated["mean_training_score"], -0.1)
            controller.eval_max_episode_steps = 3
            evaluated = controller._evaluate_current_agent(
                FakeEnv(reward_stage_ticks=1000), ACTION_SPACE_SIZE, env_step=5)
            self.assertEqual(evaluated["mean_score"], 100)
            self.assertAlmostEqual(evaluated["mean_training_score"], 1.0)


if __name__ == "__main__":
    unittest.main()
