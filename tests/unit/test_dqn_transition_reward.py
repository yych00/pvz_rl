from dataclasses import asdict, replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from pvz_gym.action_space import ACTION_SPACE_SIZE
from pvz_gym.observation import empty_observation
from tests.unit.test_dqn_staged_reward import raw, info, zombie
from train.dqn.config import DoubleDQNConfig
from train.dqn.credit import EpisodeEventCredit
from train.dqn.replay import ReplayBuffer
from train.dqn.reward import effective_reward_settings
from train.dqn.transition_reward import TransitionDenseReward, asset_value, smooth_shaping_delta


def board(sun=0, *, mower_states=None, **kwargs):
    value = raw(sun, **kwargs)
    if mower_states is not None:
        value = replace(value, mowers=tuple({"row": row, "state": state}
                                          for row, state in enumerate(mower_states)))
    return value


class TransitionAccountingTests(unittest.TestCase):
    def step(self, tracker, state, **kwargs):
        terminated = kwargs.pop("terminated", False)
        truncated = kwargs.pop("truncated", False)
        return tracker.step(0, info(state, 100, **kwargs), terminated=terminated, truncated=truncated)

    def test_fixed_slots_excluded_and_card_conversion_preserves_assets(self):
        reusable = {"source": 1, "plant_type": 0}
        conveyor = {"source": 2, "plant_type": 0, "cost": 0}
        dropped = {"source": 3, "source_id": 7, "plant_type": 0, "cost": 0}
        self.assertEqual(asset_value(board(50, cards=(reusable, conveyor, dropped),
            drops=({"id": 7, "type": 0, "coin_type": 16},))), 250)
        self.assertEqual(asset_value(board(cards=(conveyor,))),
                         asset_value(board(plants=({"type": 0},))))
        self.assertEqual(asset_value(board(100)), asset_value(board(plants=({"type": 0},))))

    def test_net_balance_is_bounded_and_positive_negative_changes_cancel(self):
        tracker = TransitionDenseReward(info(board(), 0))
        total = 0.0
        for sun, balance in [(1000, 2/15), (3000, .24), (5000, 2/7),
                             (1000, 2/15), (0, 0), (5000, 2/7), (0, 0)]:
            total += self.step(tracker, board(sun))
            self.assertAlmostEqual(total, balance)
            self.assertAlmostEqual(tracker.shaping_balance, balance)
            self.assertLess(abs(total), .4)
        self.assertAlmostEqual(tracker.positive, tracker.negative)
        np.testing.assert_allclose(tracker.observe(empty_observation())["reward_context"], [0, 1, 1])

    def test_long_sequences_do_not_exhaust_positive_or_negative_rewards(self):
        tracker = TransitionDenseReward(info(board(), 0))
        total = 0.0
        previous = float("inf")
        for sun in range(500, 10500, 500):
            reward = self.step(tracker, board(sun))
            self.assertGreater(reward, 0)
            self.assertLess(reward, previous)
            previous = reward
            total += reward
            self.assertLess(total, .4)
        for sun in range(9500, -500, -500):
            reward = self.step(tracker, board(sun))
            self.assertLess(reward, 0)
            total += reward
            self.assertLess(abs(total), .4)
        self.assertAlmostEqual(total, 0)
        negative = TransitionDenseReward(info(board(10000), 0))
        for sun in range(9500, -500, -500):
            self.assertLess(self.step(negative, board(sun)), 0)
            self.assertGreater(negative.shaping_balance, -.4)

    def test_large_step_softly_approaches_bound_and_next_step_still_receives_reward(self):
        tracker = TransitionDenseReward(info(board(), 0))
        self.assertAlmostEqual(self.step(tracker, board(10000)), 1/3)
        self.assertGreater(self.step(tracker, board(10500)), 0)
        self.assertLess(tracker.shaping_balance, .4)
        self.assertLess(self.step(tracker, board()), 0)
        self.assertAlmostEqual(tracker.shaping_balance, 0)

    def test_near_bound_delta_does_not_vanish_from_subtracting_equal_floats(self):
        self.assertGreater(smooth_shaping_delta(1e12, .01), 0)
        self.assertLess(smooth_shaping_delta(-1e12, -.01), 0)
        self.assertGreater(smooth_shaping_delta(-1e12, .01), 0)

    def test_raw_total_can_cross_zero_and_reverse_the_balance(self):
        tracker = TransitionDenseReward(info(board(5000), 0))
        total = self.step(tracker, board(10000))
        self.assertAlmostEqual(total, 2/7)
        total += self.step(tracker, board())
        self.assertAlmostEqual(total, -2/7)
        self.assertAlmostEqual(tracker.shaping_balance, -2/7)
        self.assertAlmostEqual(tracker.raw_shaping_total, -1)
        np.testing.assert_allclose(tracker.observe(empty_observation())["reward_context"], [0, 12/7, 2/7])

    def test_many_signed_changes_preserve_net_bound_and_path_independence(self):
        rng = np.random.default_rng(7)
        tracker = TransitionDenseReward(info(board(5000), 0))
        paid = 0.0
        for sun in rng.integers(0, 20000, size=1000):
            paid += self.step(tracker, board(int(sun)))
            self.assertLess(abs(paid), .4)
            # With assets alone, any path ending at the same resource value
            # has the same undiscounted net return, including repeated cycles.
            self.assertAlmostEqual(paid, tracker.shaping_balance)
        paid += self.step(tracker, board(5000))
        self.assertAlmostEqual(paid, 0)

    def test_historical_episode_budget_remains_available_for_old_checkpoints(self):
        tracker = TransitionDenseReward(info(board(), 0), shaping_limit_mode="episode_budget")
        for sun, reward in [(1000, .2), (3000, .2), (0, -.4), (5000, 0), (0, 0)]:
            self.assertAlmostEqual(self.step(tracker, board(sun)), reward)
        np.testing.assert_allclose(tracker.observe(empty_observation())["reward_context"], [0, 0, 0])

    def test_first_and_later_mowers_are_outside_net_shaping_bound(self):
        tracker = TransitionDenseReward(info(board(), 0))
        self.assertAlmostEqual(self.step(tracker, board(10000), mowers=2), 1/3 - .6)
        self.assertAlmostEqual(self.step(tracker, board(20000), mowers=1), 4/11 - 1/3 - .1)
        self.assertAlmostEqual(self.step(tracker, board(20000), mowers=0), 0)
        self.assertEqual(tracker.mowers_used, 3)
        self.assertAlmostEqual(tracker.last_breakdown["mower_reward"], 0)

    def test_native_combat_events_charge_once_and_disappearance_is_not_evidence(self):
        tracker = TransitionDenseReward(info(board(mower_states=["READY"] * 2), 0))
        self.assertAlmostEqual(self.step(tracker, board(mower_states=["ROLLING", "READY"]), mowers=1), -.5)
        self.assertEqual(self.step(tracker, board(mower_states=["LOST", "READY"]), mowers=0), 0)
        self.assertAlmostEqual(self.step(tracker, board(mower_states=["LOST", "LOST"]), mowers=1), -.1)

    def test_victory_cleanup_while_playing_never_charges_unused_mowers(self):
        tracker = TransitionDenseReward(info(board(mower_states=["READY"] * 5), 0))
        self.assertEqual(self.step(tracker, board(mower_states=["LOST"] * 5)), 0)
        self.assertEqual(tracker.mowers_used, 0)
        self.assertFalse(tracker.first_mower_seen)
        self.assertEqual(self.step(tracker, board(mower_states=["LOST"] * 5),
                                   terminated=True, result="WON"), 1)

    def test_true_trigger_can_coincide_with_cleanup_of_other_rows(self):
        initial = (zombie(row=0), zombie(2, row=1))
        tracker = TransitionDenseReward(info(board(health=initial, mower_states=["READY"] * 5), 0))
        details = info(board(health=(zombie(row=0, hp=100), zombie(2, row=1, hp=100)),
                             mower_states=["LOST"] * 5), 150, mowers=1)
        details["events"]["mower_triggered_rows"] = [0]
        tracker.step(0, details, terminated=False, truncated=False)
        self.assertEqual(tracker.mowers_used, 1)
        self.assertEqual(tracker.damage, 170)
        self.assertEqual(tracker.last_breakdown["damage_suppressed_rows"], [0])

    def test_old_loaded_dll_is_rejected_before_new_reward_semantics_are_used(self):
        with self.assertRaisesRegex(ValueError, "restart Workbench"):
            TransitionDenseReward(info(board(board_id=1234), 0))

    def test_damage_counts_effective_pools_with_no_overkill_or_healing_farm(self):
        initial = zombie(row=0, helm_hp=100, helm_max_hp=100)
        tracker = TransitionDenseReward(info(board(health=(initial,)), 0))
        for hp, helm in ((100, 0), (200, 50), (100, 0)):
            self.step(tracker, board(health=({**initial, "body_hp": hp, "helm_hp": helm},)))
        self.assertEqual(tracker.damage, 270)
        self.step(tracker, board(health=({**initial, "body_hp": 0, "helm_hp": 0,
                                        "is_dead_or_dying": True, "phase": 1},)))
        self.assertEqual(tracker.damage, 370)
        self.assertAlmostEqual(tracker.last_breakdown["damage_fraction"], 370 / 2700)
        self.assertAlmostEqual(tracker.raw_shaping_total, .3 * 370 / 2700)

    def test_spawns_disappearance_and_terminal_cleanup_are_not_damage(self):
        tracker = TransitionDenseReward(info(board(), 0))
        self.step(tracker, board(health=(zombie(hp=100),)))
        self.step(tracker, board(), killed=1)
        self.assertEqual(tracker.damage, 0)
        self.step(tracker, board(health=(zombie(2),)))
        self.step(tracker, board(health=(zombie(2, hp=0, dead=True, phase=1),)),
                  terminated=True, result="WON")
        self.assertEqual(tracker.damage, 0)

    def test_mower_rows_suppressed_while_other_rows_still_receive_damage(self):
        initial = (zombie(row=0), zombie(2, row=1))
        tracker = TransitionDenseReward(info(board(health=initial, mower_states=["READY"] * 2), 0))
        next_health = (zombie(row=0, hp=100), zombie(2, row=1, hp=100))
        self.step(tracker, board(health=next_health, mower_states=["ROLLING", "READY"]), mowers=1)
        self.assertEqual(tracker.damage, 170)
        self.step(tracker, board(health=(zombie(row=0, hp=0, dead=True, phase=1),
                                        zombie(2, row=1, hp=100)),
                                 mower_states=["LOST", "READY"]))
        self.assertEqual(tracker.damage, 170)

    def test_fallback_ignores_stale_assets_and_damage_but_settles_terminal(self):
        tracker = TransitionDenseReward(info(board(100, health=(zombie(),)), 0))
        self.assertEqual(self.step(tracker, board(10000, health=(zombie(hp=0),)),
                                   fallback=True, terminated=True, result="LOST"), -1)
        self.assertEqual(tracker.asset, 100)
        self.assertEqual(tracker.damage, 0)
        with self.assertRaisesRegex(ValueError, "finalized"):
            self.step(tracker, board())

    def test_truncation_has_no_loss_and_terminal_does_not_add_damage_bonus(self):
        tracker = TransitionDenseReward(info(board(), 0))
        self.assertAlmostEqual(self.step(tracker, board(500), truncated=True), .08)
        self.assertEqual(tracker.last_breakdown["terminal_reward"], 0)
        for outcome, expected in (("WON", 1), ("LOST", -1)):
            tracker = TransitionDenseReward(info(board(), 0))
            self.assertEqual(self.step(tracker, board(), terminated=True, result=outcome), expected)

    def test_terminal_and_tick_contract_validation(self):
        for ticks in (-1, 1.5, True, 2**63):
            tracker = TransitionDenseReward(info(board(), 0))
            with self.assertRaises(ValueError):
                tracker.step(0, info(board(), ticks), terminated=False, truncated=False)
        tracker = TransitionDenseReward(info(board(), 0))
        with self.assertRaisesRegex(ValueError, "WON/LOST"):
            self.step(tracker, board(), terminated=True)

    def test_threat_motion_sign_and_both_step_caps(self):
        tracker = TransitionDenseReward(info(board(health=(zombie(x=900),)), 0))
        self.assertAlmostEqual(self.step(tracker, board(health=(zombie(x=450),))), -.4 * .0025 / .4025)
        self.assertAlmostEqual(tracker.last_breakdown["threat_value"], .025)
        self.assertTrue(tracker.last_breakdown["threat_enabled"])
        tracker = TransitionDenseReward(info(board(health=(zombie(x=900, hp=2700, maximum=2700),)), 0))
        self.assertAlmostEqual(self.step(tracker, board(health=(zombie(x=0, hp=2700, maximum=2700),))), -.4 / 41)
        self.assertAlmostEqual(self.step(tracker, board(health=(zombie(x=900, hp=2700, maximum=2700),))), .4 / 41)

    def test_nearby_damage_reduces_more_threat_than_distant_damage(self):
        rewards = []
        for x in (0, 450):
            tracker = TransitionDenseReward(info(board(health=(zombie(x=x),)), 0))
            self.step(tracker, board(health=(zombie(x=x, hp=170),)))
            rewards.append(tracker.last_breakdown["threat_reward"])
        self.assertAlmostEqual(rewards[0], .1 * 100 / 2700)
        self.assertAlmostEqual(rewards[0], 4 * rewards[1])

    def test_threat_does_not_reward_spawns_missing_ids_or_faction_changes(self):
        tracker = TransitionDenseReward(info(board(), 0))
        self.assertEqual(self.step(tracker, board(health=(zombie(x=0),))), 0)
        self.assertGreater(tracker.threat, 0)
        self.assertEqual(self.step(tracker, board(), killed=1), 0)
        self.step(tracker, board(health=(zombie(2, x=0),)))
        self.assertEqual(self.step(tracker, board(health=(zombie(2, x=0, is_mind_controlled=True),))), 0)

    def test_observed_death_reduces_threat_but_terminal_and_fallback_do_not(self):
        initial = zombie(x=0)
        dead = zombie(x=0, hp=0, dead=True, phase=1)
        tracker = TransitionDenseReward(info(board(health=(initial,)), 0))
        self.step(tracker, board(health=(dead,)))
        self.assertAlmostEqual(tracker.last_breakdown["threat_reward"], .01)
        for options in ({"terminated": True, "result": "WON"}, {"fallback": True},
                        {"terminated": True, "result": "LOST"}):
            tracker = TransitionDenseReward(info(board(health=(initial,)), 0))
            self.step(tracker, board(health=(dead,)), **options)
            self.assertEqual(tracker.last_breakdown["threat_reward"], 0)
        tracker = TransitionDenseReward(info(board(health=(zombie(x=900),)), 0))
        self.assertAlmostEqual(self.step(tracker, board(health=(zombie(x=0),)), truncated=True), -.4 / 41)

    def test_mower_suppression_updates_threat_baseline_without_deferred_bonus(self):
        initial = (zombie(x=0, row=0), zombie(2, x=900, row=1))
        tracker = TransitionDenseReward(info(board(health=initial, mower_states=["READY"] * 2), 0))
        after = (zombie(x=0, row=0, hp=100), zombie(2, x=0, row=1))
        self.step(tracker, board(health=after, mower_states=["ROLLING", "READY"]), mowers=1)
        self.assertAlmostEqual(tracker.last_breakdown["threat_reward"], -.01)
        self.step(tracker, board(health=after, mower_states=["LOST", "READY"]))
        self.assertEqual(tracker.last_breakdown["threat_reward"], 0)
        self.step(tracker, board(health=after, mower_states=["LOST", "READY"]))
        self.assertEqual(tracker.last_breakdown["threat_reward"], 0)

    def test_threat_continues_after_prior_same_direction_shaping(self):
        initial = zombie(x=0, hp=2700, maximum=2700)
        tracker = TransitionDenseReward(info(board(health=(initial,)), 0))
        self.step(tracker, board(1975, health=(initial,)))
        reward = self.step(tracker, board(1975, health=(zombie(x=900, hp=2700, maximum=2700),)))
        self.assertAlmostEqual(tracker.last_breakdown["threat_reward"], .01)
        self.assertGreater(reward, 0)
        self.assertAlmostEqual(tracker.raw_shaping_total, .405)
        tracker = TransitionDenseReward(info(board(1975, health=(zombie(x=900, hp=2700, maximum=2700),)), 0))
        self.step(tracker, board(health=(zombie(x=900, hp=2700, maximum=2700),)))
        self.assertLess(self.step(tracker, board(health=(initial,))), 0)
        self.assertAlmostEqual(tracker.raw_shaping_total, -.405)

    def test_replay_contains_before_and_after_context_and_bootstraps_truncation(self):
        config = DoubleDQNConfig()
        replay = ReplayBuffer(8, ACTION_SPACE_SIZE)
        credit = EpisodeEventCredit(replay, config, info(board(), 0))
        state = credit.observe(empty_observation())
        next_state = empty_observation()
        details = info(board(1000), 100, mowers=1)
        reward = credit.add(replay, state, 0, 0, next_state, False, True,
                            np.ones(ACTION_SPACE_SIZE, dtype=bool), details)
        self.assertAlmostEqual(reward, 2/15 - .5)
        np.testing.assert_allclose(state["reward_context"], [0, 1, 1])
        np.testing.assert_allclose(next_state["reward_context"], [1, 2/3, 4/3])
        self.assertFalse(replay._dones[0])
        self.assertTrue(replay._episode_ends[0])
        with tempfile.TemporaryDirectory() as tmp:
            saved = replay.save(Path(tmp) / "replay.npz", env_step=1, episode=0)
            restored, metadata = ReplayBuffer.load(saved)
            self.assertEqual(metadata["reward_settings"], effective_reward_settings(config))
            batch = restored.sample(1)
            # Replay stores observations as float16 and expands them for learning.
            expected = np.asarray([[1, 2/3, 4/3]], dtype=np.float16).astype(np.float32)
            np.testing.assert_allclose(batch.next_observations["reward_context"], expected)

    def test_old_replay_round_trip_and_reward_change_cannot_mix_records(self):
        old_config = DoubleDQNConfig(shaping_limit_mode="episode_budget")
        replay = ReplayBuffer(8, ACTION_SPACE_SIZE)
        credit = EpisodeEventCredit(replay, old_config, info(board(), 0))
        credit.add(replay, credit.observe(empty_observation()), 0, 0, empty_observation(), False, False,
                   np.ones(ACTION_SPACE_SIZE, dtype=bool), info(board(1000), 100))
        with tempfile.TemporaryDirectory() as tmp:
            path = replay.save(Path(tmp) / "replay.npz", env_step=1, episode=0)
            restored, metadata = ReplayBuffer.load(path)
        self.assertEqual(metadata["reward_settings"]["version"], 5)
        np.testing.assert_allclose(restored.sample(1).next_observations["reward_context"], [[0, .5, 1]])
        with self.assertRaisesRegex(ValueError, "reward settings changed"):
            EpisodeEventCredit(restored, DoubleDQNConfig(), info(board(), 0))

    def test_replay_with_historical_cleanup_penalties_is_rejected(self):
        config = DoubleDQNConfig()
        replay = ReplayBuffer(2, ACTION_SPACE_SIZE, reward_context=True)
        tracker = TransitionDenseReward(info(board(), 0))
        obs = tracker.observe(empty_observation())
        replay.add(obs, 0, 0, obs, False, [True] * ACTION_SPACE_SIZE, ticks_advanced=150)
        settings = effective_reward_settings(config)
        del settings["mower_basis"]
        settings["metrics_version"] = 2
        with tempfile.TemporaryDirectory() as tmp:
            path = replay.save(Path(tmp) / "old-replay.npz", reward_settings=settings)
            with self.assertRaisesRegex(ValueError, "mower reward semantics changed"):
                ReplayBuffer.load(path)

    def test_dynamic_spawned_zombies_with_minus_one_wave_count_towards_damage_with_budget_cap(self):
        # 初始状态：包含一个计划波次僵尸和一个 from_wave == -1 的动态生成僵尸
        initial_z = (zombie(key=1, hp=270, kind=0, from_wave=0),
                     zombie(key=2, hp=270, kind=0, from_wave=-1))
        initial_board = board(health=initial_z)
        tracker = TransitionDenseReward(info(initial_board, 0))
        self.assertEqual(len(tracker._remaining), 2)
        self.assertEqual(tracker._remaining[2], 270)

        # 动态僵尸受到 100 点伤害
        damaged_z = (zombie(key=1, hp=270, kind=0, from_wave=0),
                     zombie(key=2, hp=170, kind=0, from_wave=-1))
        self.step(tracker, board(health=damaged_z))
        self.assertAlmostEqual(tracker.damage, 100.0)
        self.assertGreater(tracker.last_breakdown["damage_reward"], 0)

        # 动态僵尸死亡，造成额外 170 点伤害
        dead_z = (zombie(key=1, hp=270, kind=0, from_wave=0),
                  zombie(key=2, hp=0, kind=0, dead=True, phase=1, from_wave=-1))
        self.step(tracker, board(health=dead_z))
        self.assertAlmostEqual(tracker.damage, 270.0)



class TransitionIntegrationTests(unittest.TestCase):
    def test_cleanup_reward_change_resumes_model_optimizer_progress_with_empty_replay(self):
        import torch
        from train.dqn.agent import DoubleDQNAgent
        from train.dqn.optimized_controller import DoubleDQNTrainerController
        config = DoubleDQNConfig(training_architecture="legacy", hidden_sizes=(16, 8),
                                 replay_capacity=8, batch_size=2, device="cpu")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = DoubleDQNTrainerController(config, run_dir=root)
            source._agent = DoubleDQNAgent(ACTION_SPACE_SIZE, config)
            source._agent.update_count = 17
            source._replay = source._new_replay(ACTION_SPACE_SIZE)
            credit = EpisodeEventCredit(source._replay, config, info(board(), 0))
            state = credit.observe(empty_observation())
            for sun in (100, 200):
                credit.add(source._replay, state, 0, 0, empty_observation(), False, False,
                           [True] * ACTION_SPACE_SIZE, info(board(sun), 150))
            source._agent.learn(source._replay.sample(2))
            source.summary.env_steps = 123
            source.summary.episodes = 4
            saved = source.save_checkpoint(root / "old.pt")
            payload = torch.load(saved, map_location="cpu", weights_only=False)
            path = root / payload["replay_file"]
            with np.load(path, allow_pickle=False) as data:
                old = {name: data[name] for name in data.files}
            settings = json.loads(str(old["reward_settings_json"].item()))
            settings.pop("mower_basis")
            settings["metrics_version"] = 2
            old["reward_settings_json"] = np.asarray(json.dumps(settings))
            np.savez(path, **old)
            events = []
            resume = DoubleDQNTrainerController(config, run_dir=root, checkpoint_path=saved,
                on_event=lambda event, data: events.append((event, data)))
            agent, step, episode, _ = resume._build_agent(ACTION_SPACE_SIZE)
            self.assertEqual((step, episode), (123, 4))
            self.assertEqual(agent.update_count, source._agent.update_count)
            self.assertTrue(agent.optimizer.state_dict()["state"])
            self.assertEqual(len(resume._restore_replay(action_dim=ACTION_SPACE_SIZE,
                start_step=step, start_episode=episode)), 0)
            self.assertTrue(any(event == "replay_restore_skipped" and "mower reward semantics" in data["reason"]
                                and data["warmup_restarted"] for event, data in events))
            for key, value in source._agent.online.state_dict().items():
                torch.testing.assert_close(agent.online.state_dict()[key], value)

    def test_legacy_checkpoint_keeps_old_semantics_and_new_rule_rebuilds_replay(self):
        import torch
        from train.dqn.agent import DoubleDQNAgent
        from train.dqn.optimized_controller import DoubleDQNTrainerController
        from train.dqn.run_state import load_run_config

        old_config = DoubleDQNConfig(shaping_limit_mode="episode_budget", training_architecture="legacy",
            hidden_sizes=(16, 8), replay_capacity=8, batch_size=2, device="cpu")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = DoubleDQNTrainerController(old_config, run_dir=root)
            source._agent = DoubleDQNAgent(ACTION_SPACE_SIZE, old_config)
            source._replay = source._new_replay(ACTION_SPACE_SIZE)
            credit = EpisodeEventCredit(source._replay, old_config, info(board(), 0))
            state = credit.observe(empty_observation())
            for sun in (1000, 3000, 5000):
                next_state = empty_observation()
                credit.add(source._replay, state, 0, 0, next_state, False, False,
                    np.ones(ACTION_SPACE_SIZE, dtype=bool), info(board(sun), 100))
                state = next_state
            source.summary.env_steps = 3
            saved = source.save_checkpoint(root / "legacy.pt")
            payload = torch.load(saved, map_location="cpu", weights_only=False)
            payload["config"].pop("shaping_limit_mode")
            torch.save(payload, saved)
            config_data = asdict(old_config)
            config_data.pop("shaping_limit_mode")
            (root / "config.json").write_text(json.dumps(config_data), encoding="utf-8")
            self.assertEqual(load_run_config(root).shaping_limit_mode, "episode_budget")
            _, loaded_config, loaded_payload = DoubleDQNAgent.from_checkpoint(saved, device="cpu")
            self.assertEqual(loaded_config.shaping_limit_mode, "episode_budget")
            self.assertEqual(loaded_payload["env_step"], 3)

            old_resume = DoubleDQNTrainerController(old_config, run_dir=root, checkpoint_path=saved)
            _, step, episode, _ = old_resume._build_agent(ACTION_SPACE_SIZE)
            self.assertEqual(len(old_resume._restore_replay(action_dim=ACTION_SPACE_SIZE,
                start_step=step, start_episode=episode)), 3)
            events = []
            new_resume = DoubleDQNTrainerController(replace(old_config, shaping_limit_mode="smooth_total"),
                run_dir=root, checkpoint_path=saved,
                on_event=lambda event, payload: events.append((event, payload)))
            agent, step, episode, _ = new_resume._build_agent(ACTION_SPACE_SIZE)
            self.assertEqual(step, 3)
            self.assertEqual(agent.config.shaping_limit_mode, "smooth_total")
            self.assertEqual(len(new_resume._restore_replay(action_dim=ACTION_SPACE_SIZE,
                start_step=step, start_episode=episode)), 0)
            self.assertFalse(new_resume.summary.replay_restored)
            self.assertTrue(any(event == "replay_restore_skipped" and "reward settings changed" in data["reason"]
                                for event, data in events))
            for key, value in payload["online"].items():
                torch.testing.assert_close(agent.online.state_dict()[key], value)

    def test_controller_training_evaluation_checkpoint_resume_and_reject_schema_switch(self):
        from train.dqn.agent import DoubleDQNAgent
        from train.dqn.optimized_controller import DoubleDQNTrainerController

        class FakeEnv:
            def __init__(self, *args, **kwargs):
                self.action_space = type("ActionSpace", (), {"n": ACTION_SPACE_SIZE})()
                self.mask = np.zeros(ACTION_SPACE_SIZE, dtype=bool)
                self.mask[0] = True
                self.asserted_ticks = kwargs.get("reward_stage_ticks")

            def reset(self, **kwargs):
                self.index = 0
                return empty_observation(), {"action_mask": self.mask, **info(board(), 0)}

            def step(self, action):
                self.index += 1
                won = self.index == 3
                details = info(board(500 * self.index), 100, mowers=int(self.index == 1),
                               result="WON" if won else "PLAYING")
                details.update(action_mask=self.mask, action_result={"success": True})
                return empty_observation(), 100 if won else 0, won, False, details

            def close(self):
                pass

        config = DoubleDQNConfig(training_architecture="legacy", hidden_sizes=(16, 8), replay_capacity=8, batch_size=2,
            total_steps=5, learning_starts=0, train_every=1, checkpoint_every=0,
            autosave_every=0, device="cpu")
        emitted = []
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            controller = DoubleDQNTrainerController(config, run_dir=root,
                fast_eval_every_episodes=0, full_eval_every_episodes=0,
                on_event=lambda event, payload: emitted.append((event, payload)))
            with patch("train.dqn.base_controller.PvZEnv", FakeEnv), patch("train.dqn.base_controller.PvZClient"):
                summary = controller.run()
            self.assertEqual((summary.env_steps, summary.wins), (5, 1))
            self.assertGreater(controller._agent.update_count, 0)
            mower_step = next(payload for event, payload in emitted if event == "step" and payload["env_step"] == 1)
            self.assertEqual(mower_step["mower_reward"], -.5)
            replay, _ = ReplayBuffer.load(root / "replay.npz")
            np.testing.assert_allclose(replay._rewards[:5], [-.42, 2/15 - .08, 1 + 6/35 - 2/15, -.42, 2/15 - .08])
            agent, restored_config, _ = DoubleDQNAgent.from_checkpoint(controller.latest_checkpoint, device="cpu")
            self.assertEqual(restored_config.reward_scheme, "transition_dense")
            self.assertTrue(agent.online.reward_context)
            resumed = DoubleDQNTrainerController(config, run_dir=root, checkpoint_path=controller.latest_checkpoint)
            _, step, episode, _ = resumed._build_agent(ACTION_SPACE_SIZE)
            self.assertEqual(len(resumed._restore_replay(action_dim=ACTION_SPACE_SIZE, start_step=step, start_episode=episode)), 5)
            self.assertTrue(resumed.summary.replay_restored)
            controller.eval_seed_count = 1
            record = controller._evaluate_current_agent(FakeEnv(), ACTION_SPACE_SIZE, env_step=5)
            self.assertAlmostEqual(record["mean_training_score"], .5 + 6/35)
            self.assertEqual(record["zero_mower_win_rate"], 0)
            self.assertEqual(record["mean_win_mowers"], 1)
            from train.dqn.evaluate import evaluate_checkpoint
            with patch("train.dqn.evaluate.PvZEnv", FakeEnv), patch("train.dqn.evaluate.PvZClient"):
                standalone = evaluate_checkpoint(controller.latest_checkpoint, seed_count=1,
                                                 device="cpu")
            self.assertEqual(standalone.wins, 1)
            self.assertEqual(standalone.mean_win_mowers, 1)
            self.assertEqual(standalone.zero_mower_win_rate, 0)
            self.assertAlmostEqual(standalone.total_training_score, .5 + 6/35)
            # Cross-contract continuation is explicitly unsupported.
            switched = DoubleDQNTrainerController(replace(config, reward_scheme="balanced"),
                run_dir=root, checkpoint_path=controller.latest_checkpoint)
            with self.assertRaisesRegex(ValueError, "start a new model"):
                switched._build_agent(ACTION_SPACE_SIZE)


if __name__ == "__main__":
    unittest.main()
