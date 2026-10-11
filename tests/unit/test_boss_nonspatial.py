"""Boss placeholders must not become lawn locations or mower damage sources."""
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from pvz_gym.action_space import ACTION_SPACE_SIZE
from pvz_gym.observation import encode_observation, PREVIOUS_OBSERVATION_SCHEMA
from train.dqn.agent import DoubleDQNAgent
from train.dqn.config import DoubleDQNConfig
from train.dqn.checkpoint_contract import declare_checkpoint
from train.dqn.networks import SpatialQNetwork
from train.dqn.optimized_controller import DoubleDQNTrainerController
from train.dqn.replay import ReplayBuffer
from train.dqn.reward import effective_reward_settings
from train.dqn.transition_reward import TransitionDenseReward
from tests.unit.test_dqn_staged_reward import zombie, info
from tests.unit.test_dqn_transition_reward import board


def snapshot(**boss_changes):
    boss = {"type": 25, "row": 0, "x": 0, "y": 0, "body_hp": 30000,
            "max_hp": 40000, "phase": 80, "phase_counter": 300,
            "target_row": 3, "target_col": 4, **boss_changes}
    return {"global_state": {"grid_rows": 5, "scene": 4}, "zombies": [boss],
            "boss_projectiles": [{"type": 0, "row": 2, "x": 650, "y": 320, "valid": True}]}


def tensors(obs):
    return {key: torch.as_tensor(value).unsqueeze(0) for key, value in obs.items()}


class BossNonspatialTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def test_placeholder_coordinates_and_row_do_not_change_observation(self):
        base = encode_observation(snapshot())
        for row, x, y in [(4, 800, 500), (-1, -100, 80), (99, 2000, 700)]:
            other = encode_observation(snapshot(row=row, x=x, y=y))
            for key in base:
                np.testing.assert_array_equal(base[key], other[key], err_msg=key)
        self.assertEqual(int(base["zombie_mask"][0]), 1)
        self.assertEqual(int(base["zombie_row"][0]), 0)
        self.assertEqual(int(base["zombie_target_row"][0]), 4)
        self.assertEqual(int(base["zombie_target_col"][0]), 5)
        self.assertEqual(float(base["zombie_grid"].sum()), 0)
        normal = snapshot()
        normal["zombies"].append({"type": 0, "row": 0, "x": 80, "body_hp": 270})
        grid = encode_observation(normal)["zombie_grid"]
        self.assertGreater(float(grid[0, 0].sum()), 0)
        self.assertEqual(float(grid[1:].sum()), 0)

    def test_boss_reads_global_only_and_retains_state_and_attack_target(self):
        torch.manual_seed(3)
        net = SpatialQNetwork(hidden_sizes=(16, 8), boss_projectiles=True).eval()
        base = tensors(encode_observation(snapshot()))
        encoded = net._encode_action_state(base)
        self.assertTrue(bool(encoded[5][0, 0]))
        self.assertFalse(bool(net._cell_zombie_visibility(encoded[-1], encoded[5])[..., 0].any()))
        with torch.no_grad():
            branches = net._action_branches(base)
            for changes in ({"body_hp": 10000}, {"phase": 85},
                            {"target_row": 1, "target_col": 7}):
                other = net._action_branches(tensors(encode_observation(snapshot(**changes))))
                torch.testing.assert_close(branches[2], other[2], rtol=0, atol=0)
                self.assertGreater(float((branches[1] - other[1]).abs().max()), 1e-6)
            q = net(base)
            selected = net.forward_selected_actions(base, torch.arange(ACTION_SPACE_SIZE).unsqueeze(0))
            torch.testing.assert_close(q, selected, rtol=2e-5, atol=2e-6)
            dirty = {key: value.clone() for key, value in base.items()}
            dirty["zombie_row"][0, 0] = 99
            dirty["zombies"][0, 0, :2] = float("nan")
            torch.testing.assert_close(q, net(dirty), rtol=0, atol=0)
        net(base).square().mean().backward()
        self.assertGreater(float(net.type_embedding.weight.grad[26].abs().sum()), 0)

    def test_boss_damage_ignores_mowers_but_normal_damage_does_not(self):
        for reported in (1, 2):  # Second case lacks a complete mower-row list.
            initial = [zombie(1, 1000, kind=25, row=0, x=0), zombie(2, 270, row=0, x=100)]
            after = [zombie(1, 900, kind=25, row=0, x=800), zombie(2, 170, row=0, x=50)]
            tracker = TransitionDenseReward(info(board(health=initial, mower_states=["READY"]), 0))
            details = info(board(health=after, mower_states=["ROLLING"]), 150, mowers=reported)
            details["events"]["mower_triggered_rows"] = [0]
            tracker.step(0, details, terminated=False, truncated=False)
            self.assertEqual(tracker.last_breakdown["damage_delta_hp"], 100)
            self.assertEqual(tracker.last_breakdown["threat_reward"], 0)
            self.assertGreater(tracker.last_breakdown["damage_reward"], 0)

    def test_boss_motion_death_and_fallback_do_not_award_threat(self):
        initial = board(health=[zombie(1, 1000, kind=25, row=0, x=0)])
        for terminal, fallback in ((False, False), (True, False), (False, True)):
            tracker = TransitionDenseReward(info(initial, 0))
            after = board(health=[zombie(1, 0, kind=25, row=4, x=900, dead=True, phase=1)])
            tracker.step(0, info(after, 150, result="WON" if terminal else "PLAYING", fallback=fallback),
                         terminated=terminal, truncated=False)
            self.assertEqual(tracker.last_breakdown["threat_reward"], 0)
            if terminal or fallback:
                self.assertEqual(tracker.last_breakdown["damage_delta_hp"], 0)

    def test_known_checkpoint_migration_retains_state_and_skips_old_replay(self):
        config = DoubleDQNConfig(device="cpu", hidden_sizes=(16, 8), batch_size=2, replay_capacity=8,
                                 reward_scheme="transition_dense")
        source = DoubleDQNAgent(ACTION_SPACE_SIZE, config)
        source.update_count = 7
        first_parameter = next(source.online.parameters())
        first_parameter.grad = torch.ones_like(first_parameter)
        source.optimizer.step()
        with tempfile.TemporaryDirectory() as tmp:
            path = source.save(Path(tmp) / "old.pt", env_step=30, episode=2,
                               episode_rng_state=np.random.default_rng(0).bit_generator.state)
            saved = torch.load(path, weights_only=False)
            saved["observation_schema"] = PREVIOUS_OBSERVATION_SCHEMA + "_transition_dense_context_v1"
            saved["replay_file"] = "unused.npz"
            declare_checkpoint(saved, replay={})
            torch.save(saved, path)
            with self.assertRaisesRegex(ValueError, "observation schema"):
                DoubleDQNAgent.from_checkpoint(path, device="cpu", strict_resume=True)
            loaded, _, payload = DoubleDQNAgent.from_checkpoint(path, device="cpu")
            self.assertEqual(loaded.update_count, 7)
            self.assertEqual(payload["env_step"], 30)
            self.assertIn("observation_migration", payload)
            for key, value in source.online.state_dict().items():
                torch.testing.assert_close(loaded.online.state_dict()[key], value)
            source_state = source.optimizer.state_dict()["state"]
            loaded_state = loaded.optimizer.state_dict()["state"]
            self.assertEqual(source_state.keys(), loaded_state.keys())
            for param_id, values in source_state.items():
                for key, value in values.items():
                    torch.testing.assert_close(loaded_state[param_id][key], value)
            events = []
            controller = DoubleDQNTrainerController(config, run_dir=tmp, checkpoint_path=path,
                checkpoint_load_mode="resume", eval_every_episodes=0,
                on_event=lambda event, data: events.append((event, data)))
            controller._agent, step, episode, _ = controller._build_agent(ACTION_SPACE_SIZE)
            self.assertEqual((step, episode), (30, 2))
            replay = controller._restore_replay(action_dim=ACTION_SPACE_SIZE, start_step=30, start_episode=2)
            self.assertEqual(len(replay), 0)
            self.assertFalse(controller._summary.replay_restored)
            self.assertTrue(any(event == "observation_migration" for event, _ in events))
            self.assertTrue(any(data.get("warmup_restarted") for _, data in events))

    def test_old_reward_metadata_cannot_mix_into_current_replay(self):
        replay = ReplayBuffer(1, ACTION_SPACE_SIZE, reward_context=True)
        replay.reward_settings = effective_reward_settings(DoubleDQNConfig(reward_scheme="transition_dense"))
        replay.reward_settings.pop("boss_basis")
        replay.reward_settings["metrics_version"] = 3
        replay.reward_settings["version"] = 4
        with tempfile.TemporaryDirectory() as tmp:
            path = replay.save(Path(tmp) / "old_reward.npz")
            with self.assertRaisesRegex(ValueError, "transition_dense replay reward settings"):
                ReplayBuffer.load(path)


if __name__ == "__main__":
    unittest.main()
