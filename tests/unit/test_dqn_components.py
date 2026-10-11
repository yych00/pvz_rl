import tempfile
import unittest
from pathlib import Path

import numpy as np

from pvz_gym.action_space import (
    ACTION_SPACE_SIZE,
    COB_BASE,
    INTERACT_BASE,
    PLANT_BASE,
    SHOVEL_NORMAL_BASE,
)
from pvz_gym.observation import OBSERVATION_SPEC, empty_observation
from train.common import EnvConfig
from train.dqn import agent as agent_module
from train.dqn.agent import DoubleDQNAgent
from train.dqn.config import DoubleDQNConfig
from train.dqn.evaluate import EvaluationSummary
from train.dqn.exploration import (
    ACTION_FAMILY_RANGES,
    parse_action_family_weights,
    sample_balanced_legal_action,
)
from train.dqn.features import normalize_observation, validate_observation
from train.dqn.networks import NETWORK_SPATIAL, SpatialQNetwork
from train.dqn.replay import ReplayBuffer
from train.dqn.reward import training_reward


class DoubleDQNComponentTest(unittest.TestCase):
    def test_evaluation_summary_reports_win_rate_score_and_stop_state(self):
        summary = EvaluationSummary(
            requested_episodes=10,
            episodes=4,
            wins=3,
            losses=1,
            unknown=0,
            total_score=6.0,
            total_steps=100,
            stopped=True,
            tick_samples=4,
            ticks_advanced_sum=136,
            ticks_advanced_sq_sum=4640,
            long_tick_jumps=1,
        )
        self.assertAlmostEqual(summary.win_rate, 0.75)
        self.assertAlmostEqual(summary.mean_score, 1.5)
        self.assertAlmostEqual(summary.mean_steps, 25.0)
        self.assertAlmostEqual(summary.mean_ticks_advanced, 34.0)
        self.assertAlmostEqual(summary.std_ticks_advanced, 2.0)
        self.assertAlmostEqual(summary.long_tick_jump_rate, 0.25)
        self.assertTrue(summary.stopped)

    def test_model_ready_observation_path_is_zero_copy_and_debug_validation_is_explicit(self):
        observation = empty_observation()
        normalized = normalize_observation(observation)
        self.assertIs(normalized, observation)
        self.assertEqual(set(normalized), set(OBSERVATION_SPEC))
        validate_observation(normalized)

        malformed = dict(observation)
        malformed["global"] = np.zeros((1,), dtype=observation["global"].dtype)
        with self.assertRaisesRegex(ValueError, "unexpected global shape"):
            validate_observation(malformed)

    def test_replay_preserves_terminal_mask_and_promotes_samples_to_float32(self):
        replay = ReplayBuffer(capacity=4, action_dim=5, seed=7)
        observation = empty_observation()
        next_observation = empty_observation()
        observation["global"][0] = 1.0
        next_observation["global"][0] = 2.0
        terminal_mask = np.zeros(5, dtype=np.bool_)
        replay.add(observation, 2, 1.0, next_observation, True, terminal_mask, ticks_advanced=150)

        batch = replay.sample(1)
        self.assertEqual(batch.observations["global"].dtype, np.float32)
        self.assertEqual(batch.observations["plant_type"].dtype, np.int64)
        self.assertEqual(batch.next_observations["global"].dtype, np.float32)
        self.assertEqual(int(batch.actions[0]), 2)
        self.assertTrue(bool(batch.dones[0]))
        self.assertFalse(bool(batch.next_action_masks[0].any()))

    def test_long_horizon_default_and_reward_scales_are_explicit(self):
        config = DoubleDQNConfig(reward_scheme="staged")
        self.assertGreater(config.gamma, 0.99)
        self.assertEqual(config.gamma, 0.999)
        self.assertEqual(config.gamma_reference_ticks, 100)
        self.assertAlmostEqual(config.reward_scale, 0.01)
        self.assertAlmostEqual(config.mower_penalty, 0.1)
        self.assertEqual(config.env.level_id, 4)
        self.assertEqual(config.env.simulation_mode, "headless")

    def test_training_simulation_mode_can_still_be_overridden(self):
        stepped = DoubleDQNConfig(reward_scheme="staged", env=EnvConfig(simulation_mode="stepped"))
        continuous = DoubleDQNConfig(reward_scheme="staged", env=EnvConfig(simulation_mode="continuous"))
        self.assertEqual(stepped.env.simulation_mode, "stepped")
        self.assertEqual(continuous.env.simulation_mode, "continuous")

    def test_mower_penalty_is_immediate_and_additive(self):
        # mower_penalty 字段仅对 sparse scheme 生效；balanced 有自己的 mower_used=-0.20
        sparse_config = DoubleDQNConfig(reward_scheme="sparse")
        self.assertAlmostEqual(training_reward(0.0, {"mowers_used": 1}, sparse_config), -0.1)
        self.assertAlmostEqual(training_reward(0.0, {"mowers_used": 2}, sparse_config), -0.2)
        self.assertAlmostEqual(training_reward(100.0, {"mowers_used": 1}, sparse_config), 0.9)
        self.assertAlmostEqual(training_reward(-100.0, {"mowers_used": 1}, sparse_config), -1.1)
        self.assertAlmostEqual(training_reward(100.0, {"mowers_used": 0}, sparse_config), 1.0)
        # balanced 方案使用自己的 mower_used=-0.20，与 mower_penalty 字段无关
        balanced_config = DoubleDQNConfig(reward_scheme="balanced")
        self.assertAlmostEqual(training_reward(0.0, {"mowers_used": 1}, balanced_config), -0.2)
        self.assertAlmostEqual(training_reward(0.0, {"mowers_used": 2}, balanced_config), -0.4)

    def test_invalid_training_capacity_and_optimizer_values_fail_fast(self):
        with self.assertRaisesRegex(ValueError, "replay_capacity must be >= batch_size"):
            DoubleDQNConfig(reward_scheme="staged", replay_capacity=32, batch_size=64)
        with self.assertRaisesRegex(ValueError, "learning_starts must be non-negative"):
            DoubleDQNConfig(reward_scheme="staged", learning_starts=-1)
        with self.assertRaisesRegex(ValueError, "learning_rate must be positive"):
            DoubleDQNConfig(reward_scheme="staged", learning_rate=0.0)
        with self.assertRaisesRegex(ValueError, "max_grad_norm must be positive"):
            DoubleDQNConfig(reward_scheme="staged", max_grad_norm=0.0)
        with self.assertRaisesRegex(ValueError, "checkpoint_every must be non-negative"):
            DoubleDQNConfig(reward_scheme="staged", checkpoint_every=-1)
        with self.assertRaisesRegex(ValueError, "five comma-separated"):
            parse_action_family_weights("1,1,1")
        with self.assertRaisesRegex(ValueError, "at least one"):
            DoubleDQNConfig(reward_scheme="staged", exploration_type_weights=(0.0, 0.0, 0.0, 0.0, 0.0))

    def test_epsilon_exploration_balances_action_families_not_flat_action_count(self):
        mask = np.zeros(ACTION_SPACE_SIZE, dtype=np.bool_)
        legal_counts = (1, 200, 2, 54, 54)
        for count, (start, end) in zip(legal_counts, ACTION_FAMILY_RANGES):
            mask[start:min(start + count, end)] = True

        rng = np.random.default_rng(7)
        family_counts = np.zeros(5, dtype=np.int64)
        for _ in range(5000):
            action = sample_balanced_legal_action(mask, rng)
            for family, (start, end) in enumerate(ACTION_FAMILY_RANGES):
                if start <= action < end:
                    family_counts[family] += 1
                    break

        self.assertTrue(np.all(family_counts > 850), family_counts)
        self.assertTrue(np.all(family_counts < 1150), family_counts)

    def test_epsilon_exploration_respects_configured_type_weights(self):
        mask = np.zeros(ACTION_SPACE_SIZE, dtype=np.bool_)
        mask[0] = True
        mask[SHOVEL_NORMAL_BASE] = True
        weights = parse_action_family_weights("0,0,1,0,0")
        rng = np.random.default_rng(11)

        actions = [sample_balanced_legal_action(mask, rng, weights) for _ in range(50)]

        self.assertEqual(set(actions), {SHOVEL_NORMAL_BASE})

    def test_checkpoint_persists_instantiated_network_architecture(self):
        if agent_module.torch is None:
            self.skipTest("PyTorch is optional in core CI")

        original = DoubleDQNConfig(reward_scheme="staged",
            hidden_sizes=(32, 16),
            replay_capacity=64,
            batch_size=16,
            device="cpu",
        )
        agent = DoubleDQNAgent(ACTION_SPACE_SIZE, original, device="cpu")

        agent.config = DoubleDQNConfig(reward_scheme="staged",
            hidden_sizes=(512, 256),
            replay_capacity=64,
            batch_size=16,
            device="cpu",
        )

        with tempfile.TemporaryDirectory() as tmp:
            training_summary = {"wins": 2, "losses": 1, "win_rate": 2 / 3}
            path = agent.save(
                Path(tmp) / "checkpoint.pt",
                env_step=12,
                episode=3,
                training_summary=training_summary,
            )
            try:
                payload = agent_module.torch.load(path, map_location="cpu", weights_only=False)
            except TypeError:
                payload = agent_module.torch.load(path, map_location="cpu")
            self.assertEqual(tuple(payload["config"]["hidden_sizes"]), (32, 16))
            self.assertEqual(payload["training_summary"], training_summary)

    @unittest.skipIf(agent_module.torch is None, "PyTorch is required for the spatial Q test")
    def test_legacy_observation_checkpoint_is_explicitly_rejected(self):
        agent = DoubleDQNAgent(
            ACTION_SPACE_SIZE,
            DoubleDQNConfig(reward_scheme="staged", hidden_sizes=(32, 16)),
            device="cpu",
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = agent.save(Path(tmp) / "legacy.pt", env_step=0, episode=0)
            payload = agent_module.torch.load(path, map_location="cpu", weights_only=False)
            payload["observation_schema"] = "pvz_observation"
            agent_module.torch.save(payload, path)
            with self.assertRaisesRegex(ValueError, "checkpoint observation schema"):
                DoubleDQNAgent.from_checkpoint(path, device="cpu")

    @unittest.skipIf(agent_module.torch is None, "PyTorch is required for the spatial Q test")
    def test_spatial_q_network_consumes_structured_gym_contract(self):
        network = SpatialQNetwork(hidden_sizes=(64, 32))
        observations = {
            name: agent_module.torch.as_tensor(np.stack([value, value]))
            for name, value in empty_observation().items()
        }

        q_values = network(observations)

        self.assertEqual(tuple(q_values.shape), (2, ACTION_SPACE_SIZE))
        self.assertTrue(bool(agent_module.torch.isfinite(q_values).all()))
        q_values[:, (0, PLANT_BASE, SHOVEL_NORMAL_BASE, COB_BASE, INTERACT_BASE)].sum().backward()
        self.assertTrue(
            all(parameter.grad is not None for parameter in network.parameters())
        )

    @unittest.skipIf(agent_module.torch is None, "PyTorch is required for the spatial Q test")
    def test_native_agent_checkpoints_spatial_network_type(self):
        config = DoubleDQNConfig(reward_scheme="staged",
            credit_assignment="one_step",
            hidden_sizes=(64, 32),
            replay_capacity=16,
            batch_size=4,
            epsilon_start=1.0,
            epsilon_end=1.0,
            exploration_mode="epsilon_random",
            exploration_type_weights=(0.0, 0.0, 1.0, 0.0, 0.0),
            device="cpu",
        )
        agent = DoubleDQNAgent(
            ACTION_SPACE_SIZE,
            config,
            device="cpu",
        )
        self.assertEqual(agent.network_type, NETWORK_SPATIAL)
        mask = np.zeros(ACTION_SPACE_SIZE, dtype=np.bool_)
        mask[0] = True
        mask[SHOVEL_NORMAL_BASE] = True
        action = agent.select_action(
            empty_observation(),
            mask,
            env_step=0,
        )
        self.assertEqual(action, SHOVEL_NORMAL_BASE)

        with tempfile.TemporaryDirectory() as tmp:
            path = agent.save(Path(tmp) / "spatial.pt", env_step=0, episode=0)
            restored, _config, payload = DoubleDQNAgent.from_checkpoint(path, device="cpu")

        self.assertEqual(payload["network_type"], NETWORK_SPATIAL)
        self.assertEqual(restored.network_type, NETWORK_SPATIAL)
        self.assertIsInstance(restored.online, SpatialQNetwork)


if __name__ == "__main__":
    unittest.main()
