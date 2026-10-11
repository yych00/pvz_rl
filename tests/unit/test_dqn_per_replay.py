"""优先经验回放 (PER) 与动态热切换的全面单元与集成测试。"""

from __future__ import annotations

import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np

from pvz_gym.action_space import ACTION_SPACE_SIZE
from pvz_gym.observation import empty_observation
from train.dqn.agent import DoubleDQNAgent
from train.dqn.config import DoubleDQNConfig
from train.dqn.controller import DoubleDQNTrainerController
from train.dqn.replay import ReplayBatch, ReplayBuffer
from train.dqn.sum_tree import SumTree


class DQNPERReplayTest(unittest.TestCase):
    def setUp(self):
        self.capacity = 20
        self.action_dim = ACTION_SPACE_SIZE
        self.mask = np.ones(self.action_dim, dtype=np.bool_)

    def _fill_buffer(self, replay: ReplayBuffer, count: int):
        for i in range(count):
            obs = empty_observation()
            obs["global"][0] = float(i * 25)
            next_obs = empty_observation()
            next_obs["global"][0] = float((i + 1) * 25)
            replay.add(
                obs,
                action=i % self.action_dim,
                reward=float(i),
                next_observation=next_obs,
                done=False,
                next_action_mask=self.mask,
                ticks_advanced=10,
            )

    def test_per_initialization_and_weights(self):
        """测试 PER 初始化、新增样本的最高优先级赋予与初始采样。"""
        replay = ReplayBuffer(
            self.capacity,
            self.action_dim,
            seed=42,
            scheme="per",
            alpha=0.6,
            beta=0.4,
        )
        self.assertTrue(replay.is_per)
        self.assertEqual(len(replay), 0)

        self._fill_buffer(replay, 10)
        self.assertEqual(len(replay), 10)
        self.assertIsNotNone(replay._tree)
        # 10 个样本初始优先级均为 1.0 ** 0.6 = 1.0，总和为 10.0
        self.assertAlmostEqual(replay._tree.total, 10.0)

        # 采样 4 个样本
        batch = replay.sample(4, beta=0.4)
        self.assertIsInstance(batch, ReplayBatch)
        self.assertEqual(len(batch.actions), 4)
        self.assertIsNotNone(batch.indices)
        self.assertIsNotNone(batch.weights)
        self.assertEqual(len(batch.indices), 4)
        self.assertEqual(len(batch.weights), 4)
        # 重要性采样权重经 max 归一化，最大值应为 1.0
        self.assertAlmostEqual(float(np.max(batch.weights)), 1.0, places=5)
        self.assertTrue((batch.weights > 0).all())

    def test_n_step_per_uses_only_eligible_priorities_and_matching_weights(self):
        for count in (4, 6):  # Include a wrapped ring (oldest is index 2).
            replay = ReplayBuffer(4, self.action_dim, seed=42, scheme="per", alpha=1, beta=1)
            self._fill_buffer(replay, count)
            oldest = replay._position
            order = (oldest + np.arange(4)) % 4
            replay.update_priorities(order, [1, 9, 1000, 1000])
            before = replay._tree.get_priorities(np.arange(4))
            counts = np.zeros(4, dtype=int)
            mixed = 0
            for _ in range(500):
                batch = replay.sample(2, n_step=3, beta=1)
                np.add.at(counts, batch.indices, 1)
                self.assertTrue(np.isin(batch.indices, order[:2]).all())
                if len(np.unique(batch.indices)) == 2:
                    mixed += 1
                    expected = np.power(before[batch.indices], -1)
                    expected /= expected.max()
                    np.testing.assert_allclose(batch.weights, expected, rtol=1e-6)
            self.assertGreater(mixed, 0)
            self.assertAlmostEqual(counts[order[0]] / counts.sum(), 0.1, delta=0.025)
            np.testing.assert_array_equal(before, replay._tree.get_priorities(np.arange(4)))
            replay.end_episode()
            self.assertEqual(replay._eligible_count(3), 4)
            batch = replay.sample(4, n_step=3)
            self.assertTrue(np.isin(batch.indices, order[2:]).all())

    def test_sum_tree_zero_target_skips_masked_leaves(self):
        tree = SumTree(4)
        tree.set_all([0, 0, 1, 9], 4)
        indices, _ = tree.sample([0, 1, np.nextafter(10., 0.)], 4)
        np.testing.assert_array_equal(indices, [2, 3, 3])

    def test_n_step_mask_is_restored_when_sampling_raises(self):
        replay = ReplayBuffer(4, self.action_dim, scheme="per")
        self._fill_buffer(replay, 4)
        before = replay._tree.get_priorities(np.arange(4))
        with patch.object(replay._tree, "sample", side_effect=RuntimeError("sampling failed")):
            with self.assertRaisesRegex(RuntimeError, "sampling failed"):
                replay.sample(2, n_step=3)
        np.testing.assert_array_equal(before, replay._tree.get_priorities(np.arange(4)))

    def test_alpha_round_trip_and_eps_update_rebuild_existing_priorities(self):
        replay = ReplayBuffer(4, self.action_dim, scheme="per", alpha=1, eps=0.1)
        self._fill_buffer(replay, 4)
        replay.update_priorities(np.arange(4), [1, 2, 3, 4])
        replay.set_scheme("per", alpha=0)
        np.testing.assert_allclose(replay._tree.get_priorities(np.arange(4)), 1)
        replay.set_scheme("uniform", alpha=0.5, eps=0.2)
        replay.set_scheme("per")
        np.testing.assert_allclose(replay._tree.get_priorities(np.arange(4)), np.sqrt([1.2, 2.2, 3.2, 4.2]))

    def test_controller_parameter_only_updates_replay_and_rejects_invalid_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            controller = DoubleDQNTrainerController(DoubleDQNConfig(reward_scheme="staged", replay_scheme="per"), run_dir=Path(tmp))
            controller._replay = ReplayBuffer(4, self.action_dim, scheme="per")
            controller.set_replay_settings(alpha=0.2, beta=0.7, eps=0.123)
            self.assertEqual(controller._replay.alpha, 0.2)
            self.assertEqual(controller._replay.beta, 0.7)
            self.assertEqual(controller._replay.eps, 0.123)
            with self.assertRaises(ValueError):
                controller.set_replay_settings(alpha=2)
            self.assertEqual(controller.config.per_alpha, 0.2)
            self.assertEqual(controller._replay.alpha, 0.2)

    def test_custom_eps_and_raw_priorities_survive_wrapped_resize_and_alpha_zero(self):
        with tempfile.TemporaryDirectory() as tmp:
            replay = ReplayBuffer(4, self.action_dim, scheme="per", alpha=0, eps=0.123)
            self._fill_buffer(replay, 6)
            replay.update_priorities(np.arange(4), [1, 2, 3, 4])
            path = replay.save(Path(tmp) / "replay.npz")
            restored, _ = ReplayBuffer.load(path, capacity=3)
            self.assertEqual(restored.eps, 0.123)
            restored.set_scheme("per", alpha=1)
            np.testing.assert_allclose(restored._tree.get_priorities(np.arange(3)), [4.123, 1.123, 2.123])

    def test_legacy_per_priorities_recover_raw_values(self):
        with tempfile.TemporaryDirectory() as tmp:
            replay = ReplayBuffer(4, self.action_dim, scheme="per", alpha=0.5)
            self._fill_buffer(replay, 4)
            replay.update_priorities(np.arange(4), [1, 4, 9, 16])
            path = replay.save(Path(tmp) / "replay.npz")
            with np.load(path) as payload:
                legacy = {k: payload[k] for k in payload.files if k not in ("per_eps", "per_raw_priorities")}
            np.savez(path, **legacy)
            restored, _ = ReplayBuffer.load(path)
            restored.set_scheme("per", alpha=1)
            np.testing.assert_allclose(restored._tree.get_priorities(np.arange(4)), np.array([1, 4, 9, 16]) + 1e-5)

    def test_controller_resume_applies_config_eps_for_legacy_per_snapshot(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            replay = ReplayBuffer(4, self.action_dim, scheme="per", alpha=0.6)
            self._fill_buffer(replay, 4)
            replay.save(root / "replay.npz", env_step=10, episode=2)
            with np.load(root / "replay.npz") as payload:
                legacy = {k: payload[k] for k in payload.files if k not in ("per_eps", "per_raw_priorities")}
            np.savez(root / "replay.npz", **legacy)
            config = DoubleDQNConfig(reward_scheme="staged", replay_scheme="per", replay_capacity=4, batch_size=2, per_eps=0.123)
            controller = DoubleDQNTrainerController(config, run_dir=root, checkpoint_path=root / "latest.pt")
            restored = controller._restore_replay(action_dim=self.action_dim, start_step=10, start_episode=2)
            self.assertTrue(controller.summary.replay_restored)
            self.assertEqual(restored.eps, 0.123)

    def test_importance_weights_scale_huber_loss_but_not_td_errors(self):
        import torch

        class FixedQ(torch.nn.Module):
            def __init__(self, action_dim):
                super().__init__()
                self.q = torch.nn.Parameter(torch.zeros(action_dim))

            def forward(self, observations):
                return self.q.expand(observations["global"].shape[0], -1)

        replay = ReplayBuffer(4, self.action_dim)
        self._fill_buffer(replay, 2)
        batch = replace(replay.sample(2), rewards=np.array([1., 2.], dtype=np.float32),
                        dones=np.ones(2, dtype=bool), weights=np.array([1., .25], dtype=np.float32))
        agent = DoubleDQNAgent(self.action_dim, DoubleDQNConfig(reward_scheme="staged"), device="cpu")
        agent.online = FixedQ(self.action_dim)
        agent.target = FixedQ(self.action_dim)
        loss, errors = agent._one_step_loss(batch, return_td_errors=True)
        self.assertAlmostEqual(float(loss.detach()), (0.5 + 1.5 * .25) / 2)
        np.testing.assert_array_equal(errors.numpy(), [1., 2.])

    def test_per_priority_updates_bias_sampling(self):
        """测试优先级更新后，高 TD-error 样本被采样概率显著提高。"""
        replay = ReplayBuffer(
            self.capacity,
            self.action_dim,
            seed=123,
            scheme="per",
            alpha=1.0,  # 线性采样以便统计检验
            beta=0.5,
        )
        self._fill_buffer(replay, 10)

        # 将索引 3 的优先级更新为极大值 1000.0，其余保持接近 0 (0.01)
        low_prios = np.full(10, 0.01, dtype=np.float64)
        replay.update_priorities(np.arange(10), low_prios)
        replay.update_priorities([3], [1000.0])

        self.assertGreater(replay.max_priority, 900.0)

        # 采样 100 次，统计索引 3 出现的次数
        hit_target = 0
        total_samples = 200
        for _ in range(total_samples // 4):
            batch = replay.sample(4)
            hit_target += int(np.count_nonzero(batch.indices == 3))

        # 权重占比 > 99%，目标样本采样频次应占绝大多数
        self.assertGreater(hit_target / total_samples, 0.90)

    def test_dynamic_scheme_hot_switching(self):
        """测试暂停时的动态平滑热切换 (Uniform -> PER -> Uniform)。"""
        replay = ReplayBuffer(self.capacity, self.action_dim, seed=77, scheme="uniform")
        self.assertFalse(replay.is_per)
        self.assertIsNone(replay._tree)

        self._fill_buffer(replay, 12)
        batch_u1 = replay.sample(4)
        # uniform 模式下 weights 均为全 1
        np.testing.assert_allclose(batch_u1.weights, np.ones(4, dtype=np.float32))

        # 热切换到 PER
        replay.set_scheme("per", alpha=0.7, beta=0.5)
        self.assertTrue(replay.is_per)
        self.assertIsNotNone(replay._tree)
        # 存量样本被正确赋予默认最大权重 1.0 ** 0.7 = 1.0
        self.assertAlmostEqual(replay._tree.total, 12.0)

        # 切换后再插入新样本
        self._fill_buffer(replay, 3)
        self.assertEqual(len(replay), 15)
        self.assertAlmostEqual(replay._tree.total, 15.0)

        # 采样验证 PER 权重有效
        batch_per = replay.sample(5, beta=0.6)
        self.assertIsNotNone(batch_per.indices)
        self.assertIsNotNone(batch_per.weights)
        self.assertEqual(len(batch_per.weights), 5)
        self.assertAlmostEqual(float(np.max(batch_per.weights)), 1.0, places=5)

        # 热切换回 Uniform
        replay.set_scheme("uniform")
        self.assertFalse(replay.is_per)
        batch_u2 = replay.sample(4)
        np.testing.assert_allclose(batch_u2.weights, np.ones(4, dtype=np.float32))

    def test_per_save_and_load_persistence(self):
        """测试 PER 回放缓存的持久化与无损恢复。"""
        with tempfile.TemporaryDirectory() as tmp:
            save_path = Path(tmp) / "per_replay.npz"
            replay = ReplayBuffer(
                self.capacity,
                self.action_dim,
                seed=999,
                scheme="per",
                alpha=0.6,
                beta=0.45,
            )
            self._fill_buffer(replay, 8)
            # 自定义不同优先级
            custom_prios = np.array([0.1, 0.5, 2.0, 1.5, 0.8, 3.2, 0.4, 1.1])
            replay.update_priorities(np.arange(8), custom_prios)
            expected_total = replay._tree.total

            replay.save(save_path, env_step=200, episode=5)

            # 恢复
            restored, meta = ReplayBuffer.load(save_path)
            self.assertEqual(meta["env_step"], 200)
            self.assertEqual(meta["episode"], 5)
            self.assertEqual(meta["replay_scheme"], "per")
            self.assertTrue(restored.is_per)
            self.assertEqual(restored.scheme, "per")
            self.assertAlmostEqual(restored.alpha, 0.6)
            self.assertAlmostEqual(restored.beta, 0.45)
            self.assertIsNotNone(restored._tree)
            self.assertAlmostEqual(restored._tree.total, expected_total, places=5)
            np.testing.assert_allclose(
                restored._tree.get_priorities(np.arange(8)),
                replay._tree.get_priorities(np.arange(8)),
                rtol=1e-5,
            )

    def test_legacy_uniform_checkpoint_backward_compatibility(self):
        """测试旧版本（无 replay_scheme 字段）存档向下兼容作为 uniform 正确恢复。"""
        with tempfile.TemporaryDirectory() as tmp:
            legacy_path = Path(tmp) / "legacy_replay.npz"
            replay = ReplayBuffer(self.capacity, self.action_dim, seed=1, scheme="uniform")
            self._fill_buffer(replay, 6)
            replay.save(legacy_path, env_step=50, episode=2)

            # 模拟旧版存档：移除可能存在的 replay_scheme
            with np.load(legacy_path) as payload:
                data = {k: payload[k] for k in payload.files if not k.startswith("per_") and k != "replay_scheme"}
            np.savez_compressed(legacy_path, **data)

            restored, meta = ReplayBuffer.load(legacy_path)
            self.assertNotIn("replay_scheme", meta)
            self.assertFalse(restored.is_per)
            self.assertEqual(restored.scheme, "uniform")
            self.assertIsNone(restored._tree)

            # 旧存档恢复后，可通过 set_scheme 平滑升级为 PER
            restored.set_scheme("per", alpha=0.6, beta=0.4)
            self.assertTrue(restored.is_per)
            self.assertAlmostEqual(restored._tree.total, 6.0)

    def test_agent_weighted_loss_and_td_errors(self):
        """测试 DoubleDQNAgent 正确计算重要性采样加权损失并输出 TD-errors。"""
        config = DoubleDQNConfig(reward_scheme="staged", seed=42, replay_scheme="per")
        agent = DoubleDQNAgent(self.action_dim, config, device="cpu")

        replay = ReplayBuffer(self.capacity, self.action_dim, seed=42, scheme="per")
        self._fill_buffer(replay, 6)
        batch = replay.sample(4)

        loss, td_errors = agent.learn(batch, return_td_errors=True)
        self.assertIsInstance(loss, float)
        self.assertTrue(np.isfinite(loss))
        self.assertIsInstance(td_errors, np.ndarray)
        self.assertEqual(td_errors.shape, (4,))
        self.assertTrue((td_errors >= 0).all())
        self.assertIsNotNone(agent.last_td_errors)
        np.testing.assert_allclose(td_errors, agent.last_td_errors)

        # 默认 learn(batch) 依然向后兼容返回单个 float
        float_loss = agent.learn(batch)
        self.assertIsInstance(float_loss, float)
        self.assertTrue(np.isfinite(float_loss))

    def test_controller_per_hot_switching_and_beta_annealing(self):
        """测试 Controller 的 set_replay_scheme 及动态配置更新。"""
        config = DoubleDQNConfig(reward_scheme="staged",
            replay_scheme="uniform",
            per_alpha=0.6,
            per_beta_start=0.4,
            per_beta_end=1.0,
            per_beta_steps=1000,
        )
        controller = DoubleDQNTrainerController(config)
        controller._replay = ReplayBuffer(10, self.action_dim, scheme="uniform")

        self.assertEqual(controller.config.replay_scheme, "uniform")
        self.assertFalse(controller._replay.is_per)

        # 模拟暂停时切换为 PER
        controller.set_replay_scheme("per", alpha=0.8, beta=0.5)
        self.assertEqual(controller.config.replay_scheme, "per")
        self.assertAlmostEqual(controller.config.per_alpha, 0.8)
        self.assertAlmostEqual(controller.config.per_beta_start, 0.5)
        self.assertTrue(controller._replay.is_per)
        self.assertAlmostEqual(controller._replay.alpha, 0.8)
        self.assertAlmostEqual(controller._replay.beta, 0.5)


if __name__ == "__main__":
    unittest.main()
