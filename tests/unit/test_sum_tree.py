"""SumTree 单元测试与向量化性能测试。"""

from __future__ import annotations

import time
import unittest
import numpy as np

from train.dqn.sum_tree import SumTree


class SumTreeTest(unittest.TestCase):
    def test_single_update_and_total(self):
        tree = SumTree(capacity=8)
        self.assertEqual(tree.total, 0.0)
        tree.update(0, 1.5)
        self.assertAlmostEqual(tree.total, 1.5)
        tree.update(1, 2.5)
        self.assertAlmostEqual(tree.total, 4.0)
        tree.update(0, 0.5)
        self.assertAlmostEqual(tree.total, 3.0)
        self.assertAlmostEqual(tree.get_priority(0), 0.5)
        self.assertAlmostEqual(tree.get_priority(1), 2.5)

    def test_update_batch_matches_single_updates(self):
        capacity = 100
        tree_single = SumTree(capacity)
        tree_batch = SumTree(capacity)

        rng = np.random.default_rng(42)
        indices = rng.choice(capacity, size=30, replace=True)
        priorities = rng.uniform(0.1, 5.0, size=30)

        # 单点逐步更新
        for idx, prio in zip(indices, priorities):
            tree_single.update(int(idx), float(prio))

        # 批量向量化更新
        tree_batch.update_batch(indices, priorities)

        self.assertAlmostEqual(tree_single.total, tree_batch.total)
        np.testing.assert_allclose(
            tree_single.get_priorities(np.arange(capacity)),
            tree_batch.get_priorities(np.arange(capacity)),
        )

    def test_set_all_rebuilds_exact_tree(self):
        capacity = 64
        tree = SumTree(capacity)
        priorities = np.full(capacity, 2.0)
        tree.set_all(priorities, size=50)

        self.assertAlmostEqual(tree.total, 100.0)
        self.assertAlmostEqual(tree.get_priority(0), 2.0)
        self.assertAlmostEqual(tree.get_priority(49), 2.0)
        self.assertAlmostEqual(tree.get_priority(50), 0.0)

    def test_sample_retrieves_exact_intervals(self):
        # 构造简单权重 [1.0, 2.0, 3.0, 4.0]
        # 前缀和边界为:
        # [0, 1) -> 0
        # [1, 3) -> 1
        # [3, 6) -> 2
        # [6, 10) -> 3
        tree = SumTree(4)
        tree.update_batch([0, 1, 2, 3], [1.0, 2.0, 3.0, 4.0])
        self.assertAlmostEqual(tree.total, 10.0)

        targets = [0.5, 1.5, 2.9, 3.1, 5.9, 6.1, 9.9]
        expected_indices = [0, 1, 1, 2, 2, 3, 3]
        expected_priorities = [1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0]

        indices, priorities = tree.sample(targets, active_size=4)
        np.testing.assert_array_equal(indices, expected_indices)
        np.testing.assert_allclose(priorities, expected_priorities)

    def test_sample_bounds_and_clamping(self):
        tree = SumTree(8)
        tree.update_batch([0, 1], [5.0, 5.0])
        # targets 越界或极端值
        targets = [0.0, 4.99, 5.01, 10.0, 12.0]
        indices, _ = tree.sample(targets, active_size=2)
        self.assertTrue((indices >= 0).all())
        self.assertTrue((indices < 2).all())

    def test_performance_under_50k_capacity(self):
        capacity = 50_000
        tree = SumTree(capacity)
        # 初始化 50,000 个非零叶子节点
        rng = np.random.default_rng(123)
        priorities = rng.uniform(0.1, 10.0, size=capacity)
        tree.set_all(priorities, capacity)

        batch_size = 64
        # 预热
        for _ in range(10):
            targets = rng.uniform(0, tree.total, size=batch_size)
            indices, _ = tree.sample(targets, capacity)
            tree.update_batch(indices, rng.uniform(0.1, 5.0, size=batch_size))

        # 测试 100 次批量采样的总耗时
        t0 = time.perf_counter()
        iterations = 100
        for _ in range(iterations):
            targets = rng.uniform(0, tree.total, size=batch_size)
            indices, _ = tree.sample(targets, capacity)
        sample_time_ms = (time.perf_counter() - t0) * 1000.0 / iterations

        # 每次 Batch=64 采样耗时必须 < 0.5ms (通常在 0.05ms ~ 0.1ms 左右)
        self.assertLess(sample_time_ms, 0.5, f"sample_time_ms {sample_time_ms:.4f}ms is too slow")

        # 测试 100 次批量更新耗时
        t0 = time.perf_counter()
        for _ in range(iterations):
            sample_idx = rng.choice(capacity, size=batch_size, replace=False)
            new_prios = rng.uniform(0.1, 5.0, size=batch_size)
            tree.update_batch(sample_idx, new_prios)
        update_time_ms = (time.perf_counter() - t0) * 1000.0 / iterations
        self.assertLess(update_time_ms, 0.5, f"update_time_ms {update_time_ms:.4f}ms is too slow")


if __name__ == "__main__":
    unittest.main()
