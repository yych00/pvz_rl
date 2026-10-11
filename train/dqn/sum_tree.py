"""高性能向量化线段树 (SumTree)，用于优先经验回放 (PER)。"""

from __future__ import annotations

import math
from typing import Sequence
import numpy as np


class SumTree:
    """基于完全二叉树数组的高性能线段树，所有操作向量化，杜绝低效 Python 递归。

    内部叶节点起始偏移为大于等于 capacity 的最小 2 的幂次 M：
    - 树数组总长度为 2 * M；
    - 叶节点 i 对应数组索引 M + i；
    - 根节点为 tree[1]，其值为全树所有叶节点权重总和；
    - 采样与批量更新均沿树高度（log2(M) 层）向量化推进。
    """

    def __init__(self, capacity: int) -> None:
        if type(capacity) is not int or capacity <= 0:
            raise ValueError("capacity must be a positive integer")
        self.capacity = int(capacity)
        # 大于等于 capacity 的最小 2 的幂
        self._tree_depth = int(math.ceil(math.log2(self.capacity))) if self.capacity > 1 else 1
        self._leaf_offset = 1 << self._tree_depth
        self._tree = np.zeros(2 * self._leaf_offset, dtype=np.float64)

    @property
    def total(self) -> float:
        """返回当前树所有叶节点的累积权重总和。"""
        return float(self._tree[1])

    def set_all(self, priorities: Sequence[float] | np.ndarray, size: int) -> None:
        """从下到上一遍线性构建整棵树，用于初始化或从 Uniform 热切换。"""
        if not 0 <= size <= self.capacity:
            raise ValueError(f"size {size} must be in [0, capacity={self.capacity}]")
        self._tree.fill(0.0)
        if size == 0:
            return
        vals = np.asarray(priorities[:size], dtype=np.float64)
        if (vals < 0).any() or not np.isfinite(vals).all():
            raise ValueError("priorities must be finite and non-negative")
        self._tree[self._leaf_offset : self._leaf_offset + size] = vals
        # 从最深非叶节点向上层层归约
        for i in range(self._leaf_offset - 1, 0, -1):
            self._tree[i] = self._tree[2 * i] + self._tree[2 * i + 1]

    def update(self, index: int, priority: float) -> None:
        """更新单个样本的优先级。"""
        if not 0 <= index < self.capacity:
            raise IndexError(f"index {index} out of range [0, {self.capacity})")
        val = float(priority)
        if not math.isfinite(val) or val < 0:
            raise ValueError("priority must be finite and non-negative")
        tree_idx = self._leaf_offset + index
        self._tree[tree_idx] = val
        tree_idx //= 2
        while tree_idx > 0:
            self._tree[tree_idx] = self._tree[2 * tree_idx] + self._tree[2 * tree_idx + 1]
            tree_idx //= 2

    def update_batch(self, indices: Sequence[int] | np.ndarray, priorities: Sequence[float] | np.ndarray) -> None:
        """向量化批量更新叶节点并向上传递累加。"""
        idx_arr = np.asarray(indices, dtype=np.int64)
        val_arr = np.asarray(priorities, dtype=np.float64)
        if idx_arr.shape != val_arr.shape:
            raise ValueError("indices and priorities must have matching shapes")
        if not idx_arr.size:
            return
        if (idx_arr < 0).any() or (idx_arr >= self.capacity).any():
            raise IndexError(f"indices out of range [0, {self.capacity})")
        if not np.isfinite(val_arr).all() or (val_arr < 0).any():
            raise ValueError("priorities must be finite and non-negative")

        tree_indices = self._leaf_offset + idx_arr
        self._tree[tree_indices] = val_arr

        # 向量化自底向上归约，每层去重以防同一批次包含相同索引
        curr = tree_indices // 2
        for _ in range(self._tree_depth):
            unique_nodes = np.unique(curr)
            unique_nodes = unique_nodes[unique_nodes > 0]
            if not unique_nodes.size:
                break
            left = 2 * unique_nodes
            self._tree[unique_nodes] = self._tree[left] + self._tree[left + 1]
            curr = unique_nodes // 2

    def get_priority(self, index: int) -> float:
        """获取单个样本当前存储的优先级。"""
        if not 0 <= index < self.capacity:
            raise IndexError(f"index {index} out of range [0, {self.capacity})")
        return float(self._tree[self._leaf_offset + index])

    def get_priorities(self, indices: Sequence[int] | np.ndarray) -> np.ndarray:
        """批量获取指定索引的当前优先级。"""
        idx_arr = np.asarray(indices, dtype=np.int64)
        if (idx_arr < 0).any() or (idx_arr >= self.capacity).any():
            raise IndexError(f"indices out of range [0, {self.capacity})")
        return self._tree[self._leaf_offset + idx_arr].copy()

    def sample(self, targets: Sequence[float] | np.ndarray, active_size: int) -> tuple[np.ndarray, np.ndarray]:
        """根据给定的前缀和目标值向量化二分查找叶节点索引及对应的优先级。

        Args:
            targets: 查询值数组，每个值在 [0, total) 区间内。
            active_size: 当前 Buffer 内已有的有效样本量。

        Returns:
            (indices, priorities): 对应的 Buffer 下标数组与优先级数值。
        """
        if active_size <= 0:
            raise ValueError("active_size must be positive")
        val_arr = np.asarray(targets, dtype=np.float64).copy()
        if not val_arr.size:
            return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float64)

        batch_size = val_arr.size
        curr_nodes = np.ones(batch_size, dtype=np.int64)

        # 沿树深度向下走
        for _ in range(self._tree_depth):
            left = 2 * curr_nodes
            left_vals = self._tree[left]
            go_right = val_arr >= left_vals
            curr_nodes = np.where(go_right, left + 1, left)
            val_arr = np.where(go_right, val_arr - left_vals, val_arr)

        indices = curr_nodes - self._leaf_offset
        # 边界防溢出保护：浮点精度误差可能微超有效范围
        indices = np.clip(indices, 0, active_size - 1)
        priorities = self._tree[self._leaf_offset + indices].copy()
        return indices, priorities


__all__ = ["SumTree"]
