"""Unit tests for mixed level pool generation and curriculum management."""

import unittest

try:
    from train.curriculum import (
        generate_mixed_level_pool,
        format_level_label,
        format_level_pool_summary,
        sample_level_from_pool,
    )
    from train.common import EnvConfig
except ImportError:
    from train.curriculum import (
        generate_mixed_level_pool,
        format_level_label,
        format_level_pool_summary,
        sample_level_from_pool,
    )
    from train.common import EnvConfig


class TestMixedLevelCurriculum(unittest.TestCase):
    def test_single_scene_day_regular_only(self):
        # 仅白天 (1-x)，一周目，不含趣味关和传送带
        pool = generate_mixed_level_pool(
            scenes=[1], playthroughs=[1], include_fun=False, include_conveyor=False,
        )
        # 应包含 8 关: 1-1, 1-2, 1-3, 1-4, 1-6, 1-7, 1-8, 1-9
        expected = ((1, 1), (2, 1), (3, 1), (4, 1), (6, 1), (7, 1), (8, 1), (9, 1))
        self.assertEqual(pool, expected)
        self.assertEqual(len(pool), 8)

    def test_single_scene_day_with_fun_and_conveyor(self):
        # 白天 (1-x)，勾选趣味关 (1-5) 与传送带 (1-10)
        pool = generate_mixed_level_pool(
            scenes=[1], playthroughs=[1], include_fun=True, include_conveyor=True,
        )
        self.assertEqual(len(pool), 10)
        self.assertIn((5, 1), pool)
        self.assertIn((10, 1), pool)

    def test_multi_scenes_day_and_night(self):
        # 白天 (1-x) 与 黑夜 (2-x)
        pool = generate_mixed_level_pool(
            scenes=[1, 2], playthroughs=[1], include_fun=False, include_conveyor=False,
        )
        # 每个场景 8 关，共 16 关
        self.assertEqual(len(pool), 16)
        # 2-1 对应内部 level_id = 11
        self.assertIn((11, 1), pool)
        self.assertIn((12, 1), pool)

    def test_multi_playthroughs(self):
        # 白天，一周目 + 二周目
        pool = generate_mixed_level_pool(
            scenes=[1], playthroughs=[1, 2], include_fun=False, include_conveyor=False,
        )
        self.assertEqual(len(pool), 16)
        # 既包含一周目也包含二周目
        self.assertIn((1, 1), pool)
        self.assertIn((1, 2), pool)

    def test_all_five_scenes_full(self):
        # 5 个场景全部包含，包括趣味和传送带
        pool = generate_mixed_level_pool(
            scenes=[1, 2, 3, 4, 5], playthroughs=[1], include_fun=True, include_conveyor=True,
        )
        self.assertEqual(len(pool), 50)

    def test_empty_scenes_returns_empty(self):
        pool = generate_mixed_level_pool(
            scenes=[], playthroughs=[1], include_fun=False, include_conveyor=False,
        )
        self.assertEqual(pool, ())

    def test_format_level_label(self):
        self.assertEqual(format_level_label(1), "1-1")
        self.assertEqual(format_level_label(10), "1-10")
        self.assertEqual(format_level_label(11), "2-1")
        self.assertEqual(format_level_label(20), "2-10")
        self.assertEqual(format_level_label(50), "5-10")

    def test_format_level_pool_summary(self):
        pool = ((1, 1), (2, 1), (3, 1))
        summary = format_level_pool_summary(pool)
        self.assertIn("共 3 关", summary)
        self.assertIn("1大关", summary)
        self.assertIn("1周目", summary)

    def test_sample_level_from_pool(self):
        pool = ((1, 1), (11, 1))
        sampled = set()
        for _ in range(50):
            lvl, p = sample_level_from_pool(pool)
            self.assertEqual(p, 1)
            sampled.add(lvl)
        self.assertEqual(sampled, {1, 11})

    def test_env_config_level_pool_support(self):
        config = EnvConfig(level_pool=((1, 1), (11, 1)))
        self.assertEqual(config.level_pool, ((1, 1), (11, 1)))


if __name__ == "__main__":
    unittest.main()
