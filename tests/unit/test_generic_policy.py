"""Unit tests for the enhanced tactical generic policy."""

from __future__ import annotations

import unittest

from agents.scripted.generic_policy import (
    CHERRY_BOMB,
    GRAVE_BUSTER,
    LILY_PAD,
    PEASHOOTER,
    POTATO_MINE,
    SUNFLOWER,
    WALL_NUT,
    choose_enhanced_generic_plant,
)

ACTION_SPACE_SIZE = 757


def plant_index(slot: int, row: int, col: int) -> int:
    return 1 + ((slot * 6 + row) * 9 + col)


def make_obs(cards, zombies=(), plants=(), grid_items=(), grid_rows=5):
    return {
        "global": {"grid_rows": grid_rows},
        "cards": [{"slot": slot, "plant_type": ptype} for slot, ptype in cards],
        "zombies": list(zombies),
        "plants": list(plants),
        "grid_items": list(grid_items),
    }


class GenericPolicyTest(unittest.TestCase):
    def test_economy_plant_goes_to_backline(self):
        mask = [True] * ACTION_SPACE_SIZE
        obs = make_obs([(0, SUNFLOWER)])
        choice = choose_enhanced_generic_plant(mask, obs)
        self.assertIsNotNone(choice)
        self.assertEqual(choice[4], SUNFLOWER)
        # 向日葵必须种在后排 (col 0 或 1)
        self.assertIn(choice[3], (0, 1))

    def test_shooter_covers_active_lane(self):
        mask = [True] * ACTION_SPACE_SIZE
        # 第 3 行有僵尸逼近
        obs = make_obs(
            [(0, PEASHOOTER)],
            zombies=[{"row": 3, "x": 600, "body_hp": 200}],
        )
        choice = choose_enhanced_generic_plant(mask, obs)
        self.assertIsNotNone(choice)
        self.assertEqual(choice[4], PEASHOOTER)
        # 射手必须优先部署到第 3 行，且在中后排 (col 0~3)
        self.assertEqual(choice[2], 3)
        self.assertIn(choice[3], (0, 1, 2, 3))

    def test_refuses_naked_explosives_without_threat(self):
        mask = [True] * ACTION_SPACE_SIZE
        obs = make_obs([(0, CHERRY_BOMB)])
        choice = choose_enhanced_generic_plant(mask, obs)
        # 场上无僵尸时，绝不允许空放樱桃炸弹
        self.assertIsNone(choice)

    def test_deploys_explosive_on_urgent_threat(self):
        mask = [True] * ACTION_SPACE_SIZE
        # 第 1 行僵尸逼近底线 (x=300)
        obs = make_obs(
            [(0, CHERRY_BOMB)],
            zombies=[{"row": 1, "x": 300, "body_hp": 200}],
        )
        choice = choose_enhanced_generic_plant(mask, obs)
        self.assertIsNotNone(choice)
        self.assertEqual(choice[4], CHERRY_BOMB)
        self.assertIn(choice[2], (0, 1, 2))

    def test_grave_buster_eats_grave(self):
        mask = [True] * ACTION_SPACE_SIZE
        obs = make_obs(
            [(0, GRAVE_BUSTER)],
            grid_items=[{"type": "GRAVE", "row": 2, "col": 5}],
        )
        choice = choose_enhanced_generic_plant(mask, obs)
        self.assertIsNotNone(choice)
        self.assertEqual(choice[4], GRAVE_BUSTER)
        self.assertEqual(choice[2], 2)
        self.assertEqual(choice[3], 5)

    def test_lily_pad_placed_on_water(self):
        mask = [True] * ACTION_SPACE_SIZE
        obs = make_obs([(0, LILY_PAD)], grid_rows=6)
        choice = choose_enhanced_generic_plant(mask, obs)
        self.assertIsNotNone(choice)
        self.assertEqual(choice[4], LILY_PAD)
        # 睡莲必须铺在水路 (row 2 或 3)
        self.assertIn(choice[2], (2, 3))


if __name__ == "__main__":
    unittest.main()
