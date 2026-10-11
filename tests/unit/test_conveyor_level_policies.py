"""Decision checks for the two native conveyor Adventure levels."""

from __future__ import annotations

import unittest

from agents.scripted.level_1_5 import choose_level_1_5_plant
from agents.scripted.level_1_10 import choose_level_1_10_plant
from agents.scripted.level_2_10 import choose_level_2_10_plant
from agents.scripted.level_4_5 import choose_level_4_5_plant
from workbench.live.registry import get_level_spec

ACTION_SPACE_SIZE = 757


def plant_index(slot: int, row: int, col: int) -> int:
    return 1 + ((slot * 6 + row) * 9 + col)


def observation(cards, zombies=(), plants=()):
    return {
        "global": {"grid_rows": 5},
        "cards": [{"slot": slot, "plant_type": plant_type} for slot, plant_type in cards],
        "zombies": list(zombies),
        "plants": list(plants),
    }


class ConveyorPolicyTest(unittest.TestCase):
    def test_1_5_uses_second_matching_packet_when_first_is_masked(self):
        mask = [False] * ACTION_SPACE_SIZE
        mask[plant_index(1, 3, 2)] = True
        obs = observation(
            [(0, 3), (1, 3)],
            [{"row": 3, "x": 480, "body_hp": 200}],
        )
        choice = choose_level_1_5_plant(mask, obs)
        self.assertEqual(choice[:5], (plant_index(1, 3, 2), 1, 3, 2, 3))

    def test_1_5_aims_explosive_at_close_lane(self):
        mask = [False] * ACTION_SPACE_SIZE
        mask[plant_index(0, 2, 2)] = True
        mask[plant_index(1, 2, 2)] = True
        obs = observation(
            [(0, 3), (1, 49)],
            [{"row": 2, "x": 250, "body_hp": 200}],
        )
        self.assertEqual(choose_level_1_5_plant(mask, obs)[4], 49)

    def test_1_5_waits_for_zombies_and_respects_mask(self):
        mask = [False] * ACTION_SPACE_SIZE
        mask[plant_index(0, 2, 2)] = True
        self.assertIsNone(choose_level_1_5_plant(mask, observation([(0, 3)])))
        obs = observation([(0, 3)], [{"row": 2, "x": 400}])
        self.assertIsNone(choose_level_1_5_plant([False] * ACTION_SPACE_SIZE, obs))
        mask = [False] * ACTION_SPACE_SIZE
        mask[plant_index(0, 2, 3)] = True
        self.assertIsNone(choose_level_1_5_plant(mask, obs))

    def test_1_10_covers_unprotected_live_lane(self):
        mask = [False] * ACTION_SPACE_SIZE
        mask[plant_index(0, 4, 0)] = True
        mask[plant_index(1, 4, 0)] = True
        obs = observation(
            [(0, 0), (1, 7)],
            [{"row": 4, "x": 600, "body_hp": 200}],
        )
        self.assertEqual(choose_level_1_10_plant(mask, obs)[4], 7)
        self.assertEqual(choose_level_1_10_plant(mask, obs)[2], 4)

    def test_1_10_uses_cherry_for_close_cluster(self):
        mask = [False] * ACTION_SPACE_SIZE
        mask[plant_index(0, 1, 2)] = True
        mask[plant_index(1, 1, 2)] = True
        obs = observation(
            [(0, 0), (1, 2)],
            [
                {"row": 1, "x": 270, "body_hp": 200},
                {"row": 1, "x": 290, "body_hp": 200},
            ],
        )
        self.assertEqual(choose_level_1_10_plant(mask, obs)[4], 2)

    def test_1_10_spends_mine_before_extra_shooter(self):
        mask = [False] * ACTION_SPACE_SIZE
        mask[plant_index(0, 2, 5)] = True
        mask[plant_index(1, 2, 1)] = True
        obs = observation(
            [(0, 4), (1, 0)],
            [{"row": 2, "x": 620, "body_hp": 200}],
            [{"row": 2, "col": 0, "type": 0}],
        )
        self.assertEqual(choose_level_1_10_plant(mask, obs)[4], 4)

    def test_2_10_eats_grave_when_grave_buster_available(self):
        mask = [False] * ACTION_SPACE_SIZE
        mask[plant_index(0, 2, 4)] = True
        obs = observation([(0, 11)])  # Grave Buster
        obs["grid"] = [{"row": 2, "col": 4, "has_grave": True}]
        choice = choose_level_2_10_plant(mask, obs)
        self.assertIsNotNone(choice)
        self.assertEqual(choice[4], 11)
        self.assertEqual(choice[2], 2)
        self.assertEqual(choice[3], 4)

    def test_2_10_uses_ice_shroom_on_dangerous_cluster(self):
        mask = [False] * ACTION_SPACE_SIZE
        mask[plant_index(0, 0, 0)] = True
        obs = observation(
            [(0, 14)],  # Ice-shroom
            [
                {"row": 1, "x": 250, "body_hp": 200},
                {"row": 2, "x": 260, "body_hp": 200},
            ],
        )
        choice = choose_level_2_10_plant(mask, obs)
        self.assertIsNotNone(choice)
        self.assertEqual(choice[4], 14)

    def test_4_5_refuses_naked_cherry_bomb_without_zombies(self):
        mask = [True] * ACTION_SPACE_SIZE
        obs = observation([(0, 2)])  # Cherry Bomb
        obs["global"]["grid_rows"] = 6
        # 无僵尸时，绝不允许空放樱桃炸弹
        choice = choose_level_4_5_plant(mask, obs)
        self.assertIsNone(choice)

    def test_4_5_uses_cherry_bomb_when_zombie_close(self):
        mask = [False] * ACTION_SPACE_SIZE
        # 允许在 row 0, col 2 释放
        mask[plant_index(0, 0, 2)] = True
        obs = observation(
            [(0, 2)],  # Cherry Bomb
            [{"row": 0, "x": 250, "body_hp": 200}],  # 逼近僵尸
        )
        obs["global"]["grid_rows"] = 6
        choice = choose_level_4_5_plant(mask, obs)
        self.assertIsNotNone(choice)
        self.assertEqual(choice[4], 2)  # CHERRY_BOMB
        self.assertEqual(choice[2], 0)  # row 0
        self.assertEqual(choice[3], 2)  # col 2

    def test_4_5_smashes_safest_vase_rightmost_with_shooter(self):
        mask = [False] * ACTION_SPACE_SIZE
        # 允许开 (0, 8), (1, 7), (1, 8) 的罐子
        act_0_8 = 703 + 0 * 9 + 8
        act_1_7 = 703 + 1 * 9 + 7
        act_1_8 = 703 + 1 * 9 + 8
        mask[act_0_8] = True
        mask[act_1_7] = True
        mask[act_1_8] = True

        obs = observation([])  # 手牌为空
        obs["global"]["grid_rows"] = 5
        # 场地网格物件：(0, 8), (1, 7), (1, 8) 处有罐子
        obs["grid_items"] = [
            {"type": "VASE", "row": 0, "col": 8},
            {"type": "VASE", "row": 1, "col": 7},
            {"type": "VASE", "row": 1, "col": 8},
        ]
        # 第 1 行后排已有一株双发射手 (type 7)
        obs["plants"] = [{"type": 7, "row": 1, "col": 0, "id": 101}]
        # 无威胁僵尸，场面安全
        obs["zombies"] = []

        choice = choose_level_4_5_plant(mask, obs)
        self.assertIsNotNone(choice)
        # 应优先选择已有射手防线且最靠右的罐子 (row 1, col 8)
        self.assertEqual(choice[0], act_1_8)
        self.assertEqual(choice[2], 1)  # row
        self.assertEqual(choice[3], 8)  # col

    def test_4_5_refuses_smash_when_threat_close(self):
        mask = [False] * ACTION_SPACE_SIZE
        act_1_8 = 703 + 1 * 9 + 8
        mask[act_1_8] = True

        obs = observation([])
        obs["global"]["grid_rows"] = 5
        obs["grid_items"] = [{"type": "VASE", "row": 1, "col": 8}]
        # 僵尸逼近底线 (x=400 < 480)
        obs["zombies"] = [{"row": 1, "x": 400, "body_hp": 200}]

        choice = choose_level_4_5_plant(mask, obs)
        # 安全闸门拦截，绝不开罐
        self.assertIsNone(choice)

    def test_4_5_plants_rescued_shooter_before_smashing(self):
        mask = [False] * ACTION_SPACE_SIZE
        # 允许种射手 (slot 0, row 1, col 0)
        mask[plant_index(0, 1, 0)] = True
        # 同时允许开罐 (1, 8)
        act_1_8 = 703 + 1 * 9 + 8
        mask[act_1_8] = True

        # 手里有豌豆射手 (type 0)
        obs = observation([(0, 0)])
        obs["global"]["grid_rows"] = 5
        obs["grid_items"] = [{"type": "VASE", "row": 1, "col": 8}]
        obs["zombies"] = []

        choice = choose_level_4_5_plant(mask, obs)
        self.assertIsNotNone(choice)
        # 必须优先消化手牌种植射手，而不是砸罐
        self.assertEqual(choice[4], 0)  # PEASHOOTER
        self.assertEqual(choice[2], 1)  # row 1
        self.assertEqual(choice[3], 0)  # col 0

    def test_registry_uses_conveyor_policies_without_seed_priorities(self):
        for level_id, policy in (
            (5, choose_level_1_5_plant),
            (10, choose_level_1_10_plant),
            (20, choose_level_2_10_plant),
            (35, choose_level_4_5_plant),
        ):
            with self.subTest(level_id=level_id):
                spec = get_level_spec(level_id)
                self.assertIs(spec.adventure1_policy, policy)
                self.assertIs(spec.adventure2_policy, policy)
                self.assertEqual(spec.adventure1_priority, ())
                self.assertEqual(spec.adventure2_priority, ())


if __name__ == "__main__":
    unittest.main()
