"""Native mask remains the authority in the visible BattleAction summary."""

from __future__ import annotations

import unittest

from workbench.gui.battle_action_summary import install_battle_action_summary_gui


class _Client:
    def __init__(self):
        self.mask = [False] * 757

    def get_ui(self):
        return 3, "SCENE_PLAYING"

    def get_action_mask(self, capacity):
        return 0, self.mask, capacity, ""

    def get_observation_json(self):
        return 1, "", {
            "global": {
                "level_id": 5,
                "grid_rows": 5,
                "sun": 0,
                "card_system": "conveyor",
                "is_conveyor": True,
                "game_mode_name": "Adventure 1-5",
            },
            "cards": [
                {"slot": 0, "plant_type": 3, "ready": True, "cost": 0, "cooldown_remaining": 0},
                {"slot": 1, "plant_type": 49, "ready": True, "cost": 0, "cooldown_remaining": 0},
            ],
            "plants": [],
        }

    def get_runtime_state(self):
        return {"game_mode": 0, "level_id": 5}


class _Gui:
    def _build_left_panel(self, parent):
        pass


install_battle_action_summary_gui(_Gui, object())


class BattleActionSummaryConveyorTest(unittest.TestCase):
    def test_conveyor_slots_and_special_nut_use_native_mask(self):
        gui = _Gui()
        gui.client = _Client()
        for slot, row, col in ((0, 0, 0), (0, 0, 1), (1, 2, 2)):
            gui.client.mask[1 + ((slot * 6 + row) * 9 + col)] = True

        summary = gui._collect_legal_action_summary()
        self.assertTrue(summary["conveyor"])
        self.assertEqual(summary["card_system"], "conveyor")
        self.assertEqual(summary["game_mode_name"], "Adventure 1-5")
        self.assertEqual(summary["plant_total"], 3)
        self.assertEqual(summary["plant_counts"], {3: 2, 49: 1})
        self.assertEqual([card["legal"] for card in summary["card_slots"]], [2, 1])
        formatted = gui._format_legal_action_summary(summary)
        self.assertIn("Mode: Adventure 1-5", formatted)
        self.assertIn("Card system: CONVEYOR", formatted)
        self.assertIn("Explode-o-nut", formatted)
        self.assertIn("slot  1", formatted)


if __name__ == "__main__":
    unittest.main()
