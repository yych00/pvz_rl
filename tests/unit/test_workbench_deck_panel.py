import gc
import json
import tempfile
import tkinter as tk
import unittest
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch

import workbench.main  # installs the Workbench extensions
from workbench import PvzGymUnitTestGUI
from workbench.gui.deck_panel import MODE_LABELS, parse_cards
from train.common import EnvConfig
from train.dqn.config import DoubleDQNConfig
from train.dqn.run_state import load_run_config


class WorkbenchDeckPanelTests(unittest.TestCase):
    def setUp(self):
        self.root = tk.Tk()
        self.root.withdraw()
        self.app = PvzGymUnitTestGUI(self.root)

    def tearDown(self):
        self.root.destroy()
        self.app = self.root = None
        gc.collect()

    def test_visible_modes_and_shuffled_selection(self):
        for panel in (self.app.live_deck_panel, self.app.training_deck_panel):
            self.assertEqual(tuple(panel.mode.cget("values")),
                             ("自动针对僵尸", "自动针对僵尸（顺序打乱）",
                              "每关预设卡组", "每关预设卡组（顺序打乱）"))
            self.assertEqual(panel.settings(), ("fixed", None))
            self.assertFalse(panel.edit_button.winfo_manager())
            self.assertFalse(panel.entry_row.winfo_manager())
            panel.mode_var.set(MODE_LABELS["fixed_shuffled"])
            self.assertEqual(panel.settings(), ("fixed_shuffled", None))
            self.assertIn("戴夫锁卡保持原位", panel.note.cget("text"))
            self.assertEqual(panel.selection_profiles()[panel.profile_key], ("fixed_shuffled", None))
            panel.mode_var.set(MODE_LABELS["adaptive_shuffled"])
            self.assertEqual(panel.settings(), ("adaptive_shuffled", None))
            self.assertIn("戴夫锁卡保持原位", panel.note.cget("text"))

    def test_both_panels_preview_and_dark_priority_input(self):
        for panel in (self.app.live_deck_panel, self.app.training_deck_panel):
            self.assertEqual(panel.settings(), ("fixed", None))
            self.assertEqual(panel.entry.cget("bg"), "#111a25")
            self.assertEqual(panel.entry.cget("disabledbackground"), "#161b22")
            self.assertGreater(len(panel.chips.winfo_children()), 0)
        self.app._select_live_level(31)
        self.app.var_live_deck_playthrough.set("2")
        self.assertIn("4-1", self.app.live_deck_panel.badge.cget("text"))
        self.app.var_training_level.set("5-1")
        self.assertIn("5-1", self.app.training_deck_panel.badge.cget("text"))
        self.app._select_live_scene(3)
        self.assertIn("3-1", self.app.live_deck_panel.badge.cget("text"))
        self.assertIn("3-1", self.app.lbl_live_selected.cget("text"))

    def test_custom_cards_snapshot_and_live_forwarding(self):
        for panel in (self.app.live_deck_panel, self.app.training_deck_panel):
            panel.cards_var.set("睡莲，向日葵，三叶草，裂荚射手")
            panel.mode_var.set(MODE_LABELS["custom"])
            self.assertEqual(panel.settings(), ("custom", (16, 1, 27, 28)))
        values = self.app._parse_common_training_config()
        self.assertEqual(values[-1], (16, 1, 27, 28))
        self.assertEqual(self.app._training_seed_selection_mode, "custom")
        self.assertTrue(self.app._begin_live("test"))
        self.assertEqual(self.app._active_live_seed_priority, (16, 1, 27, 28))
        self.app._finish_live("done")

    def test_locked_controls_and_reset(self):
        self.app._set_training_controls(True)
        panel = self.app.training_deck_panel
        self.assertTrue(panel.test_button.is_disabled)
        self.assertTrue(panel.edit_button.is_disabled)
        self.assertEqual(panel.entry.cget("state"), "disabled")
        self.app._set_training_controls(False)
        panel.cards_var.set("1,0")
        panel.mode_var.set(MODE_LABELS["custom"])
        self.assertEqual(panel.entry.cget("state"), "normal")
        panel.reset()
        self.assertEqual(panel.settings(), ("fixed", None))

    def test_running_preview_switches_without_changing_configuration(self):
        from pvz_gym.default_decks import PLANT_NAMES, default_deck_priority

        for panel in (self.app.live_deck_panel, self.app.training_deck_panel):
            panel.cards_var.set("1,0,4")
            panel.mode_var.set(MODE_LABELS["custom"])
            context = panel.context()
            profiles = panel.selection_profiles()
            panel.set_busy(True)
            self.assertEqual(str(panel.preview_level.cget("state")), "readonly")
            self.assertEqual(str(panel.preview_playthrough.cget("state")), "readonly")
            panel.preview_level_var.set("4-9")
            self.assertIn("4-9", panel.badge.cget("text"))
            labels = [chip.winfo_children()[0].cget("text") for chip in panel.chips.winfo_children()]
            self.assertEqual(labels, [f"{i+1:02d}  {PLANT_NAMES[p]}"
                                     for i, p in enumerate(default_deck_priority(39, 1))])
            self.assertIn("不影响当前运行配置", panel.note.cget("text"))
            panel.preview_playthrough_var.set("2")
            self.assertIn("2周目", panel.badge.cget("text"))
            panel.set_busy(True)
            self.assertEqual(panel.preview_level_var.get(), "4-9")
            self.assertEqual(panel.context(), context)
            self.assertEqual(panel.settings(), ("custom", (1, 0, 4)))
            self.assertEqual(panel.selection_profiles(), profiles)
            self.assertTrue(panel.test_button.is_disabled)
            panel.preview_level_var.set("5-10")
            self.assertIn("无需选卡", panel.note.cget("text"))
            panel.preview_level_var.set("4-4")
            self.assertGreater(len(panel.chips.winfo_children()), 1)
            panel.set_busy(False)
            self.assertFalse(panel.preview_row.winfo_manager())
            self.assertEqual(panel.settings(), ("custom", (1, 0, 4)))
            self.assertIn("向日葵", panel.chips.winfo_children()[0].winfo_children()[0].cget("text"))

    def test_training_events_preserve_running_preset_preview(self):
        self.app._set_training_controls(True)
        panel = self.app.training_deck_panel
        panel.preview_level_var.set("4-9")
        note = panel.note.cget("text")
        self.app._apply_dqn_training_event("seed_selection", {
            "level_id": 29, "actual": [16, 1, 27], "warnings": ["矿工缺口"]})
        self.assertEqual(panel.note.cget("text"), note)
        self.assertIn("4-9", panel.badge.cget("text"))
        self.app._set_training_controls(True)
        self.assertEqual(panel.preview_level_var.get(), "4-9")

    def test_scene_switch_changes_deck_and_restores_per_level_customization(self):
        panel = self.app.live_deck_panel
        panel.cards_var.set("1,0,4")
        panel.mode_var.set(MODE_LABELS["custom"])
        self.app._select_live_scene(3)
        self.assertEqual(panel.settings(), ("fixed", None))
        self.assertIn("睡莲", panel.chips.winfo_children()[0].winfo_children()[0].cget("text"))
        panel.cards_var.set("16,1,19")
        panel.mode_var.set(MODE_LABELS["custom"])
        self.app._select_live_scene(1)
        self.assertEqual(panel.settings(), ("custom", (1, 0, 4)))
        self.app._select_live_scene(3)
        self.assertEqual(panel.settings(), ("custom", (16, 1, 19)))
        self.app.var_live_deck_playthrough.set("2")
        self.assertEqual(panel.settings(), ("fixed", None))
        self.app.var_live_deck_playthrough.set("1")
        self.assertEqual(panel.settings(), ("custom", (16, 1, 19)))

    def test_deck_error_stops_live_before_worker(self):
        panel = self.app.live_deck_panel
        panel.cards_var.set("1,1")
        panel.mode_var.set(MODE_LABELS["custom"])
        self.assertFalse(self.app._begin_live("invalid"))
        self.assertFalse(self.app._live_running)
        self.assertIn("重复", panel.note.cget("text"))

    def test_mode_and_priorities_survive_run_config_and_restore_ui(self):
        config = DoubleDQNConfig(env=EnvConfig(seed_selection_mode="custom", seed_priority=(16, 1, 27)))
        with tempfile.TemporaryDirectory() as directory:
            Path(directory, "config.json").write_text(json.dumps(asdict(config)), encoding="utf-8")
            Path(directory, "final.pt").write_bytes(b"fixture")
            loaded = load_run_config(directory)
            self.assertEqual(loaded.env.seed_priority, (16, 1, 27))
            with patch("workbench.gui.resume_run.filedialog.askdirectory", return_value=directory), \
                 patch.object(self.app, "_inspect_training_checkpoint_worker"):
                self.app._resume_training_run()
        self.assertEqual(self.app.training_deck_panel.settings(), ("custom", (16, 1, 27)))

    def test_training_actual_cards_are_visible(self):
        self.app._apply_dqn_training_event("seed_selection", {"actual": [16, 1, 27], "warnings": ["矿工缺口"]})
        self.assertIn("三叶草", self.app.training_deck_panel.note.cget("text"))
        self.assertIn("矿工缺口", self.app.training_deck_panel.note.cget("text"))

    def test_fixed_lesson_candidates_and_batch_modes_are_snapshotted(self):
        self.app._select_live_level(33)
        panel = self.app.live_deck_panel
        self.assertEqual(panel.mode_var.get(), "每关预设卡组")
        self.assertIn("仙人掌", panel.note.cget("text"))
        self.assertIn("气球", panel.note.cget("text"))
        self.assertIn("候选", panel.note.cget("text"))
        panel.mode_var.set(MODE_LABELS["adaptive"])
        self.app._select_live_level(34)
        self.assertTrue(self.app._begin_live("fixed batch"))
        self.assertEqual(self.app._active_live_seed_selection_mode, "fixed")
        self.assertEqual(self.app._active_live_seed_profiles[(33, 1)], ("adaptive", None))
        self.assertEqual(self.app._active_live_seed_profiles[(34, 1)], ("fixed", None))
        self.app._finish_live("done")

    def test_fixed_mode_is_saved_and_restored_without_converting_legacy_native(self):
        for mode in ("fixed", "fixed_shuffled", "native"):
            config = DoubleDQNConfig(env=EnvConfig(seed_selection_mode=mode))
            with tempfile.TemporaryDirectory() as directory:
                Path(directory, "config.json").write_text(json.dumps(asdict(config)), encoding="utf-8")
                Path(directory, "final.pt").write_bytes(b"fixture")
                self.assertEqual(load_run_config(directory).env.seed_selection_mode, mode)
                with patch("workbench.gui.resume_run.filedialog.askdirectory", return_value=directory), \
                     patch.object(self.app, "_inspect_training_checkpoint_worker"):
                    self.app._resume_training_run()
            self.assertEqual(self.app.training_deck_panel.settings(), (mode, None))

    def test_special_levels_hide_chips_and_disable_selection_controls(self):
        panel = self.app.training_deck_panel
        self.app.var_training_level.set("5-10")
        self.assertIn("5-10", panel.badge.cget("text"))
        self.assertIn("僵王传送带", panel.badge.cget("text"))
        self.assertEqual(str(panel.mode.cget("state")), "disabled")
        self.assertTrue(panel.test_button.is_disabled)
        self.assertTrue(panel.reset_button.is_disabled)
        self.assertIn("无需选卡", panel.note.cget("text"))
        self.assertEqual(len(panel.chips.winfo_children()), 1)
        self.assertIn("无需选卡", panel.chips.winfo_children()[0].winfo_children()[0].cget("text"))

        # 切换回普通关卡恢复
        self.app.var_training_level.set("5-1")
        self.assertEqual(str(panel.mode.cget("state")), "readonly")
        self.assertFalse(panel.test_button.is_disabled)
        self.assertGreater(len(panel.chips.winfo_children()), 1)

    def test_special_level_ignores_conveyor_cards_in_seed_selection_event(self):
        panel = self.app.training_deck_panel
        self.app.var_training_level.set("5-10")
        self.assertIn("无需选卡", panel.note.cget("text"))

        # 模拟传送带关卡抛出的 seed_selection 事件（带 special 或 level_id=50）
        self.app._apply_dqn_training_event("seed_selection", {
            "mode": "special",
            "special": "僵王传送带",
            "level_id": 50,
            "actual": [],
        })
        # 面板依然保留特殊关卡的说明，没有被“本局选卡”覆盖
        self.assertIn("无需选卡", panel.note.cget("text"))
        self.assertNotIn("本局选卡", panel.note.cget("text"))


class CardParsingTests(unittest.TestCase):
    def test_names_and_validation(self):
        self.assertEqual(parse_cards("向日葵、豌豆射手，27"), (1, 0, 27))
        for raw in ("48", "1,1", "-1", "未知卡牌"):
            with self.assertRaises(ValueError):
                parse_cards(raw)


if __name__ == "__main__":
    unittest.main()
