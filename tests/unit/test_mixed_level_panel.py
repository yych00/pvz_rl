"""Unit tests for MixedLevelPanel GUI interaction."""

import gc
import tkinter as tk
import unittest

from workbench.gui.mixed_level_panel import MixedLevelPanel


class DummyUI:
    CLR_CARD = "#16202c"
    CLR_PANEL = "#0d131a"
    CLR_BORDER = "#253342"
    CLR_TEXT_MAIN = "#e2e8f0"
    CLR_TEXT_MUTED = "#8899a6"
    COLOR_BLUE = "#38bdf8"
    COLOR_GREEN = "#4ade80"
    FONT_MAIN = ("Segoe UI", 9)
    FONT_BOLD = ("Segoe UI", 9, "bold")
    FONT_TITLE = ("Segoe UI", 11, "bold")


class TestMixedLevelPanelGUI(unittest.TestCase):
    def setUp(self):
        self.root = tk.Tk()
        self.root.withdraw()
        self.changed_count = 0

    def tearDown(self):
        self.root.destroy()
        self.root = None
        gc.collect()  # Tk finalizers must run on the thread that created the root.

    def _on_change(self):
        self.changed_count += 1

    def test_default_state_is_single_mode(self):
        panel = MixedLevelPanel(self.root, DummyUI(), on_change=self._on_change)
        self.assertFalse(panel.is_mixed_mode())
        self.assertIsNone(panel.get_level_pool())

    def test_switch_to_mixed_mode_default_pool(self):
        panel = MixedLevelPanel(self.root, DummyUI(), on_change=self._on_change)
        panel.var_mode.set("mixed")
        panel._on_mode_toggled()
        self.assertTrue(panel.is_mixed_mode())

        # 默认只勾选白天 (1-x) 和 一周目，不含趣味关/传送带
        pool = panel.get_level_pool()
        self.assertIsNotNone(pool)
        self.assertEqual(len(pool), 8)
        self.assertEqual(pool, ((1, 1), (2, 1), (3, 1), (4, 1), (6, 1), (7, 1), (8, 1), (9, 1)))
        self.assertGreater(self.changed_count, 0)

    def test_select_all_and_clear_scenes(self):
        panel = MixedLevelPanel(self.root, DummyUI(), on_change=self._on_change)
        panel.var_mode.set("mixed")

        # 点击全选
        panel._select_all_scenes()
        pool = panel.get_level_pool()
        # 5 个场景 * 8 关 = 40 关
        self.assertEqual(len(pool), 40)

        # 勾选特殊关卡
        panel.var_include_fun.set(True)
        panel.var_include_conveyor.set(True)
        panel._handle_change()
        pool = panel.get_level_pool()
        # 5 个场景 * 10 关 = 50 关
        self.assertEqual(len(pool), 50)

        # 勾选二周目
        panel.playthrough_vars[2].set(True)
        panel._handle_change()
        pool = panel.get_level_pool()
        # 50 * 2 = 100 关
        self.assertEqual(len(pool), 100)

        # 清空场景
        panel._clear_all_scenes()
        pool = panel.get_level_pool()
        self.assertEqual(pool, ())

    def test_seed_selection_modes(self):
        panel = MixedLevelPanel(self.root, DummyUI(), on_change=self._on_change)
        panel.var_mode.set("mixed")

        # 默认: 每关预设卡组 + 卡槽打乱 -> fixed_shuffled
        self.assertEqual(panel.get_seed_selection_mode(), "fixed_shuffled")

        # 取消打乱 -> fixed
        panel.var_shuffle_slots.set(False)
        self.assertEqual(panel.get_seed_selection_mode(), "fixed")

        # 切换到自动针对僵尸 -> adaptive
        panel.var_deck_strategy.set("自动针对僵尸")
        self.assertEqual(panel.get_seed_selection_mode(), "adaptive")
        panel.var_shuffle_slots.set(True)
        self.assertEqual(panel.get_seed_selection_mode(), "adaptive_shuffled")

    def test_restores_exact_custom_pool_and_adaptive_shuffle_setting(self):
        panel = MixedLevelPanel(self.root, DummyUI())
        pool = ((49, 2), (41, 1), (43, 2))
        for mode in ("adaptive_shuffled", "adaptive"):
            panel.set_from_level_pool(pool, seed_selection_mode=mode)
            self.assertEqual(panel.get_level_pool(), pool)
            self.assertEqual(panel.get_seed_selection_mode(), mode)
            self.assertEqual(panel.var_shuffle_slots.get(), mode == "adaptive_shuffled")

    def test_set_from_level_pool_restores_state(self):
        panel = MixedLevelPanel(self.root, DummyUI(), on_change=self._on_change)
        # 初始为单关模式
        self.assertFalse(panel.is_mixed_mode())

        # 模拟载入包含大关 1, 3，包含趣味关(x-5)但无传送带的一周目关卡池
        test_pool = (
            (1, 1), (2, 1), (3, 1), (4, 1), (5, 1), (6, 1), (7, 1), (8, 1), (9, 1),  # 1-x含1-5
            (21, 1), (22, 1), (23, 1), (24, 1), (25, 1), (26, 1), (27, 1), (28, 1), (29, 1),  # 3-x含3-5
        )
        panel.set_from_level_pool(test_pool, seed_selection_mode="adaptive")

        self.assertTrue(panel.is_mixed_mode())
        self.assertTrue(panel.scene_vars[1].get())
        self.assertFalse(panel.scene_vars[2].get())
        self.assertTrue(panel.scene_vars[3].get())
        self.assertFalse(panel.scene_vars[4].get())
        self.assertFalse(panel.scene_vars[5].get())
        self.assertTrue(panel.playthrough_vars[1].get())
        self.assertFalse(panel.playthrough_vars[2].get())
        self.assertTrue(panel.var_include_fun.get())
        self.assertFalse(panel.var_include_conveyor.get())
        self.assertEqual(panel.get_seed_selection_mode(), "adaptive")
        self.assertEqual(panel.get_level_pool(), test_pool)

        # 传入 None 恢复单关模式
        panel.set_from_level_pool(None)
        self.assertFalse(panel.is_mixed_mode())
        self.assertIsNone(panel.get_level_pool())

    def test_set_busy_disables_interactive_controls(self):
        panel = MixedLevelPanel(self.root, DummyUI(), on_change=self._on_change)
        panel.set_busy(True)
        for w in panel._interactive_widgets:
            self.assertEqual(str(w["state"]), "disabled")

        panel.set_busy(False)
        for w in panel._interactive_widgets:
            self.assertIn(str(w["state"]), ("normal", "readonly"))


if __name__ == "__main__":
    unittest.main()
