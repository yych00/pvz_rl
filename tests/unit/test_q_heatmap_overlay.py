# -*- coding: utf-8 -*-
"""Unit tests for Q-value heatmap overlay & NOOP extraction and UI bindings."""

from __future__ import annotations

import gc
import os
import sys
import ctypes
from ctypes import wintypes
import unittest
from unittest.mock import patch
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
for _p in (_ROOT, _ROOT / "python"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import numpy as np

from pvz_gym.action_space import ACTION_SPACE_SIZE, NOOP, encode_action, Action
from train.dqn import agent as agent_module
from train.dqn.agent import DoubleDQNAgent
from train.dqn.config import DoubleDQNConfig
from workbench.gui.q_heatmap_overlay import (
    QHeatmapOverlay, grid_to_pixel, rgb, HEATMAP_COLORS, user32,
)


class TestQHeatmapOverlay(unittest.TestCase):
    def tearDown(self):
        gc.collect()  # Finalize destroyed Tk fixtures before later worker threads run.

    """Test suite for coordinate math, action extraction, and NOOP handling."""

    @classmethod
    def setUpClass(cls):
        user32.GetWindow.argtypes = [wintypes.HWND, wintypes.UINT]
        user32.GetWindow.restype = wintypes.HWND
        user32.GetWindowLongPtrW.argtypes = [wintypes.HWND, ctypes.c_int]
        user32.GetWindowLongPtrW.restype = ctypes.c_ssize_t

    def test_overlay_owned_by_game_follows_topmost_toggle(self):
        # Exercise native band changes on our own hidden windows. The game style
        # is a controlled input; no desktop foreground permission is required.
        owner = QHeatmapOverlay()
        overlay = QHeatmapOverlay()
        owner_hwnd = owner._create_window(0)
        hwnd = 0
        try:
            self.assertTrue(owner_hwnd)
            hwnd = overlay._create_window(owner_hwnd)
            self.assertTrue(hwnd)
            self.assertEqual(user32.GetWindow(hwnd, 4), owner_hwnd)  # GW_OWNER
            read_style = user32.GetWindowLongPtrW
            for game_style in (8, 0, 8, 0):
                with (
                    patch.object(user32, "GetWindowLongPtrW", side_effect=lambda h, index:
                                 game_style if h == owner_hwnd else read_style(h, index)),
                    patch.object(user32, "GetWindow", return_value=hwnd),
                ):
                    overlay._sync_game_z_order(hwnd, owner_hwnd)
                self.assertEqual(bool(read_style(hwnd, -20) & 8), bool(game_style & 8))
        finally:
            overlay._destroy_window(hwnd)
            owner._destroy_window(owner_hwnd)

    def test_z_order_stays_below_window_above_game(self):
        overlay_hwnd, game_hwnd, above_game = 101, 102, 103
        overlay = QHeatmapOverlay()
        with (
            patch.object(user32, "GetWindowLongPtrW", return_value=0),
            patch.object(user32, "GetWindow", return_value=above_game),
            patch.object(user32, "SetWindowPos", return_value=True) as position,
        ):
            overlay._sync_game_z_order(overlay_hwnd, game_hwnd)
        self.assertEqual(position.call_args.args[:2], (overlay_hwnd, above_game))
        self.assertTrue(position.call_args.args[-1] & 0x0200)  # don't raise owner
        self.assertTrue(position.call_args.args[-1] & 0x0010)  # don't activate

    def test_ordinary_game_at_band_boundary_does_not_promote_overlay(self):
        overlay_hwnd, game_hwnd, above_game = 101, 102, 103
        overlay = QHeatmapOverlay()
        with (
            patch.object(user32, "GetWindowLongPtrW", side_effect=lambda h, index:
                         8 if h == above_game else 0),
            patch.object(user32, "GetWindow", return_value=above_game),
            patch.object(user32, "SetWindowPos", return_value=True) as position,
        ):
            overlay._sync_game_z_order(overlay_hwnd, game_hwnd)
        self.assertEqual(position.call_args.args[:2], (overlay_hwnd, 0))  # HWND_TOP

    def test_overlay_recreated_after_game_owner_destroyed(self):
        first_game = QHeatmapOverlay()
        next_game = QHeatmapOverlay()
        overlay = QHeatmapOverlay()
        owner_hwnd = first_game._create_window(0)
        next_hwnd = next_game._create_window(0)
        hwnd = 0
        try:
            self.assertTrue(owner_hwnd)
            self.assertTrue(next_hwnd)
            hwnd = overlay._create_window(owner_hwnd)
            overlay._target_hwnd = owner_hwnd
            overlay.set_enabled(True)
            overlay.update_data([{"row": 0, "col": 0}], noop_q=1.5)
            first_game._destroy_window(owner_hwnd)
            self.assertFalse(user32.IsWindow(hwnd))

            hwnd = overlay._ensure_game_window(hwnd, next_hwnd)
            self.assertTrue(user32.IsWindow(hwnd))
            self.assertEqual(user32.GetWindow(hwnd, 4), next_hwnd)
            self.assertEqual(overlay._hwnd, hwnd)
            self.assertTrue(overlay.is_enabled())
            self.assertIsNone(overlay._noop_q)  # don't show the previous game's Q values
            overlay.update_data([{"row": 1, "col": 1}], noop_q=2.5)
            self.assertEqual(overlay._noop_q, 2.5)
        finally:
            overlay._destroy_window(hwnd)
            first_game._destroy_window(owner_hwnd)
            next_game._destroy_window(next_hwnd)

    def test_worker_waits_without_creating_window_when_game_absent(self):
        overlay = QHeatmapOverlay()
        overlay._running = True
        def no_game():
            overlay._running = False
            return 0
        with (
            patch("workbench.gui.q_heatmap_overlay.ensure_thread_desktop", return_value=None),
            patch.object(user32, "PeekMessageW", return_value=0),
            patch.object(overlay, "_find_game_hwnd", side_effect=no_game),
            patch.object(overlay, "_create_window") as create,
            patch("workbench.gui.q_heatmap_overlay.time.sleep"),
        ):
            overlay._overlay_worker()
        create.assert_not_called()
        self.assertIsNone(overlay._hwnd)
        self.assertIsNone(overlay._target_hwnd)

    def test_failed_attach_retries_after_interval_and_stays_in_worker(self):
        overlay = QHeatmapOverlay()
        overlay._running = True
        clock = [10.0]
        attempts = []
        def create(owner):
            attempts.append(owner)
            if len(attempts) == 1:
                return 0
            overlay._running = False
            return 789
        def wait(_delay):
            clock[0] += 0.25
            self.assertLess(clock[0], 12.0, "worker failed to retry")
        with (
            patch("workbench.gui.q_heatmap_overlay.ensure_thread_desktop", return_value=None),
            patch.object(user32, "PeekMessageW", return_value=0),
            patch.object(user32, "IsWindow", return_value=True),
            patch.object(user32, "IsWindowVisible", side_effect=lambda hwnd: hwnd == 123),
            patch.object(overlay, "_find_game_hwnd", return_value=123),
            patch.object(overlay, "_create_window", side_effect=create),
            patch.object(overlay, "_destroy_window"),
            patch("workbench.gui.q_heatmap_overlay.time.monotonic", side_effect=lambda: clock[0]),
            patch("workbench.gui.q_heatmap_overlay.time.sleep", side_effect=wait),
        ):
            overlay._overlay_worker()
        self.assertEqual(attempts, [123, 123])
        self.assertGreaterEqual(clock[0], 11.0)

    def test_game_disappearance_detaches_and_waits_for_reopen(self):
        overlay = QHeatmapOverlay()
        overlay._hwnd, overlay._target_hwnd = 456, 123
        overlay.set_enabled(True)
        overlay.update_data([{"row": 0, "col": 0}], noop_q=1.5)
        with (
            patch.object(overlay, "_destroy_window") as destroy,
            patch.object(overlay, "_create_window", return_value=789) as create,
            patch.object(user32, "IsWindow", return_value=True),
        ):
            self.assertEqual(overlay._ensure_game_window(456, 0), 0)
            destroy.assert_called_once_with(456)
            create.assert_not_called()
            self.assertIsNone(overlay._hwnd)
            self.assertIsNone(overlay._noop_q)
            self.assertTrue(overlay.is_enabled())
            self.assertEqual(overlay._ensure_game_window(0, 321), 789)
            create.assert_called_once_with(321)
            self.assertEqual(overlay._target_hwnd, 321)

    def test_detector_skips_hidden_windows(self):
        overlay = QHeatmapOverlay()
        overlay.bind_target(222)
        def enumerate_windows(callback, _):
            for hwnd in (101, 102, 103):
                callback(hwnd, 0)
            return True
        def class_name(_hwnd, buffer, _size):
            buffer.value = "MainWindow"
            return len(buffer.value)
        with (
            patch.object(user32, "EnumWindows", side_effect=enumerate_windows),
            patch.object(user32, "GetClassNameW", side_effect=class_name),
            patch.object(overlay, "_window_pid", side_effect=lambda hwnd: 222 if hwnd != 103 else 333),
            patch.object(user32, "IsWindowVisible", side_effect=lambda hwnd: hwnd == 102),
        ):
            self.assertEqual(overlay._find_game_hwnd(), 102)

    def test_native_overlay_window_creation(self):
        # Exercise class registration + CreateWindowExW, not just GDI drawing.
        # The overlay remains hidden and never attaches to the user's game.
        overlay = QHeatmapOverlay()
        hwnd = overlay._create_window(0)
        try:
            self.assertTrue(hwnd)
            self.assertTrue(user32.IsWindow(hwnd))
            self.assertFalse(user32.IsWindowVisible(hwnd))
            # Black labels must survive the actual native transparency setting.
            user32.GetLayeredWindowAttributes.argtypes = [wintypes.HWND,
                ctypes.POINTER(wintypes.COLORREF), ctypes.POINTER(wintypes.BYTE),
                ctypes.POINTER(wintypes.DWORD)]
            user32.GetLayeredWindowAttributes.restype = wintypes.BOOL
            color_key, alpha, flags = wintypes.COLORREF(), wintypes.BYTE(), wintypes.DWORD()
            self.assertTrue(user32.GetLayeredWindowAttributes(hwnd, ctypes.byref(color_key),
                            ctypes.byref(alpha), ctypes.byref(flags)))
            self.assertNotEqual(color_key.value, rgb(0, 0, 0))
        finally:
            overlay._destroy_window(hwnd)

    def test_disable_rejects_in_flight_updates_and_clear_preserves_selection(self):
        overlay = QHeatmapOverlay()
        self.assertFalse(overlay.is_enabled())
        overlay.set_visible(True)
        overlay.update_data([{"row": 0, "col": 0}], noop_q=1.5)
        overlay.clear()
        self.assertTrue(overlay.is_enabled())
        self.assertEqual(overlay._top5_actions, [])
        overlay.set_visible(False)
        overlay.update_data([{"row": 1, "col": 1}], noop_q=2.5)
        self.assertFalse(overlay.is_enabled())
        self.assertFalse(overlay._visible)
        self.assertEqual(overlay._top5_actions, [])
        self.assertIsNone(overlay._noop_q)

    def test_display_toggle_retains_snapshot_without_another_training_step(self):
        overlay = QHeatmapOverlay()
        actions = [{"row": 1, "col": 2, "q_value": 3.5, "desc": "向日葵"}]
        overlay.set_enabled(True)
        overlay.update_data(actions, noop_q=1.25, scene_id=2)
        for _ in range(10):
            overlay.set_enabled(False)
            self.assertFalse(overlay._visible)
            # An update already being computed at disable time must not replace
            # the last accepted frame or turn the display back on.
            overlay.update_data([], noop_q=9.0, scene_id=4)
            self.assertFalse(overlay._visible)
            overlay.set_enabled(True)
            self.assertTrue(overlay._visible)
            self.assertEqual(overlay._top5_actions, actions)
            self.assertEqual(overlay._noop_q, 1.25)
            self.assertEqual(overlay._scene_id, 2)

        # Explicit lifecycle clearing must still prevent stale episode data.
        overlay.set_enabled(False)
        overlay.clear()
        overlay.set_enabled(True)
        self.assertEqual(overlay._top5_actions, [])
        self.assertIsNone(overlay._noop_q)

    def test_worker_repaints_after_repeated_toggles_at_unchanged_geometry(self):
        # Use real native windows and GDI buffers, but simulate visibility so
        # this regression check never shows windows over the user's desktop.
        from workbench.gui.q_heatmap_overlay import gdi32, SW_HIDE

        owner = QHeatmapOverlay()
        overlay = QHeatmapOverlay()
        owner_hwnd = owner._create_window(0)
        hwnd = 0
        shown = [False]
        waits = [0]
        frames = []
        native_blt = gdi32.BitBlt

        def show(window, command):
            self.assertEqual(window, hwnd)
            shown[0] = command != SW_HIDE
            return True

        def blit(*args):
            frames.append(native_blt(*args))
            return frames[-1]

        def wait(_delay):
            waits[0] += 1
            if waits[0] >= 6:
                overlay._running = False
            else:
                overlay.set_enabled(waits[0] % 2 == 0)

        try:
            self.assertTrue(owner_hwnd)
            hwnd = overlay._create_window(owner_hwnd)
            self.assertTrue(hwnd)
            overlay.set_enabled(True)
            overlay.update_data(
                [{"row": 0, "col": 0, "q_value": 3.5, "desc": "向日葵"}],
                noop_q=1.25,
            )
            overlay._running = True
            with (
                patch("workbench.gui.q_heatmap_overlay.ensure_thread_desktop", return_value=None),
                patch.object(user32, "PeekMessageW", return_value=0),
                patch.object(user32, "IsWindowVisible", side_effect=lambda h:
                             True if h == owner_hwnd else shown[0]),
                patch.object(user32, "ShowWindow", side_effect=show) as visibility,
                patch.object(overlay, "_find_game_hwnd", return_value=owner_hwnd),
                patch.object(overlay, "_ensure_game_window", return_value=hwnd),
                patch.object(overlay, "_sync_game_z_order"),
                patch.object(user32, "SetWindowPos", return_value=True) as position,
                patch.object(gdi32, "BitBlt", side_effect=blit),
                patch("workbench.gui.q_heatmap_overlay.time.sleep", side_effect=wait),
            ):
                overlay._overlay_worker()
            self.assertEqual(len(frames), 3)
            self.assertTrue(all(frames), "native GDI frame copy failed")
            self.assertEqual([c.args[1] for c in visibility.call_args_list], [4, 0, 4, 0, 4, 0])
            self.assertEqual(position.call_count, 1)  # reopens without a move/resize
        finally:
            overlay._destroy_window(hwnd)
            owner._destroy_window(owner_hwnd)

    def test_grid_to_pixel_day_night(self):
        # Day scene (0): row 0, col 0
        x, y, w, h = grid_to_pixel(scene_id=0, row=0, col=0)
        self.assertEqual((x, y, w, h), (40, 80, 80, 100))

        # Night scene (1): row 4, col 8
        x, y, w, h = grid_to_pixel(scene_id=1, row=4, col=8)
        self.assertEqual(x, 40 + 8 * 80)
        self.assertEqual(y, 80 + 4 * 100)
        self.assertEqual((w, h), (80, 100))

    def test_grid_to_pixel_pool_fog(self):
        # Pool scene (2): row 0, col 0
        x, y, w, h = grid_to_pixel(scene_id=2, row=0, col=0)
        self.assertEqual((x, y, w, h), (40, 80, 80, 85))

        # Fog scene (3): row 5, col 8 (6 rows total)
        x, y, w, h = grid_to_pixel(scene_id=3, row=5, col=8)
        self.assertEqual(x, 40 + 8 * 80)
        self.assertEqual(y, 80 + 5 * 85)
        self.assertEqual((w, h), (80, 85))

    def test_grid_to_pixel_roof_slope(self):
        # Roof scene (4): col 0 has slope + (5 - 0) * 20 = 100
        x, y, w, h = grid_to_pixel(scene_id=4, row=0, col=0)
        self.assertEqual(x, 40)
        self.assertEqual(y, 70 + 0 * 85 + 100)  # 170
        self.assertEqual((w, h), (80, 85))

        # Roof scene (4): col 4 has slope + (5 - 4) * 20 = 20
        x, y, w, h = grid_to_pixel(scene_id=4, row=1, col=4)
        self.assertEqual(x, 40 + 4 * 80)
        self.assertEqual(y, 70 + 1 * 85 + 20)  # 175
        self.assertEqual((w, h), (80, 85))

        # Roof scene (4): col 5 flat slope = 0
        x, y, w, h = grid_to_pixel(scene_id=4, row=2, col=5)
        self.assertEqual(x, 40 + 5 * 80)
        self.assertEqual(y, 70 + 2 * 85 + 0)   # 240
        self.assertEqual((w, h), (80, 85))

    def test_rgb_helper(self):
        self.assertEqual(rgb(255, 0, 0), 0x0000FF)    # BGR format in Win32 COLORREF
        self.assertEqual(rgb(0, 255, 0), 0x00FF00)
        self.assertEqual(rgb(0, 0, 255), 0xFF0000)

    @unittest.skipIf(getattr(agent_module, "torch", None) is None, "PyTorch is required for DQN agent tests")
    def test_extract_q_visualization_data_excludes_noop_and_takes_top5(self):
        # Mock DoubleDQNAgent without heavy training
        agent = DoubleDQNAgent(action_dim=ACTION_SPACE_SIZE, config=DoubleDQNConfig(reward_scheme="staged"), device="cpu")

        # Fake observation dictionary
        from pvz_gym.observation import OBSERVATION_SPEC
        obs = {k: np.zeros(shape, dtype=dtype) for k, (shape, dtype) in OBSERVATION_SPEC.items()}
        obs["scene"] = np.array([5], dtype=np.int64)  # Roof: raw ID 4 + 1
        obs["card_type"][:3] = [1, 2, 3]  # native IDs 0, 1, 2 + 1

        # Legal mask: allow NOOP and 6 actions
        mask = np.zeros(ACTION_SPACE_SIZE, dtype=bool)
        mask[NOOP] = True
        act1 = encode_action(Action("plant", card_slot=0, row=0, col=1))
        act2 = encode_action(Action("plant", card_slot=1, row=1, col=2))
        act3 = encode_action(Action("plant", card_slot=2, row=2, col=3))
        act4 = encode_action(Action("shovel", row=3, col=4))
        act5 = encode_action(Action("shovel", row=4, col=5))
        act6 = encode_action(Action("interact_cell", row=0, col=0))

        for a in (act1, act2, act3, act4, act5, act6):
            mask[a] = True

        result = agent.extract_q_visualization_data(obs, mask=mask, top_k=5)

        self.assertIn("top5", result)
        self.assertIn("noop_q", result)
        self.assertIsInstance(result["noop_q"], float)
        self.assertEqual(result["scene_id"], 4)

        top5 = result["top5"]
        self.assertEqual(len(top5), 5)

        # Ensure NOOP is NOT in top5
        for item in top5:
            self.assertNotEqual(item["action_id"], NOOP)
            self.assertGreater(item["action_id"], 0)
            self.assertIn("row", item)
            self.assertIn("col", item)
            self.assertIn("q_value", item)
            self.assertIn("desc", item)
            self.assertTrue(0 <= item["row"] < 6)
            self.assertTrue(0 <= item["col"] < 9)
            expected_names = {act1: "豌豆射手", act2: "向日葵", act3: "樱桃炸弹",
                              act4: "铲植物", act5: "铲植物", act6: "点击格子"}
            self.assertEqual(item["desc"], expected_names[item["action_id"]])

    def test_ui_combobox_continuous_linkage(self):
        import tkinter as tk
        import workbench.main
        from workbench import PvzGymUnitTestGUI

        root = tk.Tk()
        root.withdraw()
        overlay = QHeatmapOverlay()
        overlay_patch = patch("workbench.gui.dqn_headless.get_q_heatmap_overlay", return_value=overlay)
        overlay_patch.start()
        try:
            app = PvzGymUnitTestGUI(root)
            app._use_visible_training_preset()
            self.assertTrue(hasattr(app, "cmb_training_q_overlay"))
            self.assertTrue(hasattr(app, "var_training_q_overlay"))

            # 默认 Headless 时为 disabled
            app.var_training_simulation_mode.set("Headless (Fast)")
            app._set_training_controls(False)
            self.assertEqual(str(app.cmb_training_q_overlay.cget("state")), "disabled")
            self.assertEqual(app.var_training_q_overlay.get(), "关闭")

            # 切换为 Continuous 时变为 readonly 可选
            app.var_training_simulation_mode.set("Continuous (No Pause)")
            app._set_training_controls(False)
            self.assertEqual(str(app.cmb_training_q_overlay.cget("state")), "readonly")

            # 用户选择 Top 5 热力图
            app.var_training_q_overlay.set("Top 5 热力图")
            self.assertEqual(app.var_training_q_overlay.get(), "Top 5 热力图")
            self.assertTrue(overlay.is_enabled())
            overlay.update_data([{"row": 0, "col": 0}], noop_q=1.5)
            app.var_training_q_overlay.set("关闭")
            self.assertFalse(overlay.is_enabled())
            app.var_training_q_overlay.set("Top 5 热力图")
            self.assertEqual(overlay._top5_actions, [{"row": 0, "col": 0}])
            self.assertEqual(overlay._noop_q, 1.5)

            # 切回 Stepped 时自动清空为 disabled 并设回 关闭
            app.var_training_simulation_mode.set("Stepped")
            app._set_training_controls(False)
            self.assertEqual(str(app.cmb_training_q_overlay.cget("state")), "disabled")
            self.assertEqual(app.var_training_q_overlay.get(), "关闭")
            self.assertFalse(overlay.is_enabled())
            self.assertEqual(overlay._top5_actions, [])
            self.assertIsNone(overlay._noop_q)
        finally:
            overlay_patch.stop()
            root.destroy()

    def test_gdi_render_pipeline_smoke(self):
        import ctypes
        from ctypes import wintypes
        from workbench.gui.q_heatmap_overlay import gdi32, user32

        hdc_screen = user32.GetDC(0)
        self.assertTrue(bool(hdc_screen))
        mem_dc = gdi32.CreateCompatibleDC(hdc_screen)
        mem_bmp = gdi32.CreateCompatibleBitmap(hdc_screen, 800, 600)
        old_bmp = gdi32.SelectObject(mem_dc, mem_bmp)

        h_font_bold = gdi32.CreateFontW(13, 0, 0, 0, 700, 0, 0, 0, 1, 0, 0, 5, 0, "Microsoft YaHei")
        old_font = gdi32.SelectObject(mem_dc, h_font_bold)

        # Test TextOutW
        test_str = "#1 Q:+18.4 卡0"
        ret = gdi32.TextOutW(mem_dc, 100, 100, test_str, len(test_str))
        self.assertTrue(bool(ret))

        # Test DrawTextW for NOOP
        noop_text = "NOOP Q: +12.34"
        rc = wintypes.RECT(0, 570, 800, 595)
        ret_draw = user32.DrawTextW(mem_dc, noop_text, -1, ctypes.byref(rc), 0x00000001 | 0x00000020)
        self.assertGreater(ret_draw, 0)

        # Cleanup
        gdi32.SelectObject(mem_dc, old_font)
        gdi32.SelectObject(mem_dc, old_bmp)
        gdi32.DeleteObject(h_font_bold)
        gdi32.DeleteObject(mem_bmp)
        gdi32.DeleteDC(mem_dc)
        user32.ReleaseDC(0, hdc_screen)

    def test_ui_combobox_user_select_continuous_auto_enables_q_overlay(self):
        import tkinter as tk
        import workbench.main
        from workbench import PvzGymUnitTestGUI

        root = tk.Tk()
        root.withdraw()
        overlay = QHeatmapOverlay()
        overlay_patch = patch("workbench.gui.dqn_headless.get_q_heatmap_overlay", return_value=overlay)
        overlay_patch.start()
        try:
            app = PvzGymUnitTestGUI(root)
            # 模拟用户在界面下拉框中选中 Continuous (No Pause)
            app.var_training_simulation_mode.set("Continuous (No Pause)")
            app.cmb_training_simulation.event_generate("<<ComboboxSelected>>")

            self.assertEqual(app.var_training_q_overlay.get(), "Top 5 热力图")
            self.assertTrue(overlay.is_enabled())

            # 模拟用户切回 Headless
            app.var_training_simulation_mode.set("Headless (Fast)")
            app.cmb_training_simulation.event_generate("<<ComboboxSelected>>")

            self.assertEqual(app.var_training_q_overlay.get(), "关闭")
            self.assertFalse(overlay.is_enabled())
        finally:
            overlay_patch.stop()
            root.destroy()

    def test_visible_training_preset_and_overlay_toggle_preserve_learner(self):
        import tkinter as tk
        import workbench.main
        from workbench import PvzGymUnitTestGUI
        from workbench.gui.training_runtime import runtime_settings
        from train.dqn.base_controller import DoubleDQNTrainerController

        root = tk.Tk(); root.withdraw()
        overlay = QHeatmapOverlay()
        with patch("workbench.gui.dqn_headless.get_q_heatmap_overlay", return_value=overlay):
            try:
                app = PvzGymUnitTestGUI(root)
                app._runtime_param_vars['num_envs'].set('4')
                app._runtime_param_vars['process_ids'].set('101,102,103,104')
                app._use_visible_training_preset()
                _game, runtime, _turbo = runtime_settings(app)
                self.assertEqual(runtime.num_envs, 1)
                self.assertEqual(runtime.process_ids, (101,))
                self.assertFalse(runtime.asynchronous)
                self.assertEqual(runtime.inference_batch_size, 1)
                self.assertTrue(runtime.visualize_q)
                self.assertFalse(runtime.game_topmost)
                self.assertEqual(app.var_training_simulation_mode.get(), 'Continuous (No Pause)')
                controller = DoubleDQNTrainerController(DoubleDQNConfig(), eval_every_episodes=0)
                controller._agent, controller._replay = object(), object()
                agent, replay = controller._agent, controller._replay
                app._training_running = True; app._training_controller = controller
                for _ in range(5):
                    app.var_training_q_overlay.set('关闭')
                    self.assertFalse(controller._visualize_q)
                    app.var_training_q_overlay.set('Top 5 热力图')
                    self.assertTrue(controller._visualize_q)
                self.assertIs(controller._agent, agent)
                self.assertIs(controller._replay, replay)
                app._training_running = False
            finally:
                root.destroy()


if __name__ == "__main__":
    unittest.main()
