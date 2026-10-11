# -*- coding: utf-8 -*-
"""Compatibility patch that exposes the remaining special-level fixes in the GUI.

The old UI called this feature "tutorial skip" and also exposed a global
"no Crazy Dave" patch.  Fast intro now uses the game's ESC/CancelIntro path, so
only the training-specific 1-1/1-2 state fixes remain user-visible.  The 5-10
crash-prevention patches stay native and always-on.
"""

import ctypes
import logging

from .c_interface import PvzGymNativeDll
from .gui import PvzGymUnitTestGUI, COLOR_GREEN, CLR_CARD, CLR_TEXT_MUTED
from workbench.gui.native_activity import native_worker_busy

logger = logging.getLogger("pvz_gym_unit_test.level_fix_gui_patch")


LEVEL_FIX_STATE_NAMES = {
    0: "OFF",
    1: "LEVEL_1_PICK_UP_PEASHOOTER",
    2: "LEVEL_1_PLANT_PEASHOOTER",
    3: "LEVEL_1_REFRESH_PEASHOOTER",
    4: "LEVEL_1_COMPLETED",
    5: "LEVEL_2_PICK_UP_SUNFLOWER",
    6: "LEVEL_2_PLANT_SUNFLOWER",
    7: "LEVEL_2_REFRESH_SUNFLOWER",
    8: "LEVEL_2_COMPLETED",
}


def _bind_optional(self, name, argtypes, restype):
    if not self.dll or not hasattr(self.dll, name):
        return None
    fn = getattr(self.dll, name)
    fn.argtypes = argtypes
    fn.restype = restype
    return fn


def _client_get_level_fixes(self) -> bool:
    try:
        fn = _bind_optional(self, "pvz_gym_get_auto_skip_tutorial", [], ctypes.c_bool)
        return bool(fn()) if fn else False
    except Exception as exc:
        logger.debug("get_level_fixes: %s", exc)
        return False


def _client_set_level_fixes(self, enable: bool) -> bool:
    try:
        fn = _bind_optional(
            self,
            "pvz_gym_set_auto_skip_tutorial",
            [ctypes.c_bool],
            None,
        )
        if not fn:
            return False
        fn(bool(enable))
        return True
    except Exception as exc:
        logger.debug("set_level_fixes: %s", exc)
        return False


def _client_get_level_fix_state(self) -> int:
    try:
        fn = _bind_optional(self, "pvz_gym_get_tutorial_state", [], ctypes.c_int)
        return int(fn()) if fn else -1
    except Exception as exc:
        logger.debug("get_level_fix_state: %s", exc)
        return -1


def _client_apply_level_fix(self) -> bool:
    try:
        fn = _bind_optional(self, "pvz_gym_skip_tutorial", [], ctypes.c_bool)
        return bool(fn()) if fn else False
    except Exception as exc:
        logger.debug("apply_level_fix: %s", exc)
        return False


def _refresh_level_fix_button(self):
    enabled = bool(getattr(self, "_auto_tutorial_enabled", True))
    try:
        if self.client.is_loaded:
            enabled = self.client.get_auto_skip_tutorial()
    except Exception:
        pass

    self._auto_tutorial_enabled = enabled
    btn = getattr(self, "btn_toggle_auto_tutorial", None)
    if btn is not None:
        btn.set_text(f"关卡修复: {'开启' if enabled else '关闭'}")
        btn.set_colors(COLOR_GREEN if enabled else CLR_CARD, "#ffffff" if enabled else CLR_TEXT_MUTED)
    return enabled


def _toggle_level_fixes(self):
    old = bool(getattr(self, "_auto_tutorial_enabled", True))
    try:
        old = self.client.get_auto_skip_tutorial()
    except Exception:
        pass
    new = not old
    if not self.client.set_auto_skip_tutorial(new):
        self.log("FAIL", "[*] 当前 DLL 不支持关卡修复，请重新编译 pvz_gym_native.dll")
        return
    self._auto_tutorial_enabled = new
    self._refresh_level_fix_button()
    self.log("PASS" if new else "WARN", f"[*] 关卡修复: {'开启' if new else '关闭'}")


def install_patch():
    if getattr(PvzGymUnitTestGUI, "_level_fix_patch_installed", False):
        return

    # Keep the native ABI stable; only the GUI terminology changes.
    PvzGymNativeDll.get_auto_skip_tutorial = _client_get_level_fixes
    PvzGymNativeDll.set_auto_skip_tutorial = _client_set_level_fixes
    PvzGymNativeDll.get_tutorial_state = _client_get_level_fix_state
    PvzGymNativeDll.skip_tutorial = _client_apply_level_fix

    original_build_ui = PvzGymUnitTestGUI._build_ui
    original_refresh_status = PvzGymUnitTestGUI.refresh_status
    original_update_fast_forward_ui = PvzGymUnitTestGUI._update_fast_forward_ui
    original_daemon_step = PvzGymUnitTestGUI._daemon_step

    def build_ui_with_level_fixes(self):
        original_build_ui(self)

        # The global Dave-suppression patch is no longer part of normal Gym
        # startup.  Force the legacy option off and remove its row from the UI.
        try:
            self.client.set_no_crazy_dave(False)
        except Exception as exc:
            logger.debug("disable no-dave compatibility patch: %s", exc)
        self._no_dave_enabled = False

        old_btn = getattr(self, "btn_toggle_no_dave", None)
        if old_btn is not None:
            try:
                old_btn.master.destroy()
            except Exception:
                pass
            self.btn_toggle_no_dave = None

        self._refresh_level_fix_button()

    def update_fast_forward_ui_with_level_fixes(self):
        result = original_update_fast_forward_ui(self)
        self._refresh_level_fix_button()
        return result

    def refresh_status_with_level_fixes(self):
        result = original_refresh_status(self)
        self._refresh_level_fix_button()
        return result

    def daemon_step_with_level_fixes(self):
        if getattr(self, "_window_closing", False) or not getattr(self, "root", None) or not self.root.winfo_exists():
            return
        if native_worker_busy(self):
            return original_daemon_step(self)
        try:
            if self.client.is_loaded:
                on, _ = self.client.is_game_on()
                if on and not getattr(self, "_static_test_running", False):
                    cfg = self.client.get_automation_config()
                    if cfg.get("enabled") and self.client.get_auto_skip_tutorial():
                        before = self.client.get_tutorial_state()
                        if self.client.skip_tutorial():
                            after = self.client.get_tutorial_state()
                            self.log(
                                "INFO",
                                "[*] 关卡修复: "
                                f"{before}({LEVEL_FIX_STATE_NAMES.get(before, 'UNKNOWN')}) -> "
                                f"{after}({LEVEL_FIX_STATE_NAMES.get(after, 'UNKNOWN')})",
                            )
        except Exception as exc:
            logger.debug("level fix daemon: %s", exc)

        return original_daemon_step(self)

    PvzGymUnitTestGUI.toggle_auto_tutorial = _toggle_level_fixes
    PvzGymUnitTestGUI._refresh_level_fix_button = _refresh_level_fix_button
    PvzGymUnitTestGUI._build_ui = build_ui_with_level_fixes
    PvzGymUnitTestGUI._update_fast_forward_ui = update_fast_forward_ui_with_level_fixes
    PvzGymUnitTestGUI.refresh_status = refresh_status_with_level_fixes
    PvzGymUnitTestGUI._daemon_step = daemon_step_with_level_fixes
    PvzGymUnitTestGUI._level_fix_patch_installed = True


install_patch()
