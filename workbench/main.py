# -*- coding: utf-8 -*-
"""PvZ Gym native developer workbench entrypoint.

Use ``python -m workbench.main`` or ``run_workbench.bat``.
``--cli`` runs the native test suite without opening Tk.
"""

from __future__ import annotations

import argparse
import logging
import sys
import threading
import time
from pathlib import Path

TOOLS_GYM = Path(__file__).resolve().parents[1]
PYTHON_DIR = TOOLS_GYM / "python"
for path in (TOOLS_GYM, PYTHON_DIR):
    value = str(path)
    if value not in sys.path:
        sys.path.insert(0, value)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("workbench.main")

from workbench.native.c_interface import PvzGymNativeDll
from workbench.native_tests.cases import NativeUnitTestSuite
from workbench.gui.main_window import PvzGymUnitTestGUI, launch_gui
from workbench.gui.native_activity import native_worker_busy


def _cmd_jump_1_1_without_prebattle(self):
    """测试 GUI 的快捷跳转只负责关卡切换，不自动联动 prebattle automation。"""
    self.log("INFO", "[*] 快捷跳转冒险模式 1-1...")
    cleaned = self.client.cleanup_target_dialogs()
    if cleaned:
        names = [self.client.DIALOG_NAMES.get(d, f"ID={d}") for d in cleaned]
        self.log("INFO", f"  -> 跳转前已清理残留弹窗: {', '.join(names)}")
    code, msg = self.client.select_level(1, 1)
    if code == 0:
        self.log("PASS", f"  -> 成功跳转 1-1！消息: {msg}")
    elif code == 1001:
        self.log("SKIP", f"  -> 游戏离线: {msg}")
    else:
        self.log("FAIL", f"  -> 跳转失败 (代码: {code}): {msg}")
    self.refresh_status()


def _start_prebattle_async(self, *, report_result: bool = False) -> bool:
    """统一异步启动 prebattle，避免阻塞 Tk 主线程并防止重复并发调用。"""
    if getattr(self, "_prebattle_advancing", False):
        if report_result:
            self.log("WARN", "[*] Prebattle 已在执行中，本次请求已忽略")
        return False

    self._prebattle_advancing = True
    if report_result:
        self.log("INFO", "[*] 异步执行 pvz_gym_handle_prebattle() 推进开场过场...")

    def _worker():
        t0 = time.perf_counter()
        ok = False
        error = None
        try:
            ok = self.client.handle_prebattle()
        except Exception as exc:
            error = exc
        dt = (time.perf_counter() - t0) * 1000.0

        def _finish_on_ui_thread():
            self._prebattle_advancing = False
            if report_result:
                if error is not None:
                    self.log("FAIL", f"  -> Prebattle 异常: {error} (耗时 {dt:.2f}ms)")
                elif ok:
                    state = self.client.get_runtime_state()
                    self.log(
                        "PASS",
                        f"  -> Prebattle 到达 {state.get('phase_name', 'UNKNOWN')} (耗时 {dt:.2f}ms)",
                    )
                else:
                    self.log("WARN", f"  -> 推进超时或异常: 返回 False (耗时 {dt:.2f}ms)")
            self.refresh_status()

        self.root.after(0, _finish_on_ui_thread)

    threading.Thread(target=_worker, daemon=True).start()
    return True


def _cmd_test_handle_prebattle_async(self):
    """测试推进开场过场；实际 DLL 调用始终在后台线程执行。"""
    self._start_prebattle_async(report_result=True)


def _daemon_step_with_shared_prebattle(self):
    """后台守护循环；所有阶段判断统一走 native RuntimePhase。"""
    if getattr(self, "_window_closing", False) or not getattr(self, "root", None) or not self.root.winfo_exists():
        return
    # Live / training / benchmark workers already own the native execution
    # path.  Do not let the 100 ms Tk watchdog synchronously contend for the
    # same DLL mutex: even a short native ownership handoff would otherwise
    # stall the workbench event loop and make the window appear frozen.
    if native_worker_busy(self):
        # Workers detect process exit themselves. Even is_game_on takes the
        # native mutex and can block here while a remote reset is waiting.
        self.root.after(100, self._daemon_step)
        return

    try:
        if self.client.is_loaded:
            on, _ = self.client.is_game_on()
            if on:
                if not getattr(self, "_last_game_on", False):
                    self._last_game_on = True
                    self.client.set_window_always_on_top(bool(getattr(self, "_is_topmost", False)))
                    if getattr(self, "_is_background_running", True):
                        self.client.set_background_running(True)

                runtime = self.client.get_runtime_state()
                phase = runtime.get("phase_name", "UNKNOWN")
                raw_scene = runtime.get("raw_scene", -1)
                cfg = self.client.get_automation_config()
                static_test_running = getattr(self, "_static_test_running", False)

                if not static_test_running and hasattr(self, "auto_cleanup_scene_dialogs_if_needed"):
                    self.auto_cleanup_scene_dialogs_if_needed(raw_scene)

                if not static_test_running and cfg.get("enabled") and (cfg.get("auto_collect") or cfg.get("auto_whack")) and phase == "BATTLE":
                    cur_clock = self.client.get_game_clock()
                    if cur_clock > 0 and cur_clock != self._last_tick_clock:
                        self._last_tick_clock = cur_clock
                        self.client.automation_tick()

                # 触发 C++ 原生开场自动快进与对白推进
                if not static_test_running and phase in ("LEVEL_INTRO", "DIALOGUE"):
                    self.client.cancel_intro()

                self._tick_counter = getattr(self, "_tick_counter", 0) + 1
                if self._tick_counter >= 4:
                    self._tick_counter = 0
                    self.refresh_status()
            else:
                was_on = getattr(self, "_last_game_on", False) or getattr(self, "_game_was_on", False)
                self._last_game_on = False
                self._game_was_on = False
                if was_on:
                    self.log("MUTED", "[状态变更] 游戏进程已退出/离线")
                    if getattr(self, "_live_running", False):
                        self.log("WARN", "[自动终止] 检测到游戏进程异常退出，已自动触发测试终止")
                        self._stop_live()
                    if getattr(self, "_static_test_running", False):
                        self._static_test_stop = True
                now = time.time()
                if now - getattr(self, "_last_attach_time", 0.0) >= 1.0:
                    self._last_attach_time = now
                    ok, msg = self.client.attach()
                    if ok:
                        self.log("PASS", f"[*] 轮询自动成功附加游戏进程: {msg}")
                        self._last_game_on = True
                        self._game_was_on = True
                        self.client.set_window_always_on_top(bool(getattr(self, "_is_topmost", False)))
                        if getattr(self, "_is_background_running", True):
                            self.client.set_background_running(True)
                    self.refresh_status()
    except Exception as exc:
        logger.debug("daemon step: %s", exc)

    self.root.after(100, self._daemon_step)


# Preserve the final behavior overrides from the old entrypoint.
PvzGymUnitTestGUI.cmd_jump_1_1 = _cmd_jump_1_1_without_prebattle
PvzGymUnitTestGUI._start_prebattle_async = _start_prebattle_async
PvzGymUnitTestGUI.cmd_test_handle_prebattle = _cmd_test_handle_prebattle_async
PvzGymUnitTestGUI._daemon_step = _daemon_step_with_shared_prebattle

# This patch must wrap the final daemon override above.
from workbench.legacy import tutorial_skip_gui_patch  # noqa: E402,F401


def run_cli_tests(dll_path: str | None = None) -> int:
    """在终端控制台无头运行全量 C++ 单元测试。"""
    logger.info("=" * 65)
    logger.info("  [PvZ Gym] C++ Native 自动化单元测试套件 (CLI Mode)")
    logger.info("=" * 65)

    client = PvzGymNativeDll(dll_path)
    suite = NativeUnitTestSuite(client)
    if not client.is_loaded:
        logger.warning("DLL 未能正常加载: %s", client.error_msg)

    logger.info("目标 DLL 路径: %s", client.dll_path or "未加载")
    tests = suite.get_all_tests()
    passed = failed = skipped = 0

    for idx, test in enumerate(tests, start=1):
        result = suite.run_single(test)
        if result.status == "PASS":
            passed += 1
            mark = "[✓ PASS]"
        elif result.status == "FAIL":
            failed += 1
            mark = "[✗ FAIL]"
        else:
            skipped += 1
            mark = "[- SKIP]"
        logger.info(
            "[%02d/%02d] %s %s | %s (%.2fms)",
            idx, len(tests), mark, result.case_id, result.name, result.elapsed_ms,
        )
        logger.info("       -> %s", result.message)

    logger.info("=" * 65)
    logger.info(
        "[COMPLETE] 测试统计: 总计: %d | 通过: %d | 失败: %d | 跳过: %d",
        len(tests), passed, failed, skipped,
    )
    logger.info("=" * 65)
    return 0 if failed == 0 else 1


def main() -> None:
    parser = argparse.ArgumentParser(description="PvZ Gym C++ Native 开发工作台")
    parser.add_argument("--cli", action="store_true", help="无头运行全部 native tests")
    parser.add_argument("--dll", type=str, default=None, help="显式指定 pvz_gym_native.dll 路径")
    args = parser.parse_args()
    if args.cli:
        raise SystemExit(run_cli_tests(args.dll))
    launch_gui()


if __name__ == "__main__":
    main()
