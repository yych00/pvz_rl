# -*- coding: utf-8 -*-
"""
PvZ Gym C++ Native DLL (pvz_gym_native.dll) ctypes 绑定与安全调用层
对齐 native/include/pvz/api.h 中定义的稳定 C ABI
统一 logging 框架、严禁滥用空的 except: pass
"""

import os
import sys
import json
import ctypes
import logging
import atexit
import signal
import threading
import weakref
from typing import Optional, Dict, Any, Tuple

logger = logging.getLogger("pvz_gym_unit_test.c_interface")

from pvz_gym.save_cleaner import clean_adventure_save, SaveScopeError



class PvzGymNativeDll:
    """C++ 原生库驱动客户端，提供异常捕获与安全 Fallback 保护"""

    UI_MAP = {
        -1: "未就绪/未连接 (-1)",
        0: "未知界面 (UNKNOWN)",
        1: "主菜单 (MENU)",
        2: "选卡准备 (CHOOSING_SEEDS)",
        3: "战斗中 (PLAYING)",
        4: "失败/僵尸进屋 (SCENE_ZOMBIES_WON)",
        5: "关卡通关奖励 (SCENE_AWARD)",
    }

    def __init__(self, dll_path: Optional[str] = None, *, save_dir=None,
                 profile_id: Optional[int] = None, instance_id: Optional[str] = None,
                 template_dir=None, clean_save: bool = False):
        self.dll = None
        self.dll_path = dll_path
        self.error_msg: Optional[str] = None
        self.save_dir = save_dir
        self.profile_id = profile_id
        self.instance_id = instance_id
        self.template_dir = template_dir
        self.clean_save = bool(clean_save)
        self._load_dll(dll_path)
        install_game_exit_guard(self)

    def _load_dll(self, explicit_path: Optional[str] = None):
        """按优先级搜索并加载 pvz_gym_native.dll"""
        search_paths = []
        if explicit_path:
            search_paths.append(explicit_path)

        cur_dir = os.path.dirname(os.path.abspath(__file__))
        repo_root = os.path.abspath(os.path.join(cur_dir, "..", ".."))

        search_paths.extend([
            # 安装器和构建脚本的统一安装位置，以及独立 CMake 构建产物。
            os.path.join(repo_root, "bin", "pvz_gym_native.dll"),
            os.path.join(repo_root, "build", "native-x64", "out", "Release", "pvz_gym_native.dll"),
        ])

        last_err = None
        for p in search_paths:
            if os.path.isfile(p):
                try:
                    self.dll = ctypes.windll.LoadLibrary(p)
                    self.dll_path = p
                    self._bind_functions()
                    self.error_msg = None
                    logger.info("成功载入 pvz_gym_native.dll: %s", p)
                    return
                except Exception as e:
                    last_err = e
                    logger.warning("尝试加载 DLL 路径 [%s] 失败: %s", p, e, exc_info=True)

        self.dll = None
        self.error_msg = f"无法加载 pvz_gym_native.dll (已搜寻: {search_paths})。错误: {last_err}"
        logger.error("所有备选路径均无法加载 pvz_gym_native.dll: %s", self.error_msg)

    def _bind_functions(self):
        """声明 C-API 签名 (对齐 native/include/pvz/api.h)"""
        if not self.dll:
            return

        def _safe_bind(name: str, argtypes: list, restype: Any):
            if hasattr(self.dll, name):
                fn = getattr(self.dll, name)
                fn.argtypes = argtypes
                fn.restype = restype
            else:
                logger.debug("DLL 符号 [%s] 当前未导出", name)

        # 1. 进程与连接接口
        _safe_bind("pvz_gym_attach", [], ctypes.c_bool)
        _safe_bind("pvz_gym_is_game_on", [], ctypes.c_bool)
        _safe_bind("pvz_gym_get_ui", [], ctypes.c_int)
        _safe_bind("pvz_gym_error_message", [ctypes.c_int], ctypes.c_char_p)

        # 2. 关卡操作接口
        _safe_bind("pvz_gym_select_adventure_level", [ctypes.c_int, ctypes.c_int], ctypes.c_int)
        _safe_bind("pvz_gym_select_adventure_level_id", [ctypes.c_int], ctypes.c_int)
        _safe_bind(
            "pvz_gym_get_current_adventure_level",
            [
                ctypes.POINTER(ctypes.c_int),
                ctypes.POINTER(ctypes.c_int),
                ctypes.POINTER(ctypes.c_int),
                ctypes.POINTER(ctypes.c_int)
            ],
            ctypes.c_int
        )

        # 3. 仿真时钟与时间推进接口
        _safe_bind("pvz_gym_pause", [], ctypes.c_bool)
        _safe_bind("pvz_gym_resume", [], ctypes.c_bool)
        _safe_bind("pvz_gym_is_paused", [], ctypes.c_bool)
        _safe_bind("pvz_gym_advance_ticks", [ctypes.c_int], ctypes.c_int)
        _safe_bind("pvz_gym_get_game_clock", [], ctypes.c_int64)
        _safe_bind("pvz_gym_begin_headless_mode", [], ctypes.c_bool)
        _safe_bind("pvz_gym_advance_ticks_headless", [ctypes.c_int], ctypes.c_int)
        _safe_bind("pvz_gym_end_headless_mode", [], ctypes.c_bool)
        _safe_bind("pvz_gym_get_frame_duration", [], ctypes.c_int)
        _safe_bind("pvz_gym_set_frame_duration", [ctypes.c_int], ctypes.c_bool)

        # 4. 只读观测接口
        _safe_bind("pvz_gym_get_observation_json", [ctypes.c_char_p, ctypes.c_int], ctypes.c_int)

        # 5. 自动化辅助模块 (Automation Helpers)
        _safe_bind("pvz_gym_set_automation_config", [ctypes.c_bool, ctypes.c_bool, ctypes.c_bool], None)
        _safe_bind("pvz_gym_get_automation_config", [ctypes.POINTER(ctypes.c_bool), ctypes.POINTER(ctypes.c_bool), ctypes.POINTER(ctypes.c_bool)], None)
        _safe_bind("pvz_gym_set_auto_whack", [ctypes.c_bool], None)
        _safe_bind("pvz_gym_get_auto_whack", [], ctypes.c_bool)
        _safe_bind("pvz_gym_handle_prebattle", [], ctypes.c_bool)
        _safe_bind("pvz_gym_automation_collect", [], ctypes.c_int)
        _safe_bind("pvz_gym_automation_whack", [], ctypes.c_bool)
        _safe_bind("pvz_gym_automation_tick", [], ctypes.c_int)
        _safe_bind("pvz_gym_set_auto_skip_tutorial", [ctypes.c_bool], None)
        _safe_bind("pvz_gym_get_auto_skip_tutorial", [], ctypes.c_bool)
        _safe_bind("pvz_gym_set_no_crazy_dave", [ctypes.c_bool], None)
        _safe_bind("pvz_gym_get_no_crazy_dave", [], ctypes.c_bool)
        _safe_bind("pvz_gym_set_auto_dialogue", [ctypes.c_bool], None)
        _safe_bind("pvz_gym_get_auto_dialogue", [], ctypes.c_bool)
        _safe_bind("pvz_gym_get_tutorial_state", [], ctypes.c_int)
        _safe_bind("pvz_gym_skip_tutorial", [], ctypes.c_bool)

        # 6. 选卡独立动作域 (Seed Selection)
        _safe_bind("pvz_gym_get_seed_type_capacity", [], ctypes.c_int)
        _safe_bind("pvz_gym_get_seed_selection_json", [ctypes.c_char_p, ctypes.c_int], ctypes.c_int)
        _safe_bind(
            "pvz_gym_get_seed_selection_mask",
            [ctypes.POINTER(ctypes.c_uint8), ctypes.c_int, ctypes.POINTER(ctypes.c_int)],
            ctypes.c_int,
        )
        _safe_bind("pvz_gym_select_seed", [ctypes.c_int, ctypes.c_char_p, ctypes.c_int], ctypes.c_int)
        _safe_bind("pvz_gym_select_imitater", [ctypes.c_int, ctypes.c_char_p, ctypes.c_int], ctypes.c_int)
        _safe_bind("pvz_gym_deselect_seed", [ctypes.c_int, ctypes.c_char_p, ctypes.c_int], ctypes.c_int)
        _safe_bind("pvz_gym_confirm_seeds", [ctypes.c_char_p, ctypes.c_int], ctypes.c_int)

        # 7. 辅助功能与作弊按键 (Auxiliary & Cheat Keys)
        _safe_bind("pvz_gym_set_window_always_on_top", [ctypes.c_bool], ctypes.c_bool)
        _safe_bind("pvz_gym_set_background_running", [ctypes.c_bool], None)
        _safe_bind("pvz_gym_set_cheat_keys", [ctypes.c_bool], ctypes.c_bool)
        _safe_bind("pvz_gym_get_cheat_keys", [], ctypes.c_bool)
        _safe_bind("pvz_gym_send_key", [ctypes.c_uint32], ctypes.c_bool)
        _safe_bind("pvz_gym_cancel_intro", [], ctypes.c_bool)
        _safe_bind("pvz_gym_set_auto_cancel_intro", [ctypes.c_bool], None)
        _safe_bind("pvz_gym_get_auto_cancel_intro", [], ctypes.c_bool)
        _safe_bind("pvz_gym_get_dialog", [ctypes.c_int], ctypes.c_uint32)
        _safe_bind("pvz_gym_kill_dialog", [ctypes.c_int], ctypes.c_bool)


    @property
    def is_loaded(self) -> bool:
        return self.dll is not None

    def get_exported_symbols(self) -> Dict[str, bool]:
        """探查 API 导出的符号清单"""
        expected = [
            "pvz_gym_attach",
            "pvz_gym_is_game_on",
            "pvz_gym_get_ui",
            "pvz_gym_select_adventure_level",
            "pvz_gym_select_adventure_level_id",
            "pvz_gym_get_current_adventure_level",
            "pvz_gym_error_message",
            # 仿真时钟与时间推进
            "pvz_gym_pause",
            "pvz_gym_resume",
            "pvz_gym_is_paused",
            "pvz_gym_advance_ticks",
            "pvz_gym_get_game_clock",
            # 后续动作与观测接口探针
            "pvz_gym_get_observation_json",
            "pvz_gym_can_execute_action",
            "pvz_gym_execute_action",
            "pvz_gym_get_action_mask",
            "pvz_gym_env_reset",
            "pvz_gym_env_step",
            # 自动化辅助模块 (Automation Helpers)
            "pvz_gym_set_automation_config",
            "pvz_gym_get_automation_config",
            "pvz_gym_handle_prebattle",
            "pvz_gym_automation_collect",
            "pvz_gym_automation_whack",
            "pvz_gym_automation_tick",
            "pvz_gym_set_auto_skip_tutorial",
            "pvz_gym_get_auto_skip_tutorial",
            "pvz_gym_set_no_crazy_dave",
            "pvz_gym_get_no_crazy_dave",
            "pvz_gym_set_auto_dialogue",
            "pvz_gym_get_auto_dialogue",
            "pvz_gym_get_tutorial_state",
            "pvz_gym_skip_tutorial",
            # 选卡独立动作域 (Seed Selection)
            "pvz_gym_get_seed_type_capacity",
            "pvz_gym_get_seed_selection_json",
            "pvz_gym_get_seed_selection_mask",
            "pvz_gym_select_seed",
            "pvz_gym_select_imitater",
            "pvz_gym_deselect_seed",
            "pvz_gym_confirm_seeds",
            # 辅助功能与作弊按键
            "pvz_gym_set_window_always_on_top",
            "pvz_gym_set_background_running",
            "pvz_gym_set_cheat_keys",
            "pvz_gym_get_cheat_keys",
            "pvz_gym_send_key",
            "pvz_gym_cancel_intro",
            "pvz_gym_set_auto_cancel_intro",
            "pvz_gym_get_auto_cancel_intro",
            "pvz_gym_get_dialog",
            "pvz_gym_kill_dialog",
        ]
        res = {}
        for sym in expected:
            res[sym] = hasattr(self.dll, sym) if self.dll else False
        return res

    # -------------------------------------------------------------------------
    # 核心调用方法封装
    # -------------------------------------------------------------------------

    def attach(self) -> Tuple[bool, str]:
        """附加到 PlantsVsZombies.exe"""
        if not self.dll:
            return False, "DLL 未加载"
        try:
            ok = bool(self.dll.pvz_gym_attach())
            return ok, "成功附加到游戏进程" if ok else "未找到游戏进程或附加失败"
        except Exception as e:
            logger.warning("pvz_gym_attach 调用异常: %s", e, exc_info=True)
            return False, f"调用异常: {e}"

    def is_game_on(self) -> Tuple[bool, str]:
        """检测游戏进程是否存在且有效"""
        if not self.dll:
            return False, "DLL 未加载"
        try:
            on = bool(self.dll.pvz_gym_is_game_on())
            return on, "游戏在线 (Game On)" if on else "游戏离线 (Game Off)"
        except Exception as e:
            logger.warning("pvz_gym_is_game_on 调用异常: %s", e, exc_info=True)
            return False, f"调用异常: {e}"

    def get_ui(self) -> Tuple[int, str]:
        """获取当前游戏 UI 代码与描述"""
        if not self.dll:
            return -1, "DLL 未加载"
        try:
            ui = int(self.dll.pvz_gym_get_ui())
            desc = self.UI_MAP.get(ui, f"未知代码 ({ui})")
            return ui, desc
        except Exception as e:
            logger.warning("pvz_gym_get_ui 调用异常: %s", e, exc_info=True)
            return -1, f"调用异常: {e}"

    def error_message(self, code: int) -> str:
        """解析错误码对应的错误消息"""
        if not self.dll:
            return "DLL 未加载"
        try:
            res_b = self.dll.pvz_gym_error_message(int(code))
            if res_b:
                return res_b.decode("utf-8", errors="ignore")
        except Exception as e:
            logger.warning("pvz_gym_error_message(%d) 解析异常: %s", code, e, exc_info=True)
            return f"解析异常: {e}"
        return f"未知错误 ({code})"

    def _clean_current_adventure_save(self):
        if getattr(self, "clean_save", False):
            return clean_adventure_save(getattr(self, "save_dir", None),
                                        profile_id=getattr(self, "profile_id", None),
                                        instance_id=getattr(self, "instance_id", None),
                                        template_dir=getattr(self, "template_dir", None))
        return []

    def select_level(self, area: int, sub_level: int) -> Tuple[int, str]:
        """选择指定冒险大关-小关 (如 1-1, 1-5)"""
        if not self.dll:
            return -1, "DLL 未加载"
        try:
            # 选关前定向清理可能残留的冒险模式中途存档 (game*_0.dat)，
            # 消除原版生成 DIALOG_CONTINUE(37) 导致的主线程等待与卡死
            self._clean_current_adventure_save()
            # E2E can be started after a different runner/client left the old
            # Board paused.  In UI=4/UI=5 that freezes the original terminal
            # scene and PreNewGame only changes the visible level metadata.
            # Resume by native state, not by Python-side pause ownership.
            if self.is_paused() and not self.resume():
                return -1, "选关前无法解除旧 Board 的暂停状态"
            # 跳转时主动清理可能存在的菜单/暂停/失败等残留弹窗
            self.cleanup_target_dialogs()
            code = int(self.dll.pvz_gym_select_adventure_level(int(area), int(sub_level)))
            msg = self.error_message(code)
            self.cleanup_target_dialogs()
            return code, msg
        except SaveScopeError as e:
            logger.warning("存档清理范围无法确认: %s", e)
            return -1, f"存档清理范围无法确认: {e}"
        except Exception as e:
            logger.warning("pvz_gym_select_adventure_level(%d, %d) 异常: %s", area, sub_level, e, exc_info=True)
            return -1, f"调用异常: {e}"

    def select_level_id(self, level_id: int) -> Tuple[int, str]:
        """根据绝对关卡 ID (1~50) 选择关卡"""
        if not self.dll:
            return -1, "DLL 未加载"
        try:
            # 选关前定向清理可能残留的冒险模式中途存档 (game*_0.dat)
            self._clean_current_adventure_save()
            if self.is_paused() and not self.resume():
                return -1, "选关前无法解除旧 Board 的暂停状态"
            # 跳转时主动清理可能存在的菜单/暂停/失败等残留弹窗
            self.cleanup_target_dialogs()
            code = int(self.dll.pvz_gym_select_adventure_level_id(int(level_id)))
            msg = self.error_message(code)
            self.cleanup_target_dialogs()
            return code, msg
        except SaveScopeError as e:
            logger.warning("存档清理范围无法确认: %s", e)
            return -1, f"存档清理范围无法确认: {e}"
        except Exception as e:
            logger.warning("pvz_gym_select_adventure_level_id(%d) 异常: %s", level_id, e, exc_info=True)
            return -1, f"调用异常: {e}"

    def get_current_level(self) -> Dict[str, Any]:
        """读取当前关卡信息与 UI 状态"""
        if not self.dll:
            return {"success": False, "message": "DLL 未加载"}
        try:
            c_area = ctypes.c_int(0)
            c_sub = ctypes.c_int(0)
            c_lvl = ctypes.c_int(0)
            c_ui = ctypes.c_int(-1)
            code = self.dll.pvz_gym_get_current_adventure_level(
                ctypes.byref(c_area),
                ctypes.byref(c_sub),
                ctypes.byref(c_lvl),
                ctypes.byref(c_ui)
            )
            msg = self.error_message(code)
            ui_val = c_ui.value
            return {
                "success": (code == 0),
                "code": code,
                "message": msg,
                "area": c_area.value,
                "sub_level": c_sub.value,
                "level_str": f"{c_area.value}-{c_sub.value}",
                "level_id": c_lvl.value,
                "ui": ui_val,
                "ui_desc": self.UI_MAP.get(ui_val, f"未知({ui_val})")
            }
        except Exception as e:
            logger.warning("pvz_gym_get_current_adventure_level 异常: %s", e, exc_info=True)
            return {"success": False, "message": f"调用异常: {e}"}

    def pause(self) -> bool:
        """暂停游戏战斗"""
        if not self.dll or not hasattr(self.dll, "pvz_gym_pause"):
            return False
        try:
            return bool(self.dll.pvz_gym_pause())
        except Exception as e:
            logger.warning("pvz_gym_pause 异常: %s", e, exc_info=True)
            return False

    def resume(self) -> bool:
        """恢复游戏战斗"""
        if not self.dll or not hasattr(self.dll, "pvz_gym_resume"):
            return False
        try:
            return bool(self.dll.pvz_gym_resume())
        except Exception as e:
            logger.warning("pvz_gym_resume 异常: %s", e, exc_info=True)
            return False

    def is_paused(self) -> bool:
        """检查当前战斗是否暂停"""
        if not self.dll or not hasattr(self.dll, "pvz_gym_is_paused"):
            return False
        try:
            return bool(self.dll.pvz_gym_is_paused())
        except Exception as e:
            logger.warning("pvz_gym_is_paused 异常: %s", e, exc_info=True)
            return False

    def end_headless_mode(self) -> bool:
        """退出无头仿真模式，还原主循环代码补丁并唤醒挂起的线程"""
        if not self.dll or not hasattr(self.dll, "pvz_gym_end_headless_mode"):
            return False
        try:
            return bool(self.dll.pvz_gym_end_headless_mode())
        except Exception as e:
            logger.warning("pvz_gym_end_headless_mode 异常: %s", e, exc_info=True)
            return False

    def set_frame_duration(self, time_ms: int = 10) -> bool:
        """设置原版引擎帧时长 (10ms 为正常速度)"""
        if not self.dll or not hasattr(self.dll, "pvz_gym_set_frame_duration"):
            return False
        try:
            return bool(self.dll.pvz_gym_set_frame_duration(int(time_ms)))
        except Exception as e:
            logger.warning("pvz_gym_set_frame_duration 异常: %s", e, exc_info=True)
            return False

    def safe_restore_game(self) -> bool:
        """安全解冻并恢复游戏正常运行状态（还原主循环补丁、唤醒挂起线程、取消暂停、恢复正常速度）"""
        if not self.dll:
            return False
        restored = False
        try:
            if hasattr(self.dll, "pvz_gym_end_headless_mode"):
                self.dll.pvz_gym_end_headless_mode()
                restored = True
        except Exception as e:
            logger.debug("safe_restore_game end_headless_mode 异常: %s", e)
        try:
            self.resume()
            restored = True
        except Exception as e:
            logger.debug("safe_restore_game resume 异常: %s", e)
        try:
            self.set_frame_duration(10)
        except Exception as e:
            logger.debug("safe_restore_game set_frame_duration 异常: %s", e)
        return restored

    def advance_ticks(self, ticks: int) -> int:
        """推进真实游戏固定数量的 Tick (1 tick = 10ms 逻辑帧)"""
        if not self.dll or not hasattr(self.dll, "pvz_gym_advance_ticks"):
            return -1
        try:
            return int(self.dll.pvz_gym_advance_ticks(int(ticks)))
        except Exception as e:
            logger.warning("pvz_gym_advance_ticks 异常: %s", e, exc_info=True)
            return -1

    def get_game_clock(self) -> int:
        """读取游戏引擎权威逻辑帧时钟"""
        if not self.dll or not hasattr(self.dll, "pvz_gym_get_game_clock"):
            return 0
        try:
            return int(self.dll.pvz_gym_get_game_clock())
        except Exception as e:
            logger.warning("pvz_gym_get_game_clock 异常: %s", e, exc_info=True)
            return 0

    def get_observation_json(self, buf_size: int = 65536) -> Tuple[int, str, Optional[Dict[str, Any]]]:
        """读取当前时刻的只读观测快照 JSON 文本并解析为 dict"""
        import json
        if not self.dll or not hasattr(self.dll, "pvz_gym_get_observation_json"):
            return -1, "DLL 未加载或未导出 pvz_gym_get_observation_json", None
        try:
            buf = ctypes.create_string_buffer(buf_size)
            written = int(self.dll.pvz_gym_get_observation_json(buf, buf_size))
            if written > 0:
                raw_str = buf.value.decode("utf-8", errors="replace")
                data = json.loads(raw_str)
                return written, raw_str, data
            elif written == 0:
                return 0, "当前没有可用 Observation (未在战斗中或未附加)", None
            else:
                return written, f"缓冲区不足或执行错误 (code: {written})", None
        except Exception as e:
            logger.warning("pvz_gym_get_observation_json 异常: %s", e, exc_info=True)
            return -1, f"调用异常: {e}", None

    # -------------------------------------------------------------------------
    # 5. 自动化辅助模块 (Automation Helpers: testing & training conveniences)
    # -------------------------------------------------------------------------

    def set_automation_config(self, enabled: bool, auto_dialogue: bool, auto_collect: bool) -> bool:
        """设置自动化辅助配置 (总开关, 自动对话, 自动收集)"""
        if not self.dll or not hasattr(self.dll, "pvz_gym_set_automation_config"):
            return False
        try:
            self.dll.pvz_gym_set_automation_config(bool(enabled), bool(auto_dialogue), bool(auto_collect))
            return True
        except Exception as e:
            logger.warning("pvz_gym_set_automation_config 异常: %s", e, exc_info=True)
            return False

    def get_automation_config(self) -> Dict[str, bool]:
        """获取当前自动化辅助配置"""
        if not self.dll or not hasattr(self.dll, "pvz_gym_get_automation_config"):
            return {"enabled": False, "auto_dialogue": False, "auto_collect": False}
        try:
            e = ctypes.c_bool()
            d = ctypes.c_bool()
            c = ctypes.c_bool()
            self.dll.pvz_gym_get_automation_config(ctypes.byref(e), ctypes.byref(d), ctypes.byref(c))
            auto_whack = self.get_auto_whack()
            return {"enabled": bool(e.value), "auto_dialogue": bool(d.value),
                    "auto_collect": bool(c.value), "auto_whack": auto_whack}
        except Exception as e:
            logger.warning("pvz_gym_get_automation_config 异常: %s", e, exc_info=True)
            return {"enabled": False, "auto_dialogue": False, "auto_collect": False}

    def set_auto_whack(self, enabled: bool) -> bool:
        if not self.dll or not hasattr(self.dll, "pvz_gym_set_auto_whack"):
            return False
        self.dll.pvz_gym_set_auto_whack(bool(enabled))
        return True

    def get_auto_whack(self) -> bool:
        if not self.dll or not hasattr(self.dll, "pvz_gym_get_auto_whack"):
            return False
        return bool(self.dll.pvz_gym_get_auto_whack())

    def handle_prebattle(self) -> bool:
        """推进开场过场；Crazy Dave 对话由游戏事件钩子自动处理。"""
        if not self.dll or not hasattr(self.dll, "pvz_gym_handle_prebattle"):
            return False
        try:
            return bool(self.dll.pvz_gym_handle_prebattle())
        except Exception as e:
            logger.warning("pvz_gym_handle_prebattle 异常: %s", e, exc_info=True)
            return False

    def automation_collect(self) -> int:
        """执行一次全屏常规掉落物自动收集 (返回本次成功标记收集数量)"""
        if not self.dll or not hasattr(self.dll, "pvz_gym_automation_collect"):
            return -1
        try:
            return int(self.dll.pvz_gym_automation_collect())
        except Exception as e:
            logger.warning("pvz_gym_automation_collect 异常: %s", e, exc_info=True)
            return -1

    def automation_whack(self) -> bool:
        """在锤僵尸关卡中尝试锤击一次。"""
        if not self.dll or not hasattr(self.dll, "pvz_gym_automation_whack"):
            return False
        return bool(self.dll.pvz_gym_automation_whack())

    def automation_tick(self) -> int:
        """执行统一自动化更新：教学归一化，并按配置自动收集。"""
        if not self.dll or not hasattr(self.dll, "pvz_gym_automation_tick"):
            return -1
        try:
            return int(self.dll.pvz_gym_automation_tick())
        except Exception as e:
            logger.warning("pvz_gym_automation_tick 异常: %s", e, exc_info=True)
            return -1

    def set_auto_skip_tutorial(self, enabled: bool) -> bool:
        """启用或关闭教学状态自动归一化。"""
        if not self.dll or not hasattr(self.dll, "pvz_gym_set_auto_skip_tutorial"):
            return False
        try:
            self.dll.pvz_gym_set_auto_skip_tutorial(bool(enabled))
            return True
        except Exception as e:
            logger.warning("pvz_gym_set_auto_skip_tutorial 异常: %s", e, exc_info=True)
            return False

    def get_auto_skip_tutorial(self) -> bool:
        if not self.dll or not hasattr(self.dll, "pvz_gym_get_auto_skip_tutorial"):
            return False
        try:
            return bool(self.dll.pvz_gym_get_auto_skip_tutorial())
        except Exception as e:
            logger.warning("pvz_gym_get_auto_skip_tutorial 异常: %s", e, exc_info=True)
            return False

    def set_no_crazy_dave(self, enabled: bool) -> bool:
        """启用或关闭关卡全局屏蔽戴夫 (补丁法)"""
        if not self.dll or not hasattr(self.dll, "pvz_gym_set_no_crazy_dave"):
            return False
        try:
            self.dll.pvz_gym_set_no_crazy_dave(bool(enabled))
            return True
        except Exception as e:
            logger.warning("pvz_gym_set_no_crazy_dave 异常: %s", e, exc_info=True)
            return False

    def get_no_crazy_dave(self) -> bool:
        """获取当前是否启用关卡全局屏蔽戴夫"""
        if not self.dll or not hasattr(self.dll, "pvz_gym_get_no_crazy_dave"):
            return False
        try:
            return bool(self.dll.pvz_gym_get_no_crazy_dave())
        except Exception as e:
            logger.warning("pvz_gym_get_no_crazy_dave 异常: %s", e, exc_info=True)
            return False

    def set_auto_dialogue(self, enabled: bool) -> bool:
        """启用或关闭自动推进戴夫对话 (按键法)"""
        if not self.dll or not hasattr(self.dll, "pvz_gym_set_auto_dialogue"):
            return False
        try:
            self.dll.pvz_gym_set_auto_dialogue(bool(enabled))
            return True
        except Exception as e:
            logger.warning("pvz_gym_set_auto_dialogue 异常: %s", e, exc_info=True)
            return False

    def get_auto_dialogue(self) -> bool:
        """获取当前是否启用自动推进戴夫对话"""
        if not self.dll or not hasattr(self.dll, "pvz_gym_get_auto_dialogue"):
            return False
        try:
            return bool(self.dll.pvz_gym_get_auto_dialogue())
        except Exception as e:
            logger.warning("pvz_gym_get_auto_dialogue 异常: %s", e, exc_info=True)
            return False

    def get_tutorial_state(self) -> int:
        if not self.dll or not hasattr(self.dll, "pvz_gym_get_tutorial_state"):
            return -1
        try:
            return int(self.dll.pvz_gym_get_tutorial_state())
        except Exception as e:
            logger.warning("pvz_gym_get_tutorial_state 异常: %s", e, exc_info=True)
            return -1

    def skip_tutorial(self) -> bool:
        if not self.dll or not hasattr(self.dll, "pvz_gym_skip_tutorial"):
            return False
        try:
            return bool(self.dll.pvz_gym_skip_tutorial())
        except Exception as e:
            logger.warning("pvz_gym_skip_tutorial 异常: %s", e, exc_info=True)
            return False

    def set_window_always_on_top(self, enable: bool) -> bool:
        """设置游戏窗口置顶状态"""
        if not self.dll or not hasattr(self.dll, "pvz_gym_set_window_always_on_top"):
            return False
        try:
            return bool(self.dll.pvz_gym_set_window_always_on_top(bool(enable)))
        except Exception as e:
            logger.warning("pvz_gym_set_window_always_on_top(%s) 异常: %s", enable, e, exc_info=True)
            return False

    def set_background_running(self, enable: bool) -> bool:
        """设置游戏后台运行（失去焦点不自动暂停）"""
        if not self.dll or not hasattr(self.dll, "pvz_gym_set_background_running"):
            return False
        try:
            self.dll.pvz_gym_set_background_running(bool(enable))
            return True
        except Exception as e:
            logger.warning("pvz_gym_set_background_running(%s) 异常: %s", enable, e, exc_info=True)
            return False

    def set_cheat_keys(self, enable: bool) -> bool:
        """设置游戏内置作弊按键 (mTodCheatKeys at LawnApp + 0x7F5)"""
        if not self.dll or not hasattr(self.dll, "pvz_gym_set_cheat_keys"):
            return False
        try:
            return bool(self.dll.pvz_gym_set_cheat_keys(bool(enable)))
        except Exception as e:
            logger.warning("pvz_gym_set_cheat_keys(%s) 异常: %s", enable, e, exc_info=True)
            return False

    def get_cheat_keys(self) -> bool:
        """获取当前游戏是否开启内置作弊按键"""
        if not self.dll or not hasattr(self.dll, "pvz_gym_get_cheat_keys"):
            return False
        try:
            return bool(self.dll.pvz_gym_get_cheat_keys())
        except Exception as e:
            logger.warning("pvz_gym_get_cheat_keys 异常: %s", e, exc_info=True)
            return False

    def send_key(self, vk_code: int) -> bool:
        """向游戏主窗口发送键盘按键消息 (WM_KEYDOWN + WM_KEYUP)"""
        if not self.dll or not hasattr(self.dll, "pvz_gym_send_key"):
            return False
        try:
            return bool(self.dll.pvz_gym_send_key(int(vk_code)))
        except Exception as e:
            logger.warning("pvz_gym_send_key(%s) 异常: %s", vk_code, e, exc_info=True)
            return False

    def cancel_intro(self) -> bool:
        """直接调用已验证版本的原版 CutScene::CancelIntro()。"""
        if not self.dll or not hasattr(self.dll, "pvz_gym_cancel_intro"):
            return False
        try:
            return bool(self.dll.pvz_gym_cancel_intro())
        except Exception as e:
            logger.warning("pvz_gym_cancel_intro 异常: %s", e, exc_info=True)
            return False

    def set_auto_cancel_intro(self, enabled: bool) -> bool:
        """设置是否自动跳过关卡开场过场动画 (CutScene::CancelIntro)。"""
        if not self.dll or not hasattr(self.dll, "pvz_gym_set_auto_cancel_intro"):
            return False
        try:
            self.dll.pvz_gym_set_auto_cancel_intro(bool(enabled))
            return True
        except Exception as e:
            logger.warning("pvz_gym_set_auto_cancel_intro 异常: %s", e, exc_info=True)
            return False

    def get_auto_cancel_intro(self) -> bool:
        """获取是否自动跳过关卡开场过场动画。"""
        if not self.dll or not hasattr(self.dll, "pvz_gym_get_auto_cancel_intro"):
            return True
        try:
            return bool(self.dll.pvz_gym_get_auto_cancel_intro())
        except Exception as e:
            logger.warning("pvz_gym_get_auto_cancel_intro 异常: %s", e, exc_info=True)
            return True

    # -------------------------------------------------------------------------
    # Dialog 弹窗管理接口与场景切换清理
    # -------------------------------------------------------------------------
    DIALOG_NEWOPTIONS = 2  # 墓碑选项/菜单设置框 (游戏右上角“菜单”按钮弹出)
    DIALOG_ALMANAC = 3     # 图鉴 (独立于 Board 的模态弹窗)
    DIALOG_GAME_OVER = 17  # 游戏失败/吃脑重新开始框
    DIALOG_PAUSED = 19     # 战斗暂停菜单框 (ESC 弹出)

    TARGET_SCENE_DIALOGS = (2, 3, 19, 17)
    DIALOG_NAMES = {
        2: "DIALOG_NEWOPTIONS(2/菜单)",
        3: "DIALOG_ALMANAC(3/图鉴)",
        17: "DIALOG_GAME_OVER(17/失败)",
        19: "DIALOG_PAUSED(19/暂停)",
    }

    def get_dialog(self, dialog_id: int) -> int:
        """查询指定 ID 的 Dialog 是否存在，返回远程指针地址 (未找到或失败返回 0)"""
        if not self.dll or not hasattr(self.dll, "pvz_gym_get_dialog"):
            return 0
        try:
            return int(self.dll.pvz_gym_get_dialog(int(dialog_id)))
        except Exception as e:
            logger.warning("pvz_gym_get_dialog(%d) 异常: %s", dialog_id, e, exc_info=True)
            return 0

    def kill_dialog(self, dialog_id: int) -> bool:
        """销毁/关闭指定 ID 的 Dialog"""
        if not self.dll or not hasattr(self.dll, "pvz_gym_kill_dialog"):
            return False
        try:
            return bool(self.dll.pvz_gym_kill_dialog(int(dialog_id)))
        except Exception as e:
            logger.warning("pvz_gym_kill_dialog(%d) 异常: %s", dialog_id, e, exc_info=True)
            return False

    def cleanup_target_dialogs(self, dialog_ids=None) -> list[int]:
        """
        主动清理目标弹窗 (默认清理 2=菜单, 3=图鉴, 19=暂停, 17=失败)。
        返回成功清理的 dialog_id 列表。
        """
        if dialog_ids is None:
            dialog_ids = self.TARGET_SCENE_DIALOGS
        cleaned = []
        for did in dialog_ids:
            try:
                ptr = self.get_dialog(did)
                if ptr != 0:
                    ok = self.kill_dialog(did)
                    if ok:
                        cleaned.append(did)
                        logger.info(
                            "主动清理残留弹窗: %s, ptr=0x%X",
                            self.DIALOG_NAMES.get(did, f"Dialog({did})"),
                            ptr,
                        )
            except Exception as e:
                logger.warning("清理 Dialog %d 失败: %s", did, e)
        if 19 in cleaned:
            try:
                if self.is_paused():
                    self.resume()
            except Exception as e:
                logger.warning("清理 DIALOG_PAUSED 后恢复运行失败: %s", e)
        return cleaned

    def cleanup_scene_dialogs(self, old_ui: int, new_ui: int) -> list[int]:
        """
        场景切换时的弹窗清理钩子:
        当在 ui(2, 3, 4, 5) 之间发生场景转换时 (且 old_ui != new_ui)，
        检测是否存在残留的菜单、图鉴、暂停、失败 Dialog 并调用 KillDialog 清理。
        UI 未变或不在 (2, 3, 4, 5) 转换范围内时不触发。
        返回被成功清理的 dialog_id 列表。
        """
        valid_scenes = {2, 3, 4, 5}
        if old_ui == new_ui:
            return []
        if old_ui not in valid_scenes or new_ui not in valid_scenes:
            return []

        return self.cleanup_target_dialogs()

    # -------------------------------------------------------------------------
    # 7. 选卡独立动作域接口 (Seed Selection)
    # -------------------------------------------------------------------------

    def get_seed_type_capacity(self) -> int:
        """获取选卡动作域植物容量 (固定 ABI 容量，默认 64)"""
        if not self.dll or not hasattr(self.dll, "pvz_gym_get_seed_type_capacity"):
            return 0
        try:
            return int(self.dll.pvz_gym_get_seed_type_capacity())
        except Exception as e:
            logger.warning("pvz_gym_get_seed_type_capacity 异常: %s", e, exc_info=True)
            return 0

    def get_seed_selection_json(self) -> Tuple[int, str, Dict[str, Any]]:
        """获取当前时刻选卡状态 JSON 快照 (包含 present, required_count, selected_seeds, options 等)"""
        if not self.dll or not hasattr(self.dll, "pvz_gym_get_seed_selection_json"):
            return -1, "DLL 未导出 pvz_gym_get_seed_selection_json", {}
        buf_size = 16384
        buf = ctypes.create_string_buffer(buf_size)
        try:
            ret = int(self.dll.pvz_gym_get_seed_selection_json(buf, buf_size))
            if ret < 0:
                return ret, "缓冲区过小", {}
            if ret == 0:
                return 0, "游戏未在线或未处于选卡状态", {}
            raw = buf.value.decode("utf-8", errors="replace")
            data = json.loads(raw)
            return ret, raw, data
        except Exception as e:
            logger.warning("pvz_gym_get_seed_selection_json 异常: %s", e, exc_info=True)
            return -1, str(e), {}

    def get_seed_selection_mask(self) -> list[bool]:
        """获取固定 193 维选卡动作掩码。"""
        if not self.dll or not hasattr(self.dll, "pvz_gym_get_seed_selection_mask"):
            return []
        capacity = self.get_seed_type_capacity()
        if capacity <= 0:
            capacity = 64
        if capacity != 64:
            return []
        expected = capacity * 3 + 1
        buf = (ctypes.c_uint8 * expected)()
        out_count = ctypes.c_int(0)
        try:
            ret = int(self.dll.pvz_gym_get_seed_selection_mask(buf, expected, ctypes.byref(out_count)))
            if ret < 0:
                return []
            if out_count.value != expected:
                return []
            return [bool(buf[i]) for i in range(expected)]
        except Exception as e:
            logger.warning("pvz_gym_get_seed_selection_mask 异常: %s", e, exc_info=True)
            return []

    def select_seed(self, plant_type: int) -> Tuple[int, str]:
        """执行选择植物卡片 (返回错误码与消息)"""
        if not self.dll or not hasattr(self.dll, "pvz_gym_select_seed"):
            return -1, "DLL 未导出 pvz_gym_select_seed"
        buf = ctypes.create_string_buffer(512)
        try:
            code = int(self.dll.pvz_gym_select_seed(int(plant_type), buf, len(buf)))
            msg = buf.value.decode("utf-8", errors="replace")
            return code, msg
        except Exception as e:
            logger.warning("pvz_gym_select_seed 异常: %s", e, exc_info=True)
            return -1, str(e)

    def deselect_seed(self, plant_type: int) -> Tuple[int, str]:
        """执行撤回已选植物卡片 (返回错误码与消息)"""
        if not self.dll or not hasattr(self.dll, "pvz_gym_deselect_seed"):
            return -1, "DLL 未导出 pvz_gym_deselect_seed"
        buf = ctypes.create_string_buffer(512)
        try:
            code = int(self.dll.pvz_gym_deselect_seed(int(plant_type), buf, len(buf)))
            msg = buf.value.decode("utf-8", errors="replace")
            return code, msg
        except Exception as e:
            logger.warning("pvz_gym_deselect_seed 异常: %s", e, exc_info=True)
            return -1, str(e)

    def select_imitater(self, target_type: int) -> Tuple[int, str]:
        """选择模仿者并指定要模仿的植物。"""
        if not self.dll or not hasattr(self.dll, "pvz_gym_select_imitater"):
            return -1, "DLL 未导出 pvz_gym_select_imitater"
        buf = ctypes.create_string_buffer(512)
        try:
            code = int(self.dll.pvz_gym_select_imitater(int(target_type), buf, len(buf)))
            return code, buf.value.decode("utf-8", errors="replace")
        except Exception as e:
            logger.warning("pvz_gym_select_imitater 异常: %s", e, exc_info=True)
            return -1, str(e)

    def confirm_seeds(self) -> Tuple[int, str]:
        """确认选卡并进入战斗 (Let's Rock，返回错误码与消息)"""
        if not self.dll or not hasattr(self.dll, "pvz_gym_confirm_seeds"):
            return -1, "DLL 未导出 pvz_gym_confirm_seeds"
        buf = ctypes.create_string_buffer(512)
        try:
            code = int(self.dll.pvz_gym_confirm_seeds(buf, len(buf)))
            msg = buf.value.decode("utf-8", errors="replace")
            return code, msg
        except Exception as e:
            logger.warning("pvz_gym_confirm_seeds 异常: %s", e, exc_info=True)
            return -1, str(e)

    def __del__(self):
        try:
            self.safe_restore_game()
        except Exception:
            pass


# -----------------------------------------------------------------------------
# 全局退出与异常状态恢复守护 (Exit & Exception Guard)
# -----------------------------------------------------------------------------
_GLOBAL_CLIENT_INSTANCES = weakref.WeakSet()
_EXIT_GUARD_INSTALLED = False
_CONSOLE_CTRL_HANDLER_REF = None


def global_restore_all_games() -> None:
    """遍历所有活跃的 PvZGymNativeDll 客户端，执行安全解冻"""
    for client in list(_GLOBAL_CLIENT_INSTANCES):
        try:
            client.safe_restore_game()
        except Exception:
            pass


def install_game_exit_guard(client: Optional['PvzGymNativeDll'] = None) -> None:
    """注册进程级退出守护：在 Python 退出、被强关、Ctrl+C 或异常时自动恢复游戏"""
    global _EXIT_GUARD_INSTALLED, _CONSOLE_CTRL_HANDLER_REF
    if client is not None:
        _GLOBAL_CLIENT_INSTANCES.add(client)

    if _EXIT_GUARD_INSTALLED:
        return
    _EXIT_GUARD_INSTALLED = True

    # 1. 注册 atexit 钩子（正常退出、sys.exit、未捕获异常退出）
    atexit.register(global_restore_all_games)

    # 2. 注册常见信号 (SIGINT, SIGTERM)
    def _sig_handler(signum, frame):
        logger.info("收到退出信号 %s，正在安全解冻游戏...", signum)
        global_restore_all_games()
        signal.signal(signum, signal.SIG_DFL)
        sys.exit(128 + signum)

    try:
        if threading.current_thread() is threading.main_thread():
            signal.signal(signal.SIGINT, _sig_handler)
            if hasattr(signal, "SIGTERM"):
                signal.signal(signal.SIGTERM, _sig_handler)
            if hasattr(signal, "SIGBREAK"):
                signal.signal(signal.SIGBREAK, _sig_handler)
    except Exception as e:
        logger.debug("注册信号钩子失败: %s", e)

    # 3. Windows 控制台关闭事件 (点击控制台黑框右上角 X)
    if sys.platform == "win32":
        try:
            from ctypes import wintypes
            HandlerRoutine = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.DWORD)

            def _console_handler(ctrl_type):
                # 0: CTRL_C_EVENT, 2: CTRL_CLOSE_EVENT, 5: CTRL_LOGOFF_EVENT, 6: CTRL_SHUTDOWN_EVENT
                global_restore_all_games()
                return False

            _CONSOLE_CTRL_HANDLER_REF = HandlerRoutine(_console_handler)
            ctypes.windll.kernel32.SetConsoleCtrlHandler(_CONSOLE_CTRL_HANDLER_REF, True)
        except Exception as e:
            logger.debug("注册 Windows 控制台钩子失败: %s", e)
