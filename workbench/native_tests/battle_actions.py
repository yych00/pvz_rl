# -*- coding: utf-8 -*-
"""Battle Action ABI bindings and GUI-visible native unit tests.

This module is installed by pvz_gym_unit_test.__init__ before the GUI creates
PvzGymNativeDll.  It keeps the Battle Action tests isolated from the already
large legacy test_cases.py while still making them appear in the same GUI list.
"""

from __future__ import annotations

import ctypes
import logging
from typing import Any

logger = logging.getLogger("pvz_gym_unit_test.battle_action_tests")

ACTION_SPACE_SIZE = 757
PLANT_BASE = 1
MAX_CARD_SLOTS = 10
MAX_ROWS = 6
COLS = 9
SHOVEL_NORMAL_BASE = 541
SHOVEL_PUMPKIN_BASE = 595
COB_BASE = 649
COB_TARGET_CELLS = MAX_ROWS * COLS

ACTION_NOOP = 0
ACTION_PLANT = 1
ACTION_SHOVEL = 2
ACTION_COB_FIRE = 3

SHOVEL_NORMAL = 0
SHOVEL_PUMPKIN = 1

ERR_NONE = 0
ERR_GAME_NOT_FOUND = 1001
ERR_INVALID_ARGUMENT = 1002
ERR_SLOT_OUT_OF_RANGE = 1005
ERR_INVALID_GRID = 1006
ERR_NO_PLANT_TO_SHOVEL = 1014
ERR_COB_TARGET_OUT_OF_ARENA = 1016
ERR_NOT_IN_BATTLE = 1017
ERR_COB_NOT_FOUND = 1018


def _client_can_execute_action(
    self,
    action_type: int,
    *,
    card_slot: int = -1,
    row: int = -1,
    col: int = -1,
    shovel_target: int = SHOVEL_NORMAL,
    target_row: int = -1,
    target_col: int = -1,
):
    """Call the authoritative seven-int battle-action legality ABI."""
    if not self.dll or not hasattr(self.dll, "pvz_gym_can_execute_action"):
        return -1, "DLL 未导出 pvz_gym_can_execute_action"
    try:
        code = int(self.dll.pvz_gym_can_execute_action(
            int(action_type), int(card_slot), int(row), int(col),
            int(shovel_target), int(target_row), int(target_col),
        ))
        return code, self.error_message(code) if code >= 0 else f"native error {code}"
    except Exception as exc:
        logger.warning("pvz_gym_can_execute_action 调用异常: %s", exc, exc_info=True)
        return -1, f"调用异常: {exc}"


def _client_execute_action(
    self,
    action_type: int,
    *,
    card_slot: int = -1,
    row: int = -1,
    col: int = -1,
    shovel_target: int = SHOVEL_NORMAL,
    target_row: int = -1,
    target_col: int = -1,
):
    """Execute one battle action without advancing Gym ticks."""
    if not self.dll or not hasattr(self.dll, "pvz_gym_execute_action"):
        return -1, "DLL 未导出 pvz_gym_execute_action"
    buf = ctypes.create_string_buffer(512)
    try:
        code = int(self.dll.pvz_gym_execute_action(
            int(action_type), int(card_slot), int(row), int(col),
            int(shovel_target), int(target_row), int(target_col),
            buf, len(buf),
        ))
        msg = buf.value.decode("utf-8", errors="replace")
        if not msg:
            msg = self.error_message(code) if code >= 0 else f"native error {code}"
        return code, msg
    except Exception as exc:
        logger.warning("pvz_gym_execute_action 调用异常: %s", exc, exc_info=True)
        return -1, f"调用异常: {exc}"


def _client_get_action_mask(self, capacity: int = ACTION_SPACE_SIZE):
    """Return (code, mask, out_count, message) for the native battle mask."""
    if not self.dll or not hasattr(self.dll, "pvz_gym_get_action_mask"):
        return -1, [], 0, "DLL 未导出 pvz_gym_get_action_mask"
    capacity = max(1, int(capacity))
    buf = (ctypes.c_uint8 * capacity)()
    out_count = ctypes.c_int(0)
    try:
        code = int(self.dll.pvz_gym_get_action_mask(buf, capacity, ctypes.byref(out_count)))
        count = max(0, int(out_count.value))
        if code == 0:
            mask = [bool(buf[i]) for i in range(min(count, capacity))]
            return code, mask, count, "success"
        if code < 0:
            return code, [], count, f"native 要求至少 {-code} 字节动作掩码缓冲区"
        return code, [], count, self.error_message(code)
    except Exception as exc:
        logger.warning("pvz_gym_get_action_mask 调用异常: %s", exc, exc_info=True)
        return -1, [], 0, f"调用异常: {exc}"


def _skip_unless_live_battle(suite: Any, result_cls: Any, case_id: str, name: str):
    if not suite.client.is_loaded:
        return result_cls(case_id, "BattleAction", name, "SKIP", "DLL 未加载", 0)

    on, _ = suite.client.is_game_on()
    if not on:
        suite.client.attach()
        on, _ = suite.client.is_game_on()
    if not on:
        return result_cls(case_id, "BattleAction", name, "SKIP", "游戏未在线", 0)

    ui, ui_desc = suite.client.get_ui()
    if ui != 3:
        return result_cls(case_id, "BattleAction", name, "SKIP", f"当前不是战斗 UI: {ui_desc}", 0)
    return None


def _test_battle_action_abi(self, result_cls):
    case_id = "TC-32"
    name = "Battle Action 7 参数 ABI 与 3 个核心符号签名测试"
    if not self.client.is_loaded:
        return result_cls(case_id, "BattleAction", name, "SKIP", "DLL 未加载", 0)

    required = [
        "pvz_gym_can_execute_action",
        "pvz_gym_execute_action",
        "pvz_gym_get_action_mask",
    ]
    missing = [sym for sym in required if not hasattr(self.client.dll, sym)]
    if missing:
        return result_cls(case_id, "BattleAction", name, "FAIL", f"缺少 Battle Action C ABI: {missing}", 0)

    actual_counts = {
        "can_execute": len(getattr(self.client.dll.pvz_gym_can_execute_action, "argtypes", []) or []),
        "execute": len(getattr(self.client.dll.pvz_gym_execute_action, "argtypes", []) or []),
        "mask": len(getattr(self.client.dll.pvz_gym_get_action_mask, "argtypes", []) or []),
    }
    expected_counts = {"can_execute": 7, "execute": 9, "mask": 3}
    if actual_counts != expected_counts:
        return result_cls(
            case_id, "BattleAction", name, "FAIL",
            f"ABI 参数数量不一致: actual={actual_counts}, expected={expected_counts}", 0,
            details={"actual": actual_counts, "expected": expected_counts},
        )

    return result_cls(
        case_id, "BattleAction", name, "PASS",
        "Battle Action ABI 已绑定为 7 个整数动作参数；execute 额外包含消息缓冲区与长度", 0,
        details={"symbols": required, "arg_counts": actual_counts},
    )


def _test_action_mask_757(self, result_cls):
    case_id = "TC-33"
    name = "757 维 Endless Action Mask 固定空间与边界测试"
    skipped = _skip_unless_live_battle(self, result_cls, case_id, name)
    if skipped:
        return skipped

    code, mask, count, msg = self.client.get_action_mask()
    if code != 0:
        return result_cls(case_id, "BattleAction", name, "FAIL", f"读取 action mask 失败: code={code}, {msg}", 0)
    if count != ACTION_SPACE_SIZE or len(mask) != ACTION_SPACE_SIZE:
        return result_cls(
            case_id, "BattleAction", name, "FAIL",
            f"动作空间尺寸错误: out_count={count}, len(mask)={len(mask)}, expected={ACTION_SPACE_SIZE}", 0,
        )
    if not mask[0]:
        return result_cls(case_id, "BattleAction", name, "FAIL", "NOOP(index=0) 在战斗状态下被错误 mask", 0)

    _, _, obs = self.client.get_observation_json()
    rows = int((obs or {}).get("global", {}).get("grid_rows", 0) or 0)
    invalid_true = []
    if 1 <= rows < MAX_ROWS:
        for slot in range(MAX_CARD_SLOTS):
            for row in range(rows, MAX_ROWS):
                for col in range(COLS):
                    idx = PLANT_BASE + ((slot * MAX_ROWS + row) * COLS + col)
                    if mask[idx]:
                        invalid_true.append(("plant", idx, slot, row, col))
        for row in range(rows, MAX_ROWS):
            for col in range(COLS):
                nidx = SHOVEL_NORMAL_BASE + row * COLS + col
                pidx = SHOVEL_PUMPKIN_BASE + row * COLS + col
                if mask[nidx]:
                    invalid_true.append(("shovel_normal", nidx, row, col))
                if mask[pidx]:
                    invalid_true.append(("shovel_pumpkin", pidx, row, col))
        for target_row in range(rows, MAX_ROWS):
            for target_col in range(COLS):
                idx = COB_BASE + target_row * COLS + target_col
                if mask[idx]:
                    invalid_true.append(("cob_fire", idx, target_row, target_col))

    if invalid_true:
        return result_cls(
            case_id, "BattleAction", name, "FAIL",
            f"无效场景行仍存在 true mask，共 {len(invalid_true)} 项", 0,
            details={"examples": invalid_true[:12], "grid_rows": rows},
        )

    return result_cls(
        case_id, "BattleAction", name, "PASS",
        f"Action Mask 固定为 {ACTION_SPACE_SIZE} 维，NOOP 可用，场景行边界正确", 0,
        details={"true_count": sum(mask), "grid_rows": rows, "count": count},
    )


def _test_noop_action(self, result_cls):
    case_id = "TC-34"
    name = "NOOP 合法性、执行成功与零状态修改测试"
    skipped = _skip_unless_live_battle(self, result_cls, case_id, name)
    if skipped:
        return skipped

    code, msg = self.client.can_execute_action(ACTION_NOOP)
    if code != ERR_NONE:
        return result_cls(case_id, "BattleAction", name, "FAIL", f"NOOP can_execute 应为 0，实际 {code}: {msg}", 0)

    was_paused = self.client.is_paused()
    paused_for_test = was_paused or self.client.pause()
    try:
        _, _, before = self.client.get_observation_json()
        exec_code, exec_msg = self.client.execute_action(ACTION_NOOP)
        _, _, after = self.client.get_observation_json()
    finally:
        if not was_paused and paused_for_test:
            self.client.resume()

    if exec_code != ERR_NONE:
        return result_cls(case_id, "BattleAction", name, "FAIL", f"NOOP execute 失败: {exec_code}: {exec_msg}", 0)

    if paused_for_test and before and after:
        before_sig = (
            before.get("global", {}).get("game_clock"),
            before.get("global", {}).get("sun"),
            tuple((p.get("id"), p.get("type"), p.get("row"), p.get("col"), p.get("hp")) for p in before.get("plants", [])),
        )
        after_sig = (
            after.get("global", {}).get("game_clock"),
            after.get("global", {}).get("sun"),
            tuple((p.get("id"), p.get("type"), p.get("row"), p.get("col"), p.get("hp")) for p in after.get("plants", [])),
        )
        if before_sig != after_sig:
            return result_cls(
                case_id, "BattleAction", name, "FAIL",
                "暂停状态下 NOOP 修改了 game_clock/sun/plants 事实", 0,
                details={"before": before_sig[:2], "after": after_sig[:2]},
            )

    return result_cls(case_id, "BattleAction", name, "PASS", "NOOP can_execute/execute 均成功，且未产生动作副作用", 0)


def _test_shovel_target_semantics(self, result_cls):
    case_id = "TC-35"
    name = "SHOVEL NORMAL/PUMPKIN 双目标 Action Mask 语义测试"
    skipped = _skip_unless_live_battle(self, result_cls, case_id, name)
    if skipped:
        return skipped

    code, mask, count, msg = self.client.get_action_mask()
    if code != 0 or count != ACTION_SPACE_SIZE:
        return result_cls(case_id, "BattleAction", name, "FAIL", f"无法读取 757 维 mask: code={code}, count={count}, {msg}", 0)

    _, obs_msg, obs = self.client.get_observation_json()
    if not obs:
        return result_cls(case_id, "BattleAction", name, "SKIP", f"未获取到 Observation: {obs_msg}", 0)

    rows = int(obs.get("global", {}).get("grid_rows", 5) or 5)
    cells = {}
    for plant in obs.get("plants", []):
        row = int(plant.get("row", -1))
        col = int(plant.get("col", -1))
        layer = int(plant.get("layer", -1))
        if 0 <= row < rows and 0 <= col < COLS:
            cells.setdefault((row, col), set()).add(layer)

    mismatches = []
    dual_target_cells = []
    for row in range(min(rows, MAX_ROWS)):
        for col in range(COLS):
            layers = cells.get((row, col), set())
            expected_normal = 1 in layers or 0 in layers
            expected_pumpkin = 2 in layers
            normal_idx = SHOVEL_NORMAL_BASE + row * COLS + col
            pumpkin_idx = SHOVEL_PUMPKIN_BASE + row * COLS + col
            actual_normal = mask[normal_idx]
            actual_pumpkin = mask[pumpkin_idx]
            if actual_normal != expected_normal:
                mismatches.append((row, col, "NORMAL", sorted(layers), expected_normal, actual_normal, normal_idx))
            if actual_pumpkin != expected_pumpkin:
                mismatches.append((row, col, "PUMPKIN", sorted(layers), expected_pumpkin, actual_pumpkin, pumpkin_idx))
            if expected_normal and expected_pumpkin:
                dual_target_cells.append((row, col, normal_idx, pumpkin_idx))

    if mismatches:
        return result_cls(
            case_id, "BattleAction", name, "FAIL",
            f"Shovel mask 与 Observation 图层语义不一致，共 {len(mismatches)} 项", 0,
            details={"examples": mismatches[:12]},
        )

    return result_cls(
        case_id, "BattleAction", name, "PASS",
        "SHOVEL NORMAL=主植物/无主植物时底垫，PUMPKIN=南瓜；两类 mask 与当前植物图层一致",
        0,
        details={"occupied_cells": len(cells), "dual_target_cells": dual_target_cells},
    )


def _test_cob_discrete_grid_contract(self, result_cls):
    case_id = "TC-36"
    name = "Cob Cannon 离散 target_row/target_col 参数路由测试"
    skipped = _skip_unless_live_battle(self, result_cls, case_id, name)
    if skipped:
        return skipped

    # Target-only ABI: invalid target and legacy source coordinates must both
    # fail without firing a cannon.
    bad_target_code, bad_target_msg = self.client.can_execute_action(
        ACTION_COB_FIRE, row=-1, col=-1, target_row=-1, target_col=0
    )
    if bad_target_code != ERR_COB_TARGET_OUT_OF_ARENA:
        return result_cls(
            case_id, "BattleAction", name, "FAIL",
            f"target_row=-1 应返回 {ERR_COB_TARGET_OUT_OF_ARENA}，实际 {bad_target_code}: {bad_target_msg}", 0,
        )

    old_source_code, old_source_msg = self.client.can_execute_action(
        ACTION_COB_FIRE, row=0, col=0, target_row=0, target_col=0
    )
    if old_source_code != ERR_INVALID_ARGUMENT:
        return result_cls(
            case_id, "BattleAction", name, "FAIL",
            f"旧版源炮参数应返回 {ERR_INVALID_ARGUMENT}，实际 {old_source_code}: {old_source_msg}", 0,
        )

    col_oob_code, _ = self.client.can_execute_action(
        ACTION_COB_FIRE, row=-1, col=-1, target_row=0, target_col=9
    )
    if col_oob_code != ERR_COB_TARGET_OUT_OF_ARENA:
        return result_cls(case_id, "BattleAction", name, "FAIL", f"target_col=9 未按离散格边界拒绝: code={col_oob_code}", 0)

    return result_cls(
        case_id, "BattleAction", name, "PASS",
        "Cob ABI 已按离散 target_row/target_col 校验目标格，未暴露连续 mouse_x/mouse_y",
        0,
        details={"invalid_target_code": bad_target_code, "legacy_source_code": old_source_code},
    )


def _test_plant_mask_vs_validator(self, result_cls):
    case_id = "TC-37"
    name = "PLANT Action Mask 与 can_execute 代表样本一致性测试"
    skipped = _skip_unless_live_battle(self, result_cls, case_id, name)
    if skipped:
        return skipped

    code, mask, count, msg = self.client.get_action_mask()
    if code != 0 or count != ACTION_SPACE_SIZE:
        return result_cls(case_id, "BattleAction", name, "FAIL", f"无法读取 action mask: code={code}, count={count}, {msg}", 0)

    _, obs_msg, obs = self.client.get_observation_json()
    if not obs:
        return result_cls(case_id, "BattleAction", name, "SKIP", f"未获取到 Observation: {obs_msg}", 0)

    rows = max(1, min(MAX_ROWS, int(obs.get("global", {}).get("grid_rows", 5) or 5)))
    slots = sorted({int(c.get("slot", -1)) for c in obs.get("cards", []) if 0 <= int(c.get("slot", -1)) < MAX_CARD_SLOTS})
    candidate_slots = []
    if slots:
        candidate_slots.extend([slots[0], slots[-1]])
    candidate_slots.append(MAX_CARD_SLOTS - 1)
    candidate_slots = sorted(set(candidate_slots))

    cells = {(0, 0), (rows - 1, COLS - 1), (rows // 2, COLS // 2)}
    checked = []
    mismatches = []
    for slot in candidate_slots:
        for row, col in sorted(cells):
            idx = PLANT_BASE + ((slot * MAX_ROWS + row) * COLS + col)
            action_code, action_msg = self.client.can_execute_action(
                ACTION_PLANT, card_slot=slot, row=row, col=col
            )
            legal = action_code == ERR_NONE
            checked.append((idx, slot, row, col, action_code, mask[idx]))
            if legal != mask[idx]:
                mismatches.append((idx, slot, row, col, action_code, action_msg, mask[idx]))

    # Five-row scenes must reject the sixth row for a real selected slot too.
    if rows < MAX_ROWS and slots:
        slot = slots[0]
        row = MAX_ROWS - 1
        col = 0
        idx = PLANT_BASE + ((slot * MAX_ROWS + row) * COLS + col)
        action_code, action_msg = self.client.can_execute_action(
            ACTION_PLANT, card_slot=slot, row=row, col=col
        )
        checked.append((idx, slot, row, col, action_code, mask[idx]))
        if action_code != ERR_INVALID_GRID or mask[idx]:
            mismatches.append((idx, slot, row, col, action_code, action_msg, mask[idx]))

    if mismatches:
        return result_cls(
            case_id, "BattleAction", name, "FAIL",
            f"PLANT mask/can_execute 不一致，共 {len(mismatches)} 个样本", 0,
            details={"mismatches": mismatches, "checked": checked},
        )

    return result_cls(
        case_id, "BattleAction", name, "PASS",
        f"PLANT 代表样本共 {len(checked)} 项，mask 与 C++ can_execute 完全一致", 0,
        details={"checked": checked},
    )


def install_battle_action_tests(client_cls: Any, suite_cls: Any, result_cls: Any) -> None:
    """Install Battle Action bindings/tests exactly once."""
    if getattr(suite_cls, "_battle_action_tests_installed", False):
        return

    original_bind = client_cls._bind_functions

    def _bind_functions_with_battle_actions(self):
        original_bind(self)
        if not self.dll:
            return

        def bind(name: str, argtypes: list, restype: Any):
            if hasattr(self.dll, name):
                fn = getattr(self.dll, name)
                fn.argtypes = argtypes
                fn.restype = restype

        action_args = [ctypes.c_int] * 7
        bind("pvz_gym_can_execute_action", action_args, ctypes.c_int)
        bind("pvz_gym_execute_action", action_args + [ctypes.c_char_p, ctypes.c_int], ctypes.c_int)
        bind(
            "pvz_gym_get_action_mask",
            [ctypes.POINTER(ctypes.c_uint8), ctypes.c_int, ctypes.POINTER(ctypes.c_int)],
            ctypes.c_int,
        )

    client_cls._bind_functions = _bind_functions_with_battle_actions
    client_cls.can_execute_action = _client_can_execute_action
    client_cls.execute_action = _client_execute_action
    client_cls.get_action_mask = _client_get_action_mask

    suite_cls.test_battle_action_abi = lambda self: _test_battle_action_abi(self, result_cls)
    suite_cls.test_action_mask_757 = lambda self: _test_action_mask_757(self, result_cls)
    suite_cls.test_noop_action = lambda self: _test_noop_action(self, result_cls)
    suite_cls.test_shovel_target_semantics = lambda self: _test_shovel_target_semantics(self, result_cls)
    suite_cls.test_cob_discrete_grid_contract = lambda self: _test_cob_discrete_grid_contract(self, result_cls)
    suite_cls.test_plant_mask_vs_validator = lambda self: _test_plant_mask_vs_validator(self, result_cls)

    original_get_all_tests = suite_cls.get_all_tests

    def get_all_tests_with_battle_actions(self):
        tests = list(original_get_all_tests(self))
        tests.extend([
            {
                "id": "TC-32",
                "category": "BattleAction",
                "name": "Battle Action 7 参数 ABI 与 3 个核心符号签名测试",
                "func": self.test_battle_action_abi,
            },
            {
                "id": "TC-33",
                "category": "BattleAction",
                "name": "757 维 Endless Action Mask 固定空间与边界测试",
                "func": self.test_action_mask_757,
            },
            {
                "id": "TC-34",
                "category": "BattleAction",
                "name": "NOOP 合法性、执行成功与零状态修改测试",
                "func": self.test_noop_action,
            },
            {
                "id": "TC-35",
                "category": "BattleAction",
                "name": "SHOVEL NORMAL/PUMPKIN 双目标 Action Mask 语义测试",
                "func": self.test_shovel_target_semantics,
            },
            {
                "id": "TC-36",
                "category": "BattleAction",
                "name": "Cob Cannon 离散 target_row/target_col 参数路由测试",
                "func": self.test_cob_discrete_grid_contract,
            },
            {
                "id": "TC-37",
                "category": "BattleAction",
                "name": "PLANT Action Mask 与 can_execute 代表样本一致性测试",
                "func": self.test_plant_mask_vs_validator,
            },
        ])
        return tests

    suite_cls.get_all_tests = get_all_tests_with_battle_actions
    suite_cls._battle_action_tests_installed = True
