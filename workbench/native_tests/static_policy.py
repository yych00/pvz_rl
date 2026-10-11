# -*- coding: utf-8 -*-
"""Pure-static policy for the regular GUI/CLI C++ unit-test suite.

``get_all_tests()`` is restricted to DLL loading, exported-symbol and ctypes
signature checks. It must not attach to PvZ, inspect game memory, or invoke an
API which can mutate a running game. Live checks remain explicit/manual.
"""

from __future__ import annotations

from typing import Any


def install_static_unit_test_policy(suite_cls: Any, result_cls: Any) -> None:
    """Restrict the normal suite to checks with no game-process access."""
    if getattr(suite_cls, "_static_unit_test_policy_installed", False):
        return

    original_get_all_tests = suite_cls.get_all_tests

    def test_seed_selection_execution_static(self):
        """Validate seed-action ctypes bindings without calling PvZ actions."""
        case_id = "TC-28"
        category = "SeedSelection / Static"
        name = "选卡执行接口静态绑定与无副作用策略测试（不操作游戏）"
        if not self.client.is_loaded:
            return result_cls(case_id, category, name, "SKIP", "DLL 未加载", 0)

        expected_arg_counts = {
            "pvz_gym_select_seed": 3,
            "pvz_gym_select_imitater": 3,
            "pvz_gym_deselect_seed": 3,
            "pvz_gym_confirm_seeds": 2,
        }
        missing = []
        bad_signatures = {}
        for symbol, expected_count in expected_arg_counts.items():
            if not hasattr(self.client.dll, symbol):
                missing.append(symbol)
                continue
            actual_count = len(getattr(getattr(self.client.dll, symbol), "argtypes", []) or [])
            if actual_count != expected_count:
                bad_signatures[symbol] = (actual_count, expected_count)

        if missing or bad_signatures:
            return result_cls(
                case_id, category, name, "FAIL",
                f"选卡 ABI 静态校验失败: missing={missing}, signatures={bad_signatures}", 0,
                details={"missing": missing, "bad_signatures": bad_signatures},
            )
        return result_cls(
            case_id, category, name, "PASS",
            "已静态校验 Select/Imitater/Deselect/Confirm 的 C ABI 绑定；未调用游戏接口",
            0,
            details={"symbols": list(expected_arg_counts), "game_action_calls": 0},
        )

    def get_all_tests_without_game_mutation(self):
        # Use a whitelist, so a newly added live test cannot accidentally enter
        # the normal suite before it has been explicitly reviewed as static.
        static_case_ids = {
            "TC-01", "TC-02", "TC-03", "TC-10", "TC-11", "TC-23",
            "TC-26", "TC-28", "TC-32",
        }
        static_tests = []
        for test in original_get_all_tests(self):
            if test.get("id") not in static_case_ids:
                continue
            if test.get("id") == "TC-28":
                test["category"] = "SeedSelection / Static"
                test["name"] = "选卡执行接口静态绑定与无副作用策略测试（不操作游戏）"
                test["func"] = self.test_seed_selection_execution_static
            static_tests.append(test)
        return static_tests

    suite_cls.test_seed_selection_safe_execution = test_seed_selection_execution_static
    suite_cls.test_seed_selection_execution_static = test_seed_selection_execution_static
    suite_cls.get_all_tests = get_all_tests_without_game_mutation
    suite_cls._static_unit_test_policy_installed = True
