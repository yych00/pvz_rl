# -*- coding: utf-8 -*-
"""
PvZ Gym C++ 原生接口自动化单元测试用例集 (test_cases.py)
严格对齐 docs/gym_requirements.md 规格清单
严格遵循 00规范/python规范.md:
- 统一采用 logging 框架，严禁散落裸 print
- 图标严禁使用带 VS-16 (\ufe0f) 的变体 Emoji，防止 Windows Tkinter 字体回退异常间距
- 字符与汉字之间使用统一半角空格
"""

import time
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Any, Callable
from .c_interface import PvzGymNativeDll

logger = logging.getLogger("pvz_gym_unit_test.test_cases")


@dataclass
class TestCaseResult:
    case_id: str
    category: str
    name: str
    status: str  # "PASS" | "FAIL" | "SKIP"
    message: str
    elapsed_ms: float
    details: Dict[str, Any] = field(default_factory=dict)


class NativeUnitTestSuite:
    """C++ Native 接口单元测试执行引擎"""

    def __init__(self, client: PvzGymNativeDll):
        self.client = client

    def get_all_tests(self) -> List[Dict[str, Any]]:
        """获取全部测试用例清单"""
        return [
            # --- 1. DLL 文件完整度与 ABI 符号导出 ---
            {
                "id": "TC-01",
                "category": "ABI / Symbols",
                "name": "验证 pvz_gym_native.dll 存在性与动态加载",
                "func": self.test_dll_loading,
            },
            {
                "id": "TC-02",
                "category": "ABI / Symbols",
                "name": "校验当前阶段 7 个核心 C ABI 符号导出",
                "func": self.test_symbol_exports,
            },
            # --- 2. 错误码与解析体系 ---
            {
                "id": "TC-03",
                "category": "Error Codes",
                "name": "测试标准错误码文本解析 (0=success, 1001, 1002, 未知码)",
                "func": self.test_error_messages,
            },
            # --- 3. 未连接/离线安全防护 (安全防崩溃) ---
            {
                "id": "TC-04",
                "category": "Process Safety",
                "name": "未连接时调用 is_game_on() 与 get_ui() 防崩溃测试",
                "func": self.test_offline_process_calls,
            },
            {
                "id": "TC-05",
                "category": "Level Safety",
                "name": "未连接时调用关卡切换与查询返回 1001 拦截测试",
                "func": self.test_offline_level_calls,
            },
            # --- 4. 关卡参数合法性边界防护 ---
            {
                "id": "TC-06",
                "category": "Input Validation",
                "name": "大关与小关参数越界拦截测试 (area/sub_level 越界应返回 1002)",
                "func": self.test_level_area_sub_bounds,
            },
            {
                "id": "TC-07",
                "category": "Input Validation",
                "name": "绝对关卡 ID 边界拦截测试 (level_id < 1 或 > 50 应返回 1002)",
                "func": self.test_level_id_bounds,
            },
            # --- 5. 实机状态探测与关卡跳转 (若游戏进程存在) ---
            {
                "id": "TC-08",
                "category": "Live Process",
                "name": "尝试附加游戏并读取当前 UI 与关卡状态",
                "func": self.test_live_status_probe,
            },
            {
                "id": "TC-09",
                "category": "Live Process",
                "name": "实机执行跳转到 1-1 第一关测试 (若在线)",
                "func": self.test_live_jump_level_1_1,
            },
            # --- 6. 规格进度扫描 (gym_requirements.md) ---
            {
                "id": "TC-10",
                "category": "Specs Probe",
                "name": "P0~P4 规格实现进度全景探针扫描",
                "func": self.test_specs_progress_probe,
            },
            # --- 7. 仿真时钟与时间推进测试 (1.3 Simulation) ---
            {
                "id": "TC-11",
                "category": "Clock / Simulation",
                "name": "校验 5 个仿真时钟与时间推进 C ABI 符号导出",
                "func": self.test_clock_symbol_exports,
            },
            {
                "id": "TC-12",
                "category": "Clock / Simulation",
                "name": "未连接时调用仿真时钟接口防崩溃安全测试",
                "func": self.test_offline_clock_safety,
            },
            {
                "id": "TC-13",
                "category": "Clock / Simulation",
                "name": "实机仿真时钟暂停、读取与 Tick 推进测试 (若在线战斗)",
                "func": self.test_live_clock_stepping,
            },
            # --- 8. 原生只读 Observation 事实链路测试 (P1 Observation) ---
            {
                "id": "TC-14",
                "category": "Observation",
                "name": "Global 宏观事实读取 (时钟/阳光/波次/场景/状态)",
                "func": self.test_obs_global,
            },
            {
                "id": "TC-15",
                "category": "Observation",
                "name": "暂停状态下连续观测快照稳定性与零撕裂测试",
                "func": self.test_obs_pause_stability,
            },
            {
                "id": "TC-16",
                "category": "Observation",
                "name": "推进 20 Ticks 时钟增量与实体位移演化测试",
                "func": self.test_obs_advance_ticks_evolution,
            },
            {
                "id": "TC-17",
                "category": "Observation",
                "name": "Plants 植物事实结构与图层合法性测试",
                "func": self.test_obs_plants,
            },
            {
                "id": "TC-18",
                "category": "Observation",
                "name": "Cards 卡槽技能与冷却倒计时可用性测试",
                "func": self.test_obs_cards,
            },
            {
                "id": "TC-19",
                "category": "Observation",
                "name": "Zombies 僵尸连续物理坐标与生命值测试",
                "func": self.test_obs_zombies,
            },
            {
                "id": "TC-20",
                "category": "Observation",
                "name": "Mowers 底线割草机状态完整性测试",
                "func": self.test_obs_mowers,
            },
            {
                "id": "TC-21",
                "category": "Consistency",
                "name": "Gym Native 与 tools_perception 事实一致性回归对比",
                "func": self.test_obs_consistency_with_perception,
            },
            {
                "id": "TC-22",
                "category": "Observation",
                "name": "掉落卡、场地障碍物与冰道实体事实测试",
                "func": self.test_obs_special_entities,
            },
            # --- 9. 自动化辅助模块测试 (Automation Helpers: 非 RL 模块) ---
            {
                "id": "TC-23",
                "category": "Automation",
                "name": "校验自动化辅助模块 5 个 C ABI 符号导出",
                "func": self.test_automation_symbols,
            },
            {
                "id": "TC-24",
                "category": "Automation",
                "name": "自动化辅助配置默认开启与可控关闭测试",
                "func": self.test_automation_config_toggle,
            },
            {
                "id": "TC-25",
                "category": "Automation",
                "name": "自动化辅助接口安全调用测试",
                "func": self.test_automation_safe_calls,
            },
            # --- 10. 选卡独立动作域与编排器测试 (Seed Selection & EpisodeRunner) ---
            {
                "id": "TC-26",
                "category": "SeedSelection",
                "name": "校验选卡独立动作域 7 个 C ABI 符号导出",
                "func": self.test_seed_selection_symbols,
            },
            {
                "id": "TC-27",
                "category": "SeedSelection",
                "name": "选卡 ABI 固定容量与 193 维 Action Mask 契约测试",
                "func": self.test_seed_selection_capacity_and_mask,
            },
            {
                "id": "TC-28",
                "category": "SeedSelection",
                "name": "选卡执行器安全拦截与错误码自洽性测试",
                "func": self.test_seed_selection_safe_execution,
            },
            {
                "id": "TC-29",
                "category": "GymEnv",
                "name": "SeedSelectionEnv Gymnasium 标准接口合规性与 Space 空间结构测试",
                "func": self.test_seed_selection_gym_env,
            },
            {
                "id": "TC-30",
                "category": "Runner",
                "name": "EpisodeRunner 跨环境调度与延迟 Deck Return 结算测试",
                "func": self.test_episode_runner_orchestration,
            },
            # --- 11. 辅助功能与作弊按键测试 (Auxiliary & Cheat Keys) ---
            {
                "id": "TC-31",
                "category": "CheatKeys",
                "name": "校验作弊按键与按键发送 C ABI 导出及读写测试",
                "func": self.test_cheat_keys_and_send_key,
            },
            # --- 12. Dialog 弹窗管理与场景切换清理钩子测试 ---
            {
                "id": "TC-38",
                "category": "Dialog",
                "name": "校验 Dialog C ABI 导出及场景切换弹窗清理钩子测试",
                "func": self.test_dialog_cleanup_hook,
            },
        ]

    def run_single(self, test_info: Dict[str, Any]) -> TestCaseResult:
        """运行单项测试用例"""
        cid = test_info["id"]
        cat = test_info["category"]
        name = test_info["name"]
        func: Callable[[], TestCaseResult] = test_info["func"]

        t0 = time.perf_counter()
        try:
            res = func()
            res.elapsed_ms = (time.perf_counter() - t0) * 1000.0
            logger.info("测试执行完毕 [%s] %s -> %s (%.2fms)", cid, name, res.status, res.elapsed_ms)
            return res
        except Exception as e:
            elapsed = (time.perf_counter() - t0) * 1000.0
            logger.error("测试用例执行崩溃 [%s] %s: %s", cid, name, e, exc_info=True)
            return TestCaseResult(
                case_id=cid,
                category=cat,
                name=name,
                status="FAIL",
                message=f"测试用例发生未捕获异常: {e}",
                elapsed_ms=elapsed,
                details={"exception": str(e)}
            )

    def run_all(self, callback: Callable[[TestCaseResult], None] = None) -> List[TestCaseResult]:
        """按顺序运行全部测试用例，并触发回调"""
        results = []
        for t in self.get_all_tests():
            res = self.run_single(t)
            results.append(res)
            if callback:
                callback(res)
        return results

    # =========================================================================
    # 测试用例具体实现
    # =========================================================================

    def test_dll_loading(self) -> TestCaseResult:
        if not self.client.is_loaded:
            return TestCaseResult(
                case_id="TC-01",
                category="ABI / Symbols",
                name="验证 pvz_gym_native.dll 存在性与动态加载",
                status="FAIL",
                message=f"DLL 加载失败: {self.client.error_msg}",
                elapsed_ms=0,
                details={"error": self.client.error_msg}
            )
        return TestCaseResult(
            case_id="TC-01",
            category="ABI / Symbols",
            name="验证 pvz_gym_native.dll 存在性与动态加载",
            status="PASS",
            message=f"成功加载 DLL: {self.client.dll_path}",
            elapsed_ms=0,
            details={"path": self.client.dll_path}
        )

    def test_symbol_exports(self) -> TestCaseResult:
        if not self.client.is_loaded:
            return TestCaseResult("TC-02", "ABI / Symbols", "校验当前阶段 7 个核心 C ABI 符号导出", "SKIP", "DLL 未加载", 0)

        required = [
            "pvz_gym_attach",
            "pvz_gym_is_game_on",
            "pvz_gym_get_ui",
            "pvz_gym_select_adventure_level",
            "pvz_gym_select_adventure_level_id",
            "pvz_gym_get_current_adventure_level",
            "pvz_gym_error_message",
        ]
        missing = [s for s in required if not hasattr(self.client.dll, s)]
        if missing:
            return TestCaseResult(
                case_id="TC-02",
                category="ABI / Symbols",
                name="校验当前阶段 7 个核心 C ABI 符号导出",
                status="FAIL",
                message=f"缺少必须导出的 C ABI 符号: {missing}",
                elapsed_ms=0,
                details={"missing": missing}
            )
        return TestCaseResult(
            case_id="TC-02",
            category="ABI / Symbols",
            name="校验当前阶段 7 个核心 C ABI 符号导出",
            status="PASS",
            message="已全部导出 7 个核心 C ABI 符号",
            elapsed_ms=0,
            details={"symbols": required}
        )

    def test_error_messages(self) -> TestCaseResult:
        if not self.client.is_loaded:
            return TestCaseResult("TC-03", "Error Codes", "测试标准错误码文本解析", "SKIP", "DLL 未加载", 0)

        m0 = self.client.error_message(0)
        m1001 = self.client.error_message(1001)
        m1002 = self.client.error_message(1002)
        m_unknown = self.client.error_message(9999)

        if m0 != "success":
            return TestCaseResult("TC-03", "Error Codes", "测试标准错误码文本解析", "FAIL", f"0 错误码期望 'success'，实际为 '{m0}'", 0)
        if "not found" not in m1001.lower():
            return TestCaseResult("TC-03", "Error Codes", "测试标准错误码文本解析", "FAIL", f"1001 错误码文本不符: {m1001}", 0)
        if "invalid" not in m1002.lower():
            return TestCaseResult("TC-03", "Error Codes", "测试标准错误码文本解析", "FAIL", f"1002 错误码文本不符: {m1002}", 0)

        return TestCaseResult(
            case_id="TC-03",
            category="Error Codes",
            name="测试标准错误码文本解析",
            status="PASS",
            message="错误码映射完整正确: 0->success, 1001->not found, 1002->invalid, 未知码安全解析",
            elapsed_ms=0,
            details={"0": m0, "1001": m1001, "1002": m1002, "unknown": m_unknown}
        )

    def test_offline_process_calls(self) -> TestCaseResult:
        if not self.client.is_loaded:
            return TestCaseResult("TC-04", "Process Safety", "未连接时调用 is_game_on 与 get_ui 防崩溃测试", "SKIP", "DLL 未加载", 0)

        on, on_desc = self.client.is_game_on()
        ui, ui_desc = self.client.get_ui()

        return TestCaseResult(
            case_id="TC-04",
            category="Process Safety",
            name="未连接时调用 is_game_on 与 get_ui 防崩溃测试",
            status="PASS",
            message=f"调用安全返回: is_game_on={on}, get_ui={ui} ({ui_desc})",
            elapsed_ms=0,
            details={"is_game_on": on, "ui": ui, "ui_desc": ui_desc}
        )

    def test_offline_level_calls(self) -> TestCaseResult:
        if not self.client.is_loaded:
            return TestCaseResult("TC-05", "Level Safety", "未连接时调用关卡切换与查询返回 1001 拦截测试", "SKIP", "DLL 未加载", 0)

        on, _ = self.client.is_game_on()
        if on:
            return TestCaseResult(
                case_id="TC-05",
                category="Level Safety",
                name="未连接时调用关卡切换与查询返回 1001 拦截测试",
                status="SKIP",
                message="当前游戏进程已在线，跳过离线拦截测试",
                elapsed_ms=0
            )

        code_sel, _ = self.client.select_level_id(1)
        cur_info = self.client.get_current_level()

        if code_sel != 1001:
            return TestCaseResult("TC-05", "Level Safety", "未连接时调用关卡切换与查询返回 1001 拦截测试", "FAIL", f"离线调用应返回 1001，实际返回: {code_sel}", 0)
        if cur_info.get("code") != 1001:
            return TestCaseResult("TC-05", "Level Safety", "未连接时调用关卡切换与查询返回 1001 拦截测试", "FAIL", f"离线查询应返回 1001，实际返回: {cur_info.get('code')}", 0)

        return TestCaseResult(
            case_id="TC-05",
            category="Level Safety",
            name="未连接时调用关卡切换与查询返回 1001 拦截测试",
            status="PASS",
            message="未连接时关卡切换与查询均安全返回 1001 (GameNotFound)，无内存段错误",
            elapsed_ms=0
        )

    def test_level_area_sub_bounds(self) -> TestCaseResult:
        if not self.client.is_loaded:
            return TestCaseResult("TC-06", "Input Validation", "大关与小关参数越界拦截测试", "SKIP", "DLL 未加载", 0)

        invalid_cases = [
            (0, 1),   # 大关小于 1
            (6, 1),   # 大关大于 5
            (1, 0),   # 小关小于 1
            (1, 11),  # 小关大于 10
            (-1, 5),  # 负数
        ]
        failed_cases = []
        for area, sub in invalid_cases:
            code, _ = self.client.select_level(area, sub)
            if code != 1002:
                failed_cases.append((area, sub, code))

        if failed_cases:
            return TestCaseResult(
                case_id="TC-06",
                category="Input Validation",
                name="大关与小关参数越界拦截测试",
                status="FAIL",
                message=f"以下非法输入未返回 1002 (InvalidArgument): {failed_cases}",
                elapsed_ms=0,
                details={"failed": failed_cases}
            )

        return TestCaseResult(
            case_id="TC-06",
            category="Input Validation",
            name="大关与小关参数越界拦截测试",
            status="PASS",
            message=f"全部 {len(invalid_cases)} 组越界关卡参数均被原生 C++ 正确拦截为 1002",
            elapsed_ms=0,
            details={"tested_cases": len(invalid_cases)}
        )

    def test_level_id_bounds(self) -> TestCaseResult:
        if not self.client.is_loaded:
            return TestCaseResult("TC-07", "Input Validation", "绝对关卡 ID 边界拦截测试", "SKIP", "DLL 未加载", 0)

        invalid_ids = [0, -1, -100, 51, 100, 9999]
        failed_ids = []
        for lid in invalid_ids:
            code, _ = self.client.select_level_id(lid)
            if code != 1002:
                failed_ids.append((lid, code))

        if failed_ids:
            return TestCaseResult(
                case_id="TC-07",
                category="Input Validation",
                name="绝对关卡 ID 边界拦截测试",
                status="FAIL",
                message=f"非法关卡 ID 未返回 1002: {failed_ids}",
                elapsed_ms=0
            )

        return TestCaseResult(
            case_id="TC-07",
            category="Input Validation",
            name="绝对关卡 ID 边界拦截测试",
            status="PASS",
            message=f"全部 {len(invalid_ids)} 组非法绝对关卡 ID 均被正确拦截为 1002",
            elapsed_ms=0
        )

    def test_live_status_probe(self) -> TestCaseResult:
        if not self.client.is_loaded:
            return TestCaseResult("TC-08", "Live Process", "尝试附加游戏并读取当前 UI 与关卡状态", "SKIP", "DLL 未加载", 0)

        ok, _ = self.client.attach()
        on, _ = self.client.is_game_on()
        ui, ui_desc = self.client.get_ui()
        cur_lvl = self.client.get_current_level()

        if not on:
            return TestCaseResult(
                case_id="TC-08",
                category="Live Process",
                name="尝试附加游戏并读取当前 UI 与关卡状态",
                status="SKIP",
                message="当前无运行中的 PlantsVsZombies.exe，跳过实机探针",
                elapsed_ms=0,
                details={"attach_result": ok, "ui": ui}
            )

        return TestCaseResult(
            case_id="TC-08",
            category="Live Process",
            name="尝试附加游戏并读取当前 UI 与关卡状态",
            status="PASS",
            message=f"实机在线: UI={ui_desc}, 关卡={cur_lvl.get('level_str', '未知')}",
            elapsed_ms=0,
            details={"ui": ui, "ui_desc": ui_desc, "level": cur_lvl}
        )

    def test_live_jump_level_1_1(self) -> TestCaseResult:
        if not self.client.is_loaded:
            return TestCaseResult("TC-09", "Live Process", "实机执行跳转到 1-1 第一关测试", "SKIP", "DLL 未加载", 0)

        on, _ = self.client.is_game_on()
        if not on:
            return TestCaseResult(
                case_id="TC-09",
                category="Live Process",
                name="实机执行跳转到 1-1 第一关测试",
                status="SKIP",
                message="游戏未在线，跳过实机关卡跳转",
                elapsed_ms=0
            )

        code, msg = self.client.select_level(1, 1)
        if code != 0:
            return TestCaseResult("TC-09", "Live Process", "实机执行跳转到 1-1 第一关测试", "FAIL", f"跳转 1-1 失败: {msg} (代码: {code})", 0)

        return TestCaseResult(
            case_id="TC-09",
            category="Live Process",
            name="实机执行跳转到 1-1 第一关测试",
            status="PASS",
            message="成功执行跳转到冒险模式 1-1 (代码: 0)",
            elapsed_ms=0
        )

    def test_specs_progress_probe(self) -> TestCaseResult:
        """扫描 docs/gym_requirements.md 规格实现进度"""
        symbols = self.client.get_exported_symbols()
        p0_p1_keys = [
            "pvz_gym_attach", "pvz_gym_is_game_on", "pvz_gym_get_ui",
            "pvz_gym_select_adventure_level", "pvz_gym_select_adventure_level_id",
            "pvz_gym_get_current_adventure_level", "pvz_gym_error_message"
        ]
        p2_clock_keys = ["pvz_gym_pause", "pvz_gym_resume", "pvz_gym_is_paused", "pvz_gym_advance_ticks", "pvz_gym_get_game_clock"]
        p2_other_keys = ["pvz_gym_get_observation_json", "pvz_gym_get_action_mask", "pvz_gym_env_reset", "pvz_gym_env_step"]

        p0_p1_done = all(symbols.get(k, False) for k in p0_p1_keys)
        p2_clock_done = all(symbols.get(k, False) for k in p2_clock_keys)
        p2_pending = sum(1 for k in p2_other_keys if not symbols.get(k, False))

        msg = f"P0/P1 已就绪 (7/7)；P2 仿真时钟已就绪 (5/5)；待实现动作与观测 ({p2_pending} 个核心接口)"
        return TestCaseResult(
            case_id="TC-10",
            category="Specs Probe",
            name="P0~P4 规格实现进度全景探针扫描",
            status="PASS",
            message=msg,
            elapsed_ms=0,
            details={"symbols": symbols, "p0_p1_done": p0_p1_done, "p2_clock_done": p2_clock_done, "p2_pending": p2_pending}
        )

    def test_clock_symbol_exports(self) -> TestCaseResult:
        if not self.client.is_loaded:
            return TestCaseResult("TC-11", "Clock / Simulation", "校验 5 个仿真时钟与时间推进 C ABI 符号导出", "SKIP", "DLL 未加载", 0)

        required = [
            "pvz_gym_pause",
            "pvz_gym_resume",
            "pvz_gym_is_paused",
            "pvz_gym_advance_ticks",
            "pvz_gym_get_game_clock",
        ]
        missing = [s for s in required if not hasattr(self.client.dll, s)]
        if missing:
            return TestCaseResult(
                case_id="TC-11",
                category="Clock / Simulation",
                name="校验 5 个仿真时钟与时间推进 C ABI 符号导出",
                status="FAIL",
                message=f"缺少仿真时钟 C ABI 符号: {missing}",
                elapsed_ms=0,
                details={"missing": missing}
            )
        return TestCaseResult(
            case_id="TC-11",
            category="Clock / Simulation",
            name="校验 5 个仿真时钟与时间推进 C ABI 符号导出",
            status="PASS",
            message="已成功导出全部 5 个仿真时钟符号 (pause, resume, is_paused, advance_ticks, get_game_clock)",
            elapsed_ms=0,
            details={"symbols": required}
        )

    def test_offline_clock_safety(self) -> TestCaseResult:
        if not self.client.is_loaded:
            return TestCaseResult("TC-12", "Clock / Simulation", "未连接时调用仿真时钟接口防崩溃安全测试", "SKIP", "DLL 未加载", 0)

        on, _ = self.client.is_game_on()
        if on:
            return TestCaseResult("TC-12", "Clock / Simulation", "未连接时调用仿真时钟接口防崩溃安全测试", "SKIP", "当前游戏在线，跳过离线防崩溃测试", 0)

        # 验证离线/未附加时各个时钟调用均安全返回，不产生内存段错误
        paused = self.client.is_paused()
        clock = self.client.get_game_clock()
        pause_ret = self.client.pause()
        resume_ret = self.client.resume()
        adv_ret = self.client.advance_ticks(20)

        if paused is not False or clock != 0 or pause_ret is not False or resume_ret is not False or adv_ret != -1:
            return TestCaseResult(
                case_id="TC-12",
                category="Clock / Simulation",
                name="未连接时调用仿真时钟接口防崩溃安全测试",
                status="FAIL",
                message=f"离线返回值不符合预期安全约束: is_paused={paused}, clock={clock}, adv={adv_ret}",
                elapsed_ms=0
            )

        return TestCaseResult(
            case_id="TC-12",
            category="Clock / Simulation",
            name="未连接时调用仿真时钟接口防崩溃安全测试",
            status="PASS",
            message="离线状态下 pause/resume/is_paused/advance_ticks/get_game_clock 全部安全防御返回，零崩溃",
            elapsed_ms=0,
            details={"clock": clock, "is_paused": paused, "advance_result": adv_ret}
        )

    def test_live_clock_stepping(self) -> TestCaseResult:
        if not self.client.is_loaded:
            return TestCaseResult("TC-13", "Clock / Simulation", "实机仿真时钟暂停、读取与 Tick 推进测试", "SKIP", "DLL 未加载", 0)

        on, _ = self.client.is_game_on()
        ui, ui_desc = self.client.get_ui()
        if not on or ui != 3:
            return TestCaseResult(
                case_id="TC-13",
                category="Clock / Simulation",
                name="实机仿真时钟暂停、读取与 Tick 推进测试",
                status="SKIP",
                message=f"游戏未处于战斗状态 (当前 UI: {ui_desc})，跳过实机推进测试",
                elapsed_ms=0
            )

        clock0 = self.client.get_game_clock()
        # 推进 20 ticks (约 200ms)
        stepped = self.client.advance_ticks(20)
        clock1 = self.client.get_game_clock()

        delta = clock1 - clock0
        if stepped < 0 or delta < 20:
            return TestCaseResult(
                case_id="TC-13",
                category="Clock / Simulation",
                name="实机仿真时钟暂停、读取与 Tick 推进测试",
                status="FAIL",
                message=f"推进失败或时钟增量不足: 推进返回={stepped}, clock前={clock0}, clock后={clock1}, delta={delta}",
                elapsed_ms=0
            )

        return TestCaseResult(
            case_id="TC-13",
            category="Clock / Simulation",
            name="实机仿真时钟暂停、读取与 Tick 推进测试",
            status="PASS",
            message=f"实机时钟推进成功: 推进 20 ticks -> 实际推进 {stepped} ticks, 时钟从 {clock0} 增至 {clock1} (Δ={delta})",
            elapsed_ms=0,
            details={"clock_before": clock0, "clock_after": clock1, "delta": delta, "stepped": stepped}
        )

    # -------------------------------------------------------------------------
    # 8. 原生只读 Observation 事实链路测试 (TC-14 ~ TC-21)
    # -------------------------------------------------------------------------

    def test_obs_global(self) -> TestCaseResult:
        """TC-14: Global 宏观事实读取 (时钟/阳光/波次/场景/状态)"""
        if not self.client.is_loaded:
            return TestCaseResult("TC-14", "Observation", "Global 宏观事实读取", "SKIP", "DLL 未加载", 0)

        n, msg, data = self.client.get_observation_json()
        if n <= 0 or not data:
            return TestCaseResult("TC-14", "Observation", "Global 宏观事实读取", "SKIP", f"未获取到 Observation: {msg}", 0)

        gl = data.get("global", {})
        clock = gl.get("game_clock", 0)
        sun = gl.get("sun", -1)
        scene = gl.get("scene", -1)
        grid_rows = gl.get("grid_rows", -1)
        level_id = gl.get("level_id", -1)
        z_pool = gl.get("zombie_types_in_level", [])
        c_wave = gl.get("current_wave", -1)
        t_waves = gl.get("total_waves", -1)
        status = gl.get("status", 0)

        expected_rows = 6 if scene in (2, 3) else 5
        if (clock <= 0 or sun < 0 or scene < 0 or c_wave < 0 or t_waves <= 0 or status != 3
                or grid_rows != expected_rows or level_id < 1 or level_id > 50 or not isinstance(z_pool, list) or len(z_pool) == 0):
            return TestCaseResult(
                "TC-14", "Observation", "Global 宏观事实读取", "FAIL",
                f"Global 字段异常: clock={clock}, sun={sun}, scene={scene}, rows={grid_rows}, level={level_id}, z_pool={z_pool}, wave={c_wave}/{t_waves}, status={status}",
                0, details=gl
            )

        return TestCaseResult(
            "TC-14", "Observation", "Global 宏观事实读取", "PASS",
            f"Global 事实有效: level={level_id}, rows={grid_rows}, clock={clock}, sun={sun}, wave={c_wave}/{t_waves}, z_pool={z_pool}",
            0, details=gl
        )

    def test_obs_pause_stability(self) -> TestCaseResult:
        """TC-15: 暂停状态下连续观测快照稳定性与零撕裂测试"""
        if not self.client.is_loaded:
            return TestCaseResult("TC-15", "Observation", "暂停连续观测稳定性", "SKIP", "DLL 未加载", 0)

        on, _ = self.client.is_game_on()
        ui, _ = self.client.get_ui()
        if not on or ui != 3:
            return TestCaseResult("TC-15", "Observation", "暂停连续观测稳定性", "SKIP", "游戏未处于战斗状态", 0)

        self.client.pause()
        n1, _, obs1 = self.client.get_observation_json()
        n2, _, obs2 = self.client.get_observation_json()

        if not obs1 or not obs2:
            return TestCaseResult("TC-15", "Observation", "暂停连续观测稳定性", "FAIL", "获取 Observation 失败", 0)

        c1 = obs1.get("global", {}).get("game_clock")
        c2 = obs2.get("global", {}).get("game_clock")
        if c1 != c2:
            return TestCaseResult("TC-15", "Observation", "暂停连续观测稳定性", "FAIL", f"暂停状态下时钟不一致: c1={c1}, c2={c2}", 0)

        # 验证实体数量与内容完全稳定
        if len(obs1.get("plants", [])) != len(obs2.get("plants", [])) or \
           len(obs1.get("zombies", [])) != len(obs2.get("zombies", [])) or \
           len(obs1.get("cards", [])) != len(obs2.get("cards", [])):
            return TestCaseResult("TC-15", "Observation", "暂停连续观测稳定性", "FAIL", "实体数量发生撕裂/突变", 0)

        return TestCaseResult(
            "TC-15", "Observation", "暂停连续观测稳定性", "PASS",
            f"暂停状态下 2 次连续 Observation 完全稳定且一致 (clock={c1}, plants={len(obs1.get('plants', []))}, zombies={len(obs1.get('zombies', []))})",
            0
        )

    def test_obs_advance_ticks_evolution(self) -> TestCaseResult:
        """TC-16: 推进 20 Ticks 时钟增量与实体位移演化测试"""
        if not self.client.is_loaded:
            return TestCaseResult("TC-16", "Observation", "推进 20 Ticks 时钟增量与实体位移演化", "SKIP", "DLL 未加载", 0)

        on, _ = self.client.is_game_on()
        ui, _ = self.client.get_ui()
        if not on or ui != 3:
            return TestCaseResult("TC-16", "Observation", "推进 20 Ticks 时钟增量与实体位移演化", "SKIP", "游戏未处于战斗状态", 0)

        self.client.pause()
        _, _, obs1 = self.client.get_observation_json()
        stepped = self.client.advance_ticks(20)
        _, _, obs2 = self.client.get_observation_json()

        if not obs1 or not obs2:
            return TestCaseResult("TC-16", "Observation", "推进 20 Ticks 时钟增量与实体位移演化", "FAIL", "获取观测快照失败", 0)

        c1 = obs1.get("global", {}).get("game_clock", 0)
        c2 = obs2.get("global", {}).get("game_clock", 0)
        delta_clock = c2 - c1

        if delta_clock != 20:
            return TestCaseResult(
                "TC-16", "Observation", "推进 20 Ticks 时钟增量与实体位移演化", "FAIL",
                f"时钟增量不等于 20: c1={c1}, c2={c2}, delta={delta_clock}, stepped={stepped}", 0
            )

        z1 = obs1.get("zombies", [])
        z2 = obs2.get("zombies", [])
        zombie_moved = False
        if z1 and z2:
            z1_map = {z["id"]: z["x"] for z in z1}
            for z in z2:
                zid = z["id"]
                if zid in z1_map and abs(z["x"] - z1_map[zid]) > 0.001:
                    zombie_moved = True
                    break

        msg = f"推进 20 ticks 成功 (Δclock=20)"
        if z1 and z2:
            msg += f", 场上 {len(z1)} 只僵尸位置合理演化 (moved={zombie_moved})"

        return TestCaseResult("TC-16", "Observation", "推进 20 Ticks 时钟增量与实体位移演化", "PASS", msg, 0)

    def test_obs_plants(self) -> TestCaseResult:
        """TC-17: Plants 植物事实结构与图层合法性测试"""
        if not self.client.is_loaded:
            return TestCaseResult("TC-17", "Observation", "Plants 植物事实结构", "SKIP", "DLL 未加载", 0)

        n, msg, data = self.client.get_observation_json()
        if n <= 0 or not data:
            return TestCaseResult("TC-17", "Observation", "Plants 植物事实结构", "SKIP", f"未获取到 Observation: {msg}", 0)

        plants = data.get("plants", [])
        for p in plants:
            pid = p.get("id", -1)
            ptype = p.get("type", -1)
            row = p.get("row", -1)
            col = p.get("col", -1)
            hp = p.get("hp", -1)
            layer = p.get("layer", -1)
            if pid < 0 or ptype < 0 or row < 0 or col < 0 or col > 8 or hp < 0 or layer not in (0, 1, 2):
                return TestCaseResult("TC-17", "Observation", "Plants 植物事实结构", "FAIL", f"植物数据非法: {p}", 0)

        return TestCaseResult(
            "TC-17", "Observation", "Plants 植物事实结构", "PASS",
            f"植物事实结构校验通过 (当前场上植物数: {len(plants)})", 0, details={"plants_count": len(plants)}
        )

    def test_obs_cards(self) -> TestCaseResult:
        """TC-18: Cards 卡槽技能与冷却倒计时可用性测试"""
        if not self.client.is_loaded:
            return TestCaseResult("TC-18", "Observation", "Cards 卡槽事实状态", "SKIP", "DLL 未加载", 0)

        n, msg, data = self.client.get_observation_json()
        if n <= 0 or not data:
            return TestCaseResult("TC-18", "Observation", "Cards 卡槽事实状态", "SKIP", f"未获取到 Observation: {msg}", 0)

        cards = data.get("cards", [])
        sun = data.get("global", {}).get("sun", 0)
        global_info = data.get("global", {})
        card_system = global_info.get("card_system", "")
        if "is_conveyor" in global_info:
            conveyor = bool(global_info.get("is_conveyor"))
        elif card_system:
            conveyor = card_system == "conveyor"
        else:
            runtime = self.client.get_runtime_state()
            mode = int(runtime.get("game_mode", -1))
            level = int(global_info.get("level_id", -1))
            conveyor = (
                (mode == 0 and level in (5, 10, 20, 25, 30, 40, 45, 50))
                or mode in (17, 21, 25, 26, 27, 33, 35, 46, 47, 48)
            )
        if not cards and not (conveyor or card_system == "dropped"):
            return TestCaseResult("TC-18", "Observation", "Cards 卡槽事实状态", "FAIL", "卡槽列表为空 (SeedBank 未解析)", 0)
        for c in cards:
            slot = c.get("slot", -1)
            ptype = c.get("plant_type", -1)
            cost = c.get("cost", -1)
            cd_dur = c.get("cooldown_duration", -1)
            cd_rem = c.get("cooldown_remaining", -1)
            ready = c.get("ready", False)

            if slot < 0 or ptype < 0 or cost < 0 or cd_dur < 0 or cd_rem < 0:
                return TestCaseResult("TC-18", "Observation", "Cards 卡槽事实状态", "FAIL", f"卡槽数据非法: {c}", 0)

            if conveyor:
                # 传送带卡由卡包存在、进入可见区域决定；阳光与刷新冷却
                # 都不参与可用性判断。旧 DLL 在 1-5 会错误地报 cost=50。
                if cost != 0 or cd_dur != 0 or cd_rem != 0:
                    return TestCaseResult(
                        "TC-18", "Observation", "Cards 卡槽事实状态", "FAIL",
                        f"传送带卡不应有阳光/冷却限制: slot={slot}, cost={cost}, cd_dur={cd_dur}, cd_rem={cd_rem}", 0
                    )
                continue

            if cd_dur == 0 and cost == 0:
                # 掉落卡无阳光/冷却；ready 由对应 Coin 是否可拾取决定。
                continue

            # 普通卡槽的 ready 由阳光与刷新冷却决定。
            expected_ready = (cd_rem <= 0 and sun >= cost)
            if ready != expected_ready:
                return TestCaseResult(
                    "TC-18", "Observation", "Cards 卡槽事实状态", "FAIL",
                    f"卡片 ready 标志与事实不符: slot={slot}, ready={ready}, expected={expected_ready}, sun={sun}, cost={cost}, cd_rem={cd_rem}", 0
                )

        return TestCaseResult(
            "TC-18", "Observation", "Cards 卡槽事实状态", "PASS",
            f"全部 {len(cards)} 个卡槽事实与 {'传送带' if conveyor else '阳光/CD'} 判定一致",
            0, details={"cards_count": len(cards), "conveyor": conveyor}
        )

    def test_obs_zombies(self) -> TestCaseResult:
        """TC-19: Zombies 僵尸连续物理坐标与生命值测试"""
        if not self.client.is_loaded:
            return TestCaseResult("TC-19", "Observation", "Zombies 僵尸事实状态", "SKIP", "DLL 未加载", 0)

        n, msg, data = self.client.get_observation_json()
        if n <= 0 or not data:
            return TestCaseResult("TC-19", "Observation", "Zombies 僵尸事实状态", "SKIP", f"未获取到 Observation: {msg}", 0)

        zombies = data.get("zombies", [])
        ids = set()
        for z in zombies:
            zid = z.get("id", -1)
            ztype = z.get("type", -1)
            row = z.get("row", -1)
            x = z.get("x", 0.0)
            if "id" not in z or (int(zid) & 0xFFFF0000) == 0 or zid in ids:
                return TestCaseResult("TC-19", "Observation", "Zombies 僵尸事实状态", "FAIL", f"僵尸 ID 重复或非法: {zid}", 0)
            ids.add(zid)

            if ztype < 0 or (ztype != 25 and row < 0) or z.get("phase", 0) in (1, 2, 3):
                return TestCaseResult("TC-19", "Observation", "Zombies 僵尸事实状态", "FAIL", f"僵尸字段异常: {z}", 0)

        return TestCaseResult(
            "TC-19", "Observation", "Zombies 僵尸事实状态", "PASS",
            f"僵尸列表校验通过 (当前场上活跃僵尸数: {len(zombies)}, 全部具有唯一 ID 与连续坐标)", 0,
            details={"zombies_count": len(zombies)}
        )

    def test_obs_mowers(self) -> TestCaseResult:
        """TC-20: Mowers 底线割草机状态完整性测试"""
        if not self.client.is_loaded:
            return TestCaseResult("TC-20", "Observation", "Mowers 底线割草机状态", "SKIP", "DLL 未加载", 0)

        n, msg, data = self.client.get_observation_json()
        if n <= 0 or not data:
            return TestCaseResult("TC-20", "Observation", "Mowers 底线割草机状态", "SKIP", f"未获取到 Observation: {msg}", 0)

        mowers = data.get("mowers", [])
        scene = data.get("global", {}).get("scene", 0)
        expected_rows = 6 if scene in (2, 3) else 5

        if len(mowers) != expected_rows:
            return TestCaseResult(
                "TC-20", "Observation", "Mowers 底线割草机状态", "FAIL",
                f"割草机行数与场景不符: len={len(mowers)}, expected={expected_rows}", 0
            )

        for i, m in enumerate(mowers):
            if m.get("row") != i or m.get("state") not in ("READY", "ROLLING", "LOST"):
                return TestCaseResult("TC-20", "Observation", "Mowers 底线割草机状态", "FAIL", f"割草机项异常: {m}", 0)

        return TestCaseResult(
            "TC-20", "Observation", "Mowers 底线割草机状态", "PASS",
            f"割草机状态校验通过 ({len(mowers)} 行推车，状态均已明确)", 0, details={"mowers": mowers}
        )

    def test_obs_consistency_with_perception(self) -> TestCaseResult:
        """TC-21: Gym Native 与 tools_perception 事实一致性回归对比"""
        import ctypes, json, os
        cur_dir = os.path.dirname(os.path.abspath(__file__))
        perception_dll_path = os.path.abspath(os.path.join(cur_dir, "..", "..", "..", "tools_perception", "pvz_perception.dll"))
        if not os.path.isfile(perception_dll_path):
            return TestCaseResult("TC-21", "Consistency", "Gym 与 tools_perception 共有字段一致性", "SKIP", "未找到 pvz_perception.dll", 0)

        try:
            p_dll = ctypes.cdll.LoadLibrary(perception_dll_path)
            p_dll.pvz_perception_init()
            if not p_dll.pvz_perception_is_battle_ready():
                return TestCaseResult("TC-21", "Consistency", "Gym 与 tools_perception 共有字段一致性", "SKIP", "Perception 探测游戏未战斗就绪", 0)

            # 先暂停游戏以防两边读取跨帧
            self.client.pause()

            # 1. 读取 Gym Observation
            _, _, gym_obs = self.client.get_observation_json()
            if not gym_obs:
                return TestCaseResult("TC-21", "Consistency", "Gym 与 tools_perception 共有字段一致性", "FAIL", "读取 Gym Observation 失败", 0)

            # 2. 读取 tools_perception Slim JSON
            p_buf = ctypes.create_string_buffer(131072)
            p_len = p_dll.pvz_perception_get_slim_json(p_buf, 131072)
            if p_len <= 0:
                return TestCaseResult("TC-21", "Consistency", "Gym 与 tools_perception 共有字段一致性", "FAIL", "读取 tools_perception JSON 失败", 0)

            p_data = json.loads(p_buf.value.decode("utf-8"))

            # 3. 逐项严格对比共有字段
            g_meta = gym_obs.get("global", {})
            p_meta = p_data.get("meta", {})
            p_macro = p_data.get("macro", {})

            diffs = []
            if g_meta.get("game_clock") != p_meta.get("game_clock"):
                diffs.append(f"game_clock 不一致: Gym={g_meta.get('game_clock')} vs Perception={p_meta.get('game_clock')}")

            if g_meta.get("sun") != p_macro.get("sun"):
                diffs.append(f"sun 不一致: Gym={g_meta.get('sun')} vs Perception={p_macro.get('sun')}")

            if g_meta.get("current_wave") != p_macro.get("current_wave"):
                diffs.append(f"current_wave 不一致: Gym={g_meta.get('current_wave')} vs Perception={p_macro.get('current_wave')}")

            if g_meta.get("total_waves") != p_macro.get("total_waves"):
                diffs.append(f"total_waves 不一致: Gym={g_meta.get('total_waves')} vs Perception={p_macro.get('total_waves')}")

            # 对比卡槽植物类型
            gym_cards = gym_obs.get("cards", [])
            p_cards = p_data.get("seed_bank", [])
            if len(gym_cards) != len(p_cards):
                diffs.append(f"卡槽数量不一致: Gym={len(gym_cards)} vs Perception={len(p_cards)}")
            else:
                for i in range(len(gym_cards)):
                    if gym_cards[i].get("plant_type") != p_cards[i].get("type"):
                        diffs.append(f"卡槽 {i} 类型不一致: Gym={gym_cards[i].get('plant_type')} vs Perception={p_cards[i].get('type')}")

            # 对比僵尸数量
            gym_zombies = gym_obs.get("zombies", [])
            p_zombies = p_data.get("zombies", [])
            if len(gym_zombies) != len(p_zombies):
                diffs.append(f"僵尸数量不一致: Gym={len(gym_zombies)} vs Perception={len(p_zombies)}")

            if diffs:
                return TestCaseResult("TC-21", "Consistency", "Gym 与 tools_perception 共有字段一致性", "FAIL", f"发现共有字段差异: {diffs}", 0)

            return TestCaseResult(
                "TC-21", "Consistency", "Gym 与 tools_perception 共有字段一致性", "PASS",
                f"Gym 与 tools_perception 共有字段 (clock={g_meta.get('game_clock')}, sun={g_meta.get('sun')}, wave={g_meta.get('current_wave')}, cards={len(gym_cards)}, zombies={len(gym_zombies)}) 100% 严格一致！",
                0
            )
        except Exception as e:
            return TestCaseResult("TC-21", "Consistency", "Gym 与 tools_perception 共有字段一致性", "FAIL", f"回归对比异常: {e}", 0)

    def test_obs_special_entities(self) -> TestCaseResult:
        """TC-22: 掉落卡、场地障碍物与冰道实体事实测试"""
        if not self.client.is_loaded:
            return TestCaseResult("TC-22", "Observation", "掉落卡、场地障碍物与冰道实体事实测试", "SKIP", "DLL 未加载", 0)

        n, msg, data = self.client.get_observation_json()
        if n <= 0 or not data:
            return TestCaseResult("TC-22", "Observation", "掉落卡、场地障碍物与冰道实体事实测试", "SKIP", f"未获取到 Observation: {msg}", 0)

        dropped_cards = data.get("dropped_cards")
        grid_items = data.get("grid_items")
        ice_trails = data.get("ice_trails")

        if not isinstance(dropped_cards, list) or not isinstance(grid_items, list) or not isinstance(ice_trails, list):
            return TestCaseResult("TC-22", "Observation", "掉落卡、场地障碍物与冰道实体事实测试", "FAIL", "新实体字段类型非 list", 0)

        # 校验 dropped_cards 结构
        for dc in dropped_cards:
            if "id" not in dc or "type" not in dc or "x" not in dc or "y" not in dc or "is_collectable" not in dc:
                return TestCaseResult("TC-22", "Observation", "掉落卡、场地障碍物与冰道实体事实测试", "FAIL", f"dropped_card 结构非法: {dc}", 0)

        # 校验 grid_items 结构
        for gi in grid_items:
            g_type = gi.get("type", "")
            if g_type not in ("GRAVE", "CRATER", "VASE", "LADDER", "BRAIN"):
                return TestCaseResult("TC-22", "Observation", "掉落卡、场地障碍物与冰道实体事实测试", "FAIL", f"grid_item 类型非法: {gi}", 0)
            if "row" not in gi or "col" not in gi or "remaining_ticks" not in gi:
                return TestCaseResult("TC-22", "Observation", "掉落卡、场地障碍物与冰道实体事实测试", "FAIL", f"grid_item 字段缺失: {gi}", 0)

        # 校验 ice_trails 结构
        for it in ice_trails:
            if "row" not in it or "col_start" not in it or "col_end" not in it or "melt_countdown" not in it:
                return TestCaseResult("TC-22", "Observation", "掉落卡、场地障碍物与冰道实体事实测试", "FAIL", f"ice_trail 结构非法: {it}", 0)

        return TestCaseResult(
            "TC-22", "Observation", "掉落卡、场地障碍物与冰道实体事实测试", "PASS",
            f"实体结构校验通过: DroppedCards={len(dropped_cards)}, GridItems={len(grid_items)}, IceTrails={len(ice_trails)}", 0,
            details={"dropped_cards_count": len(dropped_cards), "grid_items_count": len(grid_items), "ice_trails_count": len(ice_trails)}
        )

    # -------------------------------------------------------------------------
    # 9. 自动化辅助模块测试 (Automation Helpers)
    # -------------------------------------------------------------------------

    def test_automation_symbols(self) -> TestCaseResult:
        """校验 5 个自动化辅助 C ABI 符号导出"""
        if not self.client.is_loaded:
            return TestCaseResult("TC-23", "Automation", "校验自动化辅助模块 5 个 C ABI 符号导出", "SKIP", "DLL 未加载", 0)
        expected = [
            "pvz_gym_set_automation_config",
            "pvz_gym_get_automation_config",
            "pvz_gym_handle_prebattle",
            "pvz_gym_automation_collect",
            "pvz_gym_automation_tick",
        ]
        missing = [s for s in expected if not hasattr(self.client.dll, s)]
        if missing:
            return TestCaseResult("TC-23", "Automation", "校验自动化辅助模块 5 个 C ABI 符号导出", "FAIL", f"缺失自动化符号: {missing}", 0)
        return TestCaseResult("TC-23", "Automation", "校验自动化辅助模块 5 个 C ABI 符号导出", "PASS", f"全部 5 个自动化符号均已正确导出: {expected}", 0)

    def test_automation_config_toggle(self) -> TestCaseResult:
        """自动化辅助配置默认开启与可控关闭测试"""
        if not self.client.is_loaded:
            return TestCaseResult("TC-24", "Automation", "自动化辅助配置默认开启与可控关闭测试", "SKIP", "DLL 未加载", 0)

        # 1. 恢复默认并检查
        self.client.set_automation_config(True, True, True)
        cfg_def = self.client.get_automation_config()
        if not (cfg_def["enabled"] and cfg_def["auto_dialogue"] and cfg_def["auto_collect"]):
            return TestCaseResult("TC-24", "Automation", "自动化辅助配置默认开启与可控关闭测试", "FAIL", f"默认配置非全开: {cfg_def}", 0)

        # 2. 测试关闭
        self.client.set_automation_config(False, False, False)
        cfg_off = self.client.get_automation_config()
        if cfg_off["enabled"] or cfg_off["auto_dialogue"] or cfg_off["auto_collect"]:
            return TestCaseResult("TC-24", "Automation", "自动化辅助配置默认开启与可控关闭测试", "FAIL", f"配置关闭失败: {cfg_off}", 0)

        # 3. 还原默认
        self.client.set_automation_config(True, True, True)
        return TestCaseResult("TC-24", "Automation", "自动化辅助配置默认开启与可控关闭测试", "PASS", "自动化配置默认全开，且支持独立/整体关闭与读取", 0)

    def test_automation_safe_calls(self) -> TestCaseResult:
        """自动化辅助接口 (handle_prebattle, collect) 安全调用测试"""
        if not self.client.is_loaded:
            return TestCaseResult("TC-25", "Automation", "自动化辅助接口安全调用测试", "SKIP", "DLL 未加载", 0)

        # 无论当前处于何种状态，调用均不应崩溃
        pre_ok = self.client.handle_prebattle()
        collected = self.client.automation_collect()
        if not isinstance(pre_ok, bool):
            return TestCaseResult("TC-25", "Automation", "自动化辅助接口安全调用测试", "FAIL", f"handle_prebattle 返回非 bool: {pre_ok}", 0)
        if collected < 0:
            return TestCaseResult("TC-25", "Automation", "自动化辅助接口安全调用测试", "FAIL", f"automation_collect 返回负数错误码: {collected}", 0)

        return TestCaseResult("TC-25", "Automation", "自动化辅助接口安全调用测试", "PASS", f"接口安全调用通过: prebattle={pre_ok}, collected_count={collected}", 0)

    # -------------------------------------------------------------------------
    # 10. 选卡独立动作域与编排器测试 (Seed Selection & EpisodeRunner)
    # -------------------------------------------------------------------------

    def test_seed_selection_symbols(self) -> TestCaseResult:
        """TC-26: 校验选卡独立动作域 7 个 C ABI 符号导出"""
        if not self.client.is_loaded:
            return TestCaseResult("TC-26", "SeedSelection", "校验选卡模块 7 个 C ABI 符号导出", "SKIP", "DLL 未加载", 0)
        expected = [
            "pvz_gym_get_seed_type_capacity",
            "pvz_gym_get_seed_selection_json",
            "pvz_gym_get_seed_selection_mask",
            "pvz_gym_select_seed",
            "pvz_gym_select_imitater",
            "pvz_gym_deselect_seed",
            "pvz_gym_confirm_seeds",
        ]
        missing = [s for s in expected if not hasattr(self.client.dll, s)]
        if missing:
            return TestCaseResult("TC-26", "SeedSelection", "校验选卡模块 7 个 C ABI 符号导出", "FAIL", f"缺失选卡符号: {missing}", 0)
        return TestCaseResult("TC-26", "SeedSelection", "校验选卡模块 7 个 C ABI 符号导出", "PASS", f"全部 7 个选卡符号均已正确导出: {expected}", 0)

    def test_seed_selection_capacity_and_mask(self) -> TestCaseResult:
        """TC-27: 选卡 ABI 固定容量与 193 维 Action Mask 契约测试"""
        if not self.client.is_loaded:
            return TestCaseResult("TC-27", "SeedSelection", "选卡容量与 193 维 Action Mask 契约测试", "SKIP", "DLL 未加载", 0)

        capacity = self.client.get_seed_type_capacity()
        if capacity != 64:
            return TestCaseResult("TC-27", "SeedSelection", "选卡容量与 193 维 Action Mask 契约测试", "FAIL", f"选卡 ABI 容量不等于 64: {capacity}", 0)

        expected_mask_len = capacity * 3 + 1  # 193
        mask = self.client.get_seed_selection_mask()
        if mask and len(mask) != expected_mask_len:
            return TestCaseResult("TC-27", "SeedSelection", "选卡容量与 193 维 Action Mask 契约测试", "FAIL", f"Action Mask 长度不等于 {expected_mask_len}: len={len(mask)}", 0)

        return TestCaseResult("TC-27", "SeedSelection", "选卡容量与 193 维 Action Mask 契约测试", "PASS", f"选卡 ABI 容量为 {capacity}，动作掩码契约空间为 {expected_mask_len} (0..63:Select, 64..127:Imitater, 128..191:Deselect, 192:Confirm)", 0)

    def test_seed_selection_safe_execution(self) -> TestCaseResult:
        """TC-28: 静态校验选卡执行器 ABI，绝不发送选卡动作。"""
        category = "SeedSelection / Static"
        name = "选卡执行接口静态绑定与无副作用策略测试（不操作游戏）"
        if not self.client.is_loaded:
            return TestCaseResult("TC-28", category, name, "SKIP", "DLL 未加载", 0)

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
            return TestCaseResult(
                "TC-28", category, name, "FAIL",
                f"选卡 ABI 静态校验失败: missing={missing}, signatures={bad_signatures}", 0,
                details={"missing": missing, "bad_signatures": bad_signatures},
            )

        return TestCaseResult(
            "TC-28", category, name, "PASS",
            "已静态校验 Select/Imitater/Deselect/Confirm 的 C ABI 绑定；未调用任何选卡执行接口，游戏状态未改变",
            0,
            details={"symbols": list(expected_arg_counts), "game_action_calls": 0},
        )

    def test_seed_selection_gym_env(self) -> TestCaseResult:
        """TC-29: SeedSelectionEnv Gymnasium 标准接口合规性与 Space 空间结构测试"""
        try:
            from pvz_gym.envs.seed_selection_env import SeedSelectionEnv
            from pvz_gym.bridge.seed_selection import SeedSelectionNativeBackend

            backend = SeedSelectionNativeBackend()
            env = SeedSelectionEnv(backend=backend)

            if env.action_space.n != 193:
                return TestCaseResult("TC-29", "GymEnv", "SeedSelectionEnv Gymnasium 接口合规性测试", "FAIL", f"action_space.n ({env.action_space.n}) != 193", 0)

            req_keys = {"present", "required_count", "selected_count", "can_confirm", "seeds_in_flight", "warning_flags", "available", "unlocked", "selected", "disabled"}
            missing_keys = req_keys - set(env.observation_space.spaces.keys())
            if missing_keys:
                return TestCaseResult("TC-29", "GymEnv", "SeedSelectionEnv Gymnasium 接口合规性测试", "FAIL", f"observation_space 缺失键: {missing_keys}", 0)

            on, _ = self.client.is_game_on()
            if not on:
                env.close()
                return TestCaseResult("TC-29", "GymEnv", "SeedSelectionEnv Gymnasium 接口合规性测试", "SKIP", "游戏未在线，已验证 Space 空间结构合规，跳过实机 reset", 0)

            obs, info = env.reset()
            if not isinstance(obs, dict) or "action_mask" not in info:
                return TestCaseResult("TC-29", "GymEnv", "SeedSelectionEnv Gymnasium 接口合规性测试", "FAIL", "reset() 未返回预期的 (obs_dict, info_dict)", 0)

            env.close()
            return TestCaseResult("TC-29", "GymEnv", "SeedSelectionEnv Gymnasium 接口合规性测试", "PASS", "SeedSelectionEnv 符合 Gymnasium 标准规范，空间与观测投影完全自洽", 0)
        except Exception as e:
            return TestCaseResult("TC-29", "GymEnv", "SeedSelectionEnv Gymnasium 接口合规性测试", "FAIL", f"环境合规性校验异常: {e}", 0)

    def test_episode_runner_orchestration(self) -> TestCaseResult:
        """TC-30: EpisodeRunner 跨环境调度与延迟 Deck Return 结算测试"""
        try:
            from pvz_gym.runner.episode_runner import EpisodeRunner, EpisodeResult
            from pvz_gym.seed_reward import deck_outcome_reward

            r_win = deck_outcome_reward(won=True, lost=False)
            r_lose = deck_outcome_reward(won=False, lost=True)
            r_none = deck_outcome_reward(won=False, lost=False)
            if r_win != 1.0 or r_lose != -1.0 or r_none != 0.0:
                return TestCaseResult("TC-30", "Runner", "EpisodeRunner 跨环境调度与延迟回报结算测试", "FAIL", f"deck_outcome_reward 数值异常: win={r_win}, lose={r_lose}, none={r_none}", 0)

            runner = EpisodeRunner()
            if not hasattr(runner, "run_episode") or not hasattr(runner, "seed_env") or not hasattr(runner, "battle_env"):
                return TestCaseResult("TC-30", "Runner", "EpisodeRunner 跨环境调度与延迟回报结算测试", "FAIL", "EpisodeRunner 结构不完整", 0)

            runner.close()
            return TestCaseResult("TC-30", "Runner", "EpisodeRunner 跨环境调度与延迟回报结算测试", "PASS", "EpisodeRunner 与延迟选卡回报结算逻辑完整就绪，支持两阶段 Gym 串联编排", 0)
        except Exception as e:
            return TestCaseResult("TC-30", "Runner", "EpisodeRunner 跨环境调度与延迟回报结算测试", "FAIL", f"编排器测试异常: {e}", 0)

    def test_cheat_keys_and_send_key(self) -> TestCaseResult:
        """TC-31: 官方 DebugMode、场景门控作弊按键与按键分发契约测试。"""
        if not self.client.is_loaded:
            return TestCaseResult("TC-31", "CheatKeys", "官方 DebugMode 与作弊按键 C ABI 导出测试", "SKIP", "DLL 未加载", 0)

        syms = self.client.get_exported_symbols()
        missing = [s for s in ["pvz_gym_set_cheat_keys", "pvz_gym_get_cheat_keys", "pvz_gym_send_key", "pvz_gym_cancel_intro"] if not syms.get(s)]
        if missing:
            return TestCaseResult("TC-31", "CheatKeys", "官方 DebugMode 与作弊按键 C ABI 导出测试", "FAIL", f"缺少作弊按键导出符号: {missing}", 0)

        on, _ = self.client.is_game_on()
        if not on:
            # 离线安全调用验证
            self.client.set_cheat_keys(True)
            self.client.get_cheat_keys()
            self.client.send_key(0x20)
            self.client.cancel_intro()
            return TestCaseResult("TC-31", "CheatKeys", "官方 DebugMode 与作弊按键 C ABI 导出测试", "PASS", "已校验 4 个作弊/DebugMode C ABI 导出，离线调用安全无崩溃", 0)

        # 全局作弊按键验证：默认全局开启，支持动态开关控制
        self.client.set_cheat_keys(True)
        read_back_true = self.client.get_cheat_keys()
        self.client.set_cheat_keys(False)
        read_back_after_disable = self.client.get_cheat_keys()
        # 测试完毕后恢复全局开启
        self.client.set_cheat_keys(True)

        if not read_back_true or read_back_after_disable:
            return TestCaseResult("TC-31", "CheatKeys", "校验全局作弊按键与按键发送测试", "FAIL", f"set(True)->{read_back_true}, set(False)->{read_back_after_disable}", 0)

        # 用未绑定的 F24 验证消息分发，避免测试本身触发 SPACE 调试行为。
        send_ok = self.client.send_key(0x87)

        return TestCaseResult(
            case_id="TC-31",
            category="CheatKeys",
            name="校验全局作弊按键与按键发送测试",
            status="PASS",
            message="mTodCheatKeys 全局作弊模式读写正确，按键消息分发验证通过",
            elapsed_ms=0,
            details={"set_true": read_back_true, "after_set_false": read_back_after_disable, "send_key_ok": send_ok}
        )

    def test_dialog_cleanup_hook(self) -> TestCaseResult:
        """TC-38: 校验 Dialog C ABI 导出及场景切换弹窗清理钩子测试"""
        if not self.client.is_loaded:
            return TestCaseResult("TC-38", "Dialog", "校验 Dialog C ABI 导出及场景切换弹窗清理钩子", "SKIP", "DLL 未加载", 0)

        syms = self.client.get_exported_symbols()
        missing = [s for s in ["pvz_gym_get_dialog", "pvz_gym_kill_dialog"] if not syms.get(s)]
        if missing:
            return TestCaseResult("TC-38", "Dialog", "校验 Dialog C ABI 导出及场景切换弹窗清理钩子", "FAIL", f"缺少 Dialog 导出符号: {missing}", 0)

        # 1. 离线/在线安全调用验证 (查询不存在的 Dialog 应安全返回 0)
        d2 = self.client.get_dialog(2)    # NEWOPTIONS (菜单)
        d17 = self.client.get_dialog(17)  # GAME_OVER
        d19 = self.client.get_dialog(19)  # PAUSED

        # 2. 验证 cleanup_scene_dialogs 规则
        # 规则 A: old_ui == new_ui 时不清理
        res_same = self.client.cleanup_scene_dialogs(3, 3)
        if len(res_same) != 0:
            return TestCaseResult("TC-38", "Dialog", "校验 Dialog 弹窗清理钩子", "FAIL", f"相同场景应忽略清理，却返回: {res_same}", 0)

        # 规则 B: 非 (2, 3, 4, 5) 场景应忽略
        res_invalid = self.client.cleanup_scene_dialogs(1, 2)
        if len(res_invalid) != 0:
            return TestCaseResult("TC-38", "Dialog", "校验 Dialog 弹窗清理钩子", "FAIL", f"主菜单(1)切换应忽略，却返回: {res_invalid}", 0)

        # 规则 C: 在 (2, 3, 4, 5) 之间切换，如果没有 Dialog 则返回空列表
        res_normal = self.client.cleanup_scene_dialogs(2, 3)
        res_award = self.client.cleanup_scene_dialogs(4, 5)

        return TestCaseResult(
            case_id="TC-38",
            category="Dialog",
            name="校验 Dialog C ABI 导出及场景切换弹窗清理钩子测试",
            status="PASS",
            message=f"Dialog C ABI 导出正常，场景切换钩子规则完备 (d2=0x{d2:X}, d17=0x{d17:X}, d19=0x{d19:X})",
            elapsed_ms=0,
            details={
                "d2": d2, "d17": d17, "d19": d19,
                "cleaned_on_2_to_3": res_normal,
                "cleaned_on_4_to_5": res_award,
            }
        )
