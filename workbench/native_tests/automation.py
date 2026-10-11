# -*- coding: utf-8 -*-
"""
自动化辅助模块 (Automation Helpers) 单元测试集 (test_automation.py)
严格对齐规格要求：
- 确认自动化辅助模块是训练与测试便利工具，绝不进入 RL 动作空间
- 验证默认配置为开启 (enabled=True, auto_dialogue=True, auto_collect=True)
- 验证配置可读可写可关闭
- 验证 handle_prebattle 与 automation_collect 在安全边界下调用正常
- 验证 PvzGymEnv 实例化支持配置关闭且不改变 action_space / observation_space
"""

import unittest
from pvz_gym.bridge.native import NativePvzBackend
from pvz_gym.env import PvzGymEnv


class TestAutomationHelpers(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.backend = NativePvzBackend()

    def setUp(self):
        # 每次测试前恢复默认自动化配置
        self.backend.set_automation_config(enabled=True, auto_dialogue=True, auto_collect=True)
        self.backend.set_auto_whack(True)

    def tearDown(self):
        # 每次测试后恢复默认自动化配置
        self.backend.set_automation_config(enabled=True, auto_dialogue=True, auto_collect=True)
        self.backend.set_auto_whack(True)

    def test_01_default_config(self):
        """验证自动化模块默认配置全开"""
        cfg = self.backend.get_automation_config()
        self.assertTrue(cfg["enabled"], "默认应该开启自动化总开关")
        self.assertTrue(cfg["auto_dialogue"], "默认应该开启自动处理对话")
        self.assertTrue(cfg["auto_collect"], "默认应该开启自动收集掉落物")
        self.assertTrue(cfg["auto_whack"], "默认应该开启锤僵尸关卡自动锤击")

    def test_auto_whack_independent_config(self):
        self.backend.set_auto_whack(False)
        self.backend.set_automation_config(enabled=True, auto_dialogue=True, auto_collect=True)
        self.assertFalse(self.backend.get_automation_config()["auto_whack"])
        self.backend.set_auto_whack(True)
        self.assertTrue(self.backend.get_automation_config()["auto_whack"])

    def test_02_set_and_get_config(self):
        """验证配置读写与关闭能力"""
        # 测试全关
        self.backend.set_automation_config(enabled=False, auto_dialogue=False, auto_collect=False)
        cfg = self.backend.get_automation_config()
        self.assertFalse(cfg["enabled"])
        self.assertFalse(cfg["auto_dialogue"])
        self.assertFalse(cfg["auto_collect"])

        # 测试独立关闭自动收集
        self.backend.set_automation_config(enabled=True, auto_dialogue=True, auto_collect=False)
        cfg = self.backend.get_automation_config()
        self.assertTrue(cfg["enabled"])
        self.assertTrue(cfg["auto_dialogue"])
        self.assertFalse(cfg["auto_collect"])

        # 测试独立关闭自动对话
        self.backend.set_automation_config(enabled=True, auto_dialogue=False, auto_collect=True)
        cfg = self.backend.get_automation_config()
        self.assertTrue(cfg["enabled"])
        self.assertFalse(cfg["auto_dialogue"])
        self.assertTrue(cfg["auto_collect"])

    def test_03_handle_prebattle_safety(self):
        """验证 handle_prebattle 调用安全，不发生崩溃"""
        # 无论当前游戏是否在线，调用均应安全返回 bool
        res = self.backend.handle_prebattle()
        self.assertIsInstance(res, bool)

    def test_04_automation_collect_safety(self):
        """验证 automation_collect 调用安全，返回收集非负计数值"""
        # 未在战斗时应安全返回 0
        count = self.backend.automation_collect()
        self.assertIsInstance(count, int)
        self.assertGreaterEqual(count, 0)

    def test_05_rl_action_space_isolation(self):
        """验证自动化模块绝不污染 RL Action Space 与 Observation Space"""
        env_auto = PvzGymEnv(self.backend, automation_enabled=True)
        env_manual = PvzGymEnv(self.backend, automation_enabled=False)

        # 两个环境的动作空间与观测空间必须完全一致
        self.assertEqual(env_auto.action_space, env_manual.action_space, "动作空间不应受自动化配置影响")
        self.assertEqual(env_auto.observation_space, env_manual.observation_space, "观测空间不应受自动化配置影响")

        # 检查后端配置是否正确被置为关闭
        cfg = self.backend.get_automation_config()
        self.assertFalse(cfg["enabled"], "实例化 automation_enabled=False 时应将 backend 配置置为关闭")


if __name__ == "__main__":
    unittest.main()
