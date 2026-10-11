# -*- coding: utf-8 -*-
"""PvZ 冒险模式中途存档清理单元测试"""

import os
import hashlib
import json
import struct
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from pvz_gym.save_cleaner import (clean_adventure_save, _ADVENTURE_SAVE_PATTERN,
                                  SaveScopeError, read_save_profiles, resolve_save_profile,
                                  restore_quarantined_save)


def make_profiles(directory, profiles=((1, "First"), (2, "Second"))):
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    data = bytearray(struct.pack("<IH", 14, len(profiles)))
    for sequence, (profile_id, name) in enumerate(profiles):
        encoded = name.encode("utf-8")
        data.extend(struct.pack("<H", len(encoded)) + encoded + struct.pack("<II", sequence, profile_id))
        (root / f"user{profile_id}.dat").write_bytes(struct.pack("<4I", 12, 1, 100, 1) + bytes(804))
    (root / "users.dat").write_bytes(data)
    return root


def summary(directory):
    return {path.relative_to(directory).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in Path(directory).rglob("*") if path.is_file()}


class TestSaveCleaner(unittest.TestCase):
    def test_regex_pattern(self):
        # 必须匹配冒险模式暂存档
        self.assertTrue(_ADVENTURE_SAVE_PATTERN.match("game1_0.dat"))
        self.assertTrue(_ADVENTURE_SAVE_PATTERN.match("game2_0.dat"))
        self.assertTrue(_ADVENTURE_SAVE_PATTERN.match("game10_0.dat"))
        self.assertTrue(_ADVENTURE_SAVE_PATTERN.match("GAME1_0.DAT"))

        # 绝对不能匹配无尽模式及其他游戏模式
        self.assertFalse(_ADVENTURE_SAVE_PATTERN.match("game1_13.dat"))  # 泳池无尽
        self.assertFalse(_ADVENTURE_SAVE_PATTERN.match("game2_13.dat"))  # 泳池无尽
        self.assertFalse(_ADVENTURE_SAVE_PATTERN.match("game1_10.dat"))  # 生存模式
        self.assertFalse(_ADVENTURE_SAVE_PATTERN.match("game2_16.dat"))  # 生存模式
        self.assertFalse(_ADVENTURE_SAVE_PATTERN.match("game2_53.dat"))  # 解谜/迷你游戏
        self.assertFalse(_ADVENTURE_SAVE_PATTERN.match("user1.dat"))     # 核心玩家档案
        self.assertFalse(_ADVENTURE_SAVE_PATTERN.match("user2.dat"))     # 核心玩家档案
        self.assertFalse(_ADVENTURE_SAVE_PATTERN.match("users.dat"))     # 用户列表

    def test_clean_adventure_save(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            test_files = [
                "game1_0.dat",   # 待清理
                "game2_0.dat",   # 待清理
                "game1_13.dat",  # 必须保留 (无尽)
                "game2_13.dat",  # 必须保留 (无尽)
                "game2_10.dat",  # 必须保留 (生存)
                "user1.dat",     # 必须保留 (档案)
                "user2.dat",     # 必须保留 (档案)
                "users.dat",     # 必须保留
            ]
            for name in test_files:
                with open(os.path.join(tmp_dir, name), "w", encoding="utf-8") as f:
                    f.write("mock")

            make_profiles(tmp_dir)
            before = summary(tmp_dir)
            cleaned = clean_adventure_save(tmp_dir, profile_id=1, instance_id="slot-0")
            cleaned_names = [os.path.basename(p) for p in cleaned]
            remaining = set(os.listdir(tmp_dir))

            self.assertEqual(set(cleaned_names), {"game1_0.dat"})
            self.assertEqual(
                remaining,
                {"game2_0.dat", "game1_13.dat", "game2_13.dat", "game2_10.dat", "user1.dat", "user2.dat", "users.dat", ".quarantine"}
            )
            after = summary(tmp_dir)
            for filename, digest in before.items():
                if filename != "game1_0.dat":
                    self.assertEqual(after[filename], digest)
            quarantined = next(Path(tmp_dir).glob(".quarantine/*/game1_0.dat"))
            self.assertEqual(quarantined.read_bytes(), b"mock")
            receipt = json.loads((quarantined.parent / "receipt.json").read_text(encoding="utf-8"))
            self.assertEqual((receipt["profile_id"], receipt["instance_id"]), (1, "slot-0"))

    def test_double_profile_double_instance_reset_never_changes_template_or_other_instance(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            template = make_profiles(root / "template")
            left, right = make_profiles(root / "left"), make_profiles(root / "right")
            for directory in (template, left, right):
                for filename in ("game1_0.dat", "game2_0.dat", "game1_13.dat"):
                    (directory / filename).write_bytes((str(directory) + filename).encode())
            template_before, right_before = summary(template), summary(right)
            left_before = summary(left)
            clean_adventure_save(left, profile_id=2, instance_id="left", template_dir=template)
            self.assertEqual(summary(template), template_before)
            self.assertEqual(summary(right), right_before)
            for filename, digest in left_before.items():
                if filename != "game2_0.dat":
                    self.assertEqual(summary(left)[filename], digest)
            clean_adventure_save(right, profile_id=1, instance_id="right", template_dir=template)
            self.assertTrue((left / "game1_0.dat").is_file())
            self.assertTrue((right / "game2_0.dat").is_file())
            self.assertEqual(summary(template), template_before)

    def test_unknown_directory_profile_and_multi_user_fail_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = make_profiles(tmp)
            (directory / "game1_0.dat").write_bytes(b"unchanged")
            before = summary(directory)
            for kwargs in ({}, {"profile_id": 3}, {"profile_id": True}):
                with self.subTest(kwargs=kwargs), self.assertRaises(SaveScopeError):
                    clean_adventure_save(directory, **kwargs)
            with patch("pvz_gym.save_cleaner.get_pvz_userdata_dirs", return_value=[str(directory)]):
                with self.assertRaises(SaveScopeError):
                    clean_adventure_save()
            self.assertEqual(summary(directory), before)

    def test_template_cannot_be_cleaned_even_with_a_known_profile(self):
        with tempfile.TemporaryDirectory() as tmp:
            template = make_profiles(tmp, ((2, "Training"),))
            (template / "game2_0.dat").write_bytes(b"original")
            before = summary(template)
            with self.assertRaises(SaveScopeError):
                clean_adventure_save(template, profile_id=2, template_dir=template)
            self.assertEqual(summary(template), before)

    def test_orphan_user_file_is_not_an_active_profile(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = make_profiles(tmp, ((2, "Training"),))
            (directory / "user1.dat").write_bytes(struct.pack("<4I", 12, 1, 0, 0))
            self.assertEqual([profile.profile_id for profile in read_save_profiles(directory)], [2])
            self.assertEqual(resolve_save_profile(directory).profile_id, 2)
            with self.assertRaises(SaveScopeError):
                resolve_save_profile(directory, 1)

    def test_corrupt_or_unsupported_users_file_prevents_cleanup(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = make_profiles(tmp, ((1, "Training"),))
            good = (directory / "users.dat").read_bytes()
            (directory / "game1_0.dat").write_bytes(b"original")
            for invalid in (good[:-1], good + b"extra", struct.pack("<I", 15) + good[4:]):
                (directory / "users.dat").write_bytes(invalid)
                with self.assertRaises(SaveScopeError):
                    clean_adventure_save(directory, profile_id=1)
                self.assertEqual((directory / "game1_0.dat").read_bytes(), b"original")

    def test_restoring_quarantine_refuses_a_new_save_and_wrong_profile(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = make_profiles(tmp)
            (directory / "game1_0.dat").write_bytes(b"old")
            clean_adventure_save(directory, profile_id=1)
            source = next(directory.glob(".quarantine/*/game1_0.dat"))
            with self.assertRaises(SaveScopeError):
                restore_quarantined_save(source, directory, profile_id=2)
            (directory / "game1_0.dat").write_bytes(b"new")
            with self.assertRaises(FileExistsError):
                restore_quarantined_save(source, directory, profile_id=1)
            self.assertEqual(source.read_bytes(), b"old")
            (directory / "game1_0.dat").unlink()
            restored = restore_quarantined_save(source, directory, profile_id=1)
            self.assertEqual(restored.read_bytes(), b"old")

    def test_client_scope_failure_prevents_native_reset(self):
        from pvz_gym.client import PvZClient
        client = object.__new__(PvZClient)
        client.clean_save = True
        client.save_dir = None
        client._dll = Mock()
        with self.assertRaises(SaveScopeError):
            client.reset({})
        self.assertEqual(client._dll.mock_calls, [])

    def test_workbench_default_external_attach_does_not_search_for_saves(self):
        from workbench.native.c_interface import PvzGymNativeDll
        client = object.__new__(PvzGymNativeDll)
        with patch("workbench.native.c_interface.clean_adventure_save") as cleaner:
            self.assertEqual(client._clean_current_adventure_save(), [])
            cleaner.assert_not_called()

    def test_workbench_ambiguous_scope_aborts_before_selection(self):
        from workbench.native.c_interface import PvzGymNativeDll
        client = object.__new__(PvzGymNativeDll)
        client.clean_save = True
        client.save_dir = None
        client.dll = Mock()
        for method, args in (("select_level", (1, 1)), ("select_level_id", (1,))):
            code, message = getattr(client, method)(*args)
            self.assertEqual(code, -1)
            self.assertIn("explicitly bound save_dir", message)
        self.assertEqual(client.dll.mock_calls, [])


if __name__ == "__main__":
    unittest.main()
