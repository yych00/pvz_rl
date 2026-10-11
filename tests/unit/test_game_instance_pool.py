"""Offline slot/profile isolation tests; no game process or real saves are used."""
from __future__ import annotations

import json
import multiprocessing as mp
import os
import struct
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from pvz_gym.runtime.instance_pool import (InstancePool, InstanceBusyError, atomic_json,
                                          process_creation_time, process_is_alive)
from pvz_gym.runtime.instances import GameInstance
from pvz_gym.save_cleaner import SaveScopeError
from tests.unit.test_save_cleaner import make_profiles, summary


def _lease_worker(root, replies, hold):
    pool = InstancePool(root, max_slots=2)
    lease = pool.acquire(0)
    replies.put((lease.slot, lease.product, os.getpid()))
    hold.wait(30)
    lease.release()


def _fake_executable(path):
    from tests.unit.test_public_setup import fake_game
    fake_game(path)


class InstancePoolTest(unittest.TestCase):
    def test_two_leases_are_exclusive_and_released_slot_identity_is_stable(self):
        with tempfile.TemporaryDirectory() as tmp:
            pool = InstancePool(tmp, max_slots=2)
            first = pool.acquire(0)
            second = pool.acquire(0)
            try:
                self.assertEqual((first.slot, second.slot), (0, 1))
                self.assertNotEqual(first.product, second.product)
                self.assertEqual(len(first.product), len("PlantsVsZombies"))
                with self.assertRaises(InstanceBusyError):
                    pool.acquire(0)
                identity = (first.product, first.instance_id)
                first.release()
                reused = pool.acquire(0)
                self.assertEqual((reused.product, reused.instance_id), identity)
                reused.release()
            finally:
                first.release()
                second.release()

    def test_different_roots_cannot_share_product_names(self):
        with tempfile.TemporaryDirectory() as tmp:
            left = InstancePool(Path(tmp) / "left").acquire()
            right = InstancePool(Path(tmp) / "right").acquire()
            try:
                self.assertNotEqual(left.product, right.product)
            finally:
                left.release(); right.release()

    def test_live_previous_game_or_manifest_pid_prevents_reuse(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            atomic_json(root / "env-00.lease.json", {"state": "leased", "owner_pid": 900, "game_pid": 42})
            pool = InstancePool(root, max_slots=3, process_probe=lambda pid: pid == 42,
                                executable_probe=lambda path: ())
            first = pool.acquire(0)
            self.assertEqual(first.slot, 1)
            first.release()
            (root / "env-01").mkdir()
            atomic_json(root / "env-01/instance.json", {"pid": 42, "game_creation_time_100ns": 123})
            second = pool.acquire(0)
            self.assertEqual(second.slot, 2)
            second.release()
            self.assertEqual(json.loads((root / "env-00.lease.json").read_text())["game_pid"], 42)

    def test_orphan_executable_detection_prevents_reuse_without_a_pid_record(self):
        with tempfile.TemporaryDirectory() as tmp:
            pool = InstancePool(tmp, max_slots=2, process_probe=lambda pid: False,
                                executable_probe=lambda path: (42,) if path.parent.name == "env-00" else ())
            lease = pool.acquire(0, executable_name="private.exe")
            self.assertEqual(lease.slot, 1)
            lease.release()

    def test_slot_cannot_be_released_until_the_owned_game_exits(self):
        with tempfile.TemporaryDirectory() as tmp:
            pool = InstancePool(tmp, max_slots=1, process_probe=lambda pid: pid == 42)
            lease = pool.acquire()
            lease.update(game_pid=42)
            with self.assertRaises(InstanceBusyError):
                lease.release()
            with self.assertRaises(InstanceBusyError):
                pool.acquire()
            lease.update(game_pid=None)
            lease.release()

    def test_cross_process_exclusion_and_crashed_owner_recovery(self):
        with tempfile.TemporaryDirectory() as tmp:
            context = mp.get_context("spawn")
            replies, hold = context.Queue(), context.Event()
            worker = context.Process(target=_lease_worker, args=(tmp, replies, hold))
            worker.start()
            try:
                slot, product, pid = replies.get(timeout=10)
                self.assertEqual(slot, 0)
                self.assertTrue(process_is_alive(pid))
                parent = InstancePool(tmp, max_slots=2)
                other = parent.acquire(0)
                self.assertEqual(other.slot, 1)
                other.release()
                worker.terminate()  # Only this test's Python worker, never PvZ.
                worker.join(10)
                self.assertFalse(worker.is_alive())
                recovered = parent.acquire(0)
                self.assertEqual((recovered.slot, recovered.product), (0, product))
                recovered.release()
            finally:
                if worker.is_alive():
                    hold.set()
                    worker.terminate()
                worker.join(10)
                replies.close()

    @unittest.skipUnless(os.name == "nt", "Windows FILETIME identity")
    def test_process_creation_time_is_recorded_for_current_owner(self):
        with tempfile.TemporaryDirectory() as tmp:
            lease = InstancePool(tmp).acquire()
            try:
                self.assertEqual(lease.record["owner_creation_time_100ns"], process_creation_time(os.getpid()))
                self.assertIsInstance(lease.record["owner_creation_time_100ns"], int)
            finally:
                lease.release()


class GameInstanceTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.base = Path(temporary.name)
        self.game = self.base / "game"
        self.game.mkdir()
        self.executable = self.game / "PlantsVsZombies.exe"
        _fake_executable(self.executable)
        (self.game / "main.pak").write_bytes(b"immutable resource")
        (self.game / "game.dll").write_bytes(b"library")
        (self.game / "properties").mkdir()
        (self.game / "properties/settings.xml").write_bytes(b"private settings")
        (self.game / "properties/partner.xml").write_bytes(b"vendor product override")
        self.template = make_profiles(self.base / "template", ((2, "Training"),))
        (self.template / "user1.dat").write_bytes(struct.pack("<4I", 12, 1, 0, 0))
        (self.template / "game2_0.dat").write_bytes(b"do not import intermediate save")
        self.root = self.base / "instances"
        for patcher in (patch.dict(os.environ, {"ProgramData": str(self.base / "programdata")}),
                        patch.object(GameInstance, "_apply_zero_volume_registry"),
                        patch.object(GameInstance, "_cleanup_registry")):
            patcher.start()
            self.addCleanup(patcher.stop)
        self.instances = []
        self.addCleanup(self.close_instances)

    def close_instances(self):
        for instance in self.instances:
            instance.close()

    def instance(self, environment_id=0, **kwargs):
        instance = GameInstance(self.executable, self.root, environment_id,
                                save_template=self.template, **kwargs)
        self.instances.append(instance)
        return instance

    def test_two_runs_reuse_disk_copy_and_preserve_private_writes(self):
        source_before, template_before = summary(self.game), summary(self.template)
        first = self.instance()
        executable = first.prepare()
        identity = (first.instance_id, first.product, first.save_dir)
        modified = executable.stat().st_mtime_ns
        (first.directory / "properties/settings.xml").write_bytes(b"owned writable config")
        (first.save_dir / "game2_0.dat").write_bytes(b"private resumed data")
        profile = first.save_dir / "user2.dat"
        profile.write_bytes(profile.read_bytes() + b"owned profile update")
        first.close()
        second = self.instance()
        with patch("pvz_gym.runtime.instances.shutil.copy2") as copy, \
             patch("pvz_gym.runtime.instances.shutil.copytree") as tree, \
             patch("pvz_gym.runtime.source_fingerprint._sha256") as source_hash:
            self.assertEqual(second.prepare(), executable)
            copy.assert_not_called(); tree.assert_not_called()
            source_hash.assert_not_called()
        self.assertEqual((second.instance_id, second.product, second.save_dir), identity)
        self.assertEqual(executable.stat().st_mtime_ns, modified)
        self.assertEqual((second.directory / "properties/settings.xml").read_bytes(), b"owned writable config")
        self.assertEqual((second.save_dir / "game2_0.dat").read_bytes(), b"private resumed data")
        self.assertEqual((second.save_dir / "user2.dat").read_bytes(), profile.read_bytes())
        self.assertEqual(summary(self.game), source_before)
        self.assertEqual(summary(self.template), template_before)
        self.assertFalse((second.directory / "properties/partner.xml").exists())
        self.assertTrue((self.game / "properties/partner.xml").exists())

    def test_concurrent_instances_have_independent_profiles_and_private_configuration(self):
        first, second = self.instance(), self.instance()
        first.prepare(); second.prepare()
        self.assertNotEqual(first.directory, second.directory)
        self.assertNotEqual(first.save_dir, second.save_dir)
        self.assertNotEqual(first.product, second.product)
        self.assertEqual((first.active_profile_id, second.active_profile_id), (2, 2))
        self.assertFalse((first.save_dir / "user1.dat").exists())
        self.assertFalse((first.save_dir / "game2_0.dat").exists())
        other = summary(second.save_dir)
        (first.save_dir / "user2.dat").write_bytes(b"modified owned profile")
        self.assertEqual(summary(second.save_dir), other)
        (first.directory / "properties/settings.xml").write_bytes(b"first instance only")
        self.assertEqual((second.directory / "properties/settings.xml").read_bytes(), b"private settings")

    def test_template_version_change_rebuilds_and_archives_private_saves(self):
        first = self.instance()
        first.prepare()
        save = first.save_dir / "game2_0.dat"
        save.write_bytes(b"old private state")
        first.close()
        template_user = self.template / "user2.dat"
        template_user.write_bytes(template_user.read_bytes() + b"new template version")
        next_run = self.instance()
        next_run.prepare()
        self.assertEqual((next_run.save_dir / "user2.dat").read_bytes(), template_user.read_bytes())
        self.assertFalse((next_run.save_dir / "game2_0.dat").exists())
        backup = Path(next_run.manifest["last_save_backup"])
        self.assertEqual((backup / "game2_0.dat").read_bytes(), b"old private state")
        self.assertEqual(len(list((self.root / ".archive").iterdir())), 1)

    def test_reset_save_and_rebuild_require_game_exit_and_keep_recoverable_backups(self):
        instance = self.instance()
        instance.prepare()
        old = instance.save_dir / "game2_0.dat"
        old.write_bytes(b"recoverable")
        before = summary(self.template)
        backup = Path(instance.reset_save())
        self.assertEqual((backup / old.name).read_bytes(), b"recoverable")
        self.assertFalse(old.exists())
        instance.rebuild()
        self.assertEqual(summary(self.template), before)
        live = Mock(); live.poll.return_value = None
        instance.process = live
        try:
            for operation in (instance.reset_save, instance.rebuild):
                with self.assertRaisesRegex(RuntimeError, "stop the private game"):
                    operation()
        finally:
            instance.process = None

    def test_unknown_profile_fails_before_copying_resources_or_saves(self):
        instance = self.instance(profile_id=1)
        before = summary(self.template)
        with self.assertRaises(SaveScopeError):
            instance.prepare()
        self.assertFalse(instance.directory.exists())
        self.assertFalse(instance.save_dir.exists())
        self.assertIsNone(instance.lease)
        self.assertEqual(summary(self.template), before)

    def test_reset_save_recovers_a_corrupt_private_profile_without_recopying_game(self):
        instance = self.instance()
        executable = instance.prepare()
        before = executable.stat().st_mtime_ns
        (instance.save_dir / "user2.dat").write_bytes(b"broken private profile")
        backup = Path(instance.reset_save())
        self.assertEqual((backup / "user2.dat").read_bytes(), b"broken private profile")
        self.assertEqual((instance.save_dir / "user2.dat").read_bytes(), (self.template / "user2.dat").read_bytes())
        self.assertEqual(executable.stat().st_mtime_ns, before)

    def test_unsupported_executable_does_not_replace_private_saves(self):
        instance = self.instance()
        instance.prepare()
        (instance.save_dir / "game2_0.dat").write_bytes(b"owned data")
        instance.close()
        before = summary(instance.save_dir)
        self.executable.write_bytes(b"unsupported executable")
        next_run = self.instance()
        with self.assertRaisesRegex(RuntimeError, "supported PE"):
            next_run.prepare()
        self.assertEqual(summary(instance.save_dir), before)

    def test_manifest_has_verified_source_product_actual_saves_and_profile(self):
        instance = self.instance()
        executable = instance.prepare()
        manifest = json.loads((instance.directory / "instance.json").read_text(encoding="utf-8"))
        self.assertEqual((manifest["product"], manifest["active_profile_id"]), (instance.product, 2))
        self.assertEqual(manifest["save_dir"], str(instance.save_dir))
        self.assertEqual(manifest["source"]["save_template"], str(self.template.resolve()))
        self.assertEqual(manifest["source"]["profile_ids"], [2])
        self.assertEqual(instance.template_dir, self.template.resolve())
        self.assertIn(instance.product.encode(), executable.read_bytes())
        self.assertNotIn(b"PlantsVsZombies", executable.read_bytes())
        self.assertIn(b"PlantsVsZombies", self.executable.read_bytes())

    def test_explicit_rebuild_can_repair_corrupt_manifest_after_exit(self):
        instance = self.instance()
        instance.prepare(); instance.close()
        path = instance.directory / "instance.json"
        path.write_bytes(b"corrupt")
        next_run = self.instance()
        with self.assertRaises(InstanceBusyError):
            next_run.prepare()
        executable = next_run.rebuild()
        self.assertTrue(executable.is_file())
        self.assertEqual(json.loads((next_run.directory / "instance.json").read_text())["manifest_version"], 1)

    def test_instance_root_cannot_overlap_original_game_or_template(self):
        for root in (self.game / "instances", self.template / "instances", self.base):
            with self.subTest(root=root), self.assertRaisesRegex(RuntimeError, "separate"):
                GameInstance(self.executable, root, save_template=self.template)

    def test_redirected_programdata_cannot_place_private_saves_in_original_game_or_template(self):
        before_game, before_template = summary(self.game), summary(self.template)
        for protected in (self.game, self.template):
            instance = self.instance()
            with patch.dict(os.environ, {"ProgramData": str(protected)}), \
                 self.assertRaisesRegex(SaveScopeError, "overlap"):
                instance.prepare()
            self.assertIsNone(instance.lease)
        self.assertEqual(summary(self.game), before_game)
        self.assertEqual(summary(self.template), before_template)

    def test_launch_records_pid_and_job_before_resuming_suspended_process(self):
        instance = self.instance()
        instance.prepare()
        process = Mock(pid=987654, _handle=42)
        process.poll.return_value = None
        job = Mock()
        api = Mock()

        def resume(handle):
            manifest = json.loads((instance.directory / "instance.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["pid"], process.pid)
            self.assertEqual(manifest["game_creation_time_100ns"], 123)
            self.assertEqual(instance.lease.record["game_pid"], process.pid)
            self.assertIs(instance.job, job)
            return 0

        api.NtResumeProcess.side_effect = resume
        with patch("pvz_gym.runtime.instances.subprocess.Popen", return_value=process) as launch, \
             patch("pvz_gym.runtime.instances._OwnedJob", return_value=job), \
             patch("pvz_gym.runtime.instances.process_creation_time", return_value=123), \
             patch("pvz_gym.runtime.instances.ctypes.WinDLL", return_value=api), \
             patch.object(instance, "_start_audio_mute_worker"):
            self.assertEqual(instance.start(), process.pid)
            self.assertEqual(launch.call_args.kwargs["creationflags"], 4)
            self.assertEqual(instance.start(), process.pid)
            launch.assert_called_once()
        instance.close()
        job.close.assert_called_once()
        self.assertIsNone(instance.lease)

    def test_provided_retail_game_and_template_prepare_offline_without_source_changes(self):
        repository = Path(__file__).resolve().parents[2]
        game = repository / "game/pvz-1.2.0.1065"
        template = repository / "local/save_template"
        if not (game / "PlantsVsZombies.exe").is_file() or not template.is_dir():
            self.skipTest("provided local game/template is unavailable")
        before_game, before_template = summary(game), summary(template)
        instance = GameInstance(game / "PlantsVsZombies.exe", self.root, save_template=template)
        self.instances.append(instance)
        executable = instance.prepare()  # Only copies/patches files, never calls start().
        self.assertTrue(executable.is_file())
        self.assertEqual(instance.active_profile_id, 1)
        self.assertEqual(instance.manifest["source"]["profile_ids"], [1])
        self.assertTrue((instance.save_dir / "user1.dat").exists())
        self.assertEqual(summary(game), before_game)
        self.assertEqual(summary(template), before_template)


if __name__ == "__main__":
    unittest.main()
