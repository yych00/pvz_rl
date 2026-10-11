"""Shared verification, invalidation and crash recovery without launching PvZ."""
from __future__ import annotations

import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from pvz_gym.runtime import source_fingerprint as fingerprint


def _hash_worker(cache, resource, replies, start):
    original = fingerprint._sha256

    def counted(path):
        replies.put(("hash", os.getpid()))
        return original(path)

    replies.put(("ready", os.getpid()))
    start.wait(20)
    with patch.object(fingerprint, "_sha256", side_effect=counted):
        value = fingerprint.cached_file_hashes({"resource": Path(resource)}, Path(cache))
    replies.put(("done", value["resource"]))


def _lock_worker(lock, acquired):
    with fingerprint._cache_lock(Path(lock)):
        acquired.set()
        acquired.wait(30)
        # Remain alive until the parent terminates this test-owned process.
        import time
        time.sleep(30)


class SourceFingerprintTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.resource = self.root / "main.pak"
        self.resource.write_bytes(b"resource contents")
        self.cache = self.root / "cache/source.json"
        self.paths = {"resource": self.resource}

    def test_warm_lookup_reads_no_file_contents_and_force_verifies_again(self):
        expected = fingerprint.cached_file_hashes(self.paths, self.cache)
        with patch.object(fingerprint, "_sha256", wraps=fingerprint._sha256) as hasher:
            self.assertEqual(fingerprint.cached_file_hashes(self.paths, self.cache), expected)
            hasher.assert_not_called()
            self.assertEqual(fingerprint.cached_file_hashes(self.paths, self.cache, force=True), expected)
            hasher.assert_called_once_with(self.resource)

    def test_only_changed_file_is_rehashed_and_inventory_tracks_additions_and_removals(self):
        other = self.root / "settings.xml"
        other.write_bytes(b"settings")
        paths = dict(self.paths, settings=other)
        first = fingerprint.cached_file_hashes(paths, self.cache)
        self.resource.write_bytes(b"modified resource contents")
        with patch.object(fingerprint, "_sha256", wraps=fingerprint._sha256) as hasher:
            second = fingerprint.cached_file_hashes(paths, self.cache)
            hasher.assert_called_once_with(self.resource)
        self.assertNotEqual(first["resource"], second["resource"])
        self.assertEqual(first["settings"], second["settings"])
        added = self.root / "new.dll"
        added.write_bytes(b"new file")
        with patch.object(fingerprint, "_sha256", wraps=fingerprint._sha256) as hasher:
            fingerprint.cached_file_hashes(dict(paths, added=added), self.cache)
            hasher.assert_called_once_with(added)
        fingerprint.cached_file_hashes(self.paths, self.cache)
        self.assertEqual(set(json.loads(self.cache.read_text())["files"]), {"resource"})

    def test_replacement_with_identical_size_and_mtime_invalidates_file_identity(self):
        first = fingerprint.cached_file_hashes(self.paths, self.cache)
        metadata = self.resource.stat()
        replacement = self.root / "replacement.pak"
        replacement.write_bytes(b"different content")
        self.assertEqual(replacement.stat().st_size, metadata.st_size)
        os.utime(replacement, ns=(metadata.st_atime_ns, metadata.st_mtime_ns))
        os.replace(replacement, self.resource)
        second = fingerprint.cached_file_hashes(self.paths, self.cache)
        self.assertNotEqual(first, second)

    def test_corrupt_cache_is_rebuilt(self):
        expected = fingerprint.cached_file_hashes(self.paths, self.cache)
        for content in ("broken json", "[]", '{"version":1,"files":[]}'):
            self.cache.write_text(content)
            with patch.object(fingerprint, "_sha256", wraps=fingerprint._sha256) as hasher:
                self.assertEqual(fingerprint.cached_file_hashes(self.paths, self.cache), expected)
                hasher.assert_called_once()

    def test_changes_during_hashing_are_not_published(self):
        original = fingerprint._sha256

        def changing(path):
            value = original(path)
            path.write_bytes(path.read_bytes() + b"x")
            return value

        with patch.object(fingerprint, "_sha256", side_effect=changing):
            with self.assertRaisesRegex(RuntimeError, "changed repeatedly"):
                fingerprint.cached_file_hashes(self.paths, self.cache)
        self.assertFalse(self.cache.exists())
        expected = hashlib.sha256(self.resource.read_bytes()).hexdigest()
        self.assertEqual(fingerprint.cached_file_hashes(self.paths, self.cache)["resource"], expected)

    def test_eight_workers_share_one_resource_read(self):
        context = mp.get_context("spawn")
        replies, start = context.Queue(), context.Event()
        workers = [context.Process(target=_hash_worker,
                   args=(str(self.cache), str(self.resource), replies, start)) for _ in range(8)]
        try:
            for worker in workers:
                worker.start()
            for _ in workers:
                self.assertEqual(replies.get(timeout=30)[0], "ready")
            start.set()
            hashes, done = 0, []
            while len(done) < len(workers):
                kind, value = replies.get(timeout=30)
                if kind == "hash":
                    hashes += 1
                elif kind == "done":
                    done.append(value)
            self.assertEqual(hashes, 1)
            self.assertEqual(set(done), {hashlib.sha256(self.resource.read_bytes()).hexdigest()})
            for worker in workers:
                worker.join(10)
                self.assertEqual(worker.exitcode, 0)
        finally:
            start.set()
            for worker in workers:
                if worker.is_alive():
                    worker.terminate()
                if worker.pid:
                    worker.join(10)
            replies.close()

    def test_crashed_worker_releases_cache_lock(self):
        self.cache.parent.mkdir()
        context = mp.get_context("spawn")
        acquired = context.Event()
        worker = context.Process(target=_lock_worker, args=(str(self.cache.with_suffix(".lock")), acquired))
        worker.start()
        try:
            self.assertTrue(acquired.wait(20))
            worker.terminate()
            worker.join(10)
            self.assertFalse(worker.is_alive())
            self.assertIn("resource", fingerprint.cached_file_hashes(self.paths, self.cache))
        finally:
            if worker.is_alive():
                worker.terminate()
            worker.join(10)


if __name__ == "__main__":
    unittest.main()
