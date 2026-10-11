"""Reject mismatched archives before replacing an installed native DLL."""
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import struct
import tempfile
import unittest
import zipfile

from scripts.install_native import install_archive
from tests.unit.test_public_setup import fake_game


class NativeReleaseInstallTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        data = bytearray(fake_game(self.root / "fixture" / "dummy.exe").read_bytes())
        struct.pack_into("<H", data, 132, 0x8664)
        self.dll = bytes(data)
        self.destination = self.root / "bin"
        self.destination.mkdir()
        (self.destination / "pvz_gym_native.dll").write_bytes(b"keep-current-DLL")

    def archive(self, *, data=None, changes=None, extra=None):
        data = self.dll if data is None else data
        metadata = {"version": version("pvz-rl"), "machine": "x64", "runtime_interface_version": 1,
                    "dll_sha256": hashlib.sha256(data).hexdigest(), "source_commit": "fixture"}
        metadata.update(changes or {})
        path = self.root / "native.zip"
        with zipfile.ZipFile(path, "w") as package:
            package.writestr("pvz_gym_native.dll", data)
            package.writestr("metadata.json", json.dumps(metadata))
            package.writestr("LICENSE", "test notice")
            package.writestr("THIRD_PARTY_NOTICES.md", "test notice")
            if extra:
                package.writestr(extra, "unexpected")
        Path(str(path) + ".sha256").write_text(hashlib.sha256(path.read_bytes()).hexdigest(), encoding="ascii")
        return path

    def retained(self):
        self.assertEqual((self.destination / "pvz_gym_native.dll").read_bytes(), b"keep-current-DLL")
        self.assertFalse((self.destination / "metadata.json").exists())
        self.assertFalse((self.destination / "pvz_gym_native.dll.incoming").exists())

    def test_matching_archive_installs_only_dll_and_metadata(self):
        install_archive(self.archive(), self.destination)
        self.assertEqual((self.destination / "pvz_gym_native.dll").read_bytes(), self.dll)
        self.assertEqual({p.name for p in self.destination.iterdir()}, {"pvz_gym_native.dll", "metadata.json"})

    def test_archive_checksum_and_paths_are_verified_before_replace(self):
        archive = self.archive()
        archive.write_bytes(archive.read_bytes() + b"modified")
        with self.assertRaisesRegex(ValueError, "SHA256"):
            install_archive(archive, self.destination)
        self.retained()
        with self.assertRaisesRegex(ValueError, "contents"):
            install_archive(self.archive(extra="../outside.txt"), self.destination)
        self.retained()

    def test_wrong_version_interface_or_dll_hash_retains_existing_dll(self):
        for changes in ({"version": "0.0.0"}, {"runtime_interface_version": 99}, {"dll_sha256": "bad"}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                install_archive(self.archive(changes=changes), self.destination)
            self.retained()

    def test_metadata_cannot_mislabel_x86_dll_as_x64(self):
        data = bytearray(self.dll)
        struct.pack_into("<H", data, 132, 0x14C)
        with self.assertRaisesRegex(ValueError, "x64 PE"):
            install_archive(self.archive(data=bytes(data)), self.destination)
        self.retained()
