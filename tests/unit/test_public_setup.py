"""Independent negative-case coverage for public installation and save scope."""
import hashlib
import os
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import patch
from pvz_gym.runtime.game_validation import validate_game, GameCompatibilityError, LAYOUT_SIGNATURES
from pvz_gym.runtime.save_template import prepare_training_save, training_profile_bytes
from pvz_gym.save_cleaner import read_save_profiles, SaveScopeError
from pvz_gym.env_file import repository_root, default_env_path, resolve_game_paths, find_env_file

def fake_game(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    image = bytearray(0x50000)
    image[:2] = b"MZ"
    struct.pack_into("<I", image, 60, 128)
    image[128:132] = b"PE\0\0"
    struct.pack_into("<HHI", image, 132, 0x14C, 1, 0x4A37D6AF)
    struct.pack_into("<H", image, 148, 224)
    struct.pack_into("<H", image, 152, 0x10B)
    struct.pack_into("<I", image, 180, 0x400000)
    struct.pack_into("<8sIIII", image, 376, b".text", len(image)-512, 0x1000, len(image)-512, 512)
    image[200:215] = b"PlantsVsZombies"
    image[240:258] = b"Plants vs. Zombies"
    for address, signature in LAYOUT_SIGNATURES.items():
        offset = address - 0x401000 + 512
        image[offset:offset+len(signature)] = signature
    path.write_bytes(image)
    (path.parent / "main.pak").write_bytes(b"synthetic-test-resource")
    return path

class PublicSetupTest(unittest.TestCase):
    def test_external_chinese_space_path_version_resource_and_signature_checks(self):
        with tempfile.TemporaryDirectory(prefix="中文 game ") as tmp:
            exe = fake_game(Path(tmp)/"source"/"PlantsVsZombies.exe")
            validate_game(exe)
            original = exe.read_bytes()
            for offset, value, pattern in ((136, 0x49ECF563, "Unsupported"), (132, 0x8664, "Unsupported")):
                image = bytearray(original)
                struct.pack_into("<I" if offset==136 else "<H", image, offset, value)
                exe.write_bytes(image)
                with self.assertRaisesRegex(GameCompatibilityError, pattern):
                    validate_game(exe)
            exe.write_bytes(original)
            image = bytearray(original); image[0x415D50-0x401000+512] ^= 1; exe.write_bytes(image)
            with self.assertRaisesRegex(GameCompatibilityError, "signature"):
                validate_game(exe)
            exe.write_bytes(original)
            (exe.parent/"main.pak").unlink()
            with self.assertRaisesRegex(GameCompatibilityError, "Missing"):
                validate_game(exe)
            for n in ("images", "sounds", "reanim"):
                (exe.parent/n).mkdir(); (exe.parent/n/"asset").write_text("synthetic")
            validate_game(exe)

    def test_generated_profile_idempotence_and_original_files_unchanged(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {}, clear=True):
            root=Path(tmp); exe=fake_game(root/"game"/"PlantsVsZombies.exe")
            before=hashlib.sha256(exe.read_bytes()).hexdigest()
            target=root/"local"/"save"
            self.assertEqual(prepare_training_save(exe,target),target.resolve())
            self.assertEqual([(p.profile_id,p.name) for p in read_save_profiles(target)],[(1,"PvZRL")])
            snapshot={p.name:p.read_bytes() for p in target.iterdir()}
            prepare_training_save(exe,target)
            self.assertEqual(snapshot,{p.name:p.read_bytes() for p in target.iterdir()})
            self.assertEqual(before,hashlib.sha256(exe.read_bytes()).hexdigest())
            values=struct.unpack("<205I",(target/"user1.dat").read_bytes())
            self.assertEqual(values[104:113],(1,)*9)
            self.assertEqual(values[125],4)
            self.assertEqual(len(training_profile_bytes()[1]),820)

    def test_refuses_personal_original_or_corrupt_save_without_overwriting(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {"ProgramData":tmp}, clear=True):
            root=Path(tmp); exe=fake_game(root/"game"/"PlantsVsZombies.exe")
            for dest in (exe.parent,root/"PopCap Games"/"PlantsVsZombies"/"userdata",root/"existing"):
                dest.mkdir(parents=True,exist_ok=True)
                (dest/"personal.txt").write_text("retain")
                with self.assertRaises(SaveScopeError):
                    prepare_training_save(exe,dest)
                self.assertEqual((dest/"personal.txt").read_text(),"retain")
            target=root/"template"; prepare_training_save(exe,target)
            (target/"user1.dat").write_bytes(b"corrupt")
            with self.assertRaises(SaveScopeError):
                prepare_training_save(exe,target)
            self.assertEqual((target/"user1.dat").read_bytes(),b"corrupt")

    def test_default_configuration_independent_of_launch_directory(self):
        before=Path.cwd()
        with tempfile.TemporaryDirectory() as tmp:
            try:
                os.chdir(tmp)
                self.assertEqual(default_env_path(),repository_root()/".env")
                self.assertEqual(resolve_game_paths().repository_root,repository_root())
            finally:
                os.chdir(before)
        with tempfile.TemporaryDirectory() as tmp:
            parent = Path(tmp)
            (parent / ".env").write_text("PVZ_GAME_EXECUTABLE=another-project.exe")
            checkout = parent / "source archive"
            checkout.mkdir()
            (checkout / "pyproject.toml").write_text("[project]\nname='pvz-rl'\n")
            nested = checkout / "python/pvz_gym"
            nested.mkdir(parents=True)
            self.assertIsNone(find_env_file(nested))
            (checkout / ".env").write_text("PVZ_GAME_EXECUTABLE=own-game.exe")
            self.assertEqual(find_env_file(nested), checkout / ".env")

if __name__ == "__main__":
    unittest.main()
