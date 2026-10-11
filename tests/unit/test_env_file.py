import os
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

from pvz_gym.env_file import (
    find_env_file,
    get_game_executable,
    load_dotenv,
    parse_env_file,
    set_game_executable,
    resolve_game_paths,
)


class EnvFileTest(unittest.TestCase):
    def setUp(self):
        self.environment = patch.dict(os.environ)
        self.environment.start()
        self.addCleanup(self.environment.stop)

    def test_parse_env_file_handles_comments_quotes_and_spaces(self):
        with tempfile.TemporaryDirectory() as tmp:
            env_path = Path(tmp) / ".env"
            env_path.write_text(
                "# Comment line\n"
                "\n"
                "PVZ_GAME_EXECUTABLE=\"C:\\Games\\PlantsVsZombies.exe\"\n"
                "OTHER_VAR = 'single_quoted'\n"
                "RAW_VAR=raw_value\n",
                encoding="utf-8",
            )
            parsed = parse_env_file(env_path)
            self.assertEqual(parsed["PVZ_GAME_EXECUTABLE"], "C:\\Games\\PlantsVsZombies.exe")
            self.assertEqual(parsed["OTHER_VAR"], "single_quoted")
            self.assertEqual(parsed["RAW_VAR"], "raw_value")

    def test_set_and_get_game_executable_round_trip(self):
        with tempfile.TemporaryDirectory() as tmp:
            env_path = Path(tmp) / ".env"
            set_game_executable("D:\\pvz\\test.exe", env_path=env_path)
            self.assertTrue(env_path.is_file())
            content = env_path.read_text(encoding="utf-8")
            self.assertIn('PVZ_GAME_EXECUTABLE="D:\\pvz\\test.exe"', content)

            # Update existing
            set_game_executable("E:\\pvz\\updated.exe", env_path=env_path)
            content_updated = env_path.read_text(encoding="utf-8")
            self.assertIn('PVZ_GAME_EXECUTABLE="E:\\pvz\\updated.exe"', content_updated)
            self.assertNotIn("test.exe", content_updated)

    def test_load_dotenv_injects_environment_variables(self):
        with tempfile.TemporaryDirectory() as tmp:
            env_path = Path(tmp) / ".env"
            env_path.write_text('TEST_DUMMY_VAR="hello_gym"\n', encoding="utf-8")
            try:
                os.environ.pop("TEST_DUMMY_VAR", None)
                loaded = load_dotenv(env_path)
                self.assertEqual(loaded, env_path)
                self.assertEqual(os.environ.get("TEST_DUMMY_VAR"), "hello_gym")
            finally:
                os.environ.pop("TEST_DUMMY_VAR", None)

    def test_default_game_paths_use_repository_root_independently_of_cwd(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {}, clear=True):
            base = Path(tmp).resolve()
            paths = resolve_game_paths(config_root=base)
            self.assertEqual(paths.executable, base / "game/pvz-1.2.0.1065/PlantsVsZombies.exe")
            self.assertEqual(paths.save_template, base / "local/save_template")
            self.assertEqual(paths.instance_root, base / "runtime/instances")
            self.assertFalse((base / ".env").exists())

    def test_relative_env_paths_and_explicit_overrides_share_a_config_root(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {}, clear=True):
            base = Path(tmp).resolve()
            env = base / ".env"
            env.write_text("PVZ_GAME_EXECUTABLE=games/PlantsVsZombies.exe\n"
                           "PVZ_TRAINING_SAVE_TEMPLATE=profiles/training\n"
                           "PVZ_INSTANCE_ROOT=runtime/instances\nPRIVATE_VALUE=unchanged\n", encoding="utf-8")
            before = env.read_bytes()
            paths = resolve_game_paths(config_root=base, env_path=env)
            self.assertEqual(paths.executable, base / "games/PlantsVsZombies.exe")
            self.assertEqual(paths.save_template, base / "profiles/training")
            self.assertEqual(paths.instance_root, base / "runtime/instances")
            explicit = resolve_game_paths("custom.exe", "new-profiles", "new-runtime", config_root=base, env_path=env)
            self.assertEqual(explicit.executable, base / "custom.exe")
            self.assertEqual(explicit.save_template, base / "new-profiles")
            self.assertEqual(explicit.instance_root, base / "new-runtime")
            self.assertEqual(env.read_bytes(), before)

    def test_process_environment_precedes_dotenv_without_rewriting_it(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {"PVZ_INSTANCE_ROOT": "process-pool"}, clear=True):
            base = Path(tmp).resolve()
            env = base / ".env"
            env.write_text("PVZ_INSTANCE_ROOT=file-pool\n", encoding="utf-8")
            paths = resolve_game_paths(config_root=base, env_path=env)
            self.assertEqual(paths.instance_root, base / "process-pool")
            self.assertEqual(env.read_text(encoding="utf-8"), "PVZ_INSTANCE_ROOT=file-pool\n")


if __name__ == "__main__":
    unittest.main()
