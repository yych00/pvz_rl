"""Repository-layout contract preventing Python responsibilities from drifting again."""

from __future__ import annotations

import unittest
from pathlib import Path

TOOLS_GYM = Path(__file__).resolve().parents[2]


class LayoutContractTest(unittest.TestCase):
    def test_required_layers_exist(self):
        required = [
            "native",
            "python/pvz_gym",
            "python/pvz_gym/bridge",
            "python/pvz_gym/envs",
            "python/pvz_gym/rewards",
            "python/pvz_gym/curriculum",
            "agents/scripted",
            "eval",
            "train/dqn",
            "tests/unit",
            "tests/integration/manual",
            "workbench/native",
            "workbench/gui",
            "workbench/live",
            "workbench/native_tests",
            "workbench/legacy",
            "scripts",
            "docs",
        ]
        for relative in required:
            with self.subTest(relative=relative):
                self.assertTrue((TOOLS_GYM / relative).is_dir(), relative)

    def test_legacy_apps_package_is_gone(self):
        self.assertFalse((TOOLS_GYM / "apps").exists())
        legacy_dirs = [p for p in TOOLS_GYM.rglob("pvz_gym_unit_test") if p.is_dir()]
        self.assertEqual(legacy_dirs, [])

    def test_scripts_only_contain_known_entrypoints(self):
        # Explicit CLI tools share the existing implementation layers; keep
        # unregistered application modules out of this launch directory.
        python_files = {p.relative_to(TOOLS_GYM / "scripts").as_posix()
                        for p in (TOOLS_GYM / "scripts").rglob("*.py")}
        self.assertEqual(python_files, {
            "__init__.py", "export_adventure_curriculum.py", "doctor.py",
            "prepare_training_save.py", "package_release.py", "install_native.py", "download_native.py",
        })
        self.assertTrue((TOOLS_GYM / "scripts/build_native.bat").is_file())
        self.assertTrue((TOOLS_GYM / "scripts/run_workbench.bat").is_file())

    def test_scripted_early_level_policies_are_all_present(self):
        scripted = TOOLS_GYM / "agents/scripted"
        expected = {
            "level_1_1.py",
            "level_1_1_playthrough.py",
            "level_1_2.py",
            "level_1_3.py",
            "level_1_4.py",
            "level_1_5.py",
            "level_1_10.py",
        }
        self.assertTrue(expected.issubset({p.name for p in scripted.glob("*.py")}))

    def test_formal_library_does_not_contain_workbench_or_training_dirs(self):
        root = TOOLS_GYM / "python/pvz_gym"
        forbidden = {"gui", "workbench", "agents", "train", "tests", "native_tests"}
        self.assertEqual(forbidden.intersection({p.name for p in root.iterdir() if p.is_dir()}), set())


if __name__ == "__main__":
    unittest.main()
