import os
import unittest
import winreg
import tempfile
from pathlib import Path
from pvz_gym.runtime.instances import GameInstance
from pvz_gym.bridge.native import set_process_audio_muted


class AudioMuteIsolationTest(unittest.TestCase):
    def test_registry_zero_volume_lifecycle(self):
        # Only lease a temporary disk slot; cmd.exe and PvZ are never started.
        with tempfile.TemporaryDirectory() as tmp:
            dummy_exe = Path(os.environ.get("COMSPEC", "cmd.exe"))
            instance = GameInstance(dummy_exe, Path(tmp) / "instances", 0)
            instance._acquire()
            sub_key = rf"Software\PopCap\{instance.product}"
            instance._apply_zero_volume_registry()
            try:
                with winreg.OpenKey(winreg.HKEY_CURRENT_USER, sub_key, 0, winreg.KEY_READ) as key:
                    music, _ = winreg.QueryValueEx(key, "MusicVolume")
                    sfx, _ = winreg.QueryValueEx(key, "SfxVolume")
                    muted, _ = winreg.QueryValueEx(key, "Muted")
                    self.assertEqual(music, 0)
                    self.assertEqual(sfx, 0)
                    self.assertEqual(muted, 1)
            finally:
                instance.close()
            with self.assertRaises(OSError):
                winreg.OpenKey(winreg.HKEY_CURRENT_USER, sub_key, 0, winreg.KEY_READ)

    def test_registry_configuration_rejects_an_unleased_instance(self):
        with tempfile.TemporaryDirectory() as tmp:
            instance = GameInstance(Path(os.environ.get("COMSPEC", "cmd.exe")), Path(tmp) / "instances", 0)
            with self.assertRaisesRegex(RuntimeError, "owned instance lease"):
                instance._apply_zero_volume_registry()

    def test_wasapi_mute_call(self):
        # 对当前进程调用静音与解除静音，验证 WASAPI native 接口不崩且安全返回
        pid = os.getpid()
        ok_mute = set_process_audio_muted(pid, True)
        self.assertTrue(ok_mute)
        ok_unmute = set_process_audio_muted(pid, False)
        self.assertTrue(ok_unmute)


if __name__ == "__main__":
    unittest.main()
