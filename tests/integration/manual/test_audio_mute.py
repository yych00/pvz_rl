"""Explicit native audio check requiring an installed DLL and Windows audio service."""
import os
import unittest

from pvz_gym.bridge.native import set_process_audio_muted


class NativeAudioMuteTest(unittest.TestCase):
    def test_wasapi_mute_call(self):
        # Only the current Python process is targeted; no game is launched.
        pid = os.getpid()
        try:
            self.assertTrue(set_process_audio_muted(pid, True))
        finally:
            self.assertTrue(set_process_audio_muted(pid, False))


if __name__ == '__main__':
    unittest.main()
