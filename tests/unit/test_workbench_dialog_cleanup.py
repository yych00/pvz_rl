import unittest
from unittest.mock import patch

from workbench.native.c_interface import PvzGymNativeDll


class WorkbenchDialogCleanupTest(unittest.TestCase):
    def make_client(self, open_dialogs):
        client = object.__new__(PvzGymNativeDll)
        dialogs = set(open_dialogs)
        killed = []

        class Dll:
            def pvz_gym_get_dialog(self, dialog_id):
                return 0x1234 if dialog_id in dialogs else 0

            def pvz_gym_kill_dialog(self, dialog_id):
                dialogs.remove(dialog_id)
                killed.append(dialog_id)
                return True

            def pvz_gym_select_adventure_level_id(self, level_id):
                self.assert_closed()
                return 0

            def pvz_gym_select_adventure_level(self, area, sub_level):
                self.assert_closed()
                return 0

            def assert_closed(self):
                if 3 in dialogs:
                    raise AssertionError("Almanac still open at level transition")

        client.dll = Dll()
        client.is_paused = lambda: False
        client.error_message = lambda code: str(code)
        return client, dialogs, killed

    def test_selection_closes_almanac_before_both_level_entry_points(self):
        for method, args in (("select_level_id", (11,)), ("select_level", (2, 1))):
            with self.subTest(method=method):
                client, dialogs, killed = self.make_client({2, 3})
                with patch("workbench.native.c_interface.clean_adventure_save"):
                    code, _ = getattr(client, method)(*args)
                self.assertEqual(code, 0)
                self.assertEqual(killed, [2, 3])
                self.assertFalse(dialogs)

    def test_scene_change_closes_almanac_and_preserves_unrelated_dialogs(self):
        client, dialogs, killed = self.make_client({3, 29})
        self.assertEqual(client.cleanup_scene_dialogs(2, 3), [3])
        self.assertEqual(killed, [3])
        self.assertEqual(dialogs, {29})

    def test_unchanged_scene_keeps_user_opened_almanac(self):
        client, dialogs, killed = self.make_client({3})
        self.assertEqual(client.cleanup_scene_dialogs(3, 3), [])
        self.assertEqual(dialogs, {3})
        self.assertEqual(killed, [])


if __name__ == "__main__":
    unittest.main()
