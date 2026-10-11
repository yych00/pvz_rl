import unittest
from unittest.mock import Mock

from pvz_gym.bridge.native import NativeCapabilityError
from pvz_gym.client import PvZClient
from pvz_gym.types import GameStatus, Observation, ResetFacts


class ClientLifecycleBoundaryTest(unittest.TestCase):
    def test_evaluation_closes_only_battle_menus_and_restores_clock_mode(self):
        for keep_paused in (False, True):
            with self.subTest(keep_paused=keep_paused):
                dialogs = {2, 19, 17, 3}
                client = object.__new__(PvZClient)
                client._dll = Mock()
                client._dll.pvz_gym_get_dialog.side_effect = lambda dialog_id: dialog_id in dialogs
                def kill(dialog_id):
                    dialogs.remove(dialog_id)
                    return True
                client._dll.pvz_gym_kill_dialog.side_effect = kill
                client.dismiss_battle_dialogs(keep_paused=keep_paused)
                self.assertEqual(dialogs, {17, 3})
                self.assertEqual(client._dll.pvz_gym_pause.call_count, int(keep_paused))
                self.assertEqual(client._dll.pvz_gym_resume.call_count, int(not keep_paused))

    def test_no_menu_keeps_existing_decision_pause(self):
        client = object.__new__(PvZClient)
        client._dll = Mock()
        client._dll.pvz_gym_get_dialog.return_value = 0
        client.dismiss_battle_dialogs()
        client._dll.pvz_gym_kill_dialog.assert_not_called()
        client._dll.pvz_gym_pause.assert_not_called()
        client._dll.pvz_gym_resume.assert_not_called()

    def test_failed_menu_close_does_not_resume_behind_modal(self):
        client = object.__new__(PvZClient)
        client._dll = Mock()
        client._dll.pvz_gym_get_dialog.return_value = 1
        client._dll.pvz_gym_kill_dialog.return_value = False
        with self.assertRaisesRegex(NativeCapabilityError, "could not close battle dialog"):
            client.dismiss_battle_dialogs()
        client._dll.pvz_gym_resume.assert_not_called()

    def test_close_cleanup_releases_any_paused_board(self):
        class Dll:
            resumed = 0

            @staticmethod
            def pvz_gym_is_paused():
                return True

            def pvz_gym_resume(self):
                self.resumed += 1
                return True

        client = object.__new__(PvZClient)
        client._dll = Dll()
        client._release_pause()
        self.assertEqual(client._dll.resumed, 1)

    def test_reset_game_is_only_a_compatibility_view_over_reset_facts(self):
        observation = Observation(global_state={"game_clock": 0})
        client = object.__new__(PvZClient)
        calls = []

        def reset(config):
            calls.append(dict(config))
            return ResetFacts(observation, 0, GameStatus.PLAYING)

        client.reset = reset
        result = client.reset_game({"level_id": 4})
        self.assertIs(result, observation)
        self.assertEqual(calls, [{"level_id": 4}])

    def test_reset_game_refuses_seed_phase_instead_of_starting_second_reset(self):
        client = object.__new__(PvZClient)
        client.reset = lambda _config: ResetFacts(None, 0, GameStatus.CHOOSING_SEEDS)
        with self.assertRaisesRegex(NativeCapabilityError, "stopped before battle"):
            client.reset_game({"level_id": 4, "stop_at_seed_selection": True})


if __name__ == "__main__":
    unittest.main()
