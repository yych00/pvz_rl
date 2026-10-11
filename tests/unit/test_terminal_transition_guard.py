import unittest

from workbench.live.terminal_transition_guard import classify_terminal_transition


class TerminalTransitionGuardTest(unittest.TestCase):
    def test_explicit_terminal_status_wins(self):
        state = {"phase_name": "LEVEL_COMPLETE", "game_status": 4, "level_id": 4}
        self.assertEqual(classify_terminal_transition(state, 4), "WON")

    def test_next_adventure_level_after_missing_observation_is_win(self):
        state = {"phase_name": "LEVEL_INTRO", "game_status": 0, "level_id": 5}
        self.assertEqual(classify_terminal_transition(state, 4), "WON")

    def test_same_battle_is_not_reclassified(self):
        state = {"phase_name": "BATTLE", "game_status": 0, "level_id": 4}
        self.assertEqual(classify_terminal_transition(state, 4), "")

    def test_unknown_menu_without_level_change_is_not_guessed(self):
        state = {"phase_name": "MENU", "game_status": 0, "level_id": 0}
        self.assertEqual(classify_terminal_transition(state, 4), "")


if __name__ == "__main__":
    unittest.main()
