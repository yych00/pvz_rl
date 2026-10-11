import unittest
from dataclasses import replace

from pvz_gym.adventure_curriculum import deck_profile, unlocked_plants
from pvz_gym.default_decks import plan_default_deck, select_default_deck
from tests.unit.test_default_decks import Chooser


class AdventureCurriculumTests(unittest.TestCase):
    def test_every_level_has_legal_fixed_deck_and_ordered_candidates(self):
        for playthrough in (1, 2):
            for level in range(1, 51):
                with self.subTest(playthrough=playthrough, level=level):
                    profile = deck_profile(level, playthrough)
                    allowed = set(unlocked_plants(level) if playthrough == 1 else range(40))
                    self.assertTrue(set(profile.primary + profile.candidates).issubset(allowed))
                    self.assertEqual(len(profile.primary + profile.candidates),
                                     len(set(profile.primary + profile.candidates)))
                    self.assertTrue(profile.lesson)
                    for slots in (6, 8, 10):
                        if len(allowed) < slots:
                            continue
                        state = Chooser(unlocked=allowed, slots=slots).get_seed_selection()
                        deck = plan_default_deck(state, level, playthrough, selection_mode="fixed")
                        self.assertEqual(len(deck), slots)
                        self.assertTrue(set(deck).issubset(allowed))

    def test_new_plant_is_available_next_level_not_its_reward_level(self):
        for reward_level, plant in ((21, 17), (23, 19), (26, 21), (33, 27),
                                    (35, 28), (38, 31), (41, 33), (43, 35), (46, 37)):
            self.assertNotIn(plant, unlocked_plants(reward_level))
            self.assertIn(plant, deck_profile(reward_level + 1, 1).new_plants)
            self.assertIn(plant, unlocked_plants(reward_level + 1))
        self.assertIn(26, deck_profile(33, 1).primary)
        self.assertNotIn(27, deck_profile(33, 1).primary)
        self.assertIn(28, deck_profile(36, 1).primary)
        self.assertNotIn(31, deck_profile(36, 1).primary)

    def test_fixed_deck_does_not_change_with_preview_roster(self):
        for playthrough in (1, 2):
            allowed = unlocked_plants(26) if playthrough == 1 else range(40)
            state = Chooser(unlocked=allowed, slots=10).get_seed_selection()
            decks = [plan_default_deck(replace(state, zombie_types_in_level=roster),
                                      26, playthrough, selection_mode="fixed")
                     for roster in (None, (), (0, 2), (12, 14), (16, 17, 20))]
            self.assertTrue(all(deck == decks[0] for deck in decks))

    def test_first_run_pool_fills_ten_cards_without_later_unlocks(self):
        chooser = Chooser(unlocked=range(40), slots=10)  # revisited / modified profile
        logs = []
        ready = select_default_deck(chooser, 21, 1, selection_mode="fixed", report=logs.append)
        self.assertEqual(ready.selected_seeds, deck_profile(21, 1).primary)
        self.assertEqual(ready.selected_count, 10)
        self.assertTrue(any("白天会睡觉" in line for line in logs))
        self.assertNotIn(17, ready.selected_seeds)
        self.assertNotIn(19, ready.selected_seeds)

    def test_second_run_keeps_dave_and_uses_candidates_when_primary_unavailable(self):
        profile = deck_profile(26, 2)
        chooser = Chooser(unlocked=set(range(40)) - {7, 17},
                          selected=(9, 8, 10), locked=(9, 8, 10), slots=10)
        ready = select_default_deck(chooser, 26, 2, selection_mode="fixed")
        self.assertTrue({9, 8, 10, 16, 1, 20, 21, 19}.issubset(ready.selected_seeds))
        self.assertTrue(ready.can_confirm)
        self.assertFalse(any(action == ("deselect", p) for action in chooser.actions for p in chooser.locked))
        # Without Dave there are room for actual replacements from the candidate pool.
        state = Chooser(unlocked=set(range(40)) - {7, 17}, slots=10).get_seed_selection()
        deck = plan_default_deck(state, 26, 2, selection_mode="fixed")
        self.assertTrue(set(deck).intersection(profile.candidates))

    def test_special_levels_and_second_run_yeti_are_explicit(self):
        self.assertEqual(deck_profile(15, 1).special, "打僵尸")
        self.assertEqual(deck_profile(35, 2).special, "砸罐子")
        self.assertEqual(deck_profile(50, 2).special, "僵王传送带")
        self.assertNotIn(19, deck_profile(40, 1).new_zombies)
        self.assertIn(19, deck_profile(40, 2).new_zombies)


if __name__ == "__main__":
    unittest.main()
