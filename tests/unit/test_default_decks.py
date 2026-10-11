import unittest
import random
import json
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch

from pvz_gym.default_decks import (
    DEFAULT_DECKS, deck_counter_gaps, default_deck_priority, plan_default_deck, select_default_deck,
)
from pvz_gym.seed_types import (
    CONFIRM_ID, DESELECT_BASE, SeedOptionState, SeedSelectionResult, SeedSelectionState,
)


class Chooser:
    def __init__(self, unlocked=range(40), selected=(), locked=(), slots=6):
        self.unlocked = set(unlocked)
        self.selected = list(selected)
        self.locked = set(locked)
        self.slots = slots
        self.confirmed = False
        self.actions = []

    def get_seed_selection(self):
        return SeedSelectionState(
            present=True, required_count=self.slots, selected_count=len(self.selected),
            can_confirm=len(self.selected) == self.slots,
            selected_seeds=tuple(self.selected),
            options=tuple(SeedOptionState(
                plant_type=i, unlocked=i in self.unlocked,
                available=i in self.unlocked and i not in self.selected,
                selected=i in self.selected, disabled=i in self.locked,
            ) for i in range(49)),
        )

    def get_seed_selection_mask(self):
        mask = [False] * 193
        for i in self.unlocked:
            mask[i] = i not in self.selected and len(self.selected) < self.slots
        for i in self.selected:
            mask[DESELECT_BASE + i] = i not in self.locked
        mask[CONFIRM_ID] = len(self.selected) == self.slots
        return mask

    def select_seed(self, plant):
        assert self.get_seed_selection_mask()[plant]
        self.selected.append(plant)
        self.actions.append(("select", plant))
        return SeedSelectionResult(True)

    def deselect_seed(self, plant):
        assert self.get_seed_selection_mask()[DESELECT_BASE + plant]
        self.selected.remove(plant)
        self.actions.append(("deselect", plant))
        return SeedSelectionResult(True)

    def confirm_seeds(self):
        assert self.get_seed_selection_mask()[CONFIRM_ID]
        self.confirmed = True
        return SeedSelectionResult(True)


class DefaultDeckTests(unittest.TestCase):
    def test_shuffled_adaptive_preserves_counters_and_locked_slots(self):
        chooser = Chooser(selected=(4,), locked=(4,), slots=8)
        state = replace(chooser.get_seed_selection(), zombie_types_in_level=(2, 4, 6))
        adaptive = plan_default_deck(state, 14, 2, selection_mode="adaptive")
        orders = set()
        for seed in range(12):
            shuffled = plan_default_deck(state, 14, 2, selection_mode="adaptive_shuffled",
                                         rng=random.Random(seed))
            self.assertEqual(set(shuffled), set(adaptive))
            self.assertEqual(shuffled[0], 4)
            self.assertEqual(deck_counter_gaps(state, 14, shuffled), deck_counter_gaps(state, 14, adaptive))
            self.assertEqual(shuffled, plan_default_deck(
                state, 14, 2, selection_mode="adaptive_shuffled", rng=random.Random(seed)))
            orders.add(shuffled)
        self.assertGreater(len(orders), 1)

    def test_shuffled_adaptive_reorders_existing_unlocked_picks(self):
        chooser = Chooser(selected=(4,), locked=(4,), slots=6)
        target = plan_default_deck(chooser.get_seed_selection(), 14, 2,
                                   selection_mode="adaptive_shuffled", rng=random.Random(42))
        chooser.selected = [target[0], *reversed(target[1:])]
        ready = select_default_deck(chooser, 14, 2, confirm=True,
                                    selection_mode="adaptive_shuffled", rng=random.Random(42))
        self.assertEqual(ready.selected_seeds, target)
        self.assertTrue(chooser.confirmed)
        self.assertNotIn(("deselect", 4), chooser.actions)

    def test_shuffled_preset_preserves_deck_and_locked_slots(self):
        chooser = Chooser(selected=(4,), locked=(4,), slots=6)
        state = chooser.get_seed_selection()
        fixed = plan_default_deck(state, 14, 2, selection_mode="fixed")
        orders = set()
        for seed in range(12):
            shuffled = plan_default_deck(state, 14, 2, selection_mode="fixed_shuffled",
                                         rng=random.Random(seed))
            self.assertEqual(set(shuffled), set(fixed))
            self.assertEqual(shuffled[0], 4)
            self.assertEqual(shuffled, plan_default_deck(
                state, 14, 2, selection_mode="fixed_shuffled", rng=random.Random(seed)))
            orders.add(shuffled)
        self.assertGreater(len(orders), 1)

    def test_preset_executor_reorders_existing_unlocked_cards(self):
        for mode in ("fixed", "fixed_shuffled"):
            chooser = Chooser(selected=(4, 0, 1), locked=(4,), slots=6)
            target = plan_default_deck(chooser.get_seed_selection(), 14, 2,
                                       selection_mode=mode, rng=random.Random(5))
            ready = select_default_deck(chooser, 14, 2, selection_mode=mode,
                                        rng=random.Random(5))
            self.assertEqual(tuple(ready.selected_seeds), target)
            self.assertNotIn(("deselect", 4), chooser.actions)
            self.assertIn(("deselect", 0), chooser.actions)
            self.assertFalse(chooser.confirmed)

    def test_native_preview_transport_and_legacy_unknown(self):
        from pvz_gym.bridge.seed_selection import SeedSelectionNativeBackend
        backend = object.__new__(SeedSelectionNativeBackend)
        backend._dll = SimpleNamespace(pvz_gym_attach=lambda: True)
        for payload, expected in (({}, None),
                                  ({"zombie_roster_available": True,
                                    "zombie_types_in_level": []}, ()),
                                  ({"zombie_roster_available": True,
                                    "zombie_types_in_level": [16, 17, 20]}, (16, 17, 20))):
            data = json.dumps({"present": True, **payload}).encode()
            with patch("pvz_gym.bridge.seed_selection._call_with_resizable_buffer",
                       return_value=(len(data), data)):
                self.assertEqual(backend.get_seed_selection().zombie_types_in_level, expected)

    def test_all_100_templates_and_slot_counts(self):
        for playthrough, levels in DEFAULT_DECKS.items():
            self.assertEqual(set(levels), set(range(1, 51)))
            for level, priority in levels.items():
                self.assertEqual(len(priority), len(set(priority)))
                for slots in (6, 8, 10):
                    with self.subTest(playthrough=playthrough, level=level, slots=slots):
                        deck = plan_default_deck(Chooser(slots=slots).get_seed_selection(),
                                                 level, playthrough)
                        self.assertEqual(len(deck), slots)
                        if level > 4 or playthrough == 2:
                            scene = (level - 1) // 10
                            for support in ({2: (16,), 3: (16,), 4: (33,)}.get(scene, ())):
                                self.assertIn(support, deck)

    def test_unlocked_first_run_only(self):
        chooser = Chooser(unlocked=range(6))
        self.assertEqual(plan_default_deck(chooser.get_seed_selection(), 6, 1),
                         (1, 0, 4, 3, 5, 2))
        self.assertEqual(default_deck_priority(1, 1), (0,))

    def test_first_run_pool_ten_slots_before_coffee_bean_unlock(self):
        chooser = Chooser(unlocked=range(17), slots=10)
        state = replace(chooser.get_seed_selection(), zombie_types_in_level=(0, 2))
        chooser.get_seed_selection = lambda: replace(
            Chooser.get_seed_selection(chooser), zombie_types_in_level=state.zombie_types_in_level)
        logs = []
        ready = select_default_deck(chooser, 21, 1, report=logs.append)
        self.assertEqual(ready.selected_count, 10)
        self.assertTrue(ready.can_confirm)
        self.assertEqual(set(ready.selected_seeds), {*range(8), 16, 8})
        self.assertTrue(any("白天会睡觉" in line for line in logs))
        self.assertFalse(chooser.confirmed)

    def test_locked_dave_cards_and_replace_full_existing_deck(self):
        chooser = Chooser(selected=(6, 12, 38, 0, 2, 5), locked=(6, 12, 38))
        ready = select_default_deck(chooser, 21, 2)
        self.assertEqual(set(ready.selected_seeds), {6, 12, 38, 16, 1, 7})
        self.assertFalse(chooser.confirmed)
        self.assertFalse(any(action == ("deselect", i)
                             for i in chooser.locked for action in chooser.actions))
        old_actions = list(chooser.actions)
        select_default_deck(chooser, 21, 2, confirm=True)
        self.assertEqual(old_actions, chooser.actions)
        self.assertTrue(chooser.confirmed)

    def test_insufficient_cards_fail_before_mutation(self):
        chooser = Chooser(unlocked=(0, 1), selected=(0,))
        with self.assertRaisesRegex(RuntimeError, "cannot fill"):
            select_default_deck(chooser, 6, 1)
        self.assertEqual(chooser.actions, [])

    def test_no_chooser_and_flight_timeout(self):
        chooser = Chooser()
        state = chooser.get_seed_selection()
        with self.assertRaisesRegex(RuntimeError, "no interactive"):
            plan_default_deck(replace(state, present=False), 21, 2)
        chooser.get_seed_selection = lambda: replace(state, seeds_in_flight=True)
        with self.assertRaises(TimeoutError):
            select_default_deck(chooser, 21, 2, timeout_seconds=0.001)
        self.assertEqual(chooser.actions, [])

    def test_illegal_mask_and_native_failure(self):
        chooser = Chooser()
        chooser.get_seed_selection_mask = lambda: [False] * 193
        with self.assertRaisesRegex(RuntimeError, "no longer legal"):
            select_default_deck(chooser, 21, 2)
        chooser = Chooser()
        chooser.select_seed = lambda plant: SeedSelectionResult(False, 1, "rejected")
        with self.assertRaisesRegex(RuntimeError, "rejected"):
            select_default_deck(chooser, 21, 2)

    def test_special_zombies_reserve_counters_before_generic_cards(self):
        cases = ((16, 31, {27}), (17, 31, {31}), (18, 31, {23}),
                 (20, 41, {37}), (22, 41, {37}), (12, 21, {21}),
                 (14, 21, {23}), (3, 21, {23}), (6, 11, {10}),
                 (21, 31, {31}), (23, 21, {17, 20}), (4, 11, {31}),
                 (7, 11, {31}), (15, 11, {31}), (11, 21, {19}))
        for zombie, level, expected in cases:
            with self.subTest(zombie=zombie, level=level):
                state = replace(Chooser().get_seed_selection(), zombie_types_in_level=(zombie,))
                deck = plan_default_deck(state, level, 2)
                self.assertTrue(expected.issubset(deck), deck)
                self.assertEqual(deck_counter_gaps(state, level, deck), ())

    def test_day_mushroom_dependencies_and_unavailable_counter(self):
        state = replace(Chooser().get_seed_selection(), zombie_types_in_level=(6,))
        deck = plan_default_deck(state, 21, 2)
        self.assertTrue({10, 35}.issubset(deck))
        state = replace(Chooser(unlocked=set(range(40)) - {35}).get_seed_selection(),
                        zombie_types_in_level=(6,))
        deck = plan_default_deck(state, 21, 2)
        self.assertNotIn(10, deck)
        self.assertIn(17, deck)
        state = replace(Chooser(unlocked=set(range(40)) - {27, 26}).get_seed_selection(),
                        zombie_types_in_level=(16,))
        deck = plan_default_deck(state, 31, 1)
        self.assertEqual(deck_counter_gaps(state, 31, deck), ("气球",))

    def test_multiple_threats_and_locked_slot_shortage(self):
        state = replace(Chooser(slots=10).get_seed_selection(),
                        zombie_types_in_level=(16, 17, 18, 20))
        deck = plan_default_deck(state, 31, 2)
        self.assertTrue({16, 9, 8, 27, 31, 23, 37}.issubset(deck))
        self.assertNotIn(26, deck)  # Blover already covers balloons.
        self.assertEqual(deck_counter_gaps(state, 31, deck), ())
        state = replace(Chooser(selected=(6, 12, 38), locked=(6, 12, 38)).get_seed_selection(),
                        zombie_types_in_level=(16, 17))
        deck = plan_default_deck(state, 31, 2)
        self.assertTrue({6, 12, 38, 16, 9, 8}.issubset(deck))
        self.assertEqual(deck_counter_gaps(state, 31, deck), ("气球", "矿工"))
        state = replace(Chooser(selected=(16, 9, 8), locked=(16, 9, 8)).get_seed_selection(),
                        zombie_types_in_level=(16, 17, 20))
        deck = plan_default_deck(state, 31, 2)
        self.assertEqual(set(deck), {16, 9, 8, 27, 31, 37})
        self.assertEqual(deck_counter_gaps(state, 31, deck), ())

    def test_roof_vehicles_and_unknown_preview(self):
        state = replace(Chooser().get_seed_selection(), zombie_types_in_level=(12,))
        deck = plan_default_deck(state, 41, 2)
        self.assertIn(20, deck)
        self.assertNotIn(21, deck)
        old = Chooser().get_seed_selection()
        self.assertIn("预览不可用", deck_counter_gaps(old, 21, ())[0])


if __name__ == "__main__":
    unittest.main()
