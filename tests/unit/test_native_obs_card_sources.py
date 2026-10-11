"""Card observation checks respect source-specific availability rules."""

from __future__ import annotations

import unittest

from workbench.native_tests.cases import NativeUnitTestSuite


class _Client:
    is_loaded = True

    def __init__(self, card_system: str, cards: list[dict]):
        self.data = {
            "global": {
                "card_system": card_system,
                "is_conveyor": card_system == "conveyor",
                "sun": 0,
            },
            "cards": cards,
        }

    def get_observation_json(self):
        return 1, "", self.data


class NativeObsCardSourcesTest(unittest.TestCase):
    def test_empty_conveyor_and_dropped_sources_are_valid(self):
        for source in ("conveyor", "dropped"):
            with self.subTest(source=source):
                result = NativeUnitTestSuite(_Client(source, [])).test_obs_cards()
                self.assertEqual(result.status, "PASS")

    def test_empty_seed_bank_still_fails(self):
        result = NativeUnitTestSuite(_Client("seed_bank", [])).test_obs_cards()
        self.assertEqual(result.status, "FAIL")

    def test_uncollectable_dropped_card_is_not_a_cooldown_failure(self):
        card = {
            "slot": 0,
            "plant_type": 3,
            "cost": 0,
            "cooldown_duration": 0,
            "cooldown_remaining": 0,
            "ready": False,
        }
        result = NativeUnitTestSuite(_Client("dropped", [card])).test_obs_cards()
        self.assertEqual(result.status, "PASS")


if __name__ == "__main__":
    unittest.main()
