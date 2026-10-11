import unittest

from pvz_gym.action_space import (
    ACTION_SPACE_SIZE,
    COB_BASE,
    INTERACT_BASE,
    SHOVEL_NORMAL_BASE,
    SHOVEL_PUMPKIN_BASE,
    decode_action,
    encode_action,
)
from pvz_gym.types import Action


class ActionSpaceContractTest(unittest.TestCase):
    def test_fixed_cross_level_size(self):
        self.assertEqual(ACTION_SPACE_SIZE, 757)

    def test_boundaries_and_round_trip(self):
        self.assertEqual(decode_action(0), Action())
        self.assertEqual(decode_action(1), Action("plant", 0, 0, 0))
        self.assertEqual(decode_action(540), Action("plant", 9, 5, 8))
        self.assertEqual(
            decode_action(SHOVEL_NORMAL_BASE),
            Action("shovel", row=0, col=0, shovel_target="normal"),
        )
        self.assertEqual(
            decode_action(SHOVEL_PUMPKIN_BASE + 53),
            Action("shovel", row=5, col=8, shovel_target="pumpkin"),
        )
        self.assertEqual(
            decode_action(COB_BASE),
            Action("cob_fire", target_row=0, target_col=0),
        )
        self.assertEqual(
            decode_action(702),
            Action("cob_fire", target_row=5, target_col=8),
        )
        self.assertEqual(decode_action(INTERACT_BASE), Action("interact_cell", row=0, col=0))
        self.assertEqual(decode_action(756), Action("interact_cell", row=5, col=8))
        for value in (0, 1, 54, 55, 540, 541, 594, 595, 648, 649, 702, 703, 756):
            self.assertEqual(encode_action(decode_action(value)), value)

    def test_every_action_round_trips_without_collisions(self):
        encoded = {encode_action(decode_action(value)) for value in range(ACTION_SPACE_SIZE)}
        self.assertEqual(encoded, set(range(ACTION_SPACE_SIZE)))

    def test_invalid_actions_fail(self):
        with self.assertRaises(ValueError):
            decode_action(ACTION_SPACE_SIZE)
        with self.assertRaises(ValueError):
            encode_action(Action("cob_fire", target_row=6, target_col=0))
        with self.assertRaises(ValueError):
            encode_action(Action("cob_fire", row=0, col=0, target_row=0, target_col=0))
        with self.assertRaises(ValueError):
            encode_action(Action("interact_cell", row=6, col=0))


if __name__ == "__main__":
    unittest.main()
