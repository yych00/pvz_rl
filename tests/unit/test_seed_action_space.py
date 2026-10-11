import unittest

from pvz_gym.envs.seed_selection import SeedSelectionEnv
from pvz_gym.seed_types import (
    CONFIRM_ID,
    DESELECT_BASE,
    IMITATER_BASE,
    SEED_ACTION_COUNT,
    SEED_TYPE_CAPACITY,
    SeedOptionState,
    SeedSelectionActionType,
    SeedSelectionResult,
    SeedSelectionState,
    decode_seed_action,
    encode_seed_action,
    seed_action_count,
)


class ImitaterBackend:
    def __init__(self):
        self.target = -1
        self.confirmed = False

    def get_seed_type_capacity(self):
        return SEED_TYPE_CAPACITY

    def get_seed_selection(self):
        return SeedSelectionState(
            present=not self.confirmed,
            required_count=1,
            selected_count=int(self.target >= 0),
            can_confirm=self.target >= 0,
            imitater_target=self.target,
            selected_seeds=(48,) if self.target >= 0 else (),
            options=(SeedOptionState(plant_type=48, available=self.target < 0,
                                     unlocked=True, selected=self.target >= 0),),
        )

    def get_seed_selection_mask(self):
        mask = [False] * SEED_ACTION_COUNT
        if self.target < 0:
            mask[IMITATER_BASE:IMITATER_BASE + 40] = [True] * 40
        else:
            mask[DESELECT_BASE + 48] = True
            mask[CONFIRM_ID] = True
        return mask

    def select_imitater(self, target_type):
        self.target = target_type
        return SeedSelectionResult(True)

    def confirm_seeds(self):
        self.confirmed = True
        return SeedSelectionResult(True)


class SeedActionSpaceTest(unittest.TestCase):
    def test_fixed_blocks_and_boundaries(self):
        self.assertEqual(seed_action_count(SEED_TYPE_CAPACITY), 193)
        cases = {
            0: (SeedSelectionActionType.SELECT, 0),
            63: (SeedSelectionActionType.SELECT, 63),
            64: (SeedSelectionActionType.SELECT_IMITATER, 0),
            103: (SeedSelectionActionType.SELECT_IMITATER, 39),
            128: (SeedSelectionActionType.DESELECT, 0),
            176: (SeedSelectionActionType.DESELECT, 48),
            191: (SeedSelectionActionType.DESELECT, 63),
            192: (SeedSelectionActionType.CONFIRM, None),
        }
        for action_id, expected in cases.items():
            self.assertEqual(decode_seed_action(action_id, SEED_TYPE_CAPACITY), expected)
        for action_id in range(SEED_ACTION_COUNT):
            self.assertEqual(encode_seed_action(*decode_seed_action(action_id, SEED_TYPE_CAPACITY)),
                             action_id)
        with self.assertRaises(ValueError):
            decode_seed_action(193, SEED_TYPE_CAPACITY)
        with self.assertRaises(ValueError):
            seed_action_count(65)

    def test_imitater_target_is_observable_and_confirm_is_192(self):
        env = SeedSelectionEnv(backend=ImitaterBackend())
        obs, info = env.reset()
        self.assertEqual(env.action_space.n, 193)
        self.assertEqual(obs["imitater_target"], 0)
        self.assertTrue(info["action_mask"][IMITATER_BASE + 7])
        obs, _, done, _, info = env.step(IMITATER_BASE + 7)
        self.assertFalse(done)
        self.assertEqual(obs["imitater_target"], 8)
        self.assertEqual(info["imitater_target"], 7)
        self.assertTrue(info["action_mask"][CONFIRM_ID])
        _, _, done, _, _ = env.step(CONFIRM_ID)
        self.assertTrue(done)


if __name__ == "__main__":
    unittest.main()
