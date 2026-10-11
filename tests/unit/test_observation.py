import unittest

import numpy as np

from pvz_gym.observation import (
    MAX_ZOMBIES, OBSERVATION_SPEC, ZOMBIE_CONTINUOUS_FIELDS,
    encode_observation, make_observation_space, observation_is_valid,
    zombie_truncation_info,
)


class ObservationContractTest(unittest.TestCase):
    def test_imitater_card_preserves_target_but_distinguishes_identity(self):
        card = {'slot': 0, 'plant_type': 14, 'cost': 75, 'ready': True}
        regular = encode_observation({'cards': [card]})
        imitation = encode_observation({'cards': [{**card, 'is_imitater': True}]})
        np.testing.assert_array_equal(regular['card_type'], imitation['card_type'])
        np.testing.assert_array_equal(regular['cards'][:, :5], imitation['cards'][:, :5])
        self.assertEqual(regular['cards'].shape, (10, 6))
        self.assertFalse(regular['cards'][:, 5].any())
        self.assertEqual(imitation['cards'][0, 5], 1)
        self.assertFalse(imitation['cards'][1:].any())
        self.assertTrue(make_observation_space().contains(imitation))

    def feature(self, encoded, field, slot=0):
        return float(encoded['zombies'][slot, ZOMBIE_CONTINUOUS_FIELDS.index(field)])

    def test_structured_shape_dtype_and_non_finite_cleanup(self):
        encoded = encode_observation({
            'global': {'sun': float('nan'), 'game_clock': float('inf')},
            'zombies': [{'row': 0, 'type': 2, 'x': float('nan'), 'body_hp': 100}],
        })
        self.assertEqual(set(encoded), set(OBSERVATION_SPEC))
        self.assertTrue(observation_is_valid(encoded))
        self.assertTrue(make_observation_space().contains(encoded))
        self.assertEqual(encoded['zombies'].shape, (256, 15))
        for value in encoded.values():
            if np.issubdtype(value.dtype, np.floating):
                self.assertTrue(np.isfinite(value).all())

    def test_ids_do_not_become_model_features_but_hp_and_position_do(self):
        common = {'row': 2, 'type': 7, 'x': 450, 'y': 300, 'body_hp': 100, 'max_hp': 200}
        first = encode_observation({'zombies': [{**common, 'id': 1234}]})
        second = encode_observation({'zombies': [{**common, 'id': 9999}]})
        for name in OBSERVATION_SPEC:
            np.testing.assert_array_equal(first[name], second[name])
        self.assertAlmostEqual(self.feature(first, 'x_norm'), 0.5)
        self.assertAlmostEqual(self.feature(first, 'body_hp'), np.log1p(100) / np.log1p(20000))
        self.assertEqual(int(first['zombie_row'][0]), 3)

    def test_zombie_grid_uses_native_lawn_x_offset(self):
        encoded = encode_observation({
            'zombies': [
                {'row': 0, 'x': 119, 'body_hp': 100},
                {'row': 0, 'x': 120, 'body_hp': 100},
            ]
        })
        one_zombie = float(np.log1p(1.0) / np.log1p(4.0))
        self.assertAlmostEqual(float(encoded['zombie_grid'][0, 0, 0]), one_zombie)
        self.assertAlmostEqual(float(encoded['zombie_grid'][0, 1, 0]), one_zombie)

    def test_truncation_preserves_first_entities_and_has_no_summary_features(self):
        zombies = [{'row': 5, 'type': 32, 'x': 800 - index, 'body_hp': index + 1}
                   for index in range(MAX_ZOMBIES + 10)]
        crowded = encode_observation({'zombies': zombies})
        self.assertEqual(int(crowded['zombie_mask'].sum()), 256)
        np.testing.assert_allclose(crowded['zombies'][:, 0],
                                   np.arange(800, 544, -1) / 900.0)
        self.assertNotIn('zombie_overflow', crowded)
        self.assertEqual(zombie_truncation_info({'zombies': zombies}),
                         {'zombie_truncated': True, 'zombies_dropped': 10})
        self.assertEqual(zombie_truncation_info({'zombies': zombies[:256]}),
                         {'zombie_truncated': False, 'zombies_dropped': 0})
        # No grid feature changes when zombies change.
        empty = encode_observation({})
        for name in ('grid', 'grid_terrain', 'plants', 'plant_type', 'plant_state'):
            np.testing.assert_array_equal(crowded[name], empty[name])

    def test_each_control_timer_and_health_pool_remains_separate(self):
        zombie = {'body_hp': 100, 'max_hp': 270, 'helm_hp': 200,
                  'shield_hp': 300, 'flying_hp': 400, 'phase_counter': 17,
                  'chilled_counter': 101, 'buttered_counter': 202, 'ice_trap_counter': 303}
        encoded = encode_observation({'zombies': [zombie]})
        for field in ('body_hp', 'max_hp', 'helm_hp', 'shield_hp', 'flying_hp'):
            self.assertAlmostEqual(self.feature(encoded, field), np.log1p(zombie[field]) / np.log1p(20000))
        for field in ('phase_counter', 'chilled_counter', 'buttered_counter', 'ice_trap_counter'):
            self.assertAlmostEqual(self.feature(encoded, field), np.log1p(zombie[field]) / np.log1p(1000))
        self.assertFalse(encoded['zombie_mask'][1:].any())
        self.assertFalse(encoded['zombies'][1:].any())

    def test_categories_targets_movement_and_interactions(self):
        z = {'type': 20, 'phase': 4, 'phase_counter': -1, 'height': 9,
             'row': 2, 'x': -20, 'y': 300, 'vel_x': -3, 'altitude': 3000,
             'target_row': 0, 'target_col': 8, 'is_eating': True, 'is_mind_controlled': True}
        encoded = encode_observation({'zombies': [z, {**z, 'target_row': -1, 'target_col': -1}]})
        for field, expected in (('type', 21), ('phase', 5), ('height', 10),
                                ('row', 3), ('target_row', 1), ('target_col', 9)):
            self.assertEqual(encoded['zombie_' + field][0], expected)
        self.assertEqual(encoded['zombie_target_row'][1], 0)
        self.assertEqual(encoded['zombie_target_col'][1], 0)
        self.assertAlmostEqual(self.feature(encoded, 'vel_x'), -0.6)
        self.assertAlmostEqual(self.feature(encoded, 'altitude'), 1.0)
        self.assertLess(self.feature(encoded, 'phase_counter'), 0)
        self.assertEqual(self.feature(encoded, 'is_eating'), 1)
        self.assertEqual(self.feature(encoded, 'is_mind_controlled'), 1)
        self.assertTrue(make_observation_space().contains(encoded))

    def test_same_cell_zombies_remain_distinct_and_unordered(self):
        zombies = [{'row': 0, 'x': 300, 'body_hp': hp, 'phase': phase}
                   for hp, phase in ((100, 15), (200, 16))]
        first = encode_observation({'zombies': zombies})
        reverse = encode_observation({'zombies': zombies[::-1]})
        np.testing.assert_array_equal(first['zombies'][:2], reverse['zombies'][:2][::-1])
        self.assertEqual(first['zombie_mask'].sum(), 2)
        self.assertNotEqual(first['zombie_phase'][0], first['zombie_phase'][1])
        self.assertNotEqual(self.feature(first, 'body_hp', 0), self.feature(first, 'body_hp', 1))


if __name__ == '__main__':
    unittest.main()
