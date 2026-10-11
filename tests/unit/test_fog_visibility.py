import copy
import json
import mmap
import os
import struct
import unittest
import uuid

import numpy as np

from pvz_gym.bridge.native import _parse_observation_data, _parse_step_json
from pvz_gym.observation import encode_observation, observation_is_valid
from tests.unit.test_turbo_execution import binary


def fog_snapshot(left=5, offset=0):
    return {"global": {"scene": 3, "grid_rows": 6,
                       "fog_state_available": True, "fog_offset": offset,
                       "fog_alpha": [0 if col < left else 255
                                     for col in range(9) for row in range(7)]}}


class FogVisibilityTest(unittest.TestCase):
    def test_hidden_enemy_and_hazard_facts_cannot_change_policy_fields(self):
        clear = fog_snapshot()
        hidden = copy.deepcopy(clear)
        hidden.update({
            "zombies": [{"row": 2, "x": 560, "type": 7, "body_hp": 500}],
            "zombie_health": [{"row": 2, "x": 560, "body_hp": 900}],
            "plants": [{"row": 2, "col": 6, "layer": 1, "type": 25, "hp": 300}],
            "grid": [{"row": 2, "col": 6, "terrain": 0, "has_main_plant": True,
                      "has_grave": True, "has_crater": True}],
            "grid_items": [{"row": 2, "col": 6, "remaining_ticks": 200}],
            "ice_trails": [{"row": 2, "col_start": 6, "col_end": 8,
                            "melt_countdown": 400}],
            "dropped_cards": [{"type": 0, "x": 560, "y": 260}],
            "boss_projectiles": [{"valid": True, "type": 0, "row": 2,
                                  "x": 560, "y": 260}],
        })
        # Terrain is public static map information in both snapshots.
        clear["plants"] = copy.deepcopy(hidden["plants"])
        clear["grid"] = [{"row": 2, "col": 6, "terrain": 0, "has_main_plant": True}]
        before = copy.deepcopy(hidden)
        expected, actual = encode_observation(clear), encode_observation(hidden)
        self.assertTrue(observation_is_valid(actual))
        for name in expected:
            np.testing.assert_array_equal(actual[name], expected[name], err_msg=name)
        self.assertEqual(hidden, before)  # Full reward/diagnostic facts are preserved.

    def test_plants_and_occupancy_survive_fog_with_live_state(self):
        raw = fog_snapshot()
        raw["plants"] = [
            {"row": 2, "col": 6, "layer": layer, "type": kind,
             "state": 3, "hp": 120, "max_hp": 300, "is_sleeping": True}
            for layer, kind in enumerate((33, 14, 30))
        ]
        raw["grid"] = [{"row": 2, "col": 6, "terrain": 0,
                        "has_base_plant": True, "has_main_plant": True,
                        "has_pumpkin": True}]
        raw["zombies"] = [{"row": 2, "x": 560, "type": 0, "body_hp": 100}]
        fogged = encode_observation(raw)
        raw["global"]["scene"] = 0
        clear = encode_observation(raw)
        for key in ("plant_type", "plant_state", "plants", "grid"):
            np.testing.assert_array_equal(fogged[key], clear[key], err_msg=key)
        self.assertTrue(fogged["plants"][:, 2, 6, 0].all())
        np.testing.assert_array_equal(fogged["grid"][2, 6, :4], [1, 1, 1, 1])
        self.assertFalse(fogged["zombie_mask"].any())
        self.assertFalse(fogged["zombie_grid"].any())
        raw["global"]["scene"] = 3
        raw["plants"] = []
        raw["grid"][0].update(has_base_plant=False, has_main_plant=False, has_pumpkin=False)
        removed = encode_observation(raw)
        self.assertFalse(removed["plants"].any())
        self.assertFalse(removed["grid"][2, 6, 1:4].any())

    def test_filter_before_entity_cap_and_grid_aggregation(self):
        raw = fog_snapshot()
        visible = {"row": 0, "x": 120, "type": 1, "body_hp": 100}
        raw["zombies"] = [{"row": 0, "x": 720, "body_hp": 1000}] * 300 + [visible]
        encoded = encode_observation(raw)
        self.assertEqual(encoded["zombie_mask"].sum(), 1)
        self.assertEqual(encoded["zombie_type"][0], 2)
        self.assertEqual(np.count_nonzero(encoded["zombie_grid"][:, :, 0]), 1)
        self.assertFalse(encoded["zombie_grid"][:, 5:].any())

    def test_native_lamp_clearance_and_reappearance(self):
        raw = fog_snapshot()
        raw["zombies"] = [{"row": 1, "x": 560, "body_hp": 100}]
        self.assertEqual(encode_observation(raw)["zombie_mask"].sum(), 0)
        raw["global"]["fog_alpha"][6 * 7 + 1] = 0
        lit = encode_observation(raw)
        self.assertEqual(lit["zombie_mask"].sum(), 1)
        self.assertEqual(lit["grid"].shape, (6, 9, 9))
        raw["global"]["fog_alpha"][6 * 7 + 1] = 127
        self.assertEqual(encode_observation(raw)["zombie_mask"].sum(), 1)
        raw["global"]["fog_alpha"][6 * 7 + 1] = 128
        self.assertEqual(encode_observation(raw)["zombie_mask"].sum(), 0)

    def test_blover_offset_follows_actual_position_not_countdown(self):
        raw = fog_snapshot(offset=700)
        raw["zombies"] = [{"row": 5, "x": 720, "body_hp": 100}]
        self.assertEqual(encode_observation(raw)["zombie_mask"].sum(), 1)
        raw["global"]["fog_offset"] = 160
        middle = encode_observation(raw)
        self.assertEqual(middle["zombie_mask"].sum(), 0)
        raw["global"]["fog_offset"] = 0
        self.assertEqual(encode_observation(raw)["zombie_mask"].sum(), 0)

    def test_edge_and_offboard_spawn_stay_obscured(self):
        raw = fog_snapshot(left=6)
        raw["zombies"] = [{"row": 0, "x": x} for x in (519.9, 520, 1100)]
        self.assertEqual(encode_observation(raw)["zombie_mask"].sum(), 1)
        raw["global"]["fog_alpha"] = [0] * 63  # Stormy night has no ordinary fog.
        self.assertEqual(encode_observation(raw)["zombie_mask"].sum(), 3)

    def test_nonfog_is_unchanged_and_inactive_rows_are_padded(self):
        raw = {"global": {"scene": 0, "grid_rows": 5},
               "zombies": [{"row": 0, "x": 720, "body_hp": 100}]}
        encoded = encode_observation(raw)
        self.assertEqual(encoded["zombie_mask"].sum(), 1)
        self.assertEqual(encoded["grid"].shape, (6, 9, 9))
        self.assertFalse(encoded["grid"][5].any())

    def test_missing_or_bad_native_fog_does_not_silently_reveal_enemies(self):
        with self.assertRaisesRegex(ValueError, "rebuilt native DLL"):
            encode_observation({"global": {"scene": 3}})
        for patch in ({"fog_alpha": []}, {"fog_alpha": [256] * 63},
                      {"fog_alpha": [float("nan")] * 63},
                      {"fog_offset": -1}, {"fog_offset": float("inf")}):
            raw = fog_snapshot()
            raw["global"].update(patch)
            with self.subTest(patch=patch), self.assertRaises(ValueError):
                encode_observation(raw)

    def test_json_binary_shared_transport_preserve_visibility(self):
        raw = fog_snapshot(offset=80.5)
        raw["global"].update(status=3, game_clock=100)
        raw["schema_version"] = 4
        raw["zombies"] = [{"row": 0, "x": 720, "body_hp": 100}]
        value = {"success": True, "status": 3, "game_clock": 100,
                 "observation": raw, "observation_quality": "fresh"}
        payload = b"PVZB\1" + binary(value)
        expected = encode_observation(_parse_observation_data(raw))
        facts = [_parse_step_json(json.dumps(value)), _parse_step_json(payload)]
        if os.name == "nt":
            name = "Local\\pvz-gym-fog-contract-" + uuid.uuid4().hex
            with mmap.mmap(-1, 65536, tagname=name) as shared:
                shared[:12] = struct.pack("<QI", 4, len(payload))
                shared[12:12 + len(payload)] = payload
                descriptor = {"name": name, "size": 65536, "length": len(payload), "revision": 4}
                facts.append(_parse_step_json(b"PVZM" + json.dumps(descriptor).encode()))
        for fact in facts:
            actual = encode_observation(fact.observation)
            for name in expected:
                np.testing.assert_array_equal(expected[name], actual[name])


if __name__ == "__main__":
    unittest.main()
