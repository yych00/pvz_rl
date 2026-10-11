import json
import mmap
import os
from pathlib import Path
import struct
import unittest
import uuid
from dataclasses import replace

import numpy as np

from pvz_gym.bridge.native import _parse_step_json, NativeCapabilityError
from pvz_gym.env import PvZEnv
from pvz_gym.observation import (
    MAX_BOSS_PROJECTILES, LEGACY_OBSERVATION_SPEC, OBSERVATION_SPEC,
    boss_projectile_truncation_info, empty_observation, encode_observation,
    make_observation_space, observation_is_valid,
)
from pvz_gym.types import GameStatus
from tests.fake_backend import FactClient
from tests.unit.test_turbo_execution import binary


def ball(**fields):
    return {"id": 123, "owner_id": 456, "type": 0, "row": 3,
            "x": 455.5, "y": 230.25, "valid": True, **fields}


def transition(**fields):
    return {"success": True, "status": int(GameStatus.WON), "game_clock": 109,
            "observation": {"schema_version": 4, "quality": "fresh",
                            "global": {"status": int(GameStatus.WON), "game_clock": 109},
                            "boss_projectiles": [ball()]},
            "ticks_advanced": 9, "ticks_advanced_exact": True,
            "replay_eligible": True, "observation_quality": "fresh", **fields}


class BossObservationQualityTest(unittest.TestCase):
    def test_empty_normal_level_and_explicit_legacy_shapes(self):
        fresh = encode_observation({})
        legacy = encode_observation({}, boss_projectiles=False)
        self.assertEqual(set(legacy), set(LEGACY_OBSERVATION_SPEC))
        self.assertEqual(list(OBSERVATION_SPEC)[:len(legacy)], list(LEGACY_OBSERVATION_SPEC))
        self.assertFalse(fresh["boss_projectile_mask"].any())
        self.assertTrue(observation_is_valid(fresh))
        self.assertTrue(observation_is_valid(legacy, boss_projectiles=False))
        self.assertFalse(observation_is_valid(legacy))
        self.assertTrue(make_observation_space(boss_projectiles=False).contains(legacy))
        self.assertTrue(make_observation_space(boss_projectiles=False).contains(empty_observation(boss_projectiles=False)))
        for name in legacy:
            np.testing.assert_array_equal(fresh[name], legacy[name])

    def test_separate_fire_ice_semantics_and_continuous_coordinates(self):
        raw = {"boss_projectiles": [ball(), ball(type=1, row=0, x=-179.75, y=100.125)]}
        encoded = encode_observation(raw)
        self.assertEqual(encoded["boss_projectiles"].shape, (8, 2))
        np.testing.assert_array_equal(encoded["boss_projectile_type"][:2], [2, 1])
        np.testing.assert_array_equal(encoded["boss_projectile_row"][:2], [1, 4])
        np.testing.assert_allclose(encoded["boss_projectiles"][:2],
                                   [[-179.75 / 900, 100.125 / 600], [455.5 / 900, 230.25 / 600]])
        self.assertEqual(encoded["boss_projectile_mask"].sum(), 2)
        self.assertFalse(encoded["zombie_mask"].any())
        self.assertFalse(encoded["zombie_grid"].any())
        altered = encode_observation({"boss_projectiles": [dict(p, id=999, owner_id=888) for p in raw["boss_projectiles"]]})
        for name in encoded:
            np.testing.assert_array_equal(encoded[name], altered[name])
        self.assertTrue(make_observation_space().contains(encoded))

    def test_invalid_or_released_attacks_do_not_become_live_entities(self):
        for invalid in (ball(valid=False), ball(type=25), ball(row=-1), ball(row=6),
                        ball(x=float("nan")), ball(y=float("inf")), ball(x=None)):
            with self.subTest(invalid=invalid):
                self.assertFalse(encode_observation({"boss_projectiles": [invalid]})["boss_projectile_mask"].any())
        for x in (455.5, 450.25, 120.125, -179.75):
            encoded = encode_observation({"boss_projectiles": [ball(x=x)]})
            self.assertAlmostEqual(float(encoded["boss_projectiles"][0, 0]), x / 900)
        self.assertFalse(encode_observation({"boss_projectiles": []})["boss_projectile_mask"].any())

    def test_capacity_retains_leftmost_valid_attacks_with_explicit_diagnostic(self):
        raw = {"boss_projectiles": [ball(x=100 - index) for index in range(11)] + [ball(valid=False)]}
        encoded = encode_observation(raw)
        self.assertEqual(encoded["boss_projectile_mask"].sum(), MAX_BOSS_PROJECTILES)
        np.testing.assert_allclose(encoded["boss_projectiles"][:, 0], np.arange(90, 98) / 900)
        self.assertEqual(boss_projectile_truncation_info(raw),
                         {"boss_projectiles_truncated": True, "boss_projectiles_dropped": 3})

    def test_success_without_required_observation_is_rejected(self):
        for observation in (None, {}, {"global": None}):
            with self.subTest(observation=observation), self.assertRaises(NativeCapabilityError):
                _parse_step_json(json.dumps(transition(observation=observation)))

    def test_versioned_transition_rejects_status_clock_and_quality_disagreements(self):
        for patch in ({"status": int(GameStatus.PLAYING)}, {"game_clock": 110},
                      {"observation_quality": "unavailable"}):
            with self.subTest(patch=patch), self.assertRaises(NativeCapabilityError):
                _parse_step_json(json.dumps(transition(**patch)))

    def test_json_binary_and_shared_decode_preserve_new_entity_and_quality(self):
        value = transition()
        encoded = b"PVZB\1" + binary(value)
        json_facts = _parse_step_json(json.dumps(value))
        self.assertEqual(_parse_step_json(encoded), json_facts)
        self.assertEqual(json_facts.observation.boss_projectiles[0]["type"], 0)
        self.assertEqual(json_facts.ticks_advanced, 9)
        self.assertTrue(json_facts.replay_eligible)
        if os.name == "nt":
            name = "Local\\pvz-gym-boss-contract-" + uuid.uuid4().hex
            with mmap.mmap(-1, 65536, tagname=name) as shared:
                shared[:12] = struct.pack("<QI", 4, len(encoded))
                shared[12:12+len(encoded)] = encoded
                descriptor = {"name": name, "size": 65536, "length": len(encoded), "revision": 4}
                self.assertEqual(_parse_step_json(b"PVZM" + json.dumps(descriptor).encode()), json_facts)

    def test_fallback_preserves_confirmed_clock_and_is_never_replay_eligible(self):
        value = transition(observation_quality="terminal_fallback", terminal_observation_fallback=True,
                           last_confirmed_game_clock=107, game_clock=107, ticks_advanced=7,
                           ticks_advanced_exact=False, replay_eligible=False)
        value["observation"]["quality"] = "terminal_fallback"
        value["observation"]["global"]["game_clock"] = 100
        facts = _parse_step_json(json.dumps(value))
        self.assertFalse(facts.replay_eligible)
        self.assertFalse(facts.ticks_advanced_exact)
        self.assertEqual(facts.last_confirmed_game_clock, 107)
        self.assertEqual(facts.observation.global_state["game_clock"], 100)

    def test_env_reports_fallback_quality_instead_of_assuming_requested_ticks(self):
        class ReleasedBoardClient(FactClient):
            def step(self, action, ticks):
                facts = super().step(action, ticks)
                return replace(facts, status=GameStatus.WON, game_clock=0,
                               terminal_observation_fallback=True, ticks_advanced=0,
                               ticks_advanced_exact=False, replay_eligible=False,
                               observation_quality="terminal_fallback", last_confirmed_game_clock=0)
        env = PvZEnv(ReleasedBoardClient(), ticks_per_step=150)
        try:
            env.reset()
            _, _, ended, _, info = env.step(0)
            self.assertTrue(ended)
            self.assertEqual(info["ticks_advanced"], 0)
            self.assertFalse(info["ticks_advanced_exact"])
            self.assertFalse(info["replay_eligible"])
            self.assertEqual(info["observation_quality"], "terminal_fallback")
        finally:
            env.close()

    @unittest.skipUnless(os.environ.get("PVZ_GYM_NATIVE_OBSERVATION_FIXTURE"), "native serializer fixture not supplied")
    def test_actual_native_json_binary_serializers_agree(self):
        prefix = Path(os.environ["PVZ_GYM_NATIVE_OBSERVATION_FIXTURE"])
        json_facts = _parse_step_json(prefix.with_suffix(".json").read_bytes())
        binary_facts = _parse_step_json(prefix.with_suffix(".pvzb").read_bytes())
        self.assertEqual(json_facts, binary_facts)
        self.assertTrue(json_facts.observation.cards[0]["is_imitater"])
        self.assertEqual(len(json_facts.observation.boss_projectiles), 2)
        self.assertTrue(json_facts.replay_eligible)
        self.assertTrue(json_facts.observation.global_state["fog_state_available"])
        encoded = encode_observation(json_facts.observation)
        self.assertEqual(encoded["boss_projectile_mask"].sum(), 1)
        self.assertEqual(encoded["grid"].shape, (6, 9, 9))


if __name__ == "__main__":
    unittest.main()
