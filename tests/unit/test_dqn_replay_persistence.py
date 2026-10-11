import tempfile
import unittest
from pathlib import Path

import numpy as np

from pvz_gym.observation import empty_observation, encode_observation
from train.dqn.replay import ReplayBuffer


class DqnReplayPersistenceTest(unittest.TestCase):
    def test_entity_fields_and_last_slot_survive_replay_round_trip(self):
        obs = encode_observation({"zombies": [
            {"type": 20, "phase": 4, "height": 9, "target_row": 2, "target_col": 8,
             "body_hp": 100, "helm_hp": 200, "shield_hp": 300, "flying_hp": 400,
             "phase_counter": 50, "chilled_counter": 100, "buttered_counter": 200,
             "ice_trap_counter": 300, "is_eating": True} for _ in range(256)]})
        replay = ReplayBuffer(capacity=1, action_dim=1)
        replay.add(obs, 0, 0, obs, False, [True], ticks_advanced=150)
        with tempfile.TemporaryDirectory() as tmp:
            path = replay.save(Path(tmp) / "replay.npz")
            restored, _ = ReplayBuffer.load(path)
            sampled = restored.sample(1).observations
            for name, expected in obs.items():
                if np.issubdtype(expected.dtype, np.floating):
                    np.testing.assert_allclose(sampled[name][0], expected, rtol=5e-4, atol=1e-6)
                else:
                    np.testing.assert_array_equal(sampled[name][0], expected)
            with np.load(path, allow_pickle=False) as data:
                legacy = {name: data[name] for name in data.files}
            legacy["observation_schema"] = np.asarray("pvz_observation")
            np.savez(path, **legacy)
            with self.assertRaisesRegex(ValueError, "observation schema"):
                ReplayBuffer.load(path)

    def _add(self, replay: ReplayBuffer, value: int) -> None:
        observation = empty_observation()
        next_observation = empty_observation()
        observation["global"][0] = value
        next_observation["global"][0] = value + 1
        replay.add(
            observation,
            value % replay.action_dim,
            float(value),
            next_observation,
            value % 3 == 0,
            np.asarray([(value + index) % 2 for index in range(replay.action_dim)], dtype=np.bool_),
            ticks_advanced=value * 17,
        )

    def test_round_trip_restores_wrapped_replay_and_metadata(self):
        replay = ReplayBuffer(capacity=3, action_dim=5, seed=7)
        for value in range(5):
            self._add(replay, value)

        with tempfile.TemporaryDirectory() as tmp:
            path = replay.save(Path(tmp) / "replay.npz", env_step=123, episode=9)
            restored, metadata = ReplayBuffer.load(
                path,
                capacity=3,
                action_dim=5,
                seed=99,
            )

        self.assertEqual(len(restored), 3)
        self.assertEqual(metadata["env_step"], 123)
        self.assertEqual(metadata["episode"], 9)
        np.testing.assert_array_equal(restored._ticks_advanced[:3], [34, 51, 68])
        np.testing.assert_allclose(
            restored._observations["global"][:3, 0].astype(np.float32),
            np.asarray([2.0, 3.0, 4.0], dtype=np.float32),
        )
        np.testing.assert_array_equal(
            np.unpackbits(
                restored._next_action_masks[2], count=5, bitorder="little"
            ).astype(np.bool_),
            np.asarray([False, True, False, True, False]),
        )

    def test_smaller_capacity_keeps_newest_transitions(self):
        replay = ReplayBuffer(capacity=4, action_dim=3, seed=1)
        for value in range(6):
            self._add(replay, value)

        with tempfile.TemporaryDirectory() as tmp:
            path = replay.save(Path(tmp) / "replay.npz", env_step=50, episode=2)
            restored, _metadata = ReplayBuffer.load(
                path,
                capacity=2,
                action_dim=3,
            )

        self.assertEqual(len(restored), 2)
        np.testing.assert_array_equal(restored._ticks_advanced[:2], [68, 85])
        np.testing.assert_allclose(
            restored._observations["global"][:2, 0].astype(np.float32),
            np.asarray([4.0, 5.0], dtype=np.float32),
        )


if __name__ == "__main__":
    unittest.main()
