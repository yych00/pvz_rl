"""Replay memory fixes preserve returns, ring boundaries and frozen snapshots."""
from pathlib import Path
import os
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from pvz_gym.action_space import ACTION_SPACE_SIZE
from pvz_gym.observation import empty_observation
from train.checkpoint_io import CheckpointWriter, DiskReplaySnapshot, freeze_replay
from train.dqn.replay import ReplayBuffer


class OrphanProbeEnv:
    def __init__(self, marker):
        self.marker = Path(marker)

    def reset(self, *, seed):
        return {}, {}

    def close(self):
        self.marker.write_text("closed", encoding="utf-8")


class ReplayMemoryTest(unittest.TestCase):
    def test_sampler_exits_and_closes_environment_after_parent_disappears(self):
        program = """
import multiprocessing as mp
import os, pathlib, sys
from tests.unit.test_replay_memory import OrphanProbeEnv
from train.dqn.config import DoubleDQNConfig
from train.sampler import _worker
if __name__ == '__main__':
    context = mp.get_context('spawn')
    inbox, outbox = context.Queue(), context.Queue()
    worker = context.Process(target=_worker, args=(OrphanProbeEnv, (sys.argv[1],),
        DoubleDQNConfig(), 0, inbox, outbox, context.Event()))
    worker.start()
    pathlib.Path(sys.argv[2]).write_text(str(worker.pid))
    assert outbox.get(timeout=20)[0] == 'ready'
    os._exit(0)
"""
        with tempfile.TemporaryDirectory() as folder:
            marker, pidfile = Path(folder) / "closed", Path(folder) / "pid"
            try:
                result = subprocess.run([sys.executable, "-c", program, str(marker), str(pidfile)],
                                        capture_output=True, text=True, timeout=30)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(marker.read_text(), "closed")
            finally:
                if pidfile.is_file():
                    from pvz_gym.runtime.instance_pool import process_is_alive
                    pid = int(pidfile.read_text())
                    if process_is_alive(pid):
                        os.kill(pid, 15)

    def test_incremental_eligibility_matches_returns_across_streams_and_wraps(self):
        replay = ReplayBuffer(13, 2)
        rng = np.random.default_rng(42)
        observation = empty_observation()
        for step in range(150):
            stream = int(rng.integers(4))
            replay.add(observation, 0, 0, observation, bool(rng.random() < .2),
                       [True, True], ticks_advanced=100, stream_id=stream,
                       episode_end=bool(rng.random() < .15))
            if step % 7 == 0:
                replay.end_episode(stream_id=stream)
            for n in (1, 3, 13):
                expected = sum(replay._complete_return(slot, n) for slot in range(len(replay)))
                self.assertEqual(replay._eligible_count(n), expected)
                self.assertEqual(len(replay._eligible_indices(n)), expected)

    def test_can_sample_and_per_do_not_allocate_complete_index_lists(self):
        replay = ReplayBuffer(8, 2, scheme="per")
        observation = empty_observation()
        for _ in range(8):
            replay.add(observation, 0, 1, observation, False, [True, True], ticks_advanced=100)
        with patch.object(replay, "_eligible_indices", side_effect=AssertionError("full index allocation")):
            self.assertTrue(replay.can_sample(3))
            batch = replay.sample(4, n_step=3)
        np.testing.assert_allclose(batch.rewards, 1 + .999 + .999**2)

    def test_disk_snapshot_is_independent_and_cleanup_follows_success(self):
        values = np.arange(600000, dtype=np.float32)
        with tempfile.TemporaryDirectory() as folder:
            snapshot = freeze_replay({"values": values}, directory=folder, spill_threshold=1)
            self.assertIsInstance(snapshot, DiskReplaySnapshot)
            self.assertIsInstance(snapshot["values"], np.memmap)
            directory = snapshot.directory
            values.fill(-1)
            model = dict(env_step=1, episode=0)
            writer = CheckpointWriter()
            try:
                writer.submit(Path(folder) / "full.pt", model, snapshot)
                writer.flush()
                self.assertFalse(directory.exists())
                with np.load(Path(folder) / model["replay_file"]) as stored:
                    np.testing.assert_array_equal(stored["values"], np.arange(600000, dtype=np.float32))
            finally:
                writer.close()

    def test_disk_snapshot_cleanup_follows_write_failure(self):
        with tempfile.TemporaryDirectory() as folder:
            snapshot = freeze_replay({"values": np.zeros(600000, dtype=np.float32)},
                                     directory=folder, spill_threshold=1)
            directory = snapshot.directory
            writer = CheckpointWriter()
            try:
                with patch.object(ReplayBuffer, "write_checkpoint_payload", side_effect=OSError("disk full")):
                    writer.submit(Path(folder) / "full.pt", dict(env_step=1, episode=0), snapshot)
                    with self.assertRaisesRegex(OSError, "disk full"):
                        writer.flush()
                self.assertFalse(directory.exists())
            finally:
                writer.close()

    def test_successful_restore_does_not_allocate_a_second_empty_replay(self):
        from train.dqn.agent import DoubleDQNAgent
        from train.dqn.config import DoubleDQNConfig
        from train.dqn.controller import DoubleDQNTrainerController
        config = DoubleDQNConfig(reward_scheme="sparse", n_step=1, replay_capacity=8,
                                 batch_size=2, device="cpu", hidden_sizes=(8, 4))
        with tempfile.TemporaryDirectory() as folder:
            controller = DoubleDQNTrainerController(config, run_dir=folder, eval_every_episodes=0)
            controller._agent = DoubleDQNAgent(ACTION_SPACE_SIZE, config)
            controller._replay = controller._new_replay(ACTION_SPACE_SIZE)
            observation = empty_observation()
            controller._replay.add(observation, 0, 0, observation, False, [True] * ACTION_SPACE_SIZE, ticks_advanced=100)
            checkpoint = controller.save_checkpoint(Path(folder) / "full.pt")
            controller._flush_checkpoint_writer(close=True)
            resumed = DoubleDQNTrainerController(config, run_dir=folder, checkpoint_path=checkpoint,
                                                checkpoint_load_mode="strict_resume", eval_every_episodes=0)
            resumed._agent, step, episode, _ = resumed._build_agent(ACTION_SPACE_SIZE)
            with patch.object(resumed, "_new_replay", side_effect=AssertionError("extra empty replay")):
                restored = resumed._restore_replay(action_dim=ACTION_SPACE_SIZE, start_step=step, start_episode=episode)
            self.assertEqual(len(restored), 1)


if __name__ == "__main__":
    unittest.main()
