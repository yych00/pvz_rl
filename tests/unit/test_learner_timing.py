from dataclasses import replace
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
import torch

from pvz_gym.action_space import ACTION_SPACE_SIZE
from pvz_gym.observation import empty_observation
from pvz_gym.profiling import TimingRecorder
from train.dqn.agent import DoubleDQNAgent
from train.dqn.config import DoubleDQNConfig
from train.dqn.learner_timing import LearnerProfiler, phase
from train.dqn.replay import ReplayBuffer


class LearnerTimingTest(unittest.TestCase):
    def test_periodic_snapshot_is_throttled_atomic_and_forceable(self):
        recorder = TimingRecorder(True)
        profiler = LearnerProfiler(recorder, torch.device("cpu"))
        recorder.add("learn", 1.0)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "learner_timing.json"
            with patch("train.dqn.learner_timing.perf_counter", return_value=100.0):
                self.assertTrue(profiler.save_if_due(path))
            recorder.add("learn", 2.0)
            with patch("train.dqn.learner_timing.perf_counter", return_value=105.0):
                self.assertFalse(profiler.save_if_due(path))
                self.assertEqual(json.loads(path.read_text())["learn"]["count"], 1)
            with patch("train.dqn.learner_timing.perf_counter", return_value=110.0):
                self.assertTrue(profiler.save_if_due(path))
            self.assertEqual(json.loads(path.read_text())["learn"]["count"], 2)
            recorder.add("learn", 3.0)
            with patch("train.dqn.learner_timing.perf_counter", return_value=111.0):
                self.assertTrue(profiler.save_if_due(path, force=True))
            self.assertEqual(json.loads(path.read_text())["learn"]["count"], 3)
            self.assertFalse(path.with_name(path.name + ".tmp").exists())

    def batch(self):
        replay = ReplayBuffer(4, ACTION_SPACE_SIZE, scheme="per")
        obs = empty_observation()
        mask = np.ones(ACTION_SPACE_SIZE, dtype=bool)
        # Exercise the all-masked-next-state fallback, as well as nonterminal TD.
        for i in range(4):
            replay.add(obs, 0, float(i + 1), obs, i == 3,
                       mask if i < 3 else np.zeros_like(mask), ticks_advanced=100)
        return replay.sample(4, n_step=1)

    def config(self, **updates):
        return replace(DoubleDQNConfig(device="cpu", hidden_sizes=(8, 4),
                       reward_scheme="sparse", target_update_interval=2), **updates)

    def test_cpu_profile_preserves_updates_td_errors_and_optimizer(self):
        batch = self.batch()
        for selected in (True, False):
            with self.subTest(selected=selected):
                config = self.config(selected_action_forward=selected)
                plain = DoubleDQNAgent(ACTION_SPACE_SIZE, config)
                timed = DoubleDQNAgent(ACTION_SPACE_SIZE, config)
                recorder = TimingRecorder(True)
                timed.learner_profiler = LearnerProfiler(recorder, timed.device)
                for _ in range(3):
                    expected_loss, expected_td = plain.learn(batch, return_td_errors=True)
                    rng = torch.get_rng_state().clone()
                    loss, td = timed.learn(batch, return_td_errors=True)
                    torch.testing.assert_close(torch.get_rng_state(), rng)
                    self.assertEqual(expected_loss, loss)
                    np.testing.assert_array_equal(expected_td, td)
                self.assertEqual(timed.update_count, plain.update_count)
                for name in ("online", "target"):
                    for key, value in getattr(plain, name).state_dict().items():
                        torch.testing.assert_close(value, getattr(timed, name).state_dict()[key], rtol=0, atol=0)
                for key, state in plain.optimizer.state_dict()["state"].items():
                    for field, value in state.items():
                        torch.testing.assert_close(value, timed.optimizer.state_dict()["state"][key][field], rtol=0, atol=0)
                stats = recorder.snapshot()
                for name in ("data_prepare", "tensor_prepare", "device_transfer", "current_forward",
                             "online_next_forward", "next_action", "target_forward", "target_compute",
                             "loss_compute", "zero_grad", "backward", "grad_clip", "optimizer", "result_copy"):
                    self.assertEqual(stats["learn_" + name + "_wall"]["count"], 3)
                self.assertEqual(stats["learn_target_sync_wall"]["count"], 1)
                self.assertFalse(any(key.endswith("_cuda") for key in stats))

    def test_disabled_profile_does_not_create_events_or_stats(self):
        recorder = TimingRecorder(False)
        profiler = LearnerProfiler(recorder, torch.device("cuda"))
        with patch("torch.cuda.Event") as event:
            with profiler.update(0) as timer:
                with phase(timer, "forward"):
                    self.assertIsNone(timer)
            event.assert_not_called()
        self.assertEqual(recorder.snapshot(), {})

    def test_cuda_event_resolution_is_deferred_without_phase_synchronization(self):
        recorder = TimingRecorder(True)
        profiler = LearnerProfiler(recorder, torch.device("cuda"), cuda_sample_every=32)
        start, end = Mock(), Mock()
        start.elapsed_time.return_value = 2.5
        end.query.return_value = False
        with patch("torch.cuda.current_stream", return_value=SimpleNamespace()) as stream, \
             patch("torch.cuda.Event", side_effect=[start, end]), \
             patch("torch.cuda.synchronize") as synchronize:
            with profiler.update(0) as timer:
                with phase(timer, "forward"):
                    pass
            self.assertEqual(stream.call_count, 1)
            end.synchronize.assert_not_called()
            synchronize.assert_not_called()
            self.assertNotIn("learn_forward_cuda", recorder.snapshot())
            # An unsampled update also queries, without allocating new events.
            with profiler.update(1) as timer:
                with phase(timer, "forward"):
                    pass
            end.query.return_value = True
            profiler.flush()
            self.assertEqual(recorder.snapshot()["learn_forward_cuda"]["mean_ms"], 2.5)
            self.assertEqual(recorder.snapshot()["learn_forward_wall"]["count"], 2)
            end.synchronize.assert_not_called()

    def test_failure_records_completed_wall_phase_and_releases_lock(self):
        recorder = TimingRecorder(True)
        profiler = LearnerProfiler(recorder, torch.device("cpu"))
        lock = Mock()
        with self.assertRaisesRegex(ValueError, "test failure"):
            with profiler.lock("lock_wait", lock), profiler.update(0) as timer:
                with phase(timer, "forward"):
                    raise ValueError("test failure")
        lock.release.assert_called_once()
        self.assertEqual(recorder.snapshot()["learn_forward_wall"]["count"], 1)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA not available")
    def test_real_cuda_stream_profile_and_target_sync(self):
        agent = DoubleDQNAgent(ACTION_SPACE_SIZE, self.config(device="cuda"))
        recorder = TimingRecorder(True)
        agent.learner_profiler = LearnerProfiler(recorder, agent.device, cuda_sample_every=32)
        batch = self.batch()
        stream = torch.cuda.Stream(device=agent.device)
        stream.wait_stream(torch.cuda.current_stream(agent.device))
        with torch.cuda.stream(stream):
            for _ in range(3):
                self.assertTrue(np.isfinite(agent.learn(batch)))
        agent.learner_profiler.flush(wait=True)
        stats = recorder.snapshot()
        for name in ("data_prepare", "current_forward", "online_next_forward", "target_forward",
                     "backward", "grad_clip", "optimizer", "result_copy"):
            self.assertEqual(stats["learn_" + name + "_wall"]["count"], 3)
            # First update plus forced sample when copying the target network.
            self.assertEqual(stats["learn_" + name + "_cuda"]["count"], 2)
            self.assertGreaterEqual(stats["learn_" + name + "_cuda"]["mean_ms"], 0)
        self.assertEqual(stats["learn_target_sync_cuda"]["count"], 1)
        self.assertEqual(stats["learn_pin_memory_wall"]["count"], 3)


if __name__ == "__main__":
    unittest.main()
