"""Lifecycle, queue and PID tests without attaching to a real game."""
from __future__ import annotations

from dataclasses import replace
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from train.common import EnvConfig
from train.dqn.base_controller import DoubleDQNTrainerController
from train.dqn.config import DoubleDQNConfig
from train.dqn.visualization import TrainingVisualization
from train.runtime import TrainingRuntimeConfig
from workbench.gui.q_heatmap_overlay import QHeatmapOverlay, user32
from workbench.gui.training_visualization import TrainingVisualizationBridge


class FakeRoot:
    def __init__(self):
        self.scheduled = {}
        self.counter = 0
    def after(self, delay, callback):
        self.counter += 1
        self.scheduled[self.counter] = callback
        return self.counter
    def after_cancel(self, token):
        self.scheduled.pop(token, None)
    def tick(self):
        token = min(self.scheduled)
        self.scheduled.pop(token)()


class VisualizationMailboxTest(unittest.TestCase):
    def frame(self, **updates):
        return dict(process_id=123, generation=2, worker_generation=0,
                    episode_id=1, sequence_id=0, top5=[], noop_q=1.0,
                    scene_id=0, selected_action=0, **updates)

    def test_worker_burst_uses_one_slot_and_no_tk_calls(self):
        root, overlay = FakeRoot(), Mock()
        bridge = TrainingVisualizationBridge(root, overlay=overlay)
        bridge.start("run")
        scheduled = root.counter
        def publish():
            for sequence in range(1000):
                data = self.frame()
                data["sequence_id"] = sequence
                bridge.post("run", "visualization", data)
        worker = threading.Thread(target=publish)
        worker.start(); worker.join()
        self.assertEqual(root.counter, scheduled)
        self.assertEqual(len(root.scheduled), 1)
        root.tick()
        overlay.update_frame.assert_called_once()
        self.assertEqual(overlay.update_frame.call_args.args[0]["sequence_id"], 999)
        bridge.close()
        self.assertFalse(root.scheduled)
        overlay.stop.assert_called()

    def test_clear_rejects_older_episode_and_run(self):
        root, overlay = FakeRoot(), Mock()
        bridge = TrainingVisualizationBridge(root, overlay=overlay)
        bridge.start("run")
        bridge.post("run", "visualization", self.frame())
        bridge.post("run", "visualization_clear", {"generation": 3})
        stale = self.frame(); stale["sequence_id"] = 999
        bridge.post("run", "visualization", stale)
        root.tick()
        overlay.update_frame.assert_not_called()
        bridge.start("next")
        bridge.post("run", "visualization", {**self.frame(), "generation": 50})
        root.tick()
        overlay.update_frame.assert_not_called()
        bridge.close()

    def test_publisher_limits_frequency_and_includes_runtime_identity(self):
        controller = SimpleNamespace(config=DoubleDQNConfig(
            env=EnvConfig(simulation_mode="continuous"),
            runtime=TrainingRuntimeConfig(visualize_q=True, ui_update_interval_ms=100)),
            state="RUNNING", _emit=Mock())
        visual = TrainingVisualization(controller)
        visual.bind({"runtime_process_id": 123, "runtime_instance_id": "slot-0"}, episode_id=4)
        data = {"top5": [], "noop_q": 1.0, "scene_id": 0, "selected_action": 0}
        with patch("train.dqn.visualization.time.monotonic", side_effect=[10, 10.01, 10.2]):
            for sequence in range(3):
                visual.publish(data, sequence_id=sequence, policy_version=7)
        frames = [call.kwargs for call in controller._emit.call_args_list if call.args[0] == "visualization"]
        self.assertEqual(len(frames), 2)
        self.assertEqual(frames[-1]["sequence_id"], 2)
        self.assertEqual(frames[-1]["process_id"], 123)
        self.assertEqual(frames[-1]["episode_id"], 4)
        self.assertEqual(frames[-1]["instance_id"], "slot-0")
        controller.state = "PAUSING"
        self.assertFalse(visual.enabled())

    def test_destroyed_worker_window_does_not_stop_future_polling(self):
        root, overlay = FakeRoot(), Mock()
        bridge = TrainingVisualizationBridge(root, overlay=overlay)
        bridge.start("run")
        overlay.bind_target.side_effect = [ValueError("destroyed window"), None]
        bridge.post("run", "visualization", self.frame())
        with self.assertLogs("workbench.gui.training_visualization", level="WARNING"):
            root.tick()
        overlay.update_frame.assert_not_called()
        self.assertEqual(len(root.scheduled), 1)
        bridge.post("run", "visualization", {**self.frame(), "sequence_id": 1})
        root.tick()
        overlay.update_frame.assert_called_once()
        overlay.bind_target.side_effect = None
        bridge.close()


class BoundOverlayTest(unittest.TestCase):
    def test_disabling_bound_overlay_joins_renderer_thread(self):
        overlay = QHeatmapOverlay()
        overlay.bind_target(0xffffffff)
        overlay.set_enabled(True)
        try:
            self.assertTrue(overlay._thread.is_alive())
            overlay.set_enabled(False)
            self.assertIsNone(overlay._thread)
            self.assertFalse(overlay._running)
        finally:
            overlay.stop()

    def test_unbound_overlay_never_searches_an_arbitrary_game(self):
        overlay = QHeatmapOverlay()
        with patch.object(user32, "FindWindowW") as find, patch.object(user32, "EnumWindows") as enum:
            self.assertEqual(overlay._find_game_hwnd(), 0)
        find.assert_not_called(); enum.assert_not_called()

    def test_bound_overlay_requires_exact_pid_and_rejects_stale_frame(self):
        overlay = QHeatmapOverlay()
        with patch.object(overlay, "start"), patch.object(overlay, "_window_pid", return_value=123), \
                patch.object(user32, "IsWindowVisible", return_value=True):
            overlay.bind_target(123, 456)
            overlay.set_enabled(True)
            self.assertEqual(overlay._find_game_hwnd(), 456)
            frame = dict(process_id=123, generation=2, episode_id=3, sequence_id=10,
                         top5=[], noop_q=1.25, selected_action=0)
            self.assertTrue(overlay.update_frame(frame))
            self.assertFalse(overlay.update_frame({**frame, "episode_id": 2, "sequence_id": 999}))
            self.assertFalse(overlay.update_frame({**frame, "process_id": 222, "sequence_id": 11}))
            self.assertEqual(overlay._noop_q, 1.25)
        with patch.object(overlay, "_window_pid", return_value=222):
            with self.assertRaisesRegex(ValueError, "bound process"):
                overlay.bind_target(123, 789)

    def test_topmost_is_opt_in_and_never_activates_window(self):
        overlay = QHeatmapOverlay()
        with patch.object(user32, "SetWindowPos", return_value=True) as position, \
                patch.object(user32, "GetWindowLongPtrW", return_value=0), \
                patch.object(user32, "IsWindow", return_value=True):
            overlay._apply_topmost(123)
            position.assert_not_called()
            overlay._topmost_requested = True
            overlay._apply_topmost(123)
            self.assertEqual(position.call_args.args[:2], (123, -1))
            self.assertTrue(position.call_args.args[-1] & 0x0010)
            overlay._topmost_requested = False
            overlay._apply_topmost(123)
            self.assertEqual(position.call_args.args[:2], (123, -2))


class ConfirmedTrainingControlTest(unittest.TestCase):
    def controller(self):
        controller = DoubleDQNTrainerController(DoubleDQNConfig(), eval_every_episodes=0)
        controller._state = "RUNNING"
        return controller

    def until(self, predicate):
        deadline = time.monotonic() + 3
        while not predicate() and time.monotonic() < deadline:
            time.sleep(0.005)
        self.assertTrue(predicate(), "control acknowledgement timed out")

    def test_pause_and_resume_only_report_success_after_runtime_ack(self):
        controller = self.controller()
        pause_gate, resume_gate = threading.Event(), threading.Event()
        calls = []
        def hook(paused):
            calls.append((paused, threading.get_ident()))
            (pause_gate if paused else resume_gate).wait(3)
            return [{"paused": paused, "game_clock": 150}]
        controller._runtime_pause_hook = hook
        controller.pause()
        self.assertEqual(controller.state, "PAUSING")
        result = []
        worker = threading.Thread(target=lambda: result.append(controller._wait_if_paused()))
        worker.start()
        self.until(lambda: bool(calls))
        self.assertEqual(controller.state, "PAUSING")
        pause_gate.set()
        self.until(lambda: controller.state == "PAUSED")
        controller.resume()
        self.assertEqual(controller.state, "RESUMING")
        resume_gate.set(); worker.join(3)
        self.assertFalse(worker.is_alive())
        self.assertEqual(controller.state, "RUNNING")
        self.assertEqual(result, [True])
        self.assertTrue(all(thread != threading.get_ident() for _, thread in calls))

    def test_pause_reset_continue_keeps_objects_and_stop_unblocks(self):
        controller = self.controller()
        controller._agent, controller._replay = object(), object()
        agent, replay = controller._agent, controller._replay
        controller._runtime_pause_hook = lambda paused: [{"paused": paused}]
        reset = Mock(return_value=[{"paused": True, "game_clock": 0}])
        controller._runtime_reset_hook = reset
        controller.pause()
        result = []
        worker = threading.Thread(target=lambda: result.append(controller._wait_if_paused()))
        worker.start()
        self.until(lambda: controller.state == "PAUSED")
        controller.reset_game()
        self.until(lambda: reset.called and controller.state == "PAUSED")
        self.assertIs(controller._agent, agent); self.assertIs(controller._replay, replay)
        controller.stop(); worker.join(3)
        self.assertEqual(result, [False])
        self.assertFalse(worker.is_alive())


if __name__ == "__main__":
    unittest.main()
