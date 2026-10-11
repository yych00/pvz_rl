"""Keep the Workbench trainer-controller reference scoped to the active DQN run."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from uuid import uuid4


def install_training_controller_lifecycle(gui_cls: Any) -> None:
    if getattr(gui_cls, "_training_controller_lifecycle_installed", False):
        return

    original_start_training = gui_cls._start_training
    original_finish_training = gui_cls._finish_training
    original_apply_dqn_event = gui_cls._apply_dqn_training_event

    def _start_training_scoped(self):
        # Any controller left by a completed/failed DQN run is historical state,
        # not the controller for the next run. Clear it before parsing/starting a
        # new trainer so Random Legal buttons cannot be intercepted and a config
        # error cannot leave stale DQN save/control state enabled.
        if not getattr(self, "_training_running", False):
            self._training_controller = None
            self._training_run_token = uuid4().hex
            from .training_visualization import TrainingVisualizationBridge
            from .dqn_headless import get_q_heatmap_overlay
            bridge = getattr(self, "_training_visualization_bridge", None)
            if bridge is None:
                bridge = TrainingVisualizationBridge(self.root, overlay=get_q_heatmap_overlay())
                self._training_visualization_bridge = bridge
            bridge.start(self._training_run_token)
        result = original_start_training(self)
        if not getattr(self, "_training_running", False):
            bridge = getattr(self, "_training_visualization_bridge", None)
            if bridge is not None:
                bridge.close()
        return result

    def _finish_training_scoped(self, trainer_key: str, result):
        active_controller = (
            getattr(self, "_training_controller", None)
            if trainer_key == "double_dqn"
            else None
        )

        # Let the original handler read latest_checkpoint and update the UI first.
        original_finish_training(self, trainer_key, result)
        bridge = getattr(self, "_training_visualization_bridge", None)
        if bridge is not None:
            bridge.close()
        self._training_run_token = None

        if (
            trainer_key == "double_dqn"
            and getattr(self, "_training_controller", None) is active_controller
        ):
            self._training_controller = None
            try:
                dqn_selected = self._selected_training_spec().key == "double_dqn"
                self._set_training_checkpoint_controls(dqn_selected)
            except Exception:
                pass

    def _apply_dqn_training_event_scoped(self, event: str, data: dict):
        run_token = data.get("_workbench_run_token")
        if run_token is not None and run_token != getattr(self, "_training_run_token", None):
            return
        original_apply_dqn_event(self, event, data)
        if event != "checkpoint_loaded":
            return

        path = str(data.get("path", ""))
        if not path:
            return
        replay_restored = bool(data.get("replay_restored", False))
        replay_size = max(0, int(data.get("replay_size", 0) or 0))
        if replay_restored:
            replay_note = f"replay restored: {replay_size}"
        else:
            replay_note = "replay empty"
        self.lbl_training_checkpoint.config(
            text=f"{Path(path).name} · loaded · {replay_note}"
        )

    gui_cls._start_training = _start_training_scoped
    gui_cls._finish_training = _finish_training_scoped
    gui_cls._apply_dqn_training_event = _apply_dqn_training_event_scoped
    gui_cls._training_controller_lifecycle_installed = True


__all__ = ["install_training_controller_lifecycle"]
