"""Small, rate-limited policy snapshots without importing a desktop renderer."""
from __future__ import annotations

import time
from typing import Any


class TrainingVisualization:
    """One environment owns each snapshot; lifecycle changes invalidate old data."""

    def __init__(self, controller):
        self.controller = controller
        self.generation = 0
        self.binding: dict[str, Any] = {}
        self.last_publish = -float("inf")

    def enabled(self) -> bool:
        c = self.controller
        return (c.config.runtime.num_envs == 1
                and getattr(c, "state", "RUNNING") == "RUNNING"
                and str(c.config.env.simulation_mode).lower() == "continuous"
                and bool(getattr(c, "_visualize_q", getattr(c.config.runtime, "visualize_q", False))))

    def bind(self, info, *, episode_id: int, environment_id: int = 0):
        self.clear()
        self.binding = {
            "environment_id": int(environment_id),
            "episode_id": int(episode_id),
            "process_id": int(info.get("runtime_process_id") or 0),
            "window_handle": int(info.get("runtime_window_handle") or 0),
            "instance_id": str(info.get("runtime_instance_id") or ""),
            "worker_generation": int(info.get("runtime_worker_generation") or 0),
        }

    def clear(self):
        self.generation += 1
        self.last_publish = -float("inf")
        self.controller._emit("visualization_clear", generation=self.generation)

    def publish(self, data, *, sequence_id: int, policy_version: int):
        if not self.enabled() or not data or not self.binding.get("process_id"):
            return
        now = time.monotonic()
        interval = max(0.05, self.controller.config.runtime.ui_update_interval_ms / 1000)
        if now - self.last_publish < interval:
            return
        self.last_publish = now
        self.controller._emit(
            "visualization", **self.binding,
            generation=self.generation, sequence_id=int(sequence_id),
            policy_version=int(policy_version), topmost=bool(getattr(self.controller.config.runtime, "game_topmost", False)),
            **data,
        )


__all__ = ["TrainingVisualization"]
