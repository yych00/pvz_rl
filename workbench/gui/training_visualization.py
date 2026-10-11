"""A single-slot mailbox transfers training snapshots onto the Tk thread."""
from __future__ import annotations

import threading
import logging

from .q_heatmap_overlay import get_q_heatmap_overlay

logger = logging.getLogger(__name__)


class TrainingVisualizationBridge:
    def __init__(self, root, *, interval_ms=100, overlay=None):
        self.root = root
        self.interval_ms = max(50, int(interval_ms))
        self.overlay = overlay or get_q_heatmap_overlay()
        self._lock = threading.Lock()
        self._run_token = None
        self._pending = None
        self._last_key = None
        self._after = None
        if hasattr(root, "bind"):
            root.bind("<Destroy>", lambda event: self.close() if event.widget is root else None, add="+")

    def start(self, run_token):
        self.close()
        with self._lock:
            self._run_token = run_token
        self._after = self.root.after(self.interval_ms, self._poll)

    def post(self, run_token, event, data):
        """Worker threads only replace a snapshot; they never call Tk."""
        with self._lock:
            if self._run_token is None or run_token != self._run_token:
                return
            generation = int(data.get("generation", 0))
            key = (generation, int(data.get("worker_generation", 0)),
                   int(data.get("episode_id", 0)), int(data.get("sequence_id", -1)))
            if self._last_key is not None and key < self._last_key:
                return
            self._last_key = key
            self._pending = (event, dict(data))

    def _poll(self):
        self._after = None
        with self._lock:
            if self._run_token is None:
                return
            pending, self._pending = self._pending, None
        if pending is not None:
            event, frame = pending
            try:
                if event == "visualization_clear":
                    self.overlay.clear()
                elif int(frame.get("process_id") or 0) > 0:
                    self.overlay.bind_target(frame["process_id"], frame.get("window_handle") or None,
                                             topmost=bool(frame.get("topmost", False)))
                    self.overlay.update_frame(frame)
            except (ValueError, OSError) as exc:
                # A worker window may disappear between publication and polling.
                self.overlay.clear()
                logger.warning("training visualization frame rejected: %s", exc)
        with self._lock:
            active = self._run_token is not None
        if active:
            self._after = self.root.after(self.interval_ms, self._poll)

    def close(self):
        with self._lock:
            self._run_token = None
            self._pending = None
            self._last_key = None
        if self._after is not None:
            self.root.after_cancel(self._after)
            self._after = None
        self.overlay.clear()
        self.overlay.bind_target(None)
        self.overlay.stop()


__all__ = ["TrainingVisualizationBridge"]
