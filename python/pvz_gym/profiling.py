"""Opt-in wall-clock timing; nested categories must not be added together."""
import json
import os
from collections import deque
from pathlib import Path
from time import perf_counter


class TimingRecorder:
    def __init__(self, enabled=None):
        self.enabled = os.environ.get("PVZ_GYM_PROFILE", "0") == "1" if enabled is None else enabled
        self._samples = {}
        self._totals = {}

    def add(self, name, milliseconds):
        if not self.enabled:
            return
        self._samples.setdefault(name, deque(maxlen=4096)).append(milliseconds)
        count, total = self._totals.get(name, (0, 0.0))
        self._totals[name] = count + 1, total + milliseconds

    def measure(self, name, call, *args, **kwargs):
        if not self.enabled:
            return call(*args, **kwargs)
        start = perf_counter()
        try:
            return call(*args, **kwargs)
        finally:
            self.add(name, (perf_counter() - start) * 1000)

    def snapshot(self):
        result = {}
        for name, values in self._samples.items():
            samples = sorted(values)
            count, total = self._totals[name]
            result[name] = {"count": count, "total_ms": total, "mean_ms": total / count,
                            "p50_ms": samples[int((len(samples) - 1) * .5)],
                            "p95_ms": samples[int((len(samples) - 1) * .95)],
                            "p99_ms": samples[int((len(samples) - 1) * .99)],
                            "percentile_window": len(samples)}
        return result

    def save(self, path):
        if self.enabled:
            Path(path).write_text(json.dumps(self.snapshot(), indent=2), encoding="utf-8")


bridge_timing = TimingRecorder()
