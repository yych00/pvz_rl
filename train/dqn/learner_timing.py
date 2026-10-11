"""Learner wall times and sampled CUDA stream intervals, without phase barriers.

CUDA intervals include launch gaps and stream contention, not just kernels.
Nested categories must not be added to their parent stages.
"""

from collections import deque
from contextlib import contextmanager, nullcontext
from pathlib import Path
from time import perf_counter


def phase(timer, name, *, cuda=True):
    return nullcontext() if timer is None else timer.phase(name, cuda=cuda)


class _UpdateTiming:
    def __init__(self, device, sample_cuda):
        self.device = device
        self.sample_cuda = sample_cuda
        self.wall = {}
        self.events = []

    @contextmanager
    def phase(self, name, *, cuda=True):
        pair = None
        if cuda and self.sample_cuda:
            import torch
            stream = torch.cuda.current_stream(self.device)
            pair = (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
            pair[0].record(stream)
        start = perf_counter()
        try:
            yield
        finally:
            self.wall[name] = self.wall.get(name, 0.0) + (perf_counter() - start) * 1000
            if pair is not None:
                pair[1].record(stream)
                self.events.append((name, *pair))


class LearnerProfiler:
    """One learner owns this recorder; CUDA is sampled every 32 updates.

    Repeated tensor subphases aggregate into one wall sample per update.
    """

    def __init__(self, recorder, device, *, cuda_sample_every=32):
        if cuda_sample_every < 1:
            raise ValueError("cuda_sample_every must be positive")
        self.recorder = recorder
        self.device = device
        self.cuda_sample_every = cuda_sample_every
        self._pending = deque()
        self._last_saved = None

    @contextmanager
    def update(self, update_count, *, force_cuda=False):
        if not self.recorder.enabled:
            yield None
            return
        self.flush()
        timer = _UpdateTiming(self.device, self.device.type == "cuda" and
                              (force_cuda or update_count % self.cuda_sample_every == 0))
        try:
            yield timer
        finally:
            for name, milliseconds in timer.wall.items():
                self.recorder.add("learn_" + name + "_wall", milliseconds)
            self._pending.extend(timer.events)
            self.flush()

    def flush(self, *, wait=False):
        # Training only queries. Waiting is reserved for thread shutdown.
        while self._pending:
            name, start, end = self._pending[0]
            if wait:
                end.synchronize()
            elif not end.query():
                break
            self.recorder.add("learn_" + name + "_cuda", start.elapsed_time(end))
            self._pending.popleft()

    def save_if_due(self, path, *, interval_s=10.0, force=False):
        """Publish an atomic snapshot from the learner thread, without a GPU wait."""
        if not self.recorder.enabled:
            return False
        now = perf_counter()
        if not force and self._last_saved is not None and now - self._last_saved < interval_s:
            return False
        self.flush()
        destination = Path(path)
        temporary = destination.with_name(destination.name + ".tmp")
        try:
            self.recorder.save(temporary)
            temporary.replace(destination)
        finally:
            temporary.unlink(missing_ok=True)
        self._last_saved = perf_counter()
        return True

    @contextmanager
    def lock(self, name, lock):
        self.recorder.measure(name, lock.acquire)
        try:
            yield
        finally:
            lock.release()
