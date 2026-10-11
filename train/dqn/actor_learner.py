"""Independent inference weights and a bounded, credit-driven DQN learner.

The controller owns the actor and environment messages. Only the learner thread
touches optimizer/target state. Replay locks cover copies, never GPU work. A
single immutable CPU snapshot mailbox bounds publication memory.
"""
from __future__ import annotations

import copy
from contextlib import contextmanager, nullcontext
import threading

import torch

from pvz_gym.profiling import TimingRecorder
from .learner_timing import LearnerProfiler


class AsyncActorLearner:
    def __init__(self, controller, *, restored=None):
        self.c = controller
        agent = controller._agent
        self.actor = copy.copy(agent)
        # Do not construct another agent: its constructor resets global RNGs
        # and allocates an unnecessary target network and optimizer.
        self.actor.online = copy.deepcopy(agent.online).eval()
        self.actor.online.requires_grad_(False)
        self.actor.target = self.actor.optimizer = None
        self.actor.last_visualizations = []
        # Exploration RNG is used exclusively by the actor; the usual agent
        # checkpoint still captures it without changing the checkpoint schema.
        self.actor._rng = agent._rng
        restored = restored or {}
        saved_actor = restored.get("actor_policy")
        if saved_actor is not None:
            version = int(restored["actor_policy_version"])
            if not 0 <= version <= agent.update_count:
                raise ValueError("saved actor policy version exceeds learner progress")
            # Tightening the lag on resume is a supported runtime change.
            # Start from current learner weights if the old actor is too stale.
            if agent.update_count - version <= controller.config.runtime.max_policy_lag:
                self.actor.online.load_state_dict(saved_actor)
                self.actor.update_count = version
        self._condition = threading.Condition()
        self._pending = int(restored.get("pending_learning_updates", 0))
        if self._pending < 0:
            raise ValueError("pending_learning_updates must be non-negative")
        self._version = agent.update_count
        self._minimum_version = self.actor.update_count
        self._last_published = self.actor.update_count
        self._snapshot = None
        self._busy = False
        self._suspended = 0
        self._closed = False
        self._draining = False
        self._failure = None
        self.timing = TimingRecorder(True)
        self.profiler = LearnerProfiler(self.timing, agent.device)
        self._previous_profiler = agent.learner_profiler
        agent.learner_profiler = self.profiler
        self.thread = threading.Thread(target=self._run, name="pvz-dqn-learner", daemon=False)
        # Resume may contain newer learner weights than the saved actor.
        if self._version > self.actor.update_count:
            self._publish()

    def start(self):
        self.thread.start()

    def check(self):
        with self._condition:
            failure = self._failure
        if failure is not None:
            raise RuntimeError("Actor–Learner background training failed") from failure

    def status(self):
        self.check()
        with self._condition:
            return dict(learner_policy_version=self._version,
                        actor_policy_version=self.actor.update_count,
                        pending_learning_updates=self._pending)

    def grant(self, updates):
        with self._condition:
            self._pending += updates
            self._condition.notify_all()

    def set_policy_floor(self, in_flight):
        with self._condition:
            self._minimum_version = min([self.actor.update_count] + [v[1] for v in in_flight.values()])
            self._condition.notify_all()

    def backpressured(self):
        return self.status()["pending_learning_updates"] >= self.c.config.max_pending_updates

    def wait_for_progress(self, timeout=0.002):
        # Yield the GIL when sampling is backpressured. In particular, an
        # immediately empty receive() must not starve the learner's Python
        # optimizer/publication work with a tight polling loop.
        with self._condition:
            self._condition.wait(timeout=timeout)
        self.check()

    def sync_actor(self):
        self.check()
        with self._condition:
            snapshot, self._snapshot = self._snapshot, None
        if snapshot is None:
            return False
        version, weights = snapshot
        self.actor.online.load_state_dict(weights)
        self.actor.update_count = version
        return True

    def _publish(self):
        # Called only by the learner (or before start). CPU copies finish before
        # publishing, so neither inference nor later updates can mutate them.
        weights = {key: value.detach().to("cpu", copy=True)
                   for key, value in self.c._agent.online.state_dict().items()}
        with self._condition:
            self._snapshot = (self.c._agent.update_count, weights)
            self._last_published = self.c._agent.update_count

    def _allowed(self):
        return (self._pending > 0 and not self._suspended and
                (self._draining or (not self.c.stop_event.is_set() and self.c.pause_event.is_set()
                 and self._version < self._minimum_version + self.c.config.runtime.max_policy_lag)))

    def _run(self):
        try:
            device = self.c._agent.device
            # A separate CUDA stream allows actor inference and learner GPU
            # work to be scheduled independently, even on the same device.
            stream = torch.cuda.Stream(device=device) if device.type == "cuda" else None
            if stream is not None:
                stream.wait_stream(torch.cuda.current_stream(device))
            with torch.cuda.stream(stream) if stream is not None else nullcontext():
                while True:
                    with self._condition:
                        self._condition.wait_for(lambda: self._closed or self._allowed(), timeout=0.05)
                        if self._closed:
                            break
                        if not self._allowed():
                            continue
                        self._busy = True
                    try:
                        config = self.c.config
                        replay = self.c._replay
                        with self.profiler.lock("replay_lock_wait", self.c._replay_lock):
                            warmup = max(config.batch_size, min(config.learning_starts, config.replay_capacity))
                            if len(replay) < warmup or not replay.can_sample(config.n_step):
                                batch = None
                            else:
                                frac = min(1.0, self.c._summary.env_steps / max(1, config.per_beta_steps))
                                beta = config.per_beta_start + frac * (config.per_beta_end - config.per_beta_start)
                                batch = self.timing.measure("replay_sample", replay.sample, config.batch_size,
                                    n_step=config.n_step, gamma=config.gamma,
                                    gamma_reference_ticks=config.gamma_reference_ticks, beta=beta)
                                ids = replay._transition_ids[batch.indices].copy() if batch.indices is not None else None
                        if batch is None:
                            with self._condition:
                                self._condition.wait(timeout=0.01)
                            continue
                        with self.profiler.lock("agent_lock_wait", self.c._agent_lock):
                            self.c._summary.last_loss = self.timing.measure("learn", self.c._agent.learn, batch)
                            errors = self.c._agent.last_td_errors
                        if ids is not None and errors is not None:
                            with self.profiler.lock("priority_lock_wait", self.c._replay_lock):
                                # The ring can wrap while the GPU is training.
                                # Never assign an old sample's TD error to a new
                                # transition that has reused its slot.
                                valid = replay._transition_ids[batch.indices] == ids
                                if replay.is_per and valid.any():
                                    self.timing.measure("per_priority_update", replay.update_priorities,
                                                        batch.indices[valid], errors[valid])
                        with self._condition:
                            self._version = self.c._agent.update_count
                            self._pending -= 1
                            publish = (self._version - self._last_published >= self.c.config.policy_sync_interval
                                       or self._version >= self._minimum_version + self.c.config.runtime.max_policy_lag
                                       or self._pending == 0)
                        if publish:
                            self.timing.measure("policy_publish", self._publish)
                        if getattr(self.c, "run_dir", None) is not None:
                            self.profiler.save_if_due(self.c.run_dir / "learner_timing.json")
                    finally:
                        with self._condition:
                            self._busy = False
                            self._condition.notify_all()
                if stream is not None:
                    stream.synchronize()
        except BaseException as exc:
            with self._condition:
                self._failure = exc
                self._busy = False
                self._condition.notify_all()

    def suspend(self):
        with self._condition:
            self._suspended += 1
            while self._busy:
                self._condition.wait(timeout=0.05)
        self.check()

    def resume(self):
        with self._condition:
            self._suspended -= 1
            self._condition.notify_all()

    @contextmanager
    def quiesce(self):
        self.suspend()
        try:
            yield
        finally:
            self.resume()

    def checkpoint_state(self):
        """Called while quiescent, from the controller's owning thread."""
        return dict(self.status(),
                    actor_policy={key: value.detach().to("cpu", copy=True)
                                  for key, value in self.actor.online.state_dict().items()})

    def drain(self):
        with self._condition:
            self._draining = True
            self._condition.notify_all()
            while (self._pending or self._busy) and self._failure is None and not self.c.stop_event.is_set():
                self._condition.wait(timeout=0.05)
        self.check()

    def close(self):
        with self._condition:
            self._closed = True
            self._condition.notify_all()
        if self.thread.ident is not None:
            self.thread.join()
        self.profiler.flush(wait=True)
        if self.c._agent.learner_profiler is self.profiler:
            self.c._agent.learner_profiler = self._previous_profiler
        self.check()
