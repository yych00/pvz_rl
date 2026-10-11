"""Bounded spawn-process environment workers; the learner owns every policy."""
from __future__ import annotations
import multiprocessing as mp
import queue
import time
import traceback
from dataclasses import replace
import numpy as np
from train.runtime import SampleIdentity, PolicyRequest, PolicyReply


def configure_game_window(process_id, topmost=False):
    """Set one verified PID's Z-order without showing, restoring or focusing it."""
    if not process_id:
        return None
    import ctypes
    from ctypes import wintypes
    user = ctypes.WinDLL('user32', use_last_error=True)
    callback_type = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
    user.GetWindowThreadProcessId.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.DWORD)]
    user.GetWindowThreadProcessId.restype = wintypes.DWORD
    user.GetWindow.argtypes = [wintypes.HWND, wintypes.UINT]
    user.GetWindow.restype = wintypes.HWND
    user.EnumWindows.argtypes = [callback_type, wintypes.LPARAM]
    user.EnumWindows.restype = wintypes.BOOL
    user.SetWindowPos.argtypes = [wintypes.HWND, wintypes.HWND, ctypes.c_int, ctypes.c_int,
                                 ctypes.c_int, ctypes.c_int, wintypes.UINT]
    user.SetWindowPos.restype = wintypes.BOOL
    windows = []
    @callback_type
    def visit(window, _):
        pid = wintypes.DWORD()
        user.GetWindowThreadProcessId(window, ctypes.byref(pid))
        if pid.value == process_id and not user.GetWindow(window, 4):
            windows.append(window)
        return True
    user.EnumWindows(visit, 0)
    if not windows:
        raise RuntimeError(f'game PID {process_id} has no verified top-level window')
    window = windows[0]
    if not user.SetWindowPos(window, wintypes.HWND(-1 if topmost else -2), 0, 0, 0, 0, 0x13):
        raise ctypes.WinError(ctypes.get_last_error())
    return int(window)


def native_environment(config, runtime, environment_id, dll_path, seed_types, *, max_episode_steps=None,
                       auto_select_seeds=True):
    from pvz_gym.client import PvZClient
    from pvz_gym.env import PvZEnv
    from pvz_gym.runtime.instances import GameInstance
    instance = None
    process_id = runtime.process_ids[environment_id] if runtime.process_ids else config.env.runtime.process_id
    if not runtime.game_executable and process_id is None:
        from pvz_gym.env_file import resolve_game_paths
        paths = resolve_game_paths(save_template=runtime.save_template, instance_root=runtime.instance_root)
        runtime = replace(runtime, game_executable=str(paths.executable),
                          save_template=str(paths.save_template), instance_root=str(paths.instance_root))
    if runtime.game_executable and not runtime.process_ids:
        instance = GameInstance(runtime.game_executable, runtime.instance_root, environment_id,
                                save_template=runtime.save_template, profile_id=runtime.profile_id)
        process_id = instance.start()
    client = None
    try:
        options = replace(config.env.runtime, process_id=process_id)
        deadline = time.monotonic() + runtime.worker_timeout_s
        client = PvZClient(dll_path=dll_path, runtime_config=options, auto_select_seeds=auto_select_seeds,
                           save_dir=instance.save_dir if instance else runtime.save_directory,
                           profile_id=instance.active_profile_id if instance else runtime.profile_id,
                           instance_id=instance.instance_id if instance else f"attached-{process_id}",
                           template_dir=instance.template_dir if instance else None,
                           clean_save=bool(instance or (runtime.save_directory and runtime.profile_id)))
        while not client.is_game_on() or client.get_raw_scene() == 0:
            if instance and instance.process.poll() is not None:
                raise RuntimeError(f"isolated game exited before initialization: {instance.process.returncode}")
            if time.monotonic() > deadline: raise TimeoutError("game instance did not reach an initialized scene")
            time.sleep(0.05)
        env = PvZEnv(client, level_id=config.env.level_id, playthrough=config.env.playthrough,
            seed_types=seed_types or config.env.seed_priority, seed_selection_mode=config.env.seed_selection_mode,
            ticks_per_step=config.env.ticks_per_step, max_episode_ticks=config.env.max_episode_ticks,
            max_episode_steps=max_episode_steps,
            simulation_mode=config.env.simulation_mode, runtime_config=options, turbo_config=config.env.turbo,
            reward_stage_ticks=1000 if config.reward_scheme == "staged" else None)
        env.runtime_process_id = process_id
        env.runtime_instance_id = instance.instance_id if instance else f"attached-{process_id}"
        env.runtime_window_handle = configure_game_window(process_id, runtime.game_topmost)
        return env, instance
    except BaseException:
        try:
            if client: client.close()
        finally:
            if instance: instance.close()
        raise


def set_environment_paused(env, paused, token):
    """Own only a user pause; never release a headless decision barrier."""
    client = getattr(env, "client", None)
    mode = getattr(env, "simulation_mode", "headless")
    pid = getattr(env, "runtime_process_id", None)
    if client is None:  # Offline environments model the ownership protocol.
        if callable(getattr(env, "pause" if paused else "resume", None)):
            getattr(env, "pause" if paused else "resume")()
        token["owned"] = paused
        return dict(command="pause" if paused else "resume", pause_owned=paused,
                    paused=paused, game_clock=None, runtime_process_id=pid)
    before = client.get_observation().global_state
    if paused:
        reset_facts = getattr(env, "_raw_observation", None)
        token["internal"] = bool(before.get("is_paused", False) and reset_facts is not None
                                 and (reset_facts.global_state.get("is_paused", False)
                                      or (getattr(env, "_steps", None) == 0
                                          and getattr(client, "deterministic_pause", False))))
    if paused and mode == "continuous" and not before.get("is_paused", False):
        if not client._dll.pvz_gym_pause():
            raise RuntimeError("native user pause failed")
        token["owned"] = True
        token["pause_mode"] = mode
    elif not paused and mode != "continuous":
        # A mode switch may have installed its own decision barrier while paused.
        token["owned"] = False
        token["internal"] = bool(before.get("is_paused", False))
    elif not paused and token.get("owned", False):
        if not client._dll.pvz_gym_resume():
            raise RuntimeError("native user resume failed")
        token["owned"] = False
    facts = client.get_observation().global_state
    if paused and not facts.get("is_paused", False):
        raise RuntimeError("native pause was not confirmed by the game")
    if not paused and mode == "continuous" and facts.get("is_paused", False) and not token.get("internal"):
        raise RuntimeError("game remains paused by another owner")
    return dict(command="pause" if paused else "resume", pause_owned=bool(token.get("owned")),
                paused=bool(facts.get("is_paused")), game_clock=facts.get("game_clock"),
                native_internal_pause=bool(token.get("internal")), runtime_process_id=pid)


def _worker(factory, args, config, environment_id, inbox, outbox, stop, rng_state=None, episode=0, generation=0, initial_seed=None):
    env = instance = None
    rng = np.random.default_rng(np.random.SeedSequence([config.seed, environment_id]))
    user_pause = {"owned": False}
    user_paused = False
    episode_terminal = False
    if rng_state is not None: rng.bit_generator.state = rng_state
    parent = mp.parent_process()

    def ensure_parent():
        if parent is not None and not parent.is_alive():
            raise InterruptedError("sampler parent exited")

    def send(kind, payload):
        while not stop.is_set():
            ensure_parent()
            try: outbox.put((kind, environment_id, payload, generation), timeout=0.1); return
            except queue.Full: continue
        raise InterruptedError("sampling stopped")

    def reset(seed=None):
        nonlocal episode, episode_terminal
        episode += 1
        episode_terminal = False
        began = time.perf_counter()
        user_pause["owned"] = False  # Reset creates a new Board and invalidates old ownership.
        reset_options = seed if isinstance(seed, dict) else {}
        if reset_options:
            seed = reset_options.get("seed")
        explicit_level = reset_options.get("level")
        if explicit_level is not None:
            env.level_id, env.playthrough = explicit_level
        elif getattr(config.env, "level_pool", None):
            try:
                from train.curriculum import sample_level_from_pool
            except ImportError:
                from train.curriculum import sample_level_from_pool
            probabilities = reset_options.get("level_probabilities")
            if probabilities is None:
                cur_level, cur_playthrough = sample_level_from_pool(config.env.level_pool, rng)
            else:
                pool = tuple(dict.fromkeys(config.env.level_pool))
                cur_level, cur_playthrough = pool[int(rng.choice(len(pool), p=probabilities))]
            env.level_id = cur_level
            env.playthrough = cur_playthrough
        episode_seed = int(rng.integers(0, 2**31-1)) if seed is None else seed
        state, info = env.reset(seed=episode_seed)
        if user_paused:
            set_environment_paused(env, True, user_pause)
        info = dict(info, runtime_reset_ms=(time.perf_counter()-began)*1000,
                    level_id=info.get("level_id", getattr(env, "level_id", config.env.level_id)),
                    playthrough=info.get("playthrough", getattr(env, "playthrough", config.env.playthrough)),
                    episode_seed=episode_seed,
                    runtime_process_id=instance.process.pid if instance else getattr(env, "runtime_process_id", config.env.runtime.process_id),
                    runtime_window_handle=getattr(env, "runtime_window_handle", None),
                    runtime_instance_id=instance.instance_id if instance else getattr(env, "runtime_instance_id", None))
        identity = SampleIdentity(environment_id, episode, 0)
        send("ready", (identity, state, info, rng.bit_generator.state))
        return identity

    try:
        created = factory(*args)
        env, instance = created if isinstance(created, tuple) else (created, None)
        identity = reset(initial_seed)
        while not stop.is_set():
            ensure_parent()
            try: command, payload = inbox.get(timeout=0.1)
            except queue.Empty: continue
            if command == "close": break
            if command in ("pause", "resume"):
                if episode_terminal:
                    # A terminated Board cannot acknowledge native pause. It has no
                    # further decisions until reset, which remains under controller control.
                    ack = dict(command=command, terminal_idle=True, pause_owned=False,
                        paused=command == "pause", game_clock=None,
                        runtime_process_id=getattr(env, "runtime_process_id", None))
                else:
                    ack = set_environment_paused(env, command == "pause", user_pause)
                user_paused = command == "pause"
                send("control_ack", dict(ack, environment_id=environment_id))
                continue
            if command == "reset": identity = reset(payload); continue
            if command != "step" or not isinstance(payload, PolicyReply) or payload.identity != identity:
                raise RuntimeError("sampler received a mismatched policy reply")
            began = time.perf_counter()
            state, reward, terminated, truncated, info = env.step(payload.action)
            episode_terminal = bool(terminated)
            info = dict(info, runtime_step_ms=(time.perf_counter()-began)*1000)
            from pvz_gym.profiling import bridge_timing
            if bridge_timing.enabled: info['runtime_profile'] = bridge_timing.snapshot()
            send("transition", (identity, payload.action, payload.policy_version, state, reward,
                                  terminated, truncated, info, rng.bit_generator.state))
            identity = SampleIdentity(environment_id, episode, identity.sequence_id+1)
    except InterruptedError:
        pass
    except BaseException:
        try: send("error", traceback.format_exc())
        except InterruptedError: pass
    finally:
        try:
            if env:
                try: env.close()
                except Exception: pass
        finally:
            if instance: instance.close()


class ProcessSampler:
    def __init__(self, config, *, dll_path=None, seed_types=None, factory=native_environment, factory_args=None):
        self.config = config
        self.runtime = config.runtime
        self.context = mp.get_context("spawn")
        self.output = self.context.Queue(maxsize=self.runtime.queue_capacity)
        self.stop = self.context.Event()
        self.workers, self.inputs, self.outstanding = {}, {}, {}
        self.restarts = {i: 0 for i in range(self.runtime.num_envs)}
        self.worker_episodes = {i: 0 for i in range(self.runtime.num_envs)}
        self.max_episodes_per_worker = getattr(self.runtime, "max_episodes_per_worker", 0)
        self.factory = factory
        self.arguments = factory_args
        self.dll_path, self.seed_types = dll_path, seed_types
        self.states = {}
        self.level_probabilities = None
        self.generations = {i: -1 for i in range(self.runtime.num_envs)}
        self._failed_workers: set[int] = set()
        self.paused = False
        self.closed = False

    def start(self, restored=None):
        for i in range(self.runtime.num_envs): self._start(i, (restored or {}).get(str(i)))

    def _start(self, i, restored=None, *, initial_seed=None):
        if initial_seed is None and self.level_probabilities is not None and self.config.env.level_pool:
            initial_seed = dict(seed=None, level_probabilities=list(self.level_probabilities))
        self.generations[i] += 1
        self._failed_workers.discard(i)
        # A fast peer can finish a short run before this worker reports ready.
        # Preserve every worker's initial seed/episode stream in checkpoints.
        self.states.setdefault(str(i), {"rng_state": (restored or {}).get("rng_state"),
                                       "episode": int((restored or {}).get("episode", 0))})
        inbox = self.context.Queue(maxsize=2)
        args = self.arguments[i] if self.arguments is not None else (
            self.config, self.runtime, i, self.dll_path, self.seed_types)
        process = self.context.Process(target=_worker, args=(self.factory, args, self.config, i, inbox,
            self.output, self.stop, (restored or {}).get("rng_state"), int((restored or {}).get("episode", 0)), self.generations[i], initial_seed),
            name=f"pvz-sampler-{i}")
        process.start()
        self.inputs[i], self.workers[i] = inbox, process
        self.outstanding[i] = time.monotonic()

    def receive(self, timeout=0.05):
        try:
            deadline = time.monotonic() + max(0, timeout)
            while True:
                kind, i, payload, generation = self.output.get(timeout=max(0, deadline-time.monotonic()))
                if generation != self.generations[i] or i in self._failed_workers: continue
                self.outstanding.pop(i, None)
                if kind == "error": self._failed_workers.add(i)
                return kind, i, payload
        except queue.Empty:
            now = time.monotonic()
            for i, process in self.workers.items():
                if i in self._failed_workers: continue
                alive = process.is_alive()
                elapsed = now-self.outstanding.get(i, now)
                if not alive or elapsed > self.runtime.worker_timeout_s:
                    # Report each failed generation once, even while evaluation
                    # defers recovery of a different environment.
                    self._failed_workers.add(i)
                    self.outstanding.pop(i, None)
                    reason = (f"sampler process exited (exitcode={process.exitcode})" if not alive else
                              f"sampler command timed out after {elapsed:.1f}s "
                              f"(worker_timeout_s={self.runtime.worker_timeout_s:g})")
                    return "error", i, reason
            return None

    def dispatch(self, reply: PolicyReply):
        if self.paused or self.closed: raise RuntimeError("sampler is paused or closed")
        if reply.identity.environment_id in self.outstanding: raise RuntimeError("an environment already has an outstanding command")
        self.inputs[reply.identity.environment_id].put(("step", reply), timeout=1)
        self.outstanding[reply.identity.environment_id] = time.monotonic()

    def reset(self, i, seed=None, *, level=None):
        if level is not None:
            seed = dict(seed=seed, level=tuple(level))
        elif self.level_probabilities is not None and self.config.env.level_pool:
            seed = dict(seed=seed, level_probabilities=list(self.level_probabilities))
        episodes = getattr(self, "worker_episodes", None)
        if episodes is not None:
            episodes[i] = episodes.get(i, 0) + 1
            max_episodes = getattr(self, "max_episodes_per_worker", None)
            if max_episodes and episodes[i] >= max_episodes:
                self.recycle_worker(i, seed=seed)
                return
        self.inputs[i].put(("reset", seed), timeout=1)
        self.outstanding[i] = time.monotonic()

    def recycle_worker(self, i, *, seed=None):
        episodes = getattr(self, "worker_episodes", None)
        if episodes is not None:
            episodes[i] = 0
        old = self.workers[i]
        try:
            self.inputs[i].put_nowait(("close", None))
        except (queue.Full, Exception):
            pass
        if hasattr(old, "join"):
            old.join(timeout=1.0)
        if hasattr(old, "is_alive") and old.is_alive():
            if hasattr(old, "terminate"):
                old.terminate()
            if hasattr(old, "join"):
                old.join(timeout=5)
        self.inputs[i].close()
        self._start(i, self.states.get(str(i)), initial_seed=seed)

    def restart(self, i):
        if self.restarts[i] >= self.runtime.restart_limit: raise RuntimeError(f"environment {i} exhausted restart limit")
        episodes = getattr(self, "worker_episodes", None)
        if episodes is not None:
            episodes[i] = 0
        old = self.workers[i]
        old.join(timeout=0.5)
        if old.is_alive(): old.terminate(); old.join(timeout=5)
        self.inputs[i].close()
        self.restarts[i] += 1
        self._start(i, self.states.get(str(i)))

    def close(self):
        if self.closed: return
        self.closed = True
        self.stop.set()
        for i, process in self.workers.items():
            try: self.inputs[i].put_nowait(("close", None))
            except queue.Full: pass
        for process in self.workers.values():
            process.join(timeout=8)
            if process.is_alive(): process.terminate(); process.join(timeout=5)
        for inbox in self.inputs.values(): inbox.close(); inbox.cancel_join_thread()
        self.output.close(); self.output.cancel_join_thread()

    def _control(self, command, on_message=None, stop_event=None):
        self.paused = True
        deadline = time.monotonic() + self.runtime.worker_timeout_s
        def stopped():
            return self.closed or self.stop.is_set() or (stop_event is not None and stop_event.is_set())
        # Consume every prior command result before asking workers to pause.
        while self.outstanding:
            if stopped(): raise InterruptedError("sampler control was stopped")
            if time.monotonic() >= deadline: raise TimeoutError("sampler control drain timed out")
            message = self.receive()
            if message is not None:
                if on_message is None: raise RuntimeError("control needs a callback for in-flight samples")
                on_message(message)
        pending = set(self.workers)
        for i in pending:
            if i in self._failed_workers: raise RuntimeError("cannot control a failed sampler worker")
            self.inputs[i].put((command, None), timeout=1)
            self.outstanding[i] = time.monotonic()
        acknowledgements = []
        while pending:
            if stopped(): raise InterruptedError("sampler control was stopped")
            if time.monotonic() >= deadline: raise TimeoutError("sampler native control timed out")
            message = self.receive()
            if message is None: continue
            kind, i, data = message
            if kind == "control_ack" and data["command"] == command and i in pending:
                pending.remove(i); acknowledgements.append(data)
            elif kind == "error":
                raise RuntimeError(f"sampler control failed for {i}: {data}")
            elif on_message is not None:
                on_message(message)
            else:
                raise RuntimeError("unexpected sample during native control")
        self.paused = command == "pause"
        return sorted(acknowledgements, key=lambda item: item["environment_id"])

    def pause(self, on_message=None, stop_event=None):
        return self._control("pause", on_message, stop_event)

    def resume(self, on_message=None, stop_event=None):
        return self._control("resume", on_message, stop_event)
