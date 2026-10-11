"""Central DQN learner over the shared bounded process sampler."""
from __future__ import annotations
import json
import time
from collections import deque
from contextlib import nullcontext
from dataclasses import replace
from pathlib import Path
from typing import Any
from pvz_gym.action_space import ACTION_SPACE_SIZE
from train.runtime import PolicyReply, SampleIdentity
from train.sampler import ProcessSampler
from train.curriculum import episode_level_metadata
from .credit import EpisodeEventCredit
from .features import normalize_observation
from .reward import mower_event_count
from .base_controller import _require_mask
from pvz_gym.profiling import TimingRecorder, bridge_timing


def _evaluate_parallel(
    controller,
    sampler,
    action_dim: int,
    *,
    env_step: int,
    timing: TimingRecorder,
    protocol=None,
) -> dict[str, Any]:
    """Evaluate agent across all available worker environments concurrently."""
    c = controller
    agent = c._agent
    if agent is None:
        raise RuntimeError("trainer has not created an agent yet")

    full = (protocol or {}).get("full", getattr(c, "_is_full_eval_episode", lambda: True)())
    tiered = getattr(c, "_tiered_eval", False)
    eval_count = (c.full_eval_seed_count if full else c.fast_eval_seed_count) if tiered else c.eval_seed_count
    eval_kind = "full" if (not tiered or full) else "fast"
    from .evaluation_plan import evaluation_plan, protocol_fields, record_level_result
    tasks = evaluation_plan(c, eval_count)
    requested = len(tasks)
    level_results = {}
    seed_start = tasks[0][0]

    wins = losses = unknown = total_steps = total_ticks = completed = 0
    total_score = 0.0
    total_training_score = 0.0
    zero_mower_wins = win_mowers = all_mowers = 0
    stopped = False
    policy_version = agent.update_count
    retries = 0
    started = time.perf_counter()

    c._emit(
        "evaluation_start",
        env_step=env_step,
        episode=c._summary.episodes,
        seed_start=seed_start,
        requested_episodes=requested,
        evaluation_mode=protocol_fields(c, requested)['evaluation_mode'],
        parallel_envs=min(c.config.runtime.num_envs, requested),
        policy_version=policy_version,
        kind=eval_kind,
    )

    seed_queue = deque(tasks)
    eval_active_seeds: dict[int, int] = {}
    eval_active_levels = {}
    eval_episodes: dict[int, dict[str, Any]] = {}
    eval_in_flight: dict[int, tuple[Any, Any]] = {}
    eval_ready: dict[int, bool] = {}
    num_envs = c.config.runtime.num_envs
    recovering = set()

    def reset_evaluation(env_id, seed):
        if getattr(c, '_legacy_evaluation_protocol', False) and not c.config.env.level_pool:
            sampler.reset(env_id, seed=seed)
        else:
            sampler.reset(env_id, seed=seed, level=eval_active_levels[env_id])

    for env_id in range(num_envs):
        if not seed_queue:
            break
        seed, pair = seed_queue.popleft()
        eval_active_seeds[env_id] = seed
        eval_active_levels[env_id] = pair
        reset_evaluation(env_id, seed)

    pending_resets = {}

    def consume(message):
        nonlocal wins, losses, unknown, total_steps, total_ticks, completed
        nonlocal total_score, total_training_score, zero_mower_wins, win_mowers, all_mowers, retries
        if message is not None:
            kind, env_id, payload = message
            if kind == "error":
                error = f"Worker {env_id} error: {payload}"
                retries += 1
                c._emit("evaluation_retry", environment_id=env_id,
                        seed=eval_active_seeds.get(env_id), error=error)
                eval_in_flight.pop(env_id, None)
                eval_ready.pop(env_id, None)
                eval_episodes.pop(env_id, None)
                sampler.restart(env_id)
                recovering.add(env_id)
                return
            elif kind == "ready":
                identity, state, info, rng = payload
                sampler.states[str(env_id)] = {"rng_state": rng, "episode": identity.episode_id}
                if env_id in recovering:
                    recovering.remove(env_id)
                    if env_id in eval_active_seeds:
                        pending_resets[env_id] = eval_active_seeds[env_id]
                    return
                if env_id not in eval_active_seeds:
                    raise RuntimeError("unexpected worker reset during evaluation")
                if not bool(info.get("replay_eligible", True)):
                    raise RuntimeError("evaluation reset received an unreliable observation")
                if not getattr(c, '_legacy_evaluation_protocol', False):
                    actual = (info.get('level_id'), info.get('playthrough'))
                    if actual != tuple(eval_active_levels[env_id]):
                        raise RuntimeError(f"evaluation level mismatch: {actual} != {eval_active_levels[env_id]}")
                reward_tracker = None
                if c.config.reward_scheme == "transition_dense":
                    from .transition_reward import TransitionDenseReward
                    reward_tracker = TransitionDenseReward(info, shaping_limit_mode=c.config.shaping_limit_mode)
                    state = reward_tracker.observe(state)
                elif c.config.reward_scheme == "staged":
                    from .staged_reward import StagedReward
                    reward_tracker = StagedReward(info)
                state = normalize_observation(state, boss_projectiles=c.config.boss_projectiles)
                mask = _require_mask(info, action_dim)
                eval_episodes[env_id] = {
                    "identity": identity,
                    "started": time.perf_counter(),
                    "state": state,
                    "mask": mask,
                    "steps": 0,
                    "ticks": 0,
                    "score": 0.0,
                    "training_score": 0.0,
                    "mowers": 0,
                    "first_mower_seen": False,
                    "first_mower_tick": None,
                    "reward_tracker": reward_tracker,
                    "seed": eval_active_seeds[env_id],
                    "level_pair": (info.get("level_id", eval_active_levels[env_id][0]),
                                   info.get("playthrough", eval_active_levels[env_id][1])),
                }
                eval_ready[env_id] = True
            elif kind == "transition":
                identity, action, version, state, reward, terminated, truncated, info, rng = payload
                expected = eval_in_flight.pop(env_id, None)
                if expected != (identity, action) or version != policy_version:
                    raise RuntimeError("stale or mismatched evaluation transition")
                if not bool(info.get("replay_eligible", True)):
                    raise RuntimeError("evaluation received an unreliable observation")
                sampler.states[str(env_id)] = {"rng_state": rng, "episode": identity.episode_id}
                ep = eval_episodes.get(env_id)
                if ep is not None:
                    if ep["reward_tracker"] is not None:
                        at_step_limit = (c.eval_max_episode_steps is not None and ep["steps"] + 1 >= c.eval_max_episode_steps)
                        ep["training_score"] += ep["reward_tracker"].step(
                            reward, info, terminated=terminated, truncated=truncated or at_step_limit
                        )
                    ticks = int(info.get("ticks_advanced", c.config.env.ticks_per_step))
                    ep["ticks"] += ticks
                    mowers = mower_event_count(info.get("events"))
                    ep["mowers"] += mowers
                    if not ep["first_mower_seen"] and mowers > 0:
                        ep["first_mower_seen"] = True
                        ep["first_mower_tick"] = ep["ticks"]
                    if c.config.reward_scheme == "transition_dense":
                        state = ep["reward_tracker"].observe(state)
                    ep["state"] = normalize_observation(state, boss_projectiles=c.config.boss_projectiles)
                    ep["mask"] = _require_mask(info, action_dim)
                    ep["score"] += float(reward)
                    ep["steps"] += 1
                    result = str(info.get("result", "UNKNOWN"))
                    done = bool(terminated or truncated) or (
                        c.eval_max_episode_steps is not None and ep["steps"] >= c.eval_max_episode_steps
                    )
                    if done:
                        completed += 1
                        record_level_result(level_results, ep["level_pair"], result)
                        is_won = (result == "WON")
                        is_lost = (result == "LOST")
                        wins += int(is_won)
                        losses += int(is_lost)
                        unknown += int(not is_won and not is_lost)
                        zero_mower_wins += int(is_won and ep["mowers"] == 0)
                        win_mowers += ep["mowers"] if is_won else 0
                        all_mowers += ep["mowers"]
                        total_score += ep["score"]
                        total_training_score += ep["training_score"]
                        total_steps += ep["steps"]
                        total_ticks += ep["ticks"]

                        c._emit(
                            "evaluation_episode",
                            env_step=env_step,
                            episode=c._summary.episodes,
                            seed=ep["seed"],
                            level_id=ep["level_pair"][0], playthrough=ep["level_pair"][1],
                            environment_id=env_id,
                            policy_version=policy_version,
                            parallel_envs=min(num_envs, requested),
                            wall_time_ms=(time.perf_counter()-ep["started"])*1000,
                            result=result,
                            score=ep["score"],
                            steps=ep["steps"],
                            episode_ticks=ep["ticks"],
                            first_mower_tick=ep["first_mower_tick"],
                            mowers_used=ep["mowers"],
                            completed=completed,
                            requested_episodes=requested,
                            **({"training_score": ep["training_score"],
                                "reward_breakdown": ep["reward_tracker"].last_breakdown} if ep["reward_tracker"] is not None else {}),
                        )
                        eval_episodes.pop(env_id, None)
                        eval_ready.pop(env_id, None)
                        eval_active_seeds.pop(env_id, None)

                        if seed_queue and not c.stop_event.is_set():
                            next_seed, pair = seed_queue.popleft()
                            eval_active_seeds[env_id] = next_seed
                            eval_active_levels[env_id] = pair
                            pending_resets[env_id] = next_seed
                    else:
                        eval_ready[env_id] = True


    previous_pause_hook = c._runtime_pause_hook

    def pause_evaluation(paused):
        operation = sampler.pause if paused else sampler.resume
        return operation(on_message=consume, stop_event=c.stop_event)

    c._runtime_pause_hook = pause_evaluation
    try:
        while (completed < requested or recovering or sampler.outstanding) and not c.stop_event.is_set():
            if not c._wait_if_paused():
                stopped = True
                break
            for env_id, seed in list(pending_resets.items()):
                reset_evaluation(env_id, seed)
                pending_resets.pop(env_id)
            consume(sampler.receive(0.005))
            deadline = time.monotonic() + c.config.runtime.inference_max_wait_ms / 1000
            while eval_ready and len(eval_ready) < c.config.runtime.inference_batch_size:
                message = sampler.receive(max(0, deadline-time.monotonic()))
                if message is None:
                    break
                consume(message)
                if time.monotonic() >= deadline:
                    break

            ready_indices = [eid for eid in eval_ready if eid not in eval_in_flight][
                :c.config.runtime.inference_batch_size]
            if ready_indices and not c.stop_event.is_set():
                policy = agent
                states = [eval_episodes[eid]["state"] for eid in ready_indices]
                masks = [eval_episodes[eid]["mask"] for eid in ready_indices]
                with c._agent_lock:
                    policy.config = c.config
                    actions = timing.measure(
                        "evaluation_inference",
                        policy.select_actions,
                        states,
                        masks,
                        env_step=env_step,
                        explore=False,
                    )
                    version = policy.update_count
                for eid, act in zip(ready_indices, actions):
                    ep = eval_episodes[eid]
                    eval_in_flight[eid] = (ep["identity"], act)
                    eval_ready.pop(eid, None)
                    sampler.dispatch(PolicyReply(ep["identity"], act, version))
                    ep["identity"] = replace(ep["identity"], sequence_id=ep["identity"].sequence_id + 1)

    finally:
        c._runtime_pause_hook = previous_pause_hook

    record = {
        "env_step": int(env_step),
        "episode": int(c._summary.episodes),
        "seed_start": seed_start,
        "requested_episodes": requested,
        "max_episode_steps": c.eval_max_episode_steps,
        "max_episode_ticks": c.config.env.max_episode_ticks,
        "episodes": completed,
        "wins": wins,
        "losses": losses,
        "unknown": unknown,
        "win_rate": wins / completed if completed else 0.0,
        "zero_mower_wins": zero_mower_wins,
        "zero_mower_win_rate": zero_mower_wins / completed if completed else 0.0,
        "mean_win_mowers": win_mowers / wins if wins else None,
        "mean_mowers": all_mowers / completed if completed else 0.0,
        "mean_score": total_score / completed if completed else 0.0,
        "mean_steps": total_steps / completed if completed else 0.0,
        "mean_episode_ticks": total_ticks / completed if completed else 0.0,
        "complete": completed == requested and not stopped and not c.stop_event.is_set(),
        "stopped": stopped or c.stop_event.is_set(),
        "error": None,
        "parallel_envs": min(num_envs, requested),
        "policy_version": policy_version,
        "retries": retries,
    }
    if c.config.reward_scheme in ("staged", "transition_dense"):
        record["mean_training_score"] = total_training_score / completed if completed else 0.0

    wall_time_ms = (time.perf_counter() - started) * 1000.0
    record["kind"] = eval_kind
    record.update(protocol_fields(c, requested), levels=list(level_results.values()))
    record["evaluation_wall_time_ms"] = wall_time_ms
    if tiered:
        c._emit(
            "evaluation_profile",
            kind=eval_kind,
            env_step=env_step,
            episode=c._summary.episodes,
            evaluation_wall_time_ms=wall_time_ms,
        )
    c._emit("evaluation_end", **record)
    return record


class RemoteEvaluationEnv:
    """Use one finished worker for fixed-policy eval; buffer other completions."""
    def __init__(self, sampler, environment_id, deferred, controller):
        self.sampler, self.environment_id, self.deferred, self.controller = sampler, environment_id, deferred, controller
        self.identity = None
        self.worker_error = None

    def _wait(self, expected):
        while not self.controller.stop_event.is_set():
            message = self.sampler.receive()
            if message is None: continue
            kind, i, payload = message
            if i != self.environment_id: self.deferred.append(message); continue
            if kind == "error":
                self.worker_error = payload
                raise RuntimeError(payload)
            if kind != expected: raise RuntimeError("unexpected evaluation worker message")
            identity = payload[0]
            self.sampler.states[str(i)] = {"rng_state": payload[-1], "episode": identity.episode_id}
            return payload
        raise InterruptedError("evaluation stopped")

    def reset(self, *, seed):
        level = (getattr(self, "level_id", self.controller.config.env.level_id),
                 getattr(self, "playthrough", self.controller.config.env.playthrough))
        if getattr(self.controller, '_legacy_evaluation_protocol', False) and not self.controller.config.env.level_pool:
            self.sampler.reset(self.environment_id, seed=seed)
        else:
            self.sampler.reset(self.environment_id, seed=seed, level=level)
        identity, state, info, _rng = self._wait("ready")
        self.identity = identity
        return state, info

    def step(self, action):
        self.sampler.dispatch(PolicyReply(self.identity, action, self.controller._agent.update_count))
        identity, _, _, state, reward, terminated, truncated, info, _rng = self._wait("transition")
        if identity != self.identity: raise RuntimeError("evaluation identity mismatch")
        self.identity = replace(identity, sequence_id=identity.sequence_id+1)
        return state, reward, terminated, truncated, info


def run_parallel(controller, *, sampler_factory=ProcessSampler):
    from .base_controller import _require_mask
    c = controller
    if c.config.runtime.num_envs > 1 and not (c.config.runtime.process_ids or c.config.runtime.game_executable) and sampler_factory is ProcessSampler:
        raise ValueError("multi-environment training needs distinct --process-ids or --game-exe for isolated launches")
    c.run_dir.mkdir(parents=True, exist_ok=True)
    metrics = c.run_dir / "episodes.jsonl"
    c._state = "RUNNING"; c.stop_event.clear(); c.pause_event.set(); c._emit("state", state=c._state)
    c._pause_acknowledged = False
    c._reset_requested = False
    c._visualization.clear()
    timing = TimingRecorder(True)
    sampler = None
    ready, episodes, in_flight, deferred = {}, {}, {}, deque()
    update_budget = 0
    learner = None
    learner_paused = False
    from .training_cadence import TrainingCadence
    cadence = TrainingCadence(c.config)
    pending_evaluation = None
    parked = set()
    last_ui = 0.0

    def learn():
        nonlocal update_budget
        if learner is not None:
            learner.check()
            if timing.measure("policy_sync", learner.sync_actor):
                c._emit("policy_sync", **learner.status())
            learner.set_policy_floor(in_flight)
            return
        replay = c._replay
        warmup = max(c.config.batch_size, min(c.config.learning_starts, c.config.replay_capacity))
        if len(replay) < warmup or not replay.can_sample(c.config.n_step): return
        while update_budget:
            if in_flight and c._agent.update_count >= min(v[1] for v in in_flight.values()) + c.config.runtime.max_policy_lag:
                break
            frac = min(1.0, c._summary.env_steps / max(1, c.config.per_beta_steps))
            beta = c.config.per_beta_start + frac*(c.config.per_beta_end-c.config.per_beta_start)
            with c._replay_lock:
                batch = timing.measure("replay_sample", replay.sample, c.config.batch_size,
                    n_step=c.config.n_step, gamma=c.config.gamma, gamma_reference_ticks=c.config.gamma_reference_ticks, beta=beta)
            with c._agent_lock: c._summary.last_loss = timing.measure("learn", c._agent.learn, batch)
            if replay.is_per and c._agent.last_td_errors is not None:
                replay.update_priorities(batch.indices, c._agent.last_td_errors)
            update_budget -= 1

    def has_deferred_failure(environment_id):
        return any(kind == "error" and i == environment_id for kind, i, _ in deferred)

    def ingest(message):
        nonlocal update_budget, last_ui, pending_evaluation
        kind, i, payload = message
        if kind == "error":
            c._visualization.clear()
            with c._replay_lock: c._replay.end_episode(stream_id=i)
            in_flight.pop(i, None); ready.pop(i, None); episodes.pop(i, None)
            c._emit("worker_recovery", environment_id=i, error=payload, attempt=sampler.restarts[i]+1,
                    restart_limit=c.config.runtime.restart_limit,
                    log_path=str((c.run_dir / "runtime_events.jsonl").resolve()))
            try:
                sampler.restart(i)
            except RuntimeError as exc:
                raise RuntimeError(f"{exc}; last worker error: {payload}") from exc
            return
        if kind == "ready":
            if has_deferred_failure(i): return
            identity, state, info, rng = payload
            state = normalize_observation(state, boss_projectiles=c.config.boss_projectiles)
            with c._replay_lock:
                credit = EpisodeEventCredit(c._replay, c.config, info, stream_id=i)
            state = credit.observe(state)
            episodes[i] = dict(credit=credit,
                metadata=episode_level_metadata(info, c.config.env),
                raw=0.0, training=0.0, mowers=0, invalid=0, steps=0, ticks=0, identity=identity,
                previous_replay_eligible=bool(info.get("replay_eligible", True)))
            ready[i] = (identity, state, _require_mask(info, ACTION_SPACE_SIZE))
            sampler.states[str(i)] = {"rng_state": rng, "episode": identity.episode_id}
            c._emit("seed_selection", environment_id=i, **info.get("seed_selection", {}))
            if c.config.runtime.num_envs == 1:
                c._visualization.bind(info, episode_id=identity.episode_id, environment_id=i)
            return
        if kind != "transition": raise RuntimeError(f"unknown sampler message {kind}")
        identity, action, version, state, reward, terminated, truncated, info, rng = payload
        expected = in_flight.pop(i, None)
        if expected is None or expected[0] != identity or expected[1] != version:
            raise RuntimeError("out-of-order or stale sampler transition")
        previous, mask = expected[2], expected[3]
        state = normalize_observation(state, boss_projectiles=c.config.boss_projectiles)
        next_mask = _require_mask(info, ACTION_SPACE_SIZE)
        ep = episodes[i]
        current_replay_eligible = bool(info.get("replay_eligible", True))
        transition_eligible = ep["previous_replay_eligible"] and current_replay_eligible
        with c._replay_lock:
            if transition_eligible:
                training_reward = ep["credit"].add(c._replay, previous, action, reward, state,
                                                    terminated, truncated, next_mask, info)
            else:
                c._replay.end_episode(stream_id=i)
                training_reward = 0.0
                ep["credit"] = EpisodeEventCredit(c._replay, c.config, info, stream_id=i)
                state = ep["credit"].observe(state)
        if not transition_eligible:
            c._visualization.clear()
            c._emit("transition_discarded", environment_id=i, episode_id=identity.episode_id,
                    sequence_id=identity.sequence_id, observation_quality=info.get("observation_quality"),
                    previous_observation_eligible=ep["previous_replay_eligible"],
                    error=info.get("observation_error"))
        ep["previous_replay_eligible"] = current_replay_eligible
        ep["raw"] += float(reward); ep["training"] += training_reward
        mowers = max(0, int((info.get("events") or {}).get("mowers_used", 0)))
        invalid = not bool((info.get("action_result") or {}).get("success", False))
        ep["mowers"] += mowers; ep["invalid"] += int(invalid)
        ep["steps"] += 1
        summary = c._summary
        summary.env_steps += 1; summary.replay_size = len(c._replay); summary.epsilon = c._agent.epsilon(summary.env_steps)
        summary.mowers_used += mowers; summary.invalid_actions += int(invalid)
        ticks = int(info.get("ticks_advanced", 0)); requested = int(info.get("ticks_requested", c.config.env.ticks_per_step))
        ep["ticks"] += ticks
        summary.tick_samples += 1; summary.ticks_advanced_sum += ticks; summary.ticks_advanced_sq_sum += ticks*ticks
        summary.long_tick_jumps += int(ticks > 2*requested)
        sampler.states[str(i)] = {"rng_state": rng, "episode": identity.episode_id}
        # Warmup has the same meaning as serial DQN: do not bank updates for
        # transitions received before the replay is ready for learning.
        warmup = max(c.config.batch_size, min(c.config.learning_starts, c.config.replay_capacity))
        if learner is not None and transition_eligible and len(c._replay) >= warmup:
            pending = learner.status()['pending_learning_updates']
            if cadence.update(c.config, summary.env_steps, pending):
                c._emit('training_cadence', effective_train_every=cadence.every,
                        pending_learning_updates=pending, env_step=summary.env_steps)
        with c._replay_lock:
            train_every = cadence.every if learner is not None else c.config.train_every
            eligible_update = (transition_eligible and summary.env_steps % train_every == 0
                and len(c._replay) >= warmup and c._replay.can_sample(c.config.n_step))
        if eligible_update:
            if learner is not None: learner.grant(c.config.gradient_steps)
            else: update_budget += c.config.gradient_steps
        done = bool(terminated or truncated)
        now = time.monotonic()
        if done or mowers or now-last_ui >= c.config.runtime.ui_update_interval_ms/1000:
            c._emit("step", environment_id=i, env_step=summary.env_steps, episode=summary.episodes+1,
                **ep["metadata"],
                episode_step=ep["steps"], result=info.get("result", "UNKNOWN"), game_clock=info.get("game_clock", 0),
                raw_reward=reward, training_reward=training_reward, reward_scheme=c.config.reward_scheme,
                reward_breakdown=info.get("reward_breakdown"), mowers_used=mowers,
                ticks_requested=requested, ticks_advanced=ticks, policy_version=version,
                policy_lag=c._agent.update_count-version,
                **(learner.status() if learner is not None else {}))
            last_ui = now
        if done:
            c._visualization.clear()
            result = str(info.get("result", "UNKNOWN"))
            summary.episodes += 1; summary.wins += int(result == "WON"); summary.losses += int(result == "LOST")
            summary.episode_lengths.append(ep["steps"]); summary.raw_episode_rewards.append(ep["raw"])
            summary.training_episode_rewards.append(ep["training"])
            record = dict(environment_id=i, environment_episode=identity.episode_id, episode=summary.episodes,
                **ep["metadata"],
                env_step=summary.env_steps, result=result, episode_steps=ep["steps"],
                episode_ticks=info.get("episode_ticks", ep["ticks"]), max_episode_ticks=c.config.env.max_episode_ticks,
                truncated=bool(truncated), truncation_reason=info.get("truncation_reason"),
                raw_episode_reward=ep["raw"], training_episode_reward=ep["training"],
                epsilon=summary.epsilon, replay_size=len(c._replay), loss=summary.last_loss,
                invalid_actions=ep["invalid"], mowers_used=ep["mowers"], first_mower_step=ep["credit"].first_mower_step,
                first_mower_tick=ep["credit"].first_mower_tick, credit_assignment=c.config.credit_assignment,
                win_rate=summary.win_rate, mean_ticks_advanced=summary.mean_ticks_advanced,
                std_ticks_advanced=summary.std_ticks_advanced, tick_samples=summary.tick_samples,
                long_tick_jumps=summary.long_tick_jumps, long_tick_jump_rate=summary.long_tick_jump_rate,
                reward_breakdown=info.get("reward_breakdown"), policy_version=version)
            if learner is not None:
                record.update(learner.status(), policy_lag=c._agent.update_count-version,
                              effective_train_every=cadence.every)
            with metrics.open("a", encoding="utf-8") as handle: handle.write(json.dumps(record, sort_keys=True)+"\n")
            c._emit("episode_end", **record)
            c._record_curriculum_episode(record)
            sampler.level_probabilities = (list(c._curriculum.probabilities)
                if c._curriculum.enabled else None)
            # A completion can precede its worker's error in the eval buffer.
            # Retain that transition, then let the queued recovery own its reset.
            if has_deferred_failure(i): return
            if c.config.runtime.num_envs > 1:
                if pending_evaluation is None and c._is_evaluation_due():
                    pending_evaluation = dict(full=getattr(c, "_is_full_eval_episode", lambda: True)())
                    c._emit("evaluation_pending", env_step=summary.env_steps,
                            parallel_envs=c.config.runtime.num_envs)
                if (pending_evaluation is not None and getattr(c, "_tiered_eval", False)
                        and c._is_full_eval_episode()):
                    pending_evaluation["full"] = True
                if pending_evaluation is not None:
                    episodes.pop(i, None)
                    ready.pop(i, None)
                    parked.add(i)
                    return
            elif c._is_evaluation_due() and not c.stop_event.is_set() and c.pause_event.is_set():
                remote = RemoteEvaluationEnv(sampler, i, deferred, c)
                try:
                    c._evaluation_active = True
                    with learner.quiesce() if learner is not None else nullcontext():
                        evaluated = timing.measure("evaluation", c._evaluate_current_agent, remote, ACTION_SPACE_SIZE, env_step=summary.env_steps)
                        sampling_state()
                        c._record_evaluation(evaluated, c.run_dir / "evaluations.jsonl")
                except InterruptedError:
                    if not c.stop_event.is_set(): raise
                except Exception as exc:
                    c._emit("evaluation_error", env_step=summary.env_steps, error=str(exc))
                finally:
                    c._evaluation_active = False
                sampler.states[str(i)]["episode"] = remote.identity.episode_id if remote.identity else identity.episode_id
                if remote.worker_error is not None:
                    if not c.stop_event.is_set(): ingest(("error", i, remote.worker_error))
                    return
            if summary.env_steps < c.config.total_steps and not c.stop_event.is_set(): sampler.reset(i)
        else:
            next_identity = replace(identity, sequence_id=identity.sequence_id+1)
            ready[i] = (next_identity, state, next_mask)

    def sampling_state():
        c._agent.sampling_state = {"version": 1, "workers": dict(sampler.states),
            "exact_game_resume": False, "pending_learning_updates": update_budget,
            "discarded_inflight": [vars(value[0]) for value in in_flight.values()],
            "discarded_ready": [vars(value[0]) for value in ready.values()],
            "discarded_pending_workers": sorted(sampler.outstanding),
            "training_architecture": c.config.training_architecture}
        c._agent.sampling_state['training_cadence'] = cadence.snapshot()
        if learner is not None:
            c._agent.sampling_state.update(learner.checkpoint_state())

    def save_snapshot(path, *, sync_latest=False):
        with learner.quiesce() if learner is not None else nullcontext():
            sampling_state()
            return c._save_scheduled_checkpoint(path, sync_latest=sync_latest)

    def receive_transition(message):
        timing.measure("ingest", ingest, message)
        learn()
        if message[0] != "transition": return
        step = c._summary.env_steps
        if c.config.checkpoint_every and step % c.config.checkpoint_every == 0:
            timing.measure("save_snapshot", save_snapshot,
                c.run_dir / f"checkpoint-{step:09d}.pt",
                sync_latest=bool(c.config.autosave_every and step % c.config.autosave_every == 0))
        elif c.config.autosave_every and step % c.config.autosave_every == 0:
            timing.measure("save_snapshot", save_snapshot, c.run_dir / "latest.pt")

    def pause_runtime(paused):
        nonlocal learner_paused
        if learner is not None and paused and not learner_paused:
            learner.suspend()
            learner_paused = True
        while deferred and not c.stop_event.is_set():
            receive_transition(deferred.popleft())
        operation = sampler.pause if paused else sampler.resume
        result = operation(on_message=receive_transition, stop_event=c.stop_event)
        if learner is not None and not paused and learner_paused:
            learner.resume()
            learner_paused = False
        return result

    def reset_runtime():
        nonlocal pending_evaluation
        pending_evaluation = None
        parked.clear()
        if in_flight:
            raise RuntimeError("Reset Game requires all in-flight decisions to drain")
        for i, episode in list(episodes.items()):
            with c._replay_lock: c._replay.end_episode(stream_id=i)
            c._emit("episode_aborted", environment_id=i, reason="user_reset",
                    episode_id=episode["identity"].episode_id, episode_steps=episode["steps"],
                    env_step=c._summary.env_steps)
        ready.clear(); episodes.clear()
        for i in range(c.config.runtime.num_envs):
            sampler.reset(i)
        while sampler.outstanding and not c.stop_event.is_set():
            message = sampler.receive()
            if message is not None:
                receive_transition(message)
        if c.stop_event.is_set():
            raise InterruptedError("reset stopped")
        return sampler.pause(on_message=receive_transition, stop_event=c.stop_event)

    try:
        c._agent, start_step, start_episode, payload = c._build_agent(ACTION_SPACE_SIZE)
        c._replay = c._restore_replay(action_dim=ACTION_SPACE_SIZE, start_step=start_step, start_episode=start_episode)
        c._summary.env_steps = start_step; c._summary.episodes = start_episode
        c._restore_metrics_history(metrics, start_episode); c._restore_evaluation_history(c.run_dir / "evaluations.jsonl", start_step)
        c._restore_curriculum(payload, start_episode)
        c._persist_run_config()
        if c.checkpoint_path is not None:
            c._emit("checkpoint_loaded", path=str(c.checkpoint_path),
                    env_step=start_step, episode=start_episode,
                    checkpoint_summary=(dict(payload.get("training_summary", {}))
                        if isinstance(payload.get("training_summary"), dict) else {}),
                    replay_restored=c._summary.replay_restored, replay_size=len(c._replay))
        if start_step < c.config.total_steps:
            sampler = sampler_factory(c.config, dll_path=c.dll_path, seed_types=c.seed_types)
            sampler.level_probabilities = (list(c._curriculum.probabilities)
                if c._curriculum.enabled else None)
            restored = dict(payload.get("sampling_state") or {})
            cadence = TrainingCadence(c.config, start_step, restored.get('training_cadence'))
            if not c._summary.replay_restored and restored.get("pending_learning_updates", 0):
                # Model-only/reward-incompatible resume restarts Replay warmup.
                # Old debt must not backpressure sampling before it can warm up.
                c._emit("learning_budget_reset", discarded_updates=int(restored["pending_learning_updates"]),
                        reason="Replay was not restored")
                restored["pending_learning_updates"] = 0
            update_budget = int(restored.get("pending_learning_updates", 0))
            if c.config.training_architecture == "actor_learner":
                from .actor_learner import AsyncActorLearner
                learner = AsyncActorLearner(c, restored=restored)
                c._async_learner = learner
                learner.start()
                c._emit("training_architecture", architecture="actor_learner", **learner.status())
            sampler.start(restored.get("workers", {}))
            c._runtime_pause_hook = pause_runtime
            c._runtime_reset_hook = reset_runtime
            while c._summary.env_steps < c.config.total_steps and c._wait_if_paused():
                if learner is not None: learn()
                message = deferred.popleft() if deferred else sampler.receive(0.002)
                if message is not None:
                    receive_transition(message)
                if c.stop_event.is_set(): break
                if (pending_evaluation is not None and len(parked) == c.config.runtime.num_envs
                        and not in_flight and not sampler.outstanding and not deferred):
                    c._evaluation_active = True
                    try:
                        with learner.quiesce() if learner is not None else nullcontext():
                            evaluated = timing.measure("evaluation", _evaluate_parallel,
                                c, sampler, ACTION_SPACE_SIZE, env_step=c._summary.env_steps,
                                timing=timing, protocol=pending_evaluation)
                            sampling_state()
                            c._record_evaluation(evaluated, c.run_dir / "evaluations.jsonl")
                    finally:
                        c._evaluation_active = False
                    pending_evaluation = None
                    parked.clear()
                    if c.stop_event.is_set(): break
                    if c._summary.env_steps < c.config.total_steps:
                        for i in range(c.config.runtime.num_envs): sampler.reset(i)
                if not ready: continue
                if not c.config.runtime.asynchronous and (in_flight or sampler.outstanding): continue
                if c.config.runtime.asynchronous:
                    deadline = time.monotonic()+c.config.runtime.inference_max_wait_ms/1000
                    while (len(ready) < c.config.runtime.inference_batch_size and time.monotonic() < deadline
                           and not c.stop_event.is_set()):
                        message = deferred.popleft() if deferred else sampler.receive(max(0, deadline-time.monotonic()))
                        if message is not None: receive_transition(message)
                if not c._wait_if_paused(): break
                if learner is not None:
                    learn()
                    if learner.backpressured():
                        learner.wait_for_progress()
                        continue
                remaining = c.config.total_steps-c._summary.env_steps-len(in_flight)
                indices = [i for i in ready if not has_deferred_failure(i)][:min(c.config.runtime.inference_batch_size, remaining)]
                if not indices: continue
                decisions = [ready[i] for i in indices]
                policy = learner.actor if learner is not None else c._agent
                with nullcontext() if learner is not None else c._agent_lock:
                    policy.config = c.config
                    actions = timing.measure("inference", policy.select_actions, [d[1] for d in decisions],
                        [d[2] for d in decisions], env_step=c._summary.env_steps, explore=True,
                        **({"visualize": True} if c._visualization.enabled() else {}))
                    version = policy.update_count
                if c.stop_event.is_set(): break
                visualizations = getattr(policy, "last_visualizations", ())
                for position, (i, decision, action) in enumerate(zip(indices, decisions, actions)):
                    if c._visualization.enabled() and position < len(visualizations):
                        c._visualization.publish(visualizations[position],
                            sequence_id=decision[0].sequence_id, policy_version=version)
                    sampler.dispatch(PolicyReply(decision[0], action, version))
                    in_flight[i] = (decision[0], version, decision[1], decision[2]); ready.pop(i)
                if learner is not None: learner.set_policy_floor(in_flight)
            c._summary.stopped = c.stop_event.is_set()
            discarded_inflight = [vars(value[0]) for value in in_flight.values()]
            sampler.close()
            in_flight.clear()
            with c._replay_lock:
                for i in range(c.config.runtime.num_envs): c._replay.end_episode(stream_id=i)
            if learner is not None:
                if not c._summary.stopped: learner.drain()
                learner.close()
                learner.sync_actor()
            elif not c._summary.stopped: learn()
            sampling_state()
            c._agent.sampling_state["discarded_inflight"] = discarded_inflight
            sampler = None
            c._summary.stopped = c.stop_event.is_set()
        c._state = "STOPPED" if c._summary.stopped else "FINISHED"
        destination = c.run_dir / ("stopped.pt" if c._summary.stopped else "final.pt")
        saved = c.save_checkpoint(destination); c._sync_latest_checkpoint(saved)
        c._emit("finished", state=c._state)
        return c._summary
    except BaseException as exc:
        c._state = "FAILED"; c._emit("error", error=str(exc), state=c._state); raise
    finally:
        c._runtime_pause_hook = c._runtime_reset_hook = None
        c._async_learner = None
        c._visualization.clear()
        try:
            if learner is not None: learner.close()
        finally:
            if sampler: sampler.close()
            c._flush_checkpoint_writer(close=True)
            timing.save(c.run_dir / "timing.json"); bridge_timing.save(c.run_dir / "bridge_timing.json")
            if learner is not None:
                learner.profiler.save_if_due(c.run_dir / "learner_timing.json", force=True)
