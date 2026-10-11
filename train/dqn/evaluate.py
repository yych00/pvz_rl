"""Evaluate a trained Double DQN checkpoint on a fixed seed set."""

from __future__ import annotations

import argparse
import logging
import math
import time
from collections import defaultdict
from dataclasses import dataclass, replace
from pathlib import Path
from threading import Event
from typing import Any, Callable

import numpy as np

from pvz_gym.client import PvZClient
from pvz_gym.env import PvZEnv
from pvz_gym.env_file import resolve_game_paths
from pvz_gym.runtime import RuntimeConfig
from train.runtime import TrainingRuntimeConfig
from types import SimpleNamespace

from .agent import DoubleDQNAgent
from .features import normalize_observation
from .reward import mower_event_count

logger = logging.getLogger(__name__)


class _TimingStats:
    _ORDER = (
        "reset",
        "select_action",
        "env_step",
        "normalize_observation",
    )

    def __init__(self) -> None:
        self.samples: dict[str, list[float]] = defaultdict(list)

    def add_ms(self, name: str, started: float) -> None:
        self.samples[name].append((time.perf_counter() - started) * 1000.0)

    def lines(self) -> list[str]:
        lines: list[str] = []
        for name in self._ORDER:
            values = self.samples.get(name, ())
            if not values:
                continue
            array = np.asarray(values, dtype=np.float64)
            lines.append(
                f"[BENCH] {name:24s} n={array.size:5d} "
                f"avg={array.mean():8.3f} ms "
                f"p50={np.percentile(array, 50):8.3f} "
                f"p95={np.percentile(array, 95):8.3f} "
                f"max={array.max():8.3f}"
            )
        return lines


@dataclass(frozen=True)
class EvaluationSummary:
    requested_episodes: int
    episodes: int
    wins: int
    losses: int
    unknown: int
    total_score: float
    total_steps: int
    stopped: bool = False
    tick_samples: int = 0
    ticks_advanced_sum: int = 0
    ticks_advanced_sq_sum: float = 0.0
    long_tick_jumps: int = 0
    wall_seconds: float = 0.0
    perf_lines: tuple[str, ...] = ()
    total_training_score: float = 0.0
    zero_mower_wins: int = 0
    win_mowers: int = 0

    @property
    def win_rate(self) -> float:
        return self.wins / self.episodes if self.episodes else 0.0

    @property
    def mean_score(self) -> float:
        return self.total_score / self.episodes if self.episodes else 0.0

    @property
    def zero_mower_win_rate(self) -> float:
        return self.zero_mower_wins / self.episodes if self.episodes else 0.0

    @property
    def mean_win_mowers(self) -> float | None:
        return self.win_mowers / self.wins if self.wins else None

    @property
    def mean_steps(self) -> float:
        return self.total_steps / self.episodes if self.episodes else 0.0

    @property
    def mean_ticks_advanced(self) -> float:
        return self.ticks_advanced_sum / self.tick_samples if self.tick_samples else 0.0

    @property
    def std_ticks_advanced(self) -> float:
        if not self.tick_samples:
            return 0.0
        variance = self.ticks_advanced_sq_sum / self.tick_samples - self.mean_ticks_advanced**2
        return math.sqrt(max(0.0, variance))

    @property
    def long_tick_jump_rate(self) -> float:
        return self.long_tick_jumps / self.tick_samples if self.tick_samples else 0.0

    @property
    def steps_per_second(self) -> float:
        return self.total_steps / self.wall_seconds if self.wall_seconds > 0.0 else 0.0

    @property
    def game_ticks_per_second(self) -> float:
        return self.ticks_advanced_sum / self.wall_seconds if self.wall_seconds > 0.0 else 0.0


EvaluationCallback = Callable[[str, dict[str, Any]], None]


def evaluate_checkpoint(
    checkpoint: str | Path,
    *,
    level_id: int | None = None,
    playthrough: int | None = None,
    ticks_per_step: int | None = None,
    seed_start: int = 10_001,
    seed_count: int = 20,
    max_episode_steps: int | None = None,
    max_episode_ticks: int | None = None,
    simulation_mode: str | None = None,
    device: str = "auto",
    dll_path: str | None = None,
    seed_types: tuple[int, ...] | None = None,
    seed_selection_mode: str | None = None,
    on_event: EvaluationCallback | None = None,
    stop_event: Event | None = None,
    training_runtime: TrainingRuntimeConfig | None = None,
    game_runtime: RuntimeConfig | None = None,
    visualize_q: bool | Callable[[], bool] | None = None,
) -> EvaluationSummary:
    """Evaluate greedily; None inherits the checkpoint tick limit, 0 disables it."""
    if seed_count <= 0:
        raise ValueError("seed_count must be positive")
    if max_episode_steps is not None and max_episode_steps <= 0:
        raise ValueError("max_episode_steps must be positive")
    if max_episode_ticks is not None and max_episode_ticks < 0:
        raise ValueError("max_episode_ticks must be non-negative")

    agent, config, _payload = DoubleDQNAgent.from_checkpoint(checkpoint, device=device)
    if training_runtime is not None:
        config = replace(config, runtime=training_runtime)
    if game_runtime is not None:
        config = replace(config, env=replace(config.env, runtime=game_runtime))
    display_mode = config.env.simulation_mode if simulation_mode is None else simulation_mode
    display_runtime = replace(config.runtime, num_envs=1, process_ids=config.runtime.process_ids[:1],
                              asynchronous=False, inference_batch_size=1)
    display_config = replace(config, runtime=display_runtime,
                             env=replace(config.env, simulation_mode=display_mode))
    from .visualization import TrainingVisualization
    display_owner = SimpleNamespace(config=display_config, state="RUNNING",
        _emit=lambda event, **data: on_event(event, data) if on_event is not None else None)
    visualization = TrainingVisualization(display_owner)

    def wants_visualization():
        enabled = config.runtime.visualize_q if visualize_q is None else (
            visualize_q() if callable(visualize_q) else visualize_q)
        return bool(enabled) and on_event is not None and str(display_mode).lower() == "continuous"

    def bind_visualization(info, episode_id):
        bound_info = dict(info)
        pid = getattr(env, "runtime_process_id", None) or config.env.runtime.process_id
        if not pid:
            read_snapshot = getattr(client, "read_snapshot", None)
            if callable(read_snapshot):
                try:
                    pid = read_snapshot().identity.process_id
                except Exception as exc:
                    if on_event is not None:
                        on_event("visualization_unavailable", {"error": str(exc)})
        if type(pid) is int:
            bound_info["runtime_process_id"] = pid
        for name in ("runtime_window_handle", "runtime_instance_id"):
            value = getattr(env, name, None)
            if isinstance(value, (str, int)):
                bound_info[name] = value
        visualization.bind(bound_info, episode_id=episode_id)
    if seed_types is None:
        from .run_state import load_checkpoint_seed_types
        seed_types = load_checkpoint_seed_types(checkpoint, config, _payload)
        if seed_types is None:
            seed_types = config.env.seed_priority
    target_level_id = int(level_id) if level_id is not None else int(config.env.level_id)
    instance = None
    if config.runtime.game_executable or config.runtime.process_ids:
        from train.sampler import native_environment
        options = replace(config.env, level_id=target_level_id,
            playthrough=config.env.playthrough if playthrough is None else playthrough,
            seed_selection_mode=config.env.seed_selection_mode if seed_selection_mode is None else seed_selection_mode,
            ticks_per_step=config.env.ticks_per_step if ticks_per_step is None else ticks_per_step,
            max_episode_ticks=config.env.max_episode_ticks if max_episode_ticks is None else max_episode_ticks or None,
            simulation_mode=config.env.simulation_mode if simulation_mode is None else simulation_mode)
        paths = resolve_game_paths(config.runtime.game_executable, config.runtime.save_template,
                                   config.runtime.instance_root)
        runtime = replace(config.runtime,num_envs=1,process_ids=config.runtime.process_ids[:1],
            instance_root=str(paths.instance_root.parent / 'evaluation'))
        env, instance = native_environment(replace(config,env=options),runtime,0,dll_path,
            seed_types,max_episode_steps=max_episode_steps)
        client = env.session.backend
    else:
        client = PvZClient(
            dll_path=dll_path,
            automation_enabled=True,
            auto_dialogue=True,
            auto_collect=True,
            auto_select_seeds=True,
            auto_skip_tutorial=True,
            background_running=True,
            deterministic_pause=True,
            runtime_config=config.env.runtime,
        )
        try:
            env = PvZEnv(
                client,
                level_id=target_level_id,
                playthrough=config.env.playthrough if playthrough is None else playthrough,
                seed_types=seed_types,
                seed_selection_mode=config.env.seed_selection_mode if seed_selection_mode is None else seed_selection_mode,
                ticks_per_step=config.env.ticks_per_step if ticks_per_step is None else ticks_per_step,
                max_episode_steps=max_episode_steps,
                max_episode_ticks=(config.env.max_episode_ticks if max_episode_ticks is None
                                   else max_episode_ticks or None),
                simulation_mode=config.env.simulation_mode if simulation_mode is None else simulation_mode,
                runtime_config=config.env.runtime,
                turbo_config=config.env.turbo,
                reward_stage_ticks=1000 if config.reward_scheme == "staged" else None,
            )
        except BaseException:
            client.close()
            raise
    wins = losses = unknown = total_steps = 0
    total_score = total_training_score = 0.0
    zero_mower_wins = win_mowers = 0
    tick_samples = ticks_advanced_sum = long_tick_jumps = 0
    ticks_advanced_sq_sum = 0.0
    completed = 0
    stopped = False
    timing_stats = _TimingStats()
    started = time.perf_counter()
    try:
        for offset in range(seed_count):
            if stop_event is not None and stop_event.is_set():
                stopped = True
                break
            seed = seed_start + offset

            timing = time.perf_counter()
            observation, info = env.reset(seed=seed)
            reward_tracker = None
            if config.reward_scheme == "transition_dense":
                from .transition_reward import TransitionDenseReward
                reward_tracker = TransitionDenseReward(info, shaping_limit_mode=config.shaping_limit_mode)
                observation = reward_tracker.observe(observation)
            elif config.reward_scheme == "staged":
                from .staged_reward import StagedReward
                reward_tracker = StagedReward(info)
            timing_stats.add_ms("reset", timing)

            timing = time.perf_counter()
            observation = normalize_observation(observation, boss_projectiles=config.boss_projectiles)
            timing_stats.add_ms("normalize_observation", timing)
            visualization.clear()
            visualization.binding = {}
            if wants_visualization():
                bind_visualization(info, offset + 1)

            mask = tuple(bool(value) for value in info.get("action_mask", ()))
            result = "UNKNOWN"
            steps = 0
            score = 0.0
            first_mower_seen = False
            first_mower_tick = None
            episode_ticks = 0
            episode_mowers = 0
            training_score = 0.0

            while True:
                if stop_event is not None and stop_event.is_set():
                    stopped = True
                    break

                dismiss_dialogs = getattr(client, "dismiss_battle_dialogs", None)
                if callable(dismiss_dialogs):
                    dismiss_dialogs(keep_paused=str(display_mode).lower() != "continuous")

                timing = time.perf_counter()
                display_owner._visualize_q = wants_visualization() and bool(info.get("replay_eligible", True))
                if display_owner._visualize_q and not visualization.binding.get("process_id"):
                    bind_visualization(info, offset + 1)
                action = agent.select_action(
                    observation, mask, env_step=0, explore=False,
                    **({"visualize": True} if display_owner._visualize_q else {}),
                )
                timing_stats.add_ms("select_action", timing)
                frames = getattr(agent, "last_visualizations", ())
                if display_owner._visualize_q and frames:
                    visualization.publish(frames[0], sequence_id=steps,
                                          policy_version=getattr(agent, "update_count", 0))

                timing = time.perf_counter()
                next_observation, reward, terminated, truncated, info = env.step(action)
                if not info.get("replay_eligible", True):
                    visualization.clear()
                if reward_tracker is not None:
                    training_score += reward_tracker.step(reward, info, terminated=terminated, truncated=truncated)
                    if config.reward_scheme == "transition_dense":
                        next_observation = reward_tracker.observe(next_observation)
                    info["reward_breakdown"] = dict(reward_tracker.last_breakdown)
                episode_mowers += mower_event_count(info.get("events"))
                first_event = not first_mower_seen and mower_event_count(info.get("events")) > 0
                first_mower_seen = first_mower_seen or first_event
                timing_stats.add_ms("env_step", timing)

                ticks_requested = max(
                    1,
                    int(info.get("ticks_requested", config.env.ticks_per_step) or 1),
                )
                ticks_advanced = max(
                    0,
                    int(info.get("ticks_advanced", ticks_requested) or 0),
                )
                tick_samples += 1
                episode_ticks += ticks_advanced
                if first_event:
                    first_mower_tick = episode_ticks
                ticks_advanced_sum += ticks_advanced
                ticks_advanced_sq_sum += float(ticks_advanced * ticks_advanced)
                if ticks_advanced > ticks_requested * 2:
                    long_tick_jumps += 1

                timing = time.perf_counter()
                observation = normalize_observation(next_observation, boss_projectiles=config.boss_projectiles)
                timing_stats.add_ms("normalize_observation", timing)

                mask = tuple(bool(value) for value in info.get("action_mask", ()))
                result = str(info.get("result", "UNKNOWN"))
                score += float(reward)
                steps += 1
                if terminated or truncated:
                    visualization.clear()
                    break

            if stopped:
                break

            completed += 1
            wins += int(result == "WON")
            losses += int(result == "LOST")
            unknown += int(result not in ("WON", "LOST"))
            total_score += score
            total_training_score += training_score
            zero_mower_wins += int(result == "WON" and episode_mowers == 0)
            win_mowers += episode_mowers if result == "WON" else 0
            total_steps += steps
            if on_event is not None:
                on_event(
                    "episode_end",
                    {
                        "episode": completed,
                        "episodes": seed_count,
                        "seed": seed,
                        "result": result,
                        "steps": steps,
                        "episode_ticks": episode_ticks,
                        "first_mower_tick": first_mower_tick,
                        "mowers_used": episode_mowers,
                        **({"training_score": training_score, "reward_breakdown": reward_tracker.last_breakdown}
                           if reward_tracker is not None else {}),
                        "max_episode_ticks": env.max_episode_ticks,
                        "truncated": bool(truncated),
                        "truncation_reason": info.get("truncation_reason"),
                        "score": score,
                        "wins": wins,
                        "losses": losses,
                        "win_rate": wins / completed,
                        "mean_score": total_score / completed,
                        "mean_ticks_advanced": ticks_advanced_sum / tick_samples,
                        "std_ticks_advanced": math.sqrt(
                            max(
                                0.0,
                                ticks_advanced_sq_sum / tick_samples
                                - (ticks_advanced_sum / tick_samples) ** 2,
                            )
                        ),
                        "long_tick_jumps": long_tick_jumps,
                        "long_tick_jump_rate": long_tick_jumps / tick_samples,
                    },
                )
    finally:
        try:
            visualization.clear()
        finally:
            try: env.close()
            finally:
                if instance: instance.close()

    wall_seconds = max(time.perf_counter() - started, 1e-9)
    perf_lines = timing_stats.lines()
    perf_lines.extend(
        (
            f"[BENCH] {'steps_per_second':24s} {total_steps / wall_seconds:12.2f}",
            f"[BENCH] {'game_ticks_per_second':24s} "
            f"{ticks_advanced_sum / wall_seconds:12.2f}",
            f"[BENCH] {'wall_seconds':24s} {wall_seconds:12.3f}",
        )
    )
    summary = EvaluationSummary(
        requested_episodes=seed_count,
        total_training_score=total_training_score,
        zero_mower_wins=zero_mower_wins,
        win_mowers=win_mowers,
        episodes=completed,
        wins=wins,
        losses=losses,
        unknown=unknown,
        total_score=total_score,
        total_steps=total_steps,
        stopped=stopped,
        tick_samples=tick_samples,
        ticks_advanced_sum=ticks_advanced_sum,
        ticks_advanced_sq_sum=ticks_advanced_sq_sum,
        long_tick_jumps=long_tick_jumps,
        wall_seconds=wall_seconds,
        perf_lines=tuple(perf_lines),
    )
    if on_event is not None:
        on_event("finished", {"summary": summary})
    return summary


def evaluate(
    checkpoint: str | Path,
    *,
    seed_start: int = 10_001,
    seed_count: int = 20,
    max_episode_steps: int | None = None,
    max_episode_ticks: int | None = None,
    device: str = "auto",
    simulation_mode: str | None = None,
    training_runtime: TrainingRuntimeConfig | None = None,
) -> float:
    def print_event(event: str, data: dict[str, Any]) -> None:
        if event == "episode_end":
            print(
                f"seed={data['seed']} result={data['result']} steps={data['steps']} "
                f"score={float(data['score']):.3f}"
            )

    summary = evaluate_checkpoint(
        checkpoint,
        seed_start=seed_start,
        seed_count=seed_count,
        max_episode_steps=max_episode_steps,
        max_episode_ticks=max_episode_ticks,
        device=device,
        on_event=print_event,
        simulation_mode=simulation_mode,
        training_runtime=training_runtime,
    )
    print(
        f"wins={summary.wins}/{summary.episodes} win_rate={summary.win_rate:.3f} "
        f"mean_score={summary.mean_score:.3f}"
    )
    return summary.win_rate


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a PvZ Double DQN checkpoint")
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--seed-start", type=int, default=10_001)
    parser.add_argument("--seed-count", type=int, default=20)
    parser.add_argument("--max-episode-steps", type=int, default=None)
    parser.add_argument(
        "--max-episode-ticks", "--max-window-ticks", type=int, default=None,
        help="override the checkpoint's episode tick budget; 0 disables the limit",
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--simulation-mode", choices=("stepped", "continuous", "headless", "turbo"), default=None)
    parser.add_argument("--game-exe", help="use this complete game instead of the repository's configured game")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    paths = resolve_game_paths(args.game_exe)
    runtime = TrainingRuntimeConfig(game_executable=str(paths.executable), save_template=str(paths.save_template),
                                    instance_root=str(paths.instance_root))
    evaluate(
        args.checkpoint,
        seed_start=args.seed_start,
        seed_count=args.seed_count,
        max_episode_steps=args.max_episode_steps,
        max_episode_ticks=args.max_episode_ticks,
        device=args.device,
        simulation_mode=args.simulation_mode,
        training_runtime=runtime,
    )


if __name__ == "__main__":
    main()
