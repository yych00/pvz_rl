"""Reusable legal-random Gym integration runner for CLI and GUI."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np


@dataclass(frozen=True)
class RandomAgentConfig:
    episodes: int = 100
    max_steps: int = 2000
    freeze_streak: int = 3
    random_seed: int = 0

    def __post_init__(self):
        if self.episodes <= 0:
            raise ValueError("episodes must be positive")
        if self.max_steps <= 0:
            raise ValueError("max_steps must be positive")
        if self.freeze_streak <= 0:
            raise ValueError("freeze_streak must be positive")


@dataclass
class RandomAgentSummary:
    requested_episodes: int
    completed_episodes: int = 0
    wins: int = 0
    losses: int = 0
    timeouts: int = 0
    invalid_actions: int = 0
    invalid_observations: int = 0
    reset_failures: int = 0
    game_freezes: int = 0
    auto_collected: int = 0
    terminal_observation_fallbacks: int = 0
    stopped: bool = False
    episode_lengths: list[int] = field(default_factory=list)
    episode_rewards: list[float] = field(default_factory=list)

    @property
    def terminal_episodes(self) -> int:
        return self.wins + self.losses

    @property
    def win_rate(self) -> float:
        return self.wins / self.terminal_episodes if self.terminal_episodes else 0.0

    @property
    def mean_episode_length(self) -> float:
        return float(np.mean(self.episode_lengths)) if self.episode_lengths else 0.0

    @property
    def mean_reward(self) -> float:
        return float(np.mean(self.episode_rewards)) if self.episode_rewards else 0.0

    @property
    def integrity_failures(self) -> int:
        return (
            self.timeouts + self.invalid_actions + self.invalid_observations
            + self.reset_failures + self.game_freezes
        )


EventCallback = Callable[[str, dict[str, Any]], None]


def observation_is_valid(observation: object, env: Any) -> bool:
    return bool(env.observation_space.contains(observation))


def _emit(callback: EventCallback | None, event: str, **payload: Any) -> None:
    if callback is not None:
        callback(event, payload)


def _wait_if_paused(pause_event: threading.Event | None,
                    stop_event: threading.Event | None) -> bool:
    while pause_event is not None and not pause_event.is_set():
        if stop_event is not None and stop_event.is_set():
            return False
        time.sleep(0.05)
    return stop_event is None or not stop_event.is_set()


def run_random_agent(
    env: Any,
    config: RandomAgentConfig,
    *,
    pause_event: threading.Event | None = None,
    stop_event: threading.Event | None = None,
    on_event: EventCallback | None = None,
) -> RandomAgentSummary:
    """Run legal random actions and collect integration-health statistics."""
    rng = np.random.default_rng(config.random_seed)
    summary = RandomAgentSummary(requested_episodes=config.episodes)

    for episode in range(1, config.episodes + 1):
        if not _wait_if_paused(pause_event, stop_event):
            summary.stopped = True
            break
        _emit(on_event, "episode_start", episode=episode, summary=summary)

        try:
            observation, info = env.reset()
            if "seed_selection" in info:
                _emit(on_event, "seed_selection", **info["seed_selection"])
        except Exception as exc:
            summary.reset_failures += 1
            _emit(on_event, "reset_failure", episode=episode, error=str(exc), summary=summary)
            # A reset failure means the environment never produced an episode.
            # Continuing can make the GUI look as though training is running
            # while it repeatedly remains on the same terminal UI.
            break

        if not observation_is_valid(observation, env):
            summary.invalid_observations += 1
            _emit(on_event, "invalid_observation", episode=episode, step=0, summary=summary)

        previous_clock = int(info.get("game_clock", 0))
        freeze_count = 0
        freeze_reported = False
        total_reward = 0.0
        steps = 0
        result = "UNKNOWN"

        for steps in range(1, config.max_steps + 1):
            if not _wait_if_paused(pause_event, stop_event):
                summary.stopped = True
                break

            try:
                mask = np.asarray(env.get_action_mask(), dtype=np.bool_)
            except Exception as exc:
                summary.invalid_actions += 1
                _emit(on_event, "mask_error", episode=episode, step=steps,
                      error=str(exc), summary=summary)
                break

            if mask.shape != (env.action_space.n,) or not bool(mask[0]):
                summary.invalid_actions += 1
                _emit(on_event, "invalid_mask", episode=episode, step=steps,
                      shape=mask.shape, noop_legal=bool(mask[0]) if mask.size else False,
                      summary=summary)
                break

            valid_actions = np.flatnonzero(mask)
            action = int(rng.choice(valid_actions))
            if not bool(mask[action]):
                summary.invalid_actions += 1
                _emit(on_event, "sampled_illegal_action", episode=episode,
                      step=steps, action=action, summary=summary)
                break

            try:
                observation, reward, terminated, truncated, info = env.step(action)
            except Exception as exc:
                summary.invalid_actions += 1
                _emit(on_event, "step_error", episode=episode, step=steps,
                      action=action, error=str(exc), summary=summary)
                break

            total_reward += float(reward)
            result = str(info.get("result", "UNKNOWN"))
            summary.auto_collected += max(0, int(info.get("automation_collected", 0)))
            if bool(info.get("terminal_observation_fallback", False)):
                summary.terminal_observation_fallbacks += 1
            if not observation_is_valid(observation, env):
                summary.invalid_observations += 1
                _emit(on_event, "invalid_observation", episode=episode,
                      step=steps, summary=summary)

            action_result = info.get("action_result") or {}
            if not bool(action_result.get("success", False)):
                summary.invalid_actions += 1
                _emit(on_event, "invalid_action", episode=episode, step=steps,
                      action=action, action_result=action_result, summary=summary)

            current_clock = int(info.get("game_clock", 0))
            if not terminated and not truncated:
                if previous_clock >= current_clock:
                    freeze_count += 1
                    if freeze_count >= config.freeze_streak and not freeze_reported:
                        summary.game_freezes += 1
                        freeze_reported = True
                        _emit(on_event, "game_freeze", episode=episode, step=steps,
                              previous_clock=previous_clock, current_clock=current_clock,
                              summary=summary)
                else:
                    freeze_count = 0
                previous_clock = current_clock

            _emit(on_event, "step", episode=episode, step=steps, action=action,
                  reward=float(reward), total_reward=total_reward, result=result,
                  game_clock=current_clock,
                  automation_collected=int(info.get("automation_collected", 0)),
                  terminal_observation_fallback=bool(
                      info.get("terminal_observation_fallback", False)
                  ), summary=summary)

            if terminated:
                if result == "WON":
                    summary.wins += 1
                elif result == "LOST":
                    summary.losses += 1
                break
            if truncated:
                summary.timeouts += 1
                break
        else:
            summary.timeouts += 1

        if summary.stopped:
            break
        summary.completed_episodes += 1
        summary.episode_lengths.append(steps)
        summary.episode_rewards.append(total_reward)
        _emit(on_event, "episode_end", episode=episode, steps=steps,
              reward=total_reward, result=result, summary=summary)

    _emit(on_event, "finished", summary=summary)
    return summary
