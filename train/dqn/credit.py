"""Store ordinary TD rewards and track episode diagnostics."""

from __future__ import annotations

from .config import DoubleDQNConfig
from .replay import ReplayBuffer
from .reward import mower_event_count, training_reward, effective_reward_settings


class EpisodeEventCredit:
    """Start a fresh tracker after every actual game reset, including resume.

    Reset/interruption boundaries keep n-step returns inside one game. Mower
    timing is recorded for diagnostics and does not change reward assignment.
    """

    def __init__(self, replay: ReplayBuffer, config: DoubleDQNConfig, initial_info=None, *, stream_id: int = 0):
        settings = effective_reward_settings(config)
        if replay.reward_settings is not None and replay.reward_settings != settings:
            raise ValueError("reward settings changed inside replay; rebuild replay")
        replay.reward_settings = settings
        # A reset also ends an interrupted episode restored from a checkpoint.
        # Keep bootstrap enabled on that last row, but never join two games.
        replay.end_episode(stream_id=stream_id)
        self.stream_id = stream_id
        self.config = config
        self.first_mower_seen = False
        self.first_mower_step: int | None = None
        self.first_mower_tick: int | None = None
        self.ticks = 0
        self.steps = 0
        self.reward_tracker = None
        if config.reward_scheme == "transition_dense":
            from .transition_reward import TransitionDenseReward
            self.reward_tracker = TransitionDenseReward(initial_info or {}, shaping_limit_mode=config.shaping_limit_mode)
            replay.set_reward_context(True)
        if config.reward_scheme == "staged":
            from .staged_reward import StagedReward
            self.reward_tracker = StagedReward(initial_info or {})

    def observe(self, observation):
        from .features import normalize_observation
        observation = normalize_observation(observation, boss_projectiles=self.config.boss_projectiles)
        return self.reward_tracker.observe(observation) if self.config.reward_scheme == "transition_dense" else observation

    def add(self, replay, observation, action, raw_reward, next_observation,
            terminated, truncated, next_action_mask, info) -> float:
        events = info.get("events") or {}
        if not isinstance(events, dict):
            events = {}
        ticks_advanced = info.get("ticks_advanced", self.config.env.ticks_per_step)
        if type(ticks_advanced) is not int or not 0 <= ticks_advanced < 2**63:
            raise ValueError("ticks_advanced must be a non-negative int64 integer")
        if self.reward_tracker is None:
            total = training_reward(raw_reward, events, self.config)
        else:
            total = self.reward_tracker.step(raw_reward, {**info, "ticks_advanced": ticks_advanced},
                                     terminated=bool(terminated), truncated=bool(truncated))
            info["reward_breakdown"] = dict(self.reward_tracker.last_breakdown)
        self.ticks += ticks_advanced
        self.steps += 1
        first_event = not self.first_mower_seen and mower_event_count(events) > 0
        if self.config.reward_scheme == "transition_dense":
            next_observation["reward_context"] = self.reward_tracker.observe(next_observation)["reward_context"]
        replay.add(observation, action, total, next_observation,
                   bool(terminated), next_action_mask, ticks_advanced=ticks_advanced,
                   episode_end=bool(terminated or truncated), stream_id=self.stream_id)
        if first_event:
            self.first_mower_seen = True
            self.first_mower_step = self.steps
            self.first_mower_tick = self.ticks
        return total
