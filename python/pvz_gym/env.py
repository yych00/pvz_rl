"""基于单个权威 PvZSession 的 Gymnasium 战斗环境。"""

from __future__ import annotations

import logging
import math
from collections.abc import Mapping
from typing import Any

try:
    import gymnasium as gym
except ImportError:  # pragma: no cover
    class _Env:
        metadata = {}
        def reset(self, *, seed=None, options=None): return None
    class _GymCompat:
        Env = _Env
    gym = _GymCompat()

from .action_space import ACTION_SPACE_SIZE, decode_action, make_action_space
from .client import PvZClient
from .profiling import bridge_timing
from .runtime import (
    RuntimeConfig, TurboConfig, coerce_config, normalize_simulation_mode,
    require_runtime_implementation,
)
from .observation import (
    boss_projectile_truncation_info, encode_observation, make_observation_space, zombie_truncation_info,
)
from .reward import RewardFn, calculate_reward
from .session import PvZSession
from .types import (
    DEFAULT_MAX_EPISODE_TICKS, Action, ActionResult, GameStatus, ResetFacts, StepEvents,
)


SIMULATION_STEPPED = "stepped"
SIMULATION_HEADLESS = "headless"
SIMULATION_CONTINUOUS = "continuous"
SIMULATION_TURBO = "turbo"
_DEFAULT_SIMULATION_MODE = SIMULATION_STEPPED
_LOGGER = logging.getLogger(__name__)


def set_default_simulation_mode(value: str) -> None:
    global _DEFAULT_SIMULATION_MODE
    _DEFAULT_SIMULATION_MODE = normalize_simulation_mode(value)


def get_default_simulation_mode() -> str:
    return _DEFAULT_SIMULATION_MODE


def _int_field(mapping: Mapping[str, object], key: str, default: int = 0) -> int:
    try:
        return int(mapping.get(key, default) or default)
    except (TypeError, ValueError, OverflowError):
        return default


def _field(value: object, key: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(key, default)
    return getattr(value, key, default)


def _transition_quality_info(facts: object) -> dict[str, object]:
    fallback = bool(_field(facts, "terminal_observation_fallback", False))
    quality = str(_field(facts, "observation_quality", "fresh"))
    if fallback:
        quality = "terminal_fallback"
    exact = bool(_field(facts, "ticks_advanced_exact", True)) and quality == "fresh" and not fallback
    confirmed = _field(facts, "last_confirmed_game_clock", None)
    if confirmed is None:
        confirmed = _field(facts, "game_clock", 0)
    return {"observation_quality": quality, "ticks_advanced_exact": exact,
            "last_confirmed_game_clock": int(confirmed),
            "replay_eligible": bool(_field(facts, "replay_eligible", True)) and exact,
            "native_observation_schema": int(_field(_field(facts, "observation", None), "schema_version", 3))}


def _transition_ticks(facts: object, before_clock: int) -> int:
    delta = max(0, int(_field(facts, "game_clock", 0)) - before_clock)
    native_ticks = _field(facts, "ticks_advanced", None)
    if native_ticks is None:
        return delta
    ticks = int(native_ticks)
    if ticks < 0 or (_transition_quality_info(facts)["ticks_advanced_exact"] and ticks != delta):
        raise RuntimeError("native transition duration disagrees with its confirmed clock")
    return ticks


def _float_field(value: object, key: str, default: float = 0.0) -> float:
    try:
        result = float(_field(value, key, default) or default)
    except (TypeError, ValueError, OverflowError):
        return default
    return result if math.isfinite(result) else default


def _clip01(value: float) -> float:
    return min(1.0, max(0.0, float(value)))


def _zombie_threat_score(observation: object | None) -> float:
    """Return a bounded board-threat score in [0, 1].

    This is a neutral diagnostic, not an RL reward.  Threat uses only raw game
    facts: proximity to the house and effective durability (body + armour +
    shield + flying HP).  It deliberately avoids a hand-written per-type table,
    so the same definition remains useful across Adventure levels.
    """
    if observation is None:
        return 0.0

    total = 0.0
    for zombie in _field(observation, "zombies", ()) or ():
        if bool(_field(zombie, "is_mind_controlled", False)):
            continue

        x = _float_field(zombie, "x", 900.0)
        proximity = _clip01((900.0 - x) / 900.0)

        effective_hp = sum(
            max(0.0, _float_field(zombie, name, 0.0))
            for name in ("body_hp", "helm_hp", "shield_hp", "flying_hp")
        )
        if effective_hp <= 0.0:
            effective_hp = max(0.0, _float_field(zombie, "hp", 0.0))

        durability = _clip01(math.log1p(effective_hp) / math.log1p(5000.0))
        eating_factor = 1.15 if bool(_field(zombie, "is_eating", False)) else 1.0
        total += proximity * durability * eating_factor

    # Smooth saturation prevents a huge late-game crowd from exploding the
    # reward scale while still preserving differences in ordinary waves.
    return _clip01(1.0 - math.exp(-total / 3.0))


def _terminal_plant_score(observation: object | None) -> float:
    """Healthy surviving-board occupancy in [0, 1]."""
    if observation is None:
        return 0.0
    global_state = _field(observation, "global_state", None)
    rows = int(max(1.0, min(6.0, _float_field(global_state, "grid_rows", 5.0))))
    capacity = float(rows * 9)
    health_sum = 0.0
    for plant in _field(observation, "plants", ()) or ():
        hp = max(0.0, _float_field(plant, "hp", 0.0))
        max_hp = max(0.0, _float_field(plant, "max_hp", 0.0))
        if max_hp > 0.0:
            health_sum += _clip01(hp / max_hp)
        elif hp > 0.0:
            health_sum += 1.0
    return _clip01(health_sum / capacity)


def _terminal_sun_score(observation: object | None) -> float:
    """Remaining sun in [0, 1], saturated at 1000 sun."""
    if observation is None:
        return 0.0
    global_state = _field(observation, "global_state", None)
    return _clip01(max(0.0, _float_field(global_state, "sun", 0.0)) / 1000.0)


def _reward_diagnostics(
    before: object | None,
    after: object | None,
    *,
    terminal: bool,
) -> dict[str, float]:
    before_threat = _zombie_threat_score(before)
    after_threat = _zombie_threat_score(after)
    return {
        # Positive means the board became safer. Both endpoints are bounded,
        # therefore the delta is naturally within [-1, 1].
        "zombie_threat_delta": max(-1.0, min(1.0, before_threat - after_threat)),
        "terminal_plant_score": _terminal_plant_score(after) if terminal else 0.0,
        "terminal_sun_score": _terminal_sun_score(after) if terminal else 0.0,
    }


class PvZEnv(gym.Env):
    """战斗阶段 Gym 视图；原生 ResetResult/StepResult 是权威状态来源。"""

    metadata = {"render_modes": []}

    def __init__(self, client: object | None = None, *, session: PvZSession | None = None,
                 level_id: int = 1, playthrough: int | None = 1,
                 seed_types: tuple[int, ...] | None = None,
                 seed_selection_mode: str = "native",
                 ticks_per_step: int = 30, max_episode_steps: int | None = None,
                 max_episode_ticks: int | None = DEFAULT_MAX_EPISODE_TICKS,
                 reward_fn: RewardFn = calculate_reward, automation_enabled: bool = True,
                 auto_dialogue: bool = True, auto_collect: bool = True,
                 auto_whack: bool = True,
                 auto_select_seeds: bool = True, auto_skip_tutorial: bool = True,
                 background_running: bool = True, deterministic_pause: bool = True,
                 reset_timeout_ms: int = 10000, simulation_mode: str | None = None,
                 reward_stage_ticks: int | None = None,
                 runtime_config: RuntimeConfig | Mapping[str, object] | None = None,
                 turbo_config: TurboConfig | Mapping[str, object] | None = None):
        if session is not None and client is not None:
            raise ValueError("pass session or client, not both")
        if ticks_per_step <= 0:
            raise ValueError("ticks_per_step must be positive")
        if reward_stage_ticks is not None and (
            type(reward_stage_ticks) is not int or reward_stage_ticks <= 0
        ):
            raise ValueError("reward_stage_ticks must be a positive integer or None")
        if max_episode_ticks is not None and (
            type(max_episode_ticks) is not int or max_episode_ticks <= 0
        ):
            raise ValueError("max_episode_ticks must be a positive integer or None")
        if not 1 <= level_id <= 50:
            raise ValueError("level_id must be a normal Adventure level in [1, 50]")

        self.simulation_mode = normalize_simulation_mode(
            _DEFAULT_SIMULATION_MODE if simulation_mode is None else simulation_mode
        )
        self.runtime_config = coerce_config(RuntimeConfig, runtime_config)
        self.turbo_config = coerce_config(TurboConfig, turbo_config)
        # Reject reserved features before constructing a client or changing its
        # automation settings. A registered mode is not an implemented backend.
        require_runtime_implementation(self.simulation_mode, self.runtime_config)

        if session is None:
            backend = client or PvZClient(
                automation_enabled=automation_enabled,
                auto_dialogue=auto_dialogue,
                auto_collect=auto_collect,
                auto_whack=auto_whack,
                auto_select_seeds=auto_select_seeds,
                auto_skip_tutorial=auto_skip_tutorial,
                background_running=background_running,
                deterministic_pause=deterministic_pause,
                runtime_config=self.runtime_config,
            )
            session = PvZSession(backend)
            self._owns_session = True
        else:
            self._owns_session = False
        self.session = session
        self.client = session.backend  # 兼容现有调用方和测试。
        require_runtime_implementation(self.simulation_mode, self.runtime_config, self.client)
        if runtime_config is not None:
            configure = getattr(self.client, "configure_runtime", None)
            if callable(configure):
                configure(self.runtime_config)

        if client is not None and hasattr(self.client, "set_automation_config"):
            self.client.set_automation_config(automation_enabled, auto_dialogue, auto_collect)
            if hasattr(self.client, "set_auto_whack"):
                self.client.set_auto_whack(auto_whack)

        self.level_id = level_id
        self.playthrough = playthrough
        self.seed_types = seed_types
        if seed_selection_mode not in ("native", "adaptive", "adaptive_shuffled", "custom", "fixed", "fixed_shuffled"):
            raise ValueError("invalid seed_selection_mode")
        if seed_selection_mode == "custom" and not seed_types:
            raise ValueError("custom seed selection requires a card priority")
        self.seed_selection_mode = seed_selection_mode
        self._seed_selection_info = {}
        self.ticks_per_step = ticks_per_step
        self.reward_stage_ticks = reward_stage_ticks
        self.max_episode_steps = max_episode_steps
        self.max_episode_ticks = max_episode_ticks
        self.reward_fn = reward_fn
        self.auto_select_seeds = bool(auto_select_seeds)
        self.deterministic_pause = bool(deterministic_pause)
        self.reset_timeout_ms = reset_timeout_ms
        self.action_space = make_action_space()
        self.observation_space = make_observation_space()
        self._steps = 0
        self._episode_ticks = 0
        self._window_start_game_clock = 0
        self._clock_segment_start = 0
        self._clock_segment_ticks = 0
        self._previous_game_clock = 0
        self._tick_clock_rebased = False
        self._raw_observation = None
        self._action_mask: tuple[bool, ...] = ()
        self._headless_active = False
        self._turbo_active = False
        self._tick_samples = 0
        self._tick_mismatches = 0

    def reset(self, *, seed: int | None = None,
              options: Mapping[str, Any] | None = None):
        super().reset(seed=seed)
        self._end_headless()
        self._begin_turbo_if_needed()
        config: dict[str, object] = {
            "level_id": self.level_id,
            "timeout_ms": self.reset_timeout_ms,
            "auto_select_seeds": self.auto_select_seeds,
            "deterministic_pause": (
                False if self.simulation_mode == SIMULATION_CONTINUOUS
                else self.deterministic_pause
            ),
            "stop_at_seed_selection": False,
        }
        if self.playthrough is not None:
            config["playthrough"] = self.playthrough
        if self.seed_types is not None:
            config["seed_types"] = self.seed_types
        config.update(dict(options or {}))
        if seed is not None:
            config["seed"] = int(seed)

        scripted_selection = self.seed_selection_mode != "native"
        if scripted_selection:
            config["stop_at_seed_selection"] = True
            config["auto_select_seeds"] = False
            config.pop("seed_types", None)

        facts = self.session.begin_episode(config)
        from .adventure_curriculum import SPECIAL_LEVELS
        effective_level = int(config.get("level_id", self.level_id))
        is_special = effective_level in SPECIAL_LEVELS
        if is_special:
            self._seed_selection_info = {
                "mode": "special",
                "special": SPECIAL_LEVELS[effective_level],
                "level_id": effective_level,
                "actual": [],
                "warnings": [],
            }
        else:
            self._seed_selection_info = {
                "mode": self.seed_selection_mode,
                "level_id": effective_level,
                "actual": [],
                "warnings": [],
            }

        if not is_special and scripted_selection and facts.status == GameStatus.CHOOSING_SEEDS:
            from .default_decks import select_default_deck
            messages = []
            ready = select_default_deck(
                self.session, self.level_id, self.playthrough or 1, confirm=True,
                priority_override=self.seed_types if self.seed_selection_mode == "custom" else None,
                selection_mode=self.seed_selection_mode,
                rng=self.np_random if self.seed_selection_mode.endswith("_shuffled") else None,
                report=messages.append,
            )
            self._seed_selection_info["actual"] = list(ready.selected_seeds)
            self._seed_selection_info["warnings"] = [m for m in messages if "缺口" in m or "补位提示" in m]
            facts = self.session.wait_for_battle(
                timeout_ms=self.reset_timeout_ms,
                deterministic_pause=bool(config["deterministic_pause"]))
        elif not is_special and facts.observation is not None:
            self._seed_selection_info["actual"] = [int(c.get("plant_type", -1)) for c in facts.observation.cards]

        activated = self._activate_battle_facts(facts)
        actual_level = int(facts.observation.global_state.get("level_id", config["level_id"]))
        if actual_level in SPECIAL_LEVELS:
            self._seed_selection_info["mode"] = "special"
            self._seed_selection_info["special"] = SPECIAL_LEVELS[actual_level]
            self._seed_selection_info["actual"] = []
        activated[1].update(
            level_id=actual_level,
            playthrough=config.get("playthrough", self.playthrough),
            episode_seed=config.get("seed"),
        )
        self._begin_headless_if_needed()
        return activated

    def observe_current(self):
        """不重置回合，直接把当前会话中已经准备好的战斗挂接到此环境。"""
        facts = self.session.reset_facts
        if facts is None or facts.observation is None or facts.status != GameStatus.PLAYING:
            raise RuntimeError("session is not attached to a battle-ready episode")
        activated = self._activate_battle_facts(facts)
        self._begin_headless_if_needed()
        return activated

    def _activate_battle_facts(self, facts: ResetFacts):
        if facts.observation is None or facts.status != GameStatus.PLAYING:
            raise RuntimeError(f"battle reset did not reach PLAYING: {facts.status.name}")
        self._raw_observation = facts.observation
        self._steps = 0
        self._episode_ticks = 0
        self._window_start_game_clock = int(facts.game_clock)
        self._clock_segment_start = int(facts.game_clock)
        self._clock_segment_ticks = 0
        self._previous_game_clock = int(facts.game_clock)
        self._tick_clock_rebased = False
        self._tick_samples = 0
        self._tick_mismatches = 0
        self._action_mask = self._project_action_mask(self.session.get_action_mask())
        events = self._events_dict(StepEvents())
        events.update({
            "zombie_threat_delta": 0.0,
            "terminal_plant_score": 0.0,
            "terminal_sun_score": 0.0,
        })
        return bridge_timing.measure("encode", encode_observation, self._raw_observation), {
            "raw_observation": self._raw_observation,
            "seed_selection": dict(self._seed_selection_info),
            "episode_ticks": 0,
            "window_start_game_clock": self._window_start_game_clock,
            "tick_clock_rebased": False,
            "max_episode_ticks": self.max_episode_ticks,
            "truncation_reason": None,
            "result": facts.status.name,
            "game_clock": facts.game_clock,
            "action_result": None,
            "events": events,
            "action_mask": self._action_mask,
            "rng_seed_support": int(facts.rng_seed_support),
            "rng_seed_applied": bool(facts.rng_seed_applied),
            "automation_collected": 0,
            "terminal_observation_fallback": False,
            "observation_quality": str(_field(facts.observation, "quality", "fresh")),
            "last_confirmed_game_clock": int(facts.game_clock),
            "native_observation_schema": int(_field(facts.observation, "schema_version", 3)),
            "replay_eligible": str(_field(facts.observation, "quality", "fresh")) == "fresh",
            "simulation_mode": self.simulation_mode,
            **zombie_truncation_info(self._raw_observation),
            **boss_projectile_truncation_info(self._raw_observation),
        }

    def step(self, action: int):
        if not self.action_space.contains(action):
            raise ValueError(f"action outside Discrete({ACTION_SPACE_SIZE}): {action}")
        native_action = decode_action(int(action))
        if self.simulation_mode in (SIMULATION_HEADLESS, SIMULATION_TURBO):
            return self._step_headless(native_action)
        if self.simulation_mode == SIMULATION_CONTINUOUS:
            return self._step_continuous(native_action)

        before = self._raw_observation
        before_clock = _int_field(
            before.global_state if before else {},
            "game_clock",
            0,
        )
        requested = self._requested_ticks()
        facts = self.session.step(native_action, requested)
        self._raw_observation = facts.observation
        self._action_mask = self._project_action_mask(facts.action_mask)
        reward = float(self.reward_fn(facts.status))
        self._steps += 1

        terminated = facts.status in (GameStatus.WON, GameStatus.LOST)
        ticks_advanced = _transition_ticks(facts, before_clock)
        truncated, limit_info = self._episode_limit_info(int(facts.game_clock), terminated)
        events = self._events_dict(facts.events)
        events.update(_reward_diagnostics(before, facts.observation, terminal=terminated))
        info = {
            "raw_observation": facts.observation,
            **limit_info,
            "result": facts.status.name,
            "game_clock": facts.game_clock,
            "action_result": self._action_result_dict(facts.action),
            "events": events,
            "action_mask": self._action_mask,
            "automation_collected": facts.automation_collected,
            "terminal_observation_fallback": facts.terminal_observation_fallback,
            **_transition_quality_info(facts),
            "simulation_mode": self.simulation_mode,
            **zombie_truncation_info(self._raw_observation),
            **boss_projectile_truncation_info(self._raw_observation),
            "ticks_requested": requested,
            "ticks_advanced": ticks_advanced,
        }
        return bridge_timing.measure("encode", encode_observation, facts.observation), reward, terminated, truncated, info

    def _step_headless(self, native_action: Action):
        if self.simulation_mode == SIMULATION_TURBO:
            self._begin_turbo_if_needed()
        elif not self._headless_active:
            self._begin_headless_if_needed()
        before = self._raw_observation
        if before is None:
            raise RuntimeError("headless step requires an active observation")

        step_headless = getattr(self.client, "step_turbo" if self.simulation_mode == SIMULATION_TURBO else "step_headless", None)
        if not callable(step_headless):
            raise RuntimeError("atomic headless step is unavailable on this backend")

        before_clock = _int_field(before.global_state, "game_clock", 0)
        requested = self._requested_ticks()
        facts = step_headless(native_action, requested)
        self._raw_observation = facts.observation
        self._action_mask = self._project_action_mask(facts.action_mask)
        reward = float(self.reward_fn(facts.status))
        self._steps += 1

        terminated = facts.status in (GameStatus.WON, GameStatus.LOST)
        ticks_advanced = _transition_ticks(facts, before_clock)
        truncated, limit_info = self._episode_limit_info(int(facts.game_clock), terminated)
        if terminated or truncated:
            self._end_headless()
        # A battle can end before the requested chunk completes. Its shorter
        # final transition is expected, so reserve drift warnings for steps
        # that remain in PLAYING state.
        tick_mismatch = not terminated and ticks_advanced != requested
        tick_warning = None
        self._tick_samples += 1
        if tick_mismatch:
            self._tick_mismatches += 1
            tick_warning = (
                f"headless tick drift: requested={requested} "
                f"actual={ticks_advanced}"
            )
            _LOGGER.warning(tick_warning)
        tick_mismatch_rate = self._tick_mismatches / self._tick_samples

        events = self._events_dict(facts.events)
        events.update(_reward_diagnostics(before, facts.observation, terminal=terminated))
        info = {
            "raw_observation": facts.observation,
            **limit_info,
            "result": facts.status.name,
            "game_clock": facts.game_clock,
            "action_result": self._action_result_dict(facts.action),
            "events": events,
            "action_mask": self._action_mask,
            "automation_collected": facts.automation_collected,
            "terminal_observation_fallback": facts.terminal_observation_fallback,
            **_transition_quality_info(facts),
            "simulation_mode": self.simulation_mode,
            **zombie_truncation_info(self._raw_observation),
            **boss_projectile_truncation_info(self._raw_observation),
            "ticks_requested": requested,
            "ticks_advanced": ticks_advanced,
            "tick_mismatch": tick_mismatch,
            "tick_warning": tick_warning,
            "tick_samples": self._tick_samples,
            "tick_mismatches": self._tick_mismatches,
            "tick_mismatch_rate": tick_mismatch_rate,
        }
        return bridge_timing.measure("encode", encode_observation, facts.observation), reward, terminated, truncated, info

    def _step_continuous(self, native_action: Action):
        before = self._raw_observation
        if before is None:
            raise RuntimeError("continuous step requires an active observation")
        step_continuous = getattr(self.client, "step_continuous", None)
        if not callable(step_continuous):
            raise RuntimeError("atomic continuous step is unavailable on this backend")

        before_clock = _int_field(before.global_state, "game_clock", 0)
        requested = self._requested_ticks()
        facts = step_continuous(native_action, requested)
        self._raw_observation = facts.observation
        self._action_mask = self._project_action_mask(facts.action_mask)
        reward = float(self.reward_fn(facts.status))
        self._steps += 1
        terminated = facts.status in (GameStatus.WON, GameStatus.LOST)
        ticks_advanced = _transition_ticks(facts, before_clock)
        truncated, limit_info = self._episode_limit_info(int(facts.game_clock), terminated)
        events = self._events_dict(facts.events)
        events.update(_reward_diagnostics(before, facts.observation, terminal=terminated))
        info = {
            "raw_observation": facts.observation,
            **limit_info,
            "result": facts.status.name,
            "game_clock": facts.game_clock,
            "action_result": self._action_result_dict(facts.action),
            "events": events,
            "action_mask": self._action_mask,
            "automation_collected": facts.automation_collected,
            "terminal_observation_fallback": facts.terminal_observation_fallback,
            **_transition_quality_info(facts),
            "simulation_mode": self.simulation_mode,
            **zombie_truncation_info(self._raw_observation),
            **boss_projectile_truncation_info(self._raw_observation),
            "ticks_requested": requested,
            # 从上一次观测起的真实推进量包含策略和优化器计算耗时；权威游戏时钟不做截断或改写。
            "ticks_advanced": ticks_advanced,
        }
        return bridge_timing.measure("encode", encode_observation, facts.observation), reward, terminated, truncated, info

    def _requested_ticks(self) -> int:
        requested = self.ticks_per_step
        if self.reward_stage_ticks is not None:
            requested = min(requested, self.reward_stage_ticks - self._episode_ticks % self.reward_stage_ticks)
        return requested

    def _episode_limit_info(self, game_clock: int, terminated: bool):
        # Reuse the authoritative step clocks; no extra read, timer or polling.
        if game_clock < self._previous_game_clock:
            # A new clock epoch cannot be subtracted from the old one. Retain
            # measured time and rebase; the unobserved reset interval is unknown.
            self._clock_segment_ticks = self._episode_ticks
            self._clock_segment_start = game_clock
            self._tick_clock_rebased = True
        self._episode_ticks = (
            self._clock_segment_ticks + max(0, game_clock - self._clock_segment_start)
        )
        self._previous_game_clock = game_clock
        reason = None
        if not terminated:
            if (self.max_episode_ticks is not None
                    and self._episode_ticks >= self.max_episode_ticks):
                reason = "max_episode_ticks"
            elif (self.max_episode_steps is not None
                    and self._steps >= self.max_episode_steps):
                reason = "max_episode_steps"
        return reason is not None, {
            "episode_ticks": self._episode_ticks,
            "window_start_game_clock": self._window_start_game_clock,
            "tick_clock_rebased": self._tick_clock_rebased,
            "max_episode_ticks": self.max_episode_ticks,
            "truncation_reason": reason,
        }

    def _begin_headless_if_needed(self) -> None:
        if self.simulation_mode != SIMULATION_HEADLESS or self._headless_active:
            return
        begin = getattr(self.client, "begin_headless_mode", None)
        if not callable(begin):
            raise RuntimeError("headless simulation is unavailable on this backend")
        if not bool(begin()):
            raise RuntimeError("native headless mode could not start")
        self._headless_active = True

    def _end_headless(self) -> None:
        if not self._headless_active:
            return
        end = getattr(self.client, "end_headless_mode", None)
        if not callable(end):
            raise RuntimeError("headless shutdown is unavailable on this backend")
        if end() is False:
            # Keep ownership recorded and reject reset before its first game
            # mutation. A failed release must never authorize another owner.
            raise RuntimeError("native headless mode could not stop; reset was not executed")
        self._headless_active = False

    def _begin_turbo_if_needed(self) -> None:
        if self.simulation_mode == SIMULATION_TURBO and not self._turbo_active:
            if not self.client.begin_turbo_mode(self.turbo_config):
                raise RuntimeError("native Turbo mode could not start")
            self._turbo_active = True

    def _end_turbo(self) -> None:
        if self._turbo_active:
            try: self.client.end_turbo_mode()
            finally: self._turbo_active = False

    def set_simulation_mode(self, mode: str) -> None:
        """Switch implemented modes; reject unavailable policies before teardown."""
        new_mode = normalize_simulation_mode(mode)
        require_runtime_implementation(new_mode, self.runtime_config, self.client)
        if new_mode == self.simulation_mode:
            return
        if self.simulation_mode == SIMULATION_HEADLESS:
            self._end_headless()
        if self.simulation_mode == SIMULATION_TURBO:
            self._end_turbo()
        self.simulation_mode = new_mode
        if self.simulation_mode == SIMULATION_HEADLESS:
            self._begin_headless_if_needed()
        if self.simulation_mode == SIMULATION_TURBO:
            self._begin_turbo_if_needed()

    def get_action_mask(self):
        if self._action_mask:
            return self._action_mask
        self._action_mask = self._project_action_mask(self.session.get_action_mask())
        return self._action_mask

    def close(self):
        try:
            self._end_headless()
        except Exception:
            pass
        try:
            self._end_turbo()
        except Exception:
            pass
        if self._owns_session:
            try:
                self.session.close()
            except Exception:
                pass

    def _project_action_mask(self, mask) -> tuple[bool, ...]:
        # Native StepFacts 已保证 action_mask 是 bool tuple；标准训练路径直接复用，
        # 避免每个 step 再遍历 757 个动作并重新分配 tuple。
        values = mask if isinstance(mask, tuple) else tuple(bool(value) for value in mask)
        if len(values) != self.action_space.n:
            raise RuntimeError(
                f"native action mask version mismatch: {len(values)} != {self.action_space.n}"
            )
        return values

    @staticmethod
    def _action_result_dict(result: ActionResult | object) -> dict[str, object]:
        return {
            "success": bool(getattr(result, "success", False)),
            "error": getattr(result, "error", None),
            "message": getattr(result, "message", ""),
        }

    @staticmethod
    def _events_dict(events: StepEvents) -> dict[str, object]:
        return {
            "zombies_killed": events.zombies_killed,
            "plants_lost": events.plants_lost,
            "sun_collected": events.sun_collected,
            "sun_spent": events.sun_spent,
            "mowers_used": events.mowers_used,
            "mower_triggered_rows": list(events.mower_triggered_rows),
            "waves_advanced": events.waves_advanced,
            "won": events.won,
            "lost": events.lost,
        }


PvzGymEnv = PvZEnv
