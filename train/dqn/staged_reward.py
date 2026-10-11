"""Tick-stage reward accounting from raw, untruncated game facts.

Prices are original Adventure prices, independent of a packet's current cost.
Health budgets use the actual finite Board spawn schedule, never a running
estimate based on the enemies observed so far.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any


STAGE_TICKS = 1000
STAGE_CAP = 0.1
RESOURCE_REFERENCE = 1000.0
STAGE_DAMAGE_WEIGHT = 0.5
TERMINAL_DAMAGE_WEIGHT = 0.5
MOWER_PENALTY = 0.2
STAGE_LIFETIME_CAP = 1.0  # 整局累计正向阶段辅助奖励上限（防止拖延时间无限刷分）
METRICS_VERSION = 1


# Original PlantDefinition sun prices. Index 48 resolves to its imitation target;
# special bowling nuts 49/50 and Zen sprout 51 really have original cost zero.
PLANT_PRICES = (
    100, 50, 150, 50, 25, 175, 150, 200, 0, 25, 75, 75,
    75, 25, 75, 125, 25, 50, 325, 25, 125, 100, 175, 125,
    0, 25, 125, 100, 125, 125, 125, 100, 100, 25, 100, 75,
    50, 100, 50, 300, 250, 150, 150, 225, 200, 50, 125, 500,
    0, 0, 0, 0, 200,
)
# Upgrade type -> consumed bases. These bases no longer have separate IDs.
UPGRADE_BASES = {40: (7,), 41: (1,), 42: (10,), 43: (16,), 44: (39,),
                 45: (21,), 46: (19,), 47: (34, 34)}

# ZombieInitialize, retail/GOTY Adventure: body + helmet + shield + flying HP.
# A Bobsled schedule entry creates a leader and three followers: 4*270+300.
# Source revision and special-mode boundaries are documented in rewards.md.
PLANNED_ZOMBIE_HP = (
    270, 270, 640, 500, 1370, 420, 1370, 1670, 500, 270, 270,
    270, 1350, 1380, 500, 500, 290, 370, 500, 1350, 450, 1000,
    850, 3000, 270, 40000, 270, 1370, 500, 270, 270, 2470, 6000,
)


def _field(value: object, name: str, default: Any = None) -> Any:
    return value.get(name, default) if isinstance(value, Mapping) else getattr(value, name, default)


def _number(value: object) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("staged reward requires finite game facts")
    return result


def plant_price(plant_type: int, *, planted: bool = False) -> float:
    if not 0 <= plant_type < len(PLANT_PRICES):
        raise ValueError(f"staged reward has no original sun price for plant type {plant_type}")
    price = PLANT_PRICES[plant_type]
    if planted:
        price += sum(PLANT_PRICES[t] for t in UPGRADE_BASES.get(plant_type, ()))
    return float(price)


def resource_value(raw: object) -> float:
    if raw is None:
        raise ValueError("staged reward requires raw_observation from PvZEnv")
    global_state = _field(raw, "global_state", _field(raw, "global", {}))
    total = max(0.0, _number(_field(global_state, "sun", 0)))
    for plant in _field(raw, "plants", ()):
        plant_type = int(_field(plant, "type"))
        if plant_type == 48:
            plant_type = int(_field(plant, "imitater_type", -1))
            if plant_type == 48:
                raise ValueError("staged reward needs a valid Imitater target")
        total += plant_price(plant_type, planted=True)
    # Dropped cards are present in both the coin list and action-slot aliases.
    coins = {int(_field(card, "id")): card for card in _field(raw, "dropped_cards", ())
             if int(_field(card, "coin_type", 16)) == 16}
    seen_coins = set(coins)
    for card in _field(raw, "cards", ()):
        source = int(_field(card, "source", 0))
        if source not in (1, 2, 3):
            raise ValueError("staged reward requires native card source metadata; rebuild the DLL")
        if source == 3:
            source_id = int(_field(card, "source_id", -1))
            if source_id < 0:
                raise ValueError("dropped card alias has no source_id")
            if source_id in seen_coins:
                continue
            seen_coins.add(source_id)
        total += plant_price(int(_field(card, "plant_type")))
    for card in coins.values():
        total += plant_price(int(_field(card, "type")))
    return total


def planned_health_budget(raw: object) -> float:
    global_state = _field(raw, "global_state", _field(raw, "global", {}))
    if not bool(_field(global_state, "spawn_plan_available", False)):
        raise ValueError("staged reward needs the native full zombie spawn plan; rebuild the DLL")
    if int(_field(global_state, "game_mode", 0)) != 0:
        raise ValueError("staged spawn health budgets currently support original Adventure only")
    waves = int(_field(global_state, "total_waves", 0))
    plan = tuple(_field(global_state, "zombie_spawn_plan", ()))
    if not 1 <= waves <= 100 or len(plan) != waves * 50:
        raise ValueError("invalid native zombie spawn plan dimensions")
    health = 0
    for start in range(0, len(plan), 50):
        for raw_type in plan[start:start + 50]:
            zombie_type = int(raw_type)
            if zombie_type == -1:
                break
            if not 0 <= zombie_type < len(PLANNED_ZOMBIE_HP):
                raise ValueError(f"unsupported planned zombie type {zombie_type}")
            health += PLANNED_ZOMBIE_HP[zombie_type]
    if health <= 0:
        raise ValueError("staged reward needs a non-empty finite spawn plan")
    return float(health)


def _health_records(raw: object):
    # The native ledger includes dying slots omitted from the policy observation.
    # 计入预设波次僵尸 (from_wave >= 0)、动态生成僵尸 (from_wave == -1) 以及僵王本体 (type == 25)
    records = _field(raw, "zombie_health", None)
    if records is None:
        raise ValueError("staged reward needs native zombie_health diagnostics")
    return {int(_field(z, "id")): z for z in records
            if (int(_field(z, "from_wave", 0)) >= -1 and int(_field(z, "type", -1)) >= 0)
            or int(_field(z, "type", -1)) == 25}



def _hp(z: object, *, maximum: bool = False) -> float:
    names = ("max_hp", "helm_max_hp", "shield_max_hp", "flying_max_hp") if maximum else (
        "body_hp", "helm_hp", "shield_hp", "flying_hp")
    return sum(max(0.0, _number(_field(z, name, 0))) for name in names)


class StagedReward:
    """One fresh account per actual game; payouts occur at fixed tick stages."""

    def __init__(self, initial_info: Mapping[str, Any]):
        raw = initial_info.get("raw_observation")
        self.health_budget = planned_health_budget(raw)
        self.stage_asset = resource_value(raw)
        self.asset = self.stage_asset
        self.ticks = 0
        self.stage_start_tick = 0
        self.damage = 0.0
        self.damage_fraction = self.stage_damage_fraction = 0.0
        self.stages = 0
        self.accumulated_stage_reward = 0.0  # 整局累计正向阶段辅助奖励
        self.finished = False
        self.last_breakdown: dict[str, Any] = {}
        records = _health_records(raw)
        # Existing injuries/deaths at attachment are baseline, not new damage.
        self._remaining = {key: (0.0 if bool(_field(z, "is_dead_or_dying", False)) else _hp(z))
                           for key, z in records.items()}
        self._previous = records

    def _advance_damage(self, raw: object, *, lost: bool, killed: int) -> None:
        current = _health_records(raw)
        for key, z in current.items():
            dead = bool(_field(z, "is_dead_or_dying", False))
            if bool(_field(z, "is_mind_controlled", False)) or (lost and dead):
                continue
            dying_phase = int(_field(z, "phase", 0)) in (1, 2, 3)
            remaining = 0.0 if dead and dying_phase else _hp(z)
            before = self._remaining.get(key, _hp(z, maximum=True))
            # Keep a low-water mark: healing cannot farm the same HP twice.
            self.damage += max(0.0, before - remaining)
            self._remaining[key] = min(before, remaining)
        # Instant kills may be freed before the next snapshot. Use the native
        # disappearance count only for previously observed, non-escaping foes.
        # Lost/fallback snapshots never turn terminal cleanup into damage.
        if not lost:
            missing = [key for key, z in self._previous.items()
                       if key not in current and not bool(_field(z, "is_mind_controlled", False))
                       and int(_field(z, "type", -1)) not in (19, 20)
                       and self._remaining.get(key, 0.0) > 0]
            for key in missing[:max(0, killed)]:
                self.damage += self._remaining[key]
                self._remaining[key] = 0.0
        self._previous = current
        self.damage_fraction = min(1.0, self.damage / self.health_budget)

    def step(self, raw_reward: float, info: Mapping[str, Any], *, terminated: bool,
             truncated: bool) -> float:
        if self.finished:
            raise ValueError("staged reward was already finalized; reset the episode")
        ticks = info.get("ticks_advanced")
        if type(ticks) is not int or ticks < 0:
            raise ValueError("staged reward requires non-negative integer ticks_advanced")
        events = info.get("events") or {}
        raw = info.get("raw_observation")
        lost = bool(events.get("lost")) or info.get("result") == "LOST"
        fallback = bool(info.get("terminal_observation_fallback", False))
        if not fallback:
            self._advance_damage(raw, lost=lost, killed=int(events.get("zombies_killed", 0)))
            self.asset = resource_value(raw)
        self.ticks += ticks
        stage_reward = 0.0
        resource_delta = damage_delta = 0.0
        close_stage = self.ticks >= self.stage_start_tick + STAGE_TICKS or terminated or truncated
        if close_stage:
            resource_delta = self.asset - self.stage_asset
            damage_delta = self.damage_fraction - self.stage_damage_fraction
            raw_stage_reward = min(resource_delta / RESOURCE_REFERENCE + STAGE_DAMAGE_WEIGHT * damage_delta,
                                   STAGE_CAP)
            if raw_stage_reward > 0.0:
                allowed = max(0.0, STAGE_LIFETIME_CAP - self.accumulated_stage_reward)
                if allowed <= 1e-9:
                    stage_reward = 0.0
                else:
                    stage_reward = min(raw_stage_reward, allowed)
                    self.accumulated_stage_reward = min(STAGE_LIFETIME_CAP,
                                                        self.accumulated_stage_reward + stage_reward)
            else:
                stage_reward = raw_stage_reward
            self.stage_asset = self.asset
            self.stage_damage_fraction = self.damage_fraction
            # Continuous mode can overrun a boundary; settle observed facts once,
            # never invent multiple payouts for a large unobserved clock jump.
            self.stage_start_tick = self.ticks // STAGE_TICKS * STAGE_TICKS
            self.stages += 1
        mowers = max(0, int(events.get("mowers_used", 0)))
        mower_reward = -MOWER_PENALTY * mowers
        terminal_reward = 0.0
        if terminated:
            if info.get("result") not in ("WON", "LOST"):
                raise ValueError("staged terminal reward requires a real WON/LOST result")
            terminal_reward = (1.0 if info["result"] == "WON" else -1.0)
            terminal_reward += TERMINAL_DAMAGE_WEIGHT * self.damage_fraction
        self.finished = bool(terminated or truncated)
        self.last_breakdown = {
            "stage_settled": close_stage, "stage_reward": stage_reward,
            "accumulated_stage_reward": self.accumulated_stage_reward,
            "stage_resource_delta": resource_delta, "stage_damage_delta": damage_delta,
            "mower_reward": mower_reward, "terminal_reward": terminal_reward,
            "resource_value": self.asset, "damage_fraction": self.damage_fraction,
            "damage_hp": self.damage, "planned_health_budget": self.health_budget,
            "reward_stages": self.stages,
        }
        return stage_reward + mower_reward + terminal_reward
