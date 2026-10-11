"""Per-transition assets, conservative damage and a bounded net shaping balance.

Native snapshots lack damage-source/cleanup events. Damage and threat use
conservative snapshots and exclude rows swept by a mower. Missing zombies never
imply damage or threat reduction. These are approximate snapshot metrics.
"""

from collections.abc import Mapping

import numpy as np
from pvz_gym.observation import ZOMBIE_BOSS_TYPE

from .staged_reward import (_field, _health_records, _hp, _number, resource_value,
                            planned_health_budget)

ASSET_REFERENCE = 5000.0
DAMAGE_WEIGHT = 0.30
THREAT_WEIGHT = 0.10
THREAT_STEP_CAP = 0.01
THREAT_SPAWN_X = 900.0
THREAT_HOUSE_X = 0.0
POSITIVE_CAP = NEGATIVE_CAP = 0.40
FIRST_MOWER_PENALTY = 0.50
LATER_MOWER_PENALTY = 0.10
METRICS_VERSION = 4
MOWER_BASIS = "native_trigger_counter_and_squish_v1"
BOSS_BASIS = "nonspatial_boss_no_threat_mower_independent_damage_v1"


def smooth_shaping_balance(total):
    """Odd softsign with slope 1 at zero and asymptotes +/-0.40.

    Rational saturation retains a useful derivative much farther from zero than
    tanh. It never deliberately clips finite same-direction input to zero.
    """
    return POSITIVE_CAP * (total / (POSITIVE_CAP + abs(total)))


def smooth_shaping_delta(before, delta):
    """Compute F(before + delta) - F(before) without cancellation near a bound."""
    after = before + delta
    if (before >= 0 and after >= 0) or (before <= 0 and after <= 0):
        return delta * (POSITIVE_CAP / (POSITIVE_CAP + abs(before))) * (POSITIVE_CAP / (POSITIVE_CAP + abs(after)))
    return smooth_shaping_balance(after) - smooth_shaping_balance(before)


def asset_value(raw):
    # Reuse original-price/upgrade/alias accounting, but exclude reusable slots.
    cards = tuple(c for c in _field(raw, "cards", ())
                  if int(_field(c, "source", 0)) != 1)
    return resource_value({"global_state": _field(raw, "global_state", _field(raw, "global", {})),
                           "plants": _field(raw, "plants", ()), "cards": cards,
                           "dropped_cards": _field(raw, "dropped_cards", ())})


def _mowers(raw):
    return {int(_field(m, "row", -1)): str(_field(m, "state", "LOST"))
            for m in _field(raw, "mowers", ())}


def _remaining(z):
    if bool(_field(z, "is_dead_or_dying", False)) and int(_field(z, "phase", 0)) in (1, 2, 3):
        return 0.0
    return _hp(z)


def _threat(z):
    if int(_field(z, "type", -1)) == ZOMBIE_BOSS_TYPE:
        return 0.0
    if bool(_field(z, "is_mind_controlled", False)) or bool(_field(z, "is_dead_or_dying", False)):
        return 0.0
    x = _number(_field(z, "x", THREAT_SPAWN_X))
    proximity = min(1.0, max(0.0, (THREAT_SPAWN_X - x) / (THREAT_SPAWN_X - THREAT_HOUSE_X)))
    return _hp(z) * proximity ** 2


class TransitionDenseReward:
    def __init__(self, initial_info: Mapping, *, shaping_limit_mode="smooth_total"):
        if shaping_limit_mode not in ("smooth_total", "episode_budget"):
            raise ValueError("invalid shaping_limit_mode")
        self.shaping_limit_mode = shaping_limit_mode
        raw = initial_info.get("raw_observation")
        global_state = _field(raw, "global_state", _field(raw, "global", {}))
        if _field(global_state, "board_id", 0) and _field(global_state, "triggered_mowers", None) is None:
            raise ValueError("native mower facts unavailable; restart Workbench with the rebuilt DLL")
        self.health_budget = planned_health_budget(raw)
        self.asset = asset_value(raw)
        self.damage = 0.0
        self.positive = self.negative = 0.0
        self.raw_shaping_total = self.shaping_balance = 0.0
        self.mowers_used = 0
        self.first_mower_seen = False
        self.finished = False
        self.last_breakdown = {}
        self._mowers = _mowers(raw)
        self._previous = _health_records(raw)
        self._remaining = {key: _remaining(z) for key, z in self._previous.items()}
        self.threat = sum(_threat(z) for z in self._previous.values()) / self.health_budget

    def observe(self, observation):
        # New array each time: later accounting must not mutate replay inputs.
        return {**observation, "reward_context": np.asarray((
            float(self.first_mower_seen),
            self._available_positive() / POSITIVE_CAP,
            self._available_negative() / NEGATIVE_CAP,
        ), dtype=np.float32)}

    def _available_positive(self):
        if self.shaping_limit_mode == "smooth_total":
            return POSITIVE_CAP - self.shaping_balance
        return max(0.0, POSITIVE_CAP - self.positive)

    def _available_negative(self):
        if self.shaping_limit_mode == "smooth_total":
            return NEGATIVE_CAP + self.shaping_balance
        return max(0.0, NEGATIVE_CAP - self.negative)

    def step(self, raw_reward, info, *, terminated, truncated):
        if self.finished:
            raise ValueError("transition_dense reward was already finalized; reset the episode")
        ticks = info.get("ticks_advanced")
        if type(ticks) is not int or not 0 <= ticks < 2**63:
            raise ValueError("transition_dense requires non-negative int64 ticks_advanced")
        result = info.get("result")
        if terminated and result not in ("WON", "LOST"):
            raise ValueError("transition_dense terminal reward requires a real WON/LOST result")
        events = info.get("events") or {}
        reported_mowers = max(0, int(events.get("mowers_used", 0)))
        fallback = bool(info.get("terminal_observation_fallback", False))
        asset_delta = damage_delta = 0.0
        threat_before = threat_after = 0.0
        blocked_rows = set()
        mowers = reported_mowers
        if not fallback:
            raw = info.get("raw_observation")
            current_mowers = _mowers(raw)
            changed_rows = {row for row, state in self._mowers.items()
                            if state == "READY" and current_mowers.get(row) != "READY"}
            # Only native combat events charge mower penalties. Disappearance
            # alone includes victory cleanup, even while status is PLAYING.
            event_rows = {int(row) for row in events.get("mower_triggered_rows", ())}
            swept_rows = event_rows or (changed_rows if reported_mowers else set())
            blocked_rows = swept_rows | {row for row, state in self._mowers.items() if state == "ROLLING"}
            blocked_rows |= {row for row, state in current_mowers.items() if state == "ROLLING"}
            suppress_all = reported_mowers > len(swept_rows)
            current = _health_records(raw)
            for key, z in current.items():
                remaining = _remaining(z)
                # Newly seen injured enemies are baselined: their damage source
                # is unknown. Retain low-water marks across healing/reappearance.
                before = self._remaining.get(key, remaining)
                is_boss = int(_field(z, "type", -1)) == ZOMBIE_BOSS_TYPE
                mower_safe = is_boss or (not suppress_all
                    and int(_field(z, "row", -1)) not in blocked_rows
                    and (not blocked_rows or int(_field(z, "row", -1)) >= 0))
                if (mower_safe
                        and not bool(_field(z, "is_mind_controlled", False))
                        and not (terminated and bool(_field(z, "is_dead_or_dying", False)))):
                    damage_delta += max(0.0, before - remaining)
                self._remaining[key] = min(before, remaining)
                previous = self._previous.get(key)
                # Compare the same hostile cohort. Births, unconfirmed removals,
                # faction changes and terminal cleanup never award threat delta.
                if (previous is not None and not terminated and not suppress_all
                        and not bool(_field(previous, "is_mind_controlled", False))
                        and not bool(_field(z, "is_mind_controlled", False))
                        and int(_field(previous, "row", -1)) not in blocked_rows
                        and int(_field(z, "row", -1)) not in blocked_rows
                        and (not blocked_rows or (int(_field(previous, "row", -1)) >= 0
                                                  and int(_field(z, "row", -1)) >= 0))):
                    threat_before += _threat(previous) / self.health_budget
                    threat_after += _threat(z) / self.health_budget
            self._previous = current
            self.threat = sum(_threat(z) for z in current.values()) / self.health_budget
            asset = asset_value(raw)
            asset_delta = asset - self.asset
            self.asset = asset
            self._mowers = current_mowers
        else:
            # A fallback repeats the last reliable board; no metric deltas.
            # Event count is the only available mower evidence on this boundary.
            mowers = reported_mowers
        damage_delta = min(damage_delta, max(0.0, self.health_budget - self.damage))
        self.damage += damage_delta
        asset_reward = asset_delta / ASSET_REFERENCE
        damage_reward = DAMAGE_WEIGHT * damage_delta / self.health_budget
        threat_delta = threat_before - threat_after
        threat_reward = min(THREAT_STEP_CAP, max(-THREAT_STEP_CAP, THREAT_WEIGHT * threat_delta))
        raw_shaping = asset_reward + damage_reward + threat_reward
        if self.shaping_limit_mode == "smooth_total":
            shaping = smooth_shaping_delta(self.raw_shaping_total, raw_shaping)
            self.shaping_balance = smooth_shaping_balance(self.raw_shaping_total + raw_shaping)
        else:
            shaping = (min(raw_shaping, self._available_positive()) if raw_shaping >= 0
                       else -min(-raw_shaping, self._available_negative()))
        self.raw_shaping_total += raw_shaping
        self.positive += max(0.0, shaping)
        self.negative += max(0.0, -shaping)
        if self.shaping_limit_mode == "episode_budget":
            self.positive = min(POSITIVE_CAP, self.positive)
            self.negative = min(NEGATIVE_CAP, self.negative)
            self.shaping_balance = self.positive - self.negative
        first = bool(mowers and not self.first_mower_seen)
        first_reward = -FIRST_MOWER_PENALTY if first else 0.0
        later_reward = -LATER_MOWER_PENALTY * (mowers - int(first))
        self.first_mower_seen |= first
        self.mowers_used += mowers
        terminal = (1.0 if result == "WON" else -1.0) if terminated else 0.0
        self.finished = bool(terminated or truncated)
        self.last_breakdown = {
            "asset_reward": asset_reward, "damage_reward": damage_reward,
            "threat_reward": threat_reward, "threat_enabled": True,
            "threat_value": self.threat, "threat_delta": threat_delta,
            "threat_cohort_before": threat_before, "threat_cohort_after": threat_after,
            "threat_basis": "matched_hostile_snapshot_cohort_mower_rows_excluded",
            "boss_basis": BOSS_BASIS,
            "raw_shaping_reward": raw_shaping, "shaping_reward": shaping,
            "shaping_limit_mode": self.shaping_limit_mode,
            "shaping_raw_total": self.raw_shaping_total,
            "shaping_balance": self.shaping_balance,
            "shaping_attenuation": shaping / raw_shaping if raw_shaping else 1.0,
            "positive_shaping_used": self.positive, "negative_shaping_used": self.negative,
            "positive_shaping_remaining": self._available_positive(),
            "negative_shaping_remaining": self._available_negative(),
            "first_mower_reward": first_reward, "later_mower_reward": later_reward,
            "mower_reward": first_reward + later_reward, "mowers_used": self.mowers_used,
            "mower_basis": MOWER_BASIS,
            "terminal_reward": terminal, "resource_value": self.asset,
            "asset_delta": asset_delta, "damage_delta_hp": damage_delta,
            "damage_hp": self.damage, "damage_fraction": self.damage / self.health_budget,
            "planned_health_budget": self.health_budget,
            "damage_basis": "conservative_snapshot_low_watermark",
            "damage_suppressed_rows": sorted(blocked_rows),
            "terminal_observation_fallback": fallback,
        }
        return shaping + first_reward + later_reward + terminal
