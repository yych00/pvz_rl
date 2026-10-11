"""Episode orchestrator connecting SeedSelection and Battle in one PvZSession."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable

from pvz_gym.envs.seed_selection import SeedSelectionEnv
from pvz_gym.envs.battle import BattleEnv
from pvz_gym.rewards.seed_selection import deck_outcome_reward
from pvz_gym.session import PvZSession
from pvz_gym.types import GameStatus

logger = logging.getLogger("pvz_gym.eval.episode_runner")


@dataclass
class SeedStepRecord:
    obs: Any
    action: int
    reward: float
    terminated: bool
    truncated: bool
    info: dict[str, Any]


@dataclass
class BattleStepRecord:
    obs: Any
    action: Any
    reward: float
    terminated: bool
    truncated: bool
    info: dict[str, Any]


@dataclass
class EpisodeResult:
    level_id: int
    seed_trajectory: list[SeedStepRecord] = field(default_factory=list)
    battle_trajectory: list[BattleStepRecord] = field(default_factory=list)
    won: bool = False
    lost: bool = False
    deck_return: float = 0.0
    total_battle_reward: float = 0.0
    selected_deck: list[int] = field(default_factory=list)
    imitater_target: int = -1


class EpisodeRunner:
    """Run SeedChooser -> Battle without ever resetting between the phases."""

    def __init__(
        self,
        session: PvZSession | None = None,
        seed_env: SeedSelectionEnv | None = None,
        battle_env: BattleEnv | None = None,
    ):
        if session is None and seed_env is None and battle_env is None:
            session = PvZSession()
            self._owns_session = True
        elif session is not None:
            self._owns_session = False
        else:
            inferred = getattr(battle_env, "session", None)
            seed_backend = getattr(seed_env, "backend", None)
            if inferred is None or seed_backend is not inferred:
                raise ValueError(
                    "custom seed_env and battle_env must share the same PvZSession"
                )
            session = inferred
            self._owns_session = False

        assert session is not None
        if seed_env is not None and getattr(seed_env, "backend", None) is not session:
            raise ValueError("seed_env does not use the supplied PvZSession")
        if battle_env is not None and getattr(battle_env, "session", None) is not session:
            raise ValueError("battle_env does not use the supplied PvZSession")

        self.session = session
        self.seed_env = seed_env or SeedSelectionEnv(session=session)
        self.battle_env = battle_env or BattleEnv(session=session)

    def run_episode(
        self,
        level_id: int = 1,
        deck_policy: Callable[[Any, list[bool]], int] | None = None,
        battle_policy: Callable[[Any, list[bool]], Any] | None = None,
        *,
        playthrough: int = 1,
        seed: int | None = None,
        max_seed_steps: int = 32,
        max_battle_steps: int = 1000,
    ) -> EpisodeResult:
        result = EpisodeResult(level_id=level_id)
        config: dict[str, object] = {
            "level_id": int(level_id),
            "playthrough": int(playthrough),
            "stop_at_seed_selection": True,
            "auto_select_seeds": False,
        }
        if seed is not None:
            config["seed"] = int(seed)

        prepared = self.session.begin_episode(config)

        if prepared.status == GameStatus.CHOOSING_SEEDS:
            seed_obs, seed_info = self.seed_env.reset()
            seed_finished = False
            for _ in range(max_seed_steps):
                mask = list(seed_info.get("action_mask", []))
                if deck_policy is not None:
                    action = deck_policy(seed_obs, mask)
                elif seed_obs.get("can_confirm") and mask and mask[-1]:
                    action = len(mask) - 1
                else:
                    legal_actions = [i for i, allowed in enumerate(mask) if allowed]
                    action = legal_actions[0] if legal_actions else 0

                next_obs, reward, term, trunc, info = self.seed_env.step(action)
                result.seed_trajectory.append(
                    SeedStepRecord(seed_obs, action, reward, term, trunc, info)
                )
                seed_obs, seed_info = next_obs, info
                if term:
                    seed_finished = True
                    break
                if trunc:
                    break

            result.selected_deck = list(seed_info.get("selected_seeds", []))
            result.imitater_target = int(seed_info.get("imitater_target", -1))
            if not seed_finished:
                return result

            # Critical invariant: this advances the already prepared native
            # episode. It must never call select_level/reset again.
            self.session.wait_for_battle()
        elif prepared.status != GameStatus.PLAYING:
            raise RuntimeError(
                f"episode preparation ended in unexpected phase: {prepared.status.name}"
            )

        battle_obs, battle_info = self.battle_env.observe_current()
        won = lost = False
        for _ in range(max_battle_steps):
            mask = list(battle_info.get("action_mask", []))
            action = battle_policy(battle_obs, mask) if battle_policy is not None else 0
            next_obs, reward, term, trunc, info = self.battle_env.step(action)
            result.total_battle_reward += reward
            result.battle_trajectory.append(
                BattleStepRecord(battle_obs, action, reward, term, trunc, info)
            )
            battle_obs, battle_info = next_obs, info

            status = str(info.get("result", "UNKNOWN"))
            events = info.get("events") or {}
            if status == "WON" or bool(events.get("won", False)):
                won = True
            if status == "LOST" or bool(events.get("lost", False)):
                lost = True
            if term or trunc:
                break

        if not won and not lost:
            won = self.session.status == GameStatus.WON
            lost = self.session.status == GameStatus.LOST

        result.won = won
        result.lost = lost
        result.deck_return = deck_outcome_reward(won=won, lost=lost)
        return result

    def close(self) -> None:
        for env in (self.seed_env, self.battle_env):
            try:
                env.close()
            except Exception:
                pass
        if self._owns_session:
            self.session.close()
