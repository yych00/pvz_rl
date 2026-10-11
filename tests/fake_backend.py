"""Deterministic native-fact fixture; it does not simulate PvZ rules."""

from __future__ import annotations

from collections.abc import Mapping

from pvz_gym.action_space import ACTION_SPACE_SIZE
from pvz_gym.types import (
    Action,
    ActionResult,
    GameStatus,
    Observation,
    ResetFacts,
    RngSeedSupport,
    StepEvents,
    StepFacts,
)


class FactClient:
    def __init__(self, *, terminal_observation_unavailable: bool = False):
        self.clock = 0
        self.status = GameStatus.PLAYING
        self.advance_calls = 0
        self.call_order: list[str] = []
        self.terminal_observation_unavailable = terminal_observation_unavailable
        self.closed = False
        self.last_reset_config: dict[str, object] = {}

    def _observation(self) -> Observation:
        return Observation(
            global_state={
                "sun": 150,
                "current_wave": self.advance_calls,
                "total_waves": 10,
                "wave_countdown": 500,
                "game_clock": self.clock,
                "level_id": 27,
                "status": int(self.status),
            },
            cards=({"slot": 0, "plant_type": 0, "cost": 100,
                    "cooldown_duration": 100, "cooldown_remaining": 0, "ready": True},),
            plants=({"id": 1, "type": 0, "row": 2, "col": 0, "layer": 1,
                     "hp": 300, "max_hp": 300},),
            zombies=({"id": 1, "type": 0, "row": 2, "x": 700.0, "hp": 200,
                      "max_hp": 200, "shield_hp": 0},),
            mowers=({"row": 2, "state": "READY"},),
        )

    def reset(self, config: Mapping[str, object]) -> ResetFacts:
        self.call_order.append("reset")
        self.last_reset_config = dict(config)
        self.clock = 0
        self.advance_calls = 0
        self.status = GameStatus.PLAYING
        return ResetFacts(
            observation=self._observation(),
            game_clock=0,
            status=GameStatus.PLAYING,
            rng_seed_support=RngSeedSupport.FULL,
            rng_seed_applied=("seed" in config),
        )

    def step(self, action: Action, ticks: int) -> StepFacts:
        self.call_order.append("step")
        self.clock += int(ticks)
        self.advance_calls += 1
        previous = self.status
        if self.advance_calls == 2:
            self.status = GameStatus.WON
        fallback = self.terminal_observation_unavailable and self.status in (
            GameStatus.WON, GameStatus.LOST,
        )
        return StepFacts(
            observation=self._observation(),
            events=StepEvents(won=(previous != GameStatus.WON and self.status == GameStatus.WON)),
            action=ActionResult(action.type in ("noop", "plant"), None, "fixture"),
            action_mask=self.get_action_mask() if self.status == GameStatus.PLAYING else (False,) * ACTION_SPACE_SIZE,
            status=self.status,
            game_clock=self.clock,
            automation_collected=2,
            terminal_observation_fallback=fallback,
        )

    def get_action_mask(self):
        return (True, True) + (False,) * (ACTION_SPACE_SIZE - 2)

    def close(self):
        self.closed = True


FactBackend = FactClient
