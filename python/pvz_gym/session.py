"""One authoritative owner for a real PvZ episode across lifecycle phases."""

from __future__ import annotations

from collections.abc import Mapping

from .client import PvZClient
from .types import Action, GameStatus, Observation, ResetFacts, StepFacts


class PvZSession:
    """Own one native backend and one continuous episode lifecycle.

    SeedSelectionEnv and BattleEnv can both view this object. Confirming a deck
    never creates a new session; wait_for_battle() advances the same real game
    from SeedChooser into Battle.
    """

    def __init__(self, backend: object | None = None):
        self.backend = backend or PvZClient()
        self._reset_facts: ResetFacts | None = None
        self._step_facts: StepFacts | None = None
        self._closed = False

    @property
    def status(self) -> GameStatus:
        if self._step_facts is not None:
            return self._step_facts.status
        if self._reset_facts is not None:
            return self._reset_facts.status
        getter = getattr(self.backend, "get_game_status", None)
        if callable(getter):
            return GameStatus(getter())
        return GameStatus.UNKNOWN

    @property
    def current_observation(self) -> Observation | None:
        if self._step_facts is not None:
            return self._step_facts.observation
        if self._reset_facts is not None:
            return self._reset_facts.observation
        return None

    @property
    def reset_facts(self) -> ResetFacts | None:
        return self._reset_facts

    @property
    def step_facts(self) -> StepFacts | None:
        return self._step_facts

    def begin_episode(self, config: Mapping[str, object]) -> ResetFacts:
        self._ensure_open()
        facts = self.backend.reset(dict(config))
        if not isinstance(facts, ResetFacts):
            raise TypeError("backend.reset() must return ResetFacts")
        self._reset_facts = facts
        self._step_facts = None
        return facts

    def wait_for_battle(self, *, timeout_ms: int = 10000,
                        deterministic_pause: bool = True) -> ResetFacts:
        self._ensure_open()
        waiter = getattr(self.backend, "wait_for_battle", None)
        if not callable(waiter):
            raise RuntimeError("backend does not support same-session seed-to-battle handoff")
        facts = waiter(
            timeout_ms=int(timeout_ms),
            deterministic_pause=bool(deterministic_pause),
        )
        if not isinstance(facts, ResetFacts) or facts.observation is None:
            raise RuntimeError("wait_for_battle() did not return a battle observation")
        # Preserve whether the seed was actually applied when the episode began;
        # the handoff itself must not reseed or restart the game.
        previous = self._reset_facts
        if previous is not None:
            facts = ResetFacts(
                observation=facts.observation,
                game_clock=facts.game_clock,
                status=facts.status,
                rng_seed_support=previous.rng_seed_support,
                rng_seed_applied=previous.rng_seed_applied,
            )
        self._reset_facts = facts
        self._step_facts = None
        return facts

    def step(self, action: Action, ticks: int) -> StepFacts:
        self._ensure_open()
        if self.status != GameStatus.PLAYING:
            raise RuntimeError(f"battle step requires PLAYING session, got {self.status.name}")
        facts = self.backend.step(action, int(ticks))
        if not isinstance(facts, StepFacts):
            raise TypeError("backend.step() must return StepFacts")
        self._step_facts = facts
        return facts

    def get_action_mask(self) -> tuple[bool, ...]:
        self._ensure_open()
        if self._step_facts is not None and self._step_facts.action_mask:
            return tuple(bool(v) for v in self._step_facts.action_mask)
        getter = getattr(self.backend, "get_action_mask", None)
        if not callable(getter):
            return ()
        return tuple(bool(v) for v in getter())

    # SeedChooser view. These forwards intentionally use the same backend
    # instance that owns reset/step, preventing a second DLL/session owner.
    def get_seed_type_capacity(self) -> int:
        return int(self.backend.get_seed_type_capacity())

    def get_seed_selection(self):
        return self.backend.get_seed_selection()

    def get_seed_selection_mask(self):
        return self.backend.get_seed_selection_mask()

    def select_seed(self, plant_type: int):
        return self.backend.select_seed(int(plant_type))

    def select_imitater(self, target_type: int):
        return self.backend.select_imitater(int(target_type))

    def deselect_seed(self, plant_type: int):
        return self.backend.deselect_seed(int(plant_type))

    def confirm_seeds(self):
        return self.backend.confirm_seeds()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        close = getattr(self.backend, "close", None)
        if callable(close):
            close()

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("PvZSession is closed")
