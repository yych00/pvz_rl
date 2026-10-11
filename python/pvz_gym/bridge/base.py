"""Native facts transport contract; it intentionally contains no PvZ rules."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Mapping

from ..types import Action, ResetFacts, StepFacts


class GameFactsBackend(ABC):
    @abstractmethod
    def reset(self, config: Mapping[str, object]) -> ResetFacts: ...

    @abstractmethod
    def step(self, action: Action, ticks: int) -> StepFacts: ...

    def close(self) -> None:
        pass
