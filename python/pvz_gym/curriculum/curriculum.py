"""Training policy configuration; native C++ applies the chosen configuration."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ResetConfig:
    level_id: int = 1
    seed: int | None = None
