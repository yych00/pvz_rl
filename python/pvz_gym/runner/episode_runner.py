"""Backward-compatible import path for :mod:`eval.episode_runner`."""

from __future__ import annotations

import sys
from pathlib import Path

_TOOLS_GYM = Path(__file__).resolve().parents[3]
if str(_TOOLS_GYM) not in sys.path:
    sys.path.insert(0, str(_TOOLS_GYM))

from eval.episode_runner import BattleStepRecord, EpisodeResult, EpisodeRunner, SeedStepRecord

__all__ = ["BattleStepRecord", "EpisodeResult", "EpisodeRunner", "SeedStepRecord"]
