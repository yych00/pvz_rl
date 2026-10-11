"""Backward-compatible import path for the legal-random evaluation runner."""

from __future__ import annotations

import sys
from pathlib import Path

_TOOLS_GYM = Path(__file__).resolve().parents[2]
if str(_TOOLS_GYM) not in sys.path:
    sys.path.insert(0, str(_TOOLS_GYM))

from eval.random_runner import RandomAgentConfig, RandomAgentSummary, run_random_agent

__all__ = ["RandomAgentConfig", "RandomAgentSummary", "run_random_agent"]
