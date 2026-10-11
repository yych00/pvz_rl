# -*- coding: utf-8 -*-
"""Standalone destructive LIVE E2E test for Adventure 1-1.

Usage from repository root:
    python -m workbench.live.run_level_1_1_e2e

This is intentionally not part of CI because it controls the real game.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

TOOLS_GYM = Path(__file__).resolve().parents[2]
for path in (TOOLS_GYM, TOOLS_GYM / "python"):
    value = str(path)
    if value not in sys.path:
        sys.path.insert(0, value)

from workbench import PvzGymNativeDll, run_level_1_1_e2e

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("pvz_gym.level_1_1_single_e2e")


def main() -> int:
    client = PvzGymNativeDll()

    def progress(phase: str, message: str) -> None:
        logger.info("[%s] %s", phase, message)

    logger.warning("单次 LIVE E2E：将主动切换到 1-1 并推进真实游戏")
    result = run_level_1_1_e2e(client, progress=progress)
    if result.success:
        logger.info(
            "SINGLE E2E PASS: terminal=%s path=%s ticks=%d plants=%d collected=%d elapsed=%.1fms",
            result.outcome,
            result.entry_path,
            result.ticks_advanced,
            result.plants_executed,
            result.collected_items,
            result.elapsed_ms,
        )
        return 0
    logger.error("SINGLE E2E FAIL: outcome=%s message=%s", result.outcome, result.message)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
