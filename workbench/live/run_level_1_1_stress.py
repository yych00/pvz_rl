# -*- coding: utf-8 -*-
"""Standalone destructive stress test for Adventure 1-1.

Usage from repository root:
    python -m workbench.live.run_level_1_1_stress --episodes 20
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

TOOLS_GYM = Path(__file__).resolve().parents[2]
for path in (TOOLS_GYM, TOOLS_GYM / "python"):
    value = str(path)
    if value not in sys.path:
        sys.path.insert(0, value)

from workbench import PvzGymNativeDll, run_level_1_1_stress

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("pvz_gym.level_1_1_stress")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run repeated destructive 1-1 LIVE E2E episodes.")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--max-ticks", type=int, default=60000)
    parser.add_argument("--step-ticks", type=int, default=50)
    parser.add_argument("--max-plants", type=int, default=40)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    client = PvzGymNativeDll()

    def progress(phase: str, message: str) -> None:
        logger.info("[%s] %s", phase, message)

    logger.warning("1-1 LIVE STRESS：episodes=%d；会反复切换/重开真实 1-1", max(1, args.episodes))
    result = run_level_1_1_stress(
        client,
        episodes=args.episodes,
        progress=progress,
        max_ticks=args.max_ticks,
        step_ticks=args.step_ticks,
        max_plants=args.max_plants,
    )
    logger.info(
        "STRESS SUMMARY: completed=%d/%d wins=%d losses=%d failures=%d "
        "setup_failures=%d timeouts=%d connection_or_exception=%d "
        "win_rate=%.1f%% elapsed=%.1fms",
        result.completed_episodes,
        result.requested_episodes,
        result.wins,
        result.losses,
        result.failures,
        result.setup_failures,
        result.timeouts,
        result.connection_or_exception_failures,
        result.win_rate * 100.0,
        result.elapsed_ms,
    )
    return 0 if result.success else 1


if __name__ == "__main__":
    raise SystemExit(main())
