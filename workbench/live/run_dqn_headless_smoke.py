# -*- coding: utf-8 -*-
"""Run one destructive LIVE headless Double-DQN smoke/benchmark episode.

Examples from the repository root::

    python -m workbench.live.run_dqn_headless_smoke
    python -m workbench.live.run_dqn_headless_smoke --benchmark --benchmark-steps 1000

Benchmark mode collects timings in memory and prints once at the end so console
I/O does not contaminate per-step measurements. It also exercises replay
add/sample and real DQN learn calls after one batch is available.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

TOOLS_GYM = Path(__file__).resolve().parents[2]
for path in (TOOLS_GYM, TOOLS_GYM / "python"):
    value = str(path)
    if value not in sys.path:
        sys.path.insert(0, value)

from pvz_gym.client import PvZClient
from pvz_gym.env import PvZEnv
from train.dqn.agent import DoubleDQNAgent
from train.dqn.config import DoubleDQNConfig
from train.dqn.features import normalize_observation
from train.dqn.replay import ReplayBuffer
from train.dqn.reward import training_reward

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("pvz_gym.dqn_headless_smoke")


class _Benchmark:
    def __init__(self) -> None:
        self.samples: dict[str, list[float]] = defaultdict(list)

    def add_ms(self, name: str, started: float) -> None:
        self.samples[name].append((time.perf_counter() - started) * 1000.0)

    def print(self, *, steps: int, ticks: int, wall_seconds: float) -> None:
        print("\n========== DQN SMOKE BENCHMARK ==========")
        for name in (
            "reset",
            "select_action",
            "env_step",
            "normalize_observation",
            "replay_add",
            "replay_sample",
            "learn",
        ):
            values = self.samples.get(name, [])
            if not values:
                continue
            array = np.asarray(values, dtype=np.float64)
            print(
                f"{name:24s} n={array.size:5d} avg={array.mean():8.3f} ms "
                f"p50={np.percentile(array, 50):8.3f} "
                f"p95={np.percentile(array, 95):8.3f} max={array.max():8.3f}"
            )
        wall = max(float(wall_seconds), 1e-9)
        print(f"{'steps_per_second':24s} {steps / wall:12.2f}")
        print(f"{'game_ticks_per_second':24s} {ticks / wall:12.2f}")
        print(f"{'wall_seconds':24s} {wall:12.3f}")
        print("=========================================\n")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one real-game headless DQN smoke episode")
    parser.add_argument("--level-id", type=int, default=4, help="Adventure level ID; default: 4 (1-4)")
    parser.add_argument("--playthrough", type=int, choices=(1, 2), default=1)
    parser.add_argument("--seed", type=int, default=12345, help="Gym reset seed")
    parser.add_argument("--max-steps", type=int, default=5000, help="Fail if no terminal result by this many steps")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--ticks-per-step", type=int, default=150)
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="collect reset/step/replay/learn timings and print one summary",
    )
    parser.add_argument(
        "--benchmark-steps",
        type=int,
        default=1000,
        help="fixed step budget in --benchmark mode; default: 1000",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.max_steps <= 0:
        raise SystemExit("--max-steps must be positive")
    if args.ticks_per_step <= 0:
        raise SystemExit("--ticks-per-step must be positive")
    if args.benchmark_steps <= 0:
        raise SystemExit("--benchmark-steps must be positive")

    config = DoubleDQNConfig(
        env=DoubleDQNConfig().env.__class__(
            level_id=args.level_id,
            playthrough=args.playthrough,
            ticks_per_step=args.ticks_per_step,
            simulation_mode="headless",
        ),
        total_steps=args.max_steps,
        epsilon_start=1.0,
        epsilon_end=1.0,
        epsilon_decay_steps=1,
        device=args.device,
    )

    env = PvZEnv(
        PvZClient(
            automation_enabled=True,
            auto_dialogue=True,
            auto_collect=True,
            auto_select_seeds=True,
            auto_skip_tutorial=True,
            background_running=True,
            deterministic_pause=True,
        ),
        level_id=args.level_id,
        playthrough=args.playthrough,
        ticks_per_step=args.ticks_per_step,
        max_episode_steps=args.max_steps,
        simulation_mode="headless",
    )

    benchmark = _Benchmark() if args.benchmark else None
    started = time.perf_counter()
    total_ticks = 0
    long_tick_jumps = 0
    result = "UNKNOWN"
    completed_steps = 0

    logger.warning(
        "LIVE DQN smoke: level=%d mode=headless seed=%d max_steps=%d benchmark=%s",
        args.level_id,
        args.seed,
        args.max_steps,
        args.benchmark,
    )

    try:
        reset_started = time.perf_counter()
        observation, info = env.reset(seed=args.seed)
        if benchmark is not None:
            benchmark.add_ms("reset", reset_started)
        observation = normalize_observation(observation)
        action_dim = int(env.action_space.n)
        agent = DoubleDQNAgent(action_dim, config, device=args.device)
        replay = ReplayBuffer(config.replay_capacity, action_dim, seed=args.seed + 1)
        step_limit = min(args.max_steps, args.benchmark_steps) if args.benchmark else args.max_steps

        for step in range(1, step_limit + 1):
            mask = np.asarray(info.get("action_mask", ()), dtype=np.bool_)
            if mask.shape != (action_dim,):
                raise RuntimeError(f"action mask shape {mask.shape} != {(action_dim,)}")
            if not mask.any():
                raise RuntimeError("playing state exposed no legal actions")

            timing = time.perf_counter()
            action = agent.select_action(
                observation,
                mask,
                env_step=step - 1,
                explore=True,
            )
            if benchmark is not None:
                benchmark.add_ms("select_action", timing)

            timing = time.perf_counter()
            next_observation, raw_reward, terminated, truncated, next_info = env.step(action)
            if benchmark is not None:
                benchmark.add_ms("env_step", timing)

            timing = time.perf_counter()
            next_observation = normalize_observation(next_observation)
            if benchmark is not None:
                benchmark.add_ms("normalize_observation", timing)

            done = bool(terminated or truncated)
            next_mask = np.asarray(next_info.get("action_mask", ()), dtype=np.bool_)
            reward = training_reward(
                raw_reward,
                next_info,
                reward_scale=config.reward_scale,
                mower_penalty=config.mower_penalty,
            )

            timing = time.perf_counter()
            replay.add(observation, action, reward, next_observation, done, next_mask,
                       ticks_advanced=int(next_info["ticks_advanced"]))
            if benchmark is not None:
                benchmark.add_ms("replay_add", timing)

            if benchmark is not None and len(replay) >= config.batch_size and step % config.train_every == 0:
                timing = time.perf_counter()
                batch = replay.sample(config.batch_size)
                benchmark.add_ms("replay_sample", timing)
                timing = time.perf_counter()
                agent.learn(batch)
                benchmark.add_ms("learn", timing)

            observation = next_observation
            info = next_info
            completed_steps = step
            ticks_requested = max(1, int(info.get("ticks_requested", args.ticks_per_step) or 1))
            ticks_advanced = max(0, int(info.get("ticks_advanced", ticks_requested) or 0))
            total_ticks += ticks_advanced
            if ticks_advanced > ticks_requested * 2:
                long_tick_jumps += 1

            result = str(info.get("result", "UNKNOWN"))
            if not args.benchmark and (step == 1 or step % 100 == 0 or done):
                elapsed = max(time.perf_counter() - started, 1e-9)
                logger.info(
                    "step=%d result=%s game_clock=%s ticks=%d rate=%.0f ticks/s long_jumps=%d",
                    step,
                    result,
                    info.get("game_clock", 0),
                    total_ticks,
                    total_ticks / elapsed,
                    long_tick_jumps,
                )

            if done:
                elapsed = time.perf_counter() - started
                if benchmark is not None:
                    benchmark.print(steps=completed_steps, ticks=total_ticks, wall_seconds=elapsed)
                if result in ("WON", "LOST"):
                    logger.info(
                        "DQN HEADLESS SMOKE PASS: result=%s steps=%d ticks=%d elapsed=%.2fs rate=%.0f ticks/s",
                        result,
                        step,
                        total_ticks,
                        elapsed,
                        total_ticks / max(elapsed, 1e-9),
                    )
                    return 0
                logger.error(
                    "DQN HEADLESS SMOKE FAIL: terminal/truncated without authoritative result; result=%s step=%d",
                    result,
                    step,
                )
                return 1

        elapsed = time.perf_counter() - started
        if benchmark is not None:
            benchmark.print(steps=completed_steps, ticks=total_ticks, wall_seconds=elapsed)
            logger.info(
                "DQN HEADLESS BENCHMARK PASS: fixed budget=%d steps result=%s",
                completed_steps,
                result,
            )
            return 0

        logger.error(
            "DQN HEADLESS SMOKE FAIL: no terminal result after %d steps (last result=%s)",
            args.max_steps,
            result,
        )
        return 1
    except Exception:
        logger.exception("DQN HEADLESS SMOKE FAIL")
        return 1
    finally:
        try:
            env.close()
        except Exception:
            logger.exception("failed to close smoke environment cleanly")


if __name__ == "__main__":
    raise SystemExit(main())
