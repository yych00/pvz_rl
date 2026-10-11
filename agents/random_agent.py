"""Live baseline agent sampling only actions approved by the native mask."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from types import SimpleNamespace

TOOLS_GYM = Path(__file__).resolve().parents[1]
for path in (TOOLS_GYM, TOOLS_GYM / "python"):
    value = str(path)
    if value not in sys.path:
        sys.path.insert(0, value)

from eval.random_runner import RandomAgentConfig, run_random_agent
from train.common import EnvConfig
from train.runtime import TrainingRuntimeConfig
from train.sampler import native_environment

EPISODES = 100
MAX_STEPS = 2000


def parse_seed_types(value: str | None) -> tuple[int, ...] | None:
    if not value:
        return None
    return tuple(int(item.strip()) for item in value.split(",") if item.strip())


def run(args: argparse.Namespace) -> int:
    settings = SimpleNamespace(
        env=EnvConfig(level_id=args.level_id, playthrough=args.playthrough, ticks_per_step=args.ticks_per_step),
        runtime=TrainingRuntimeConfig(game_executable=getattr(args, "game_exe", None)),
        reward_scheme="transition_dense",
    )
    env, instance = native_environment(settings, settings.runtime, 0, None, parse_seed_types(args.seed_types),
                                       auto_select_seeds=not args.no_auto_select_seeds)

    def report(event: str, data: dict) -> None:
        episode = data.get("episode", "-")
        step = data.get("step")
        if event == "episode_end":
            print(
                f"Episode {episode}/{args.episodes}: result={data['result']} "
                f"steps={data['steps']} reward={data['reward']:.1f}"
            )
        elif event not in ("episode_start", "step", "finished"):
            location = f"episode {episode}" + (f" step {step}" if step is not None else "")
            detail = data.get("error") or data.get("action_result") or event
            print(f"[{location}] {event}: {detail}", file=sys.stderr)

    try:
        summary = run_random_agent(
            env,
            RandomAgentConfig(
                episodes=args.episodes,
                max_steps=args.max_steps,
                random_seed=args.random_seed,
            ),
            on_event=report,
        )
    finally:
        try:
            env.close()
        finally:
            if instance is not None:
                instance.close()

    print(
        "\n===== Random Agent Summary =====\n"
        f"\nEpisodes: {summary.requested_episodes}\n"
        f"\nWins: {summary.wins}\n"
        f"Losses: {summary.losses}\n"
        f"Timeouts: {summary.timeouts}\n"
        f"\nWin rate: {summary.win_rate:.2%}\n"
        f"Mean episode length: {summary.mean_episode_length:.2f}\n"
        f"Mean reward: {summary.mean_reward:.2f}\n"
        f"\nInvalid actions: {summary.invalid_actions}\n"
        f"Invalid observations: {summary.invalid_observations}\n"
        f"Reset failures: {summary.reset_failures}\n"
        f"Game freezes: {summary.game_freezes}\n"
        f"Auto-collected: {summary.auto_collected}\n"
        f"Terminal observation fallbacks: {summary.terminal_observation_fallbacks}"
    )
    return 1 if summary.integrity_failures else 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--level-id", type=int, default=1, help="Adventure level ID, 1..50")
    parser.add_argument("--playthrough", type=int, choices=(1, 2), default=1)
    parser.add_argument("--seed-types", help="comma-separated plant type IDs for reset-time selection")
    parser.add_argument("--no-auto-select-seeds", action="store_true")
    parser.add_argument("--game-exe", help="complete retail 1065 game; defaults to repository configuration")
    parser.add_argument("--episodes", type=int, default=EPISODES)
    parser.add_argument("--max-steps", type=int, default=MAX_STEPS)
    parser.add_argument("--ticks-per-step", type=int, default=30)
    parser.add_argument("--random-seed", type=int, default=0)
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
