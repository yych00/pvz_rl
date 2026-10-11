"""CLI entrypoint for the controllable masked Double DQN trainer."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
from pvz_gym.runtime import SIMULATION_MODES

from .config import DoubleDQNConfig, convert_gamma
from .optimized_controller import DoubleDQNTrainerController
from .checkpoint_contract import CHECKPOINT_LOAD_MODES
from .exploration import EXPLORATION_MODE_LABELS, parse_action_family_weights, parse_exploration_ratio
from .run_state import load_checkpoint_seed_types, load_run_config, load_run_seed_types, resolve_resume_checkpoint
from .reward import REWARD_SCHEME_NAMES


def train(
    config: DoubleDQNConfig,
    run_dir: str | Path | None = None,
    checkpoint_path: str | Path | None = None,
    seed_types: tuple[int, ...] | None = None,
    dll_path: str | Path | None = None,
    checkpoint_load_mode: str = "resume",
) -> Path:
    def on_event(event: str, data: dict) -> None:
        if event == "episode_end":
            loss = data.get("loss")
            result = "TRUNCATED" if data.get("truncated") else data["result"]
            print(
                f"step={data['env_step']} episode={data['episode']} result={result} "
                f"len={data['episode_steps']} win_rate={float(data['win_rate']):.3f} "
                f"window_ticks={data.get('episode_ticks', 0)}/{data.get('max_episode_ticks')} "
                f"eps={float(data['epsilon']):.3f} "
                f"loss={float(loss) if loss is not None else float('nan'):.5f} "
                f"mowers={data['mowers_used']} "
                f"first_mower_tick={data.get('first_mower_tick')} "
                f"train_reward={float(data['training_episode_reward']):.3f} "
                f"ticks={float(data.get('mean_ticks_advanced', 0.0)):.1f}+/-"
                f"{float(data.get('std_ticks_advanced', 0.0)):.1f} "
                f"long_ticks={float(data.get('long_tick_jump_rate', 0.0)):.1%}"
            )
        elif event == "checkpoint":
            suffix = ""
            if "model_save_ms" in data:
                suffix = (
                    f" kind={data.get('kind', 'unknown')} "
                    f"model_ms={float(data.get('model_save_ms', 0.0)):.1f} "
                    f"replay_ms={float(data.get('replay_save_ms', 0.0)):.1f}"
                )
            print(f"checkpoint={data['path']}{suffix}")
        elif event == "checkpoint_loaded":
            replay_state = (
                f"restored:{int(data.get('replay_size', 0) or 0)}"
                if bool(data.get("replay_restored", False))
                else "empty"
            )
            print(
                f"loaded={data['path']} env_step={data['env_step']} "
                f"episode={data['episode']} replay={replay_state}"
            )
        elif event == "evaluation_end":
            print(
                f"eval kind={data.get('kind', 'legacy')} step={data['env_step']} "
                f"games={data['episodes']}/{data['requested_episodes']} "
                f"win_rate={float(data['win_rate']):.3f} "
                f"mean_score={float(data['mean_score']):.3f} "
                f"wall_ms={float(data.get('evaluation_wall_time_ms', 0.0)):.1f}"
            )
        elif event == "best_checkpoint":
            print(
                f"best={data['path']} win_rate={float(data['win_rate']):.3f} "
                f"mean_score={float(data['mean_score']):.3f}"
            )
        elif event in ("checkpoint_restore_contract", "replay_restore_skipped"):
            print(f"{event}: {data}")

    controller = DoubleDQNTrainerController(
        config,
        run_dir=run_dir,
        checkpoint_path=checkpoint_path,
        seed_types=seed_types,
        dll_path=dll_path,
        checkpoint_load_mode=checkpoint_load_mode,
        on_event=on_event,
    )
    summary = controller.run()
    if controller.latest_checkpoint is None:
        raise RuntimeError("training completed without producing a checkpoint")
    print(
        f"state={controller.state} step={summary.env_steps} episodes={summary.episodes} "
        f"win_rate={summary.win_rate:.3f} saved={controller.latest_checkpoint}"
    )
    if summary.best_checkpoint:
        print(
            f"best={summary.best_checkpoint} eval_win_rate={summary.best_eval_win_rate:.3f} "
            f"eval_mean_score={summary.best_eval_mean_score:.3f}"
        )
    return controller.latest_checkpoint


def _parse_seed_types(raw: str | None) -> tuple[int, ...] | None:
    text = str(raw or "").strip()
    if not text:
        return None
    values = tuple(int(item.strip()) for item in text.split(",") if item.strip())
    return values or None


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the PvZ Double DQN baseline")
    parser.add_argument("--level-id", type=int, default=None)
    parser.add_argument("--playthrough", type=int, choices=(1, 2), default=None)
    parser.add_argument("--ticks-per-step", type=int, default=None)
    parser.add_argument("--n-step", type=int, default=None,
                        help="TD return decision steps (new training: 3; resume: saved value)")
    parser.add_argument("--shaping-limit-mode", choices=("smooth_total", "episode_budget"), default=None,
                        help="smooth net episode bounds (new-run default), or historical independent budgets; changing this rebuilds Replay")
    parser.add_argument("--gamma", type=float, default=None,
                        help="new training discount per 100 ticks; overrides the converted checkpoint value")
    parser.add_argument("--gamma-reference-ticks", type=int, default=None,
                        help="legacy input unit for --gamma; converted to the fixed 100-tick unit")
    parser.add_argument(
        "--max-episode-ticks", "--max-window-ticks", type=int, default=None,
        help="game ticks per training window (default 120000); 0 disables the limit",
    )
    parser.add_argument(
        "--simulation-mode",
        choices=SIMULATION_MODES,
        default=None,
        help="simulation policy; turbo runs intro, selection and battle without normal frame rendering",
    )
    parser.add_argument("--num-envs", type=int, default=None)
    parser.add_argument("--dll", type=Path, default=None, help="native DLL used by the learner and environment workers")
    parser.add_argument("--process-ids", default=None, help="comma-separated distinct PIDs, one per environment")
    parser.add_argument("--game-exe", default=None, help="launch isolated copies of this supported retail game executable")
    parser.add_argument("--instance-root", default=None)
    parser.add_argument("--save-template", default=None)
    parser.add_argument("--save-directory", default=None, help="verified writable save directory for an explicitly attached game")
    parser.add_argument("--profile-id", type=int, default=None)
    parser.add_argument("--q-heatmap", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--game-topmost", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--selected-action-forward", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--checkpoint-load-mode", choices=CHECKPOINT_LOAD_MODES, default="resume")
    parser.add_argument("--async-sampling", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--inference-batch-size", type=int, default=None)
    parser.add_argument("--inference-max-wait-ms", type=float, default=None)
    parser.add_argument("--queue-capacity", type=int, default=None)
    parser.add_argument("--worker-timeout-s", type=float, default=None)
    parser.add_argument("--restart-limit", type=int, default=None)
    parser.add_argument("--max-policy-lag", type=int, default=None)
    parser.add_argument("--training-architecture", choices=("actor_learner", "legacy"), default=None,
                        help="independent actor/learner (new-run default) or existing serial learner")
    parser.add_argument("--policy-sync-interval", type=int, default=None)
    parser.add_argument("--max-pending-updates", type=int, default=None)
    parser.add_argument("--ui-update-interval-ms", type=int, default=None)
    parser.add_argument("--async-checkpoints", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--fast-intro", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--fast-terminal", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--mute-audio", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--executor", choices=("legacy", "resident"), default=None)
    parser.add_argument("--transport", choices=("json", "binary", "shared_memory"), default=None)
    parser.add_argument("--snapshot", choices=("legacy", "shared"), default=None)
    parser.add_argument("--total-steps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--seed-types", type=str, default=None, help="comma-separated fixed plant type IDs")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default=None)
    parser.add_argument("--autosave-every", type=int, default=None)
    parser.add_argument("--checkpoint-every", type=int, default=None)
    parser.add_argument("--mower-penalty", type=float, default=None)
    parser.add_argument("--reward-scheme", choices=REWARD_SCHEME_NAMES, default=None)
    parser.add_argument("--replay-scheme", choices=("uniform", "per"), default=None)
    parser.add_argument("--per-alpha", type=float, default=None)
    parser.add_argument("--per-beta-start", type=float, default=None)
    parser.add_argument(
        "--exploration-mode", choices=tuple(EXPLORATION_MODE_LABELS), default=None,
    )
    parser.add_argument("--exploration-ratio", type=str, default=None, metavar="Q:RANDOM",
                        help="Q:random weights within exploration; default 50:50")
    parser.add_argument(
        "--exploration-type-weights",
        type=str,
        default=None,
        metavar="NOOP,PLANT,SHOVEL,COB,INTERACT",
        help="family selection weights for random/Q guidance; neighbor excludes shovel; default 2,4,0.1,0.1,0.1",
    )
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help=(
            "load one checkpoint into a new or explicitly selected run directory; "
            "a matching replay.npz sidecar is restored automatically when available"
        ),
    )
    parser.add_argument(
        "--resume-run",
        type=Path,
        default=None,
        help="continue an existing runs/<name> directory using its config and newest safe checkpoint",
    )
    parser.add_argument("--run-dir", type=Path, default=None)
    args = parser.parse_args()
    if args.resume is not None and args.resume_run is not None:
        parser.error("--resume and --resume-run are mutually exclusive")
    if args.resume_run is not None and args.run_dir is not None:
        parser.error("--run-dir is implicit when --resume-run is used")
    if args.checkpoint_load_mode == "weights_replay":
        if args.resume_run is not None or args.resume is None:
            parser.error("weights_replay starts a new run; use --resume <paired-checkpoint.pt>")
    return args


def main() -> None:
    args = _parse_args()

    checkpoint_path = args.resume
    run_dir = args.run_dir
    seed_types = _parse_seed_types(args.seed_types)
    try:
        exploration_type_weights = (
            None
            if args.exploration_type_weights is None
            else parse_action_family_weights(args.exploration_type_weights)
        )
        exploration_ratio = (
            None if args.exploration_ratio is None else parse_exploration_ratio(args.exploration_ratio)
        )
    except ValueError as error:
        raise SystemExit(str(error)) from error
    if args.resume_run is not None:
        config = load_run_config(args.resume_run)
        checkpoint_path = resolve_resume_checkpoint(args.resume_run, mode=args.checkpoint_load_mode)
        run_dir = args.resume_run
        if args.seed_types is None:
            seed_types = load_run_seed_types(args.resume_run)
        print(f"resume_run={run_dir} checkpoint={checkpoint_path}")
    else:
        if checkpoint_path is not None:
            from .agent import DoubleDQNAgent
            _agent, config, _payload = DoubleDQNAgent.from_checkpoint(
                checkpoint_path, device="cpu",
                weights_only=args.checkpoint_load_mode in ("weights_only", "weights_replay"))
            del _agent
            if args.seed_types is None:
                seed_types = load_checkpoint_seed_types(checkpoint_path, config, _payload)
        else:
            config = DoubleDQNConfig()

    env_config = config.env
    if args.level_id is not None:
        env_config = replace(env_config, level_id=args.level_id)
    if args.playthrough is not None:
        env_config = replace(env_config, playthrough=args.playthrough)
    if args.ticks_per_step is not None:
        env_config = replace(env_config, ticks_per_step=args.ticks_per_step)
    if args.max_episode_ticks is not None:
        env_config = replace(env_config, max_episode_ticks=args.max_episode_ticks or None)
    if args.simulation_mode is not None:
        env_config = replace(env_config, simulation_mode=args.simulation_mode)

    config = replace(config, env=env_config)
    architecture_updates = {name: getattr(args, name) for name in
        ("training_architecture", "policy_sync_interval", "max_pending_updates")
        if getattr(args, name, None) is not None}
    runtime_updates = {name: getattr(args, name) for name in
        ("num_envs", "instance_root", "inference_batch_size", "inference_max_wait_ms", "queue_capacity",
         "worker_timeout_s", "restart_limit", "max_policy_lag", "ui_update_interval_ms",
         "save_template", "save_directory", "profile_id")
        if getattr(args, name, None) is not None}
    if getattr(args, "process_ids", None) is not None:
        runtime_updates["process_ids"] = tuple(int(pid.strip()) for pid in args.process_ids.split(","))
        if args.num_envs is None: runtime_updates["num_envs"] = len(runtime_updates["process_ids"])
    if getattr(args, "game_exe", None) is not None:
        from pvz_gym.env_file import resolve_game_paths
        runtime_updates["game_executable"] = str(resolve_game_paths(args.game_exe).executable)
    elif (checkpoint_path is None and not config.runtime.game_executable
          and not config.runtime.process_ids and not config.env.runtime.process_id):
        from pvz_gym.env_file import resolve_game_paths
        runtime_updates["game_executable"] = str(resolve_game_paths().executable)
    if getattr(args, "async_sampling", None) is not None: runtime_updates["asynchronous"] = args.async_sampling
    if args.async_checkpoints is not None: runtime_updates["asynchronous_checkpoints"] = args.async_checkpoints
    if args.q_heatmap is not None: runtime_updates["visualize_q"] = args.q_heatmap
    if args.game_topmost is not None: runtime_updates["game_topmost"] = args.game_topmost
    if runtime_updates.get("num_envs", 1) > 1:
        runtime_updates.setdefault("asynchronous", True)
        runtime_updates.setdefault("inference_batch_size", runtime_updates["num_envs"])
    backend_updates = {name: getattr(args, name) for name in ("executor", "transport", "snapshot")
                       if getattr(args, name, None) is not None}
    if config.env.simulation_mode == "turbo":
        backend_updates.setdefault("executor", "resident")
        backend_updates.setdefault("transport", "binary")
        backend_updates.setdefault("snapshot", "shared")
    elif runtime_updates.get("game_executable"):
        backend_updates.setdefault("executor", "resident")
    config = replace(config, **architecture_updates, runtime=replace(config.runtime, **runtime_updates),
                     env=replace(config.env, runtime=replace(config.env.runtime, **backend_updates),
                         turbo=replace(config.env.turbo, **{name: getattr(args, name) for name in
                             ("fast_intro", "fast_terminal", "mute_audio") if getattr(args, name) is not None})))
    if args.n_step is not None:
        config = replace(config, n_step=args.n_step)
    if args.shaping_limit_mode is not None:
        config = replace(config, shaping_limit_mode=args.shaping_limit_mode)
    if args.gamma is not None or args.gamma_reference_ticks is not None:
        config = replace(config, gamma=convert_gamma(
            config.gamma if args.gamma is None else args.gamma,
            config.gamma_reference_ticks if args.gamma_reference_ticks is None else args.gamma_reference_ticks,
        ), gamma_reference_ticks=100)
    if args.total_steps is not None:
        config = replace(config, total_steps=args.total_steps)
    if args.selected_action_forward is not None:
        config = replace(config, selected_action_forward=args.selected_action_forward)
    if args.seed is not None:
        config = replace(config, seed=args.seed)
    if args.device is not None:
        config = replace(config, device=args.device)
    if args.autosave_every is not None:
        config = replace(config, autosave_every=args.autosave_every)
    if args.checkpoint_every is not None:
        config = replace(config, checkpoint_every=args.checkpoint_every)
    if args.mower_penalty is not None:
        config = replace(config, mower_penalty=args.mower_penalty)
    if args.reward_scheme is not None:
        config = replace(config, reward_scheme=args.reward_scheme)
    if args.replay_scheme is not None:
        config = replace(config, replay_scheme=args.replay_scheme)
    if args.per_alpha is not None:
        config = replace(config, per_alpha=args.per_alpha)
    if args.per_beta_start is not None:
        config = replace(config, per_beta_start=args.per_beta_start)
    if exploration_type_weights is not None:
        config = replace(config, exploration_type_weights=exploration_type_weights)
    if args.exploration_mode is not None:
        config = replace(config, exploration_mode=args.exploration_mode)
    if exploration_ratio is not None:
        config = replace(config, exploration_ratio=exploration_ratio)

    train(config, run_dir, checkpoint_path, seed_types, dll_path=args.dll,
          checkpoint_load_mode=args.checkpoint_load_mode)


if __name__ == "__main__":
    main()
