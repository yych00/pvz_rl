"""Run an existing scripted policy on the same battle prepared by selection."""

from dataclasses import dataclass

from pvz_gym.action_space import ACTION_SPACE_SIZE, decode_action
from pvz_gym.types import DEFAULT_MAX_EPISODE_TICKS, GameStatus


@dataclass(frozen=True)
class BattleSummary:
    outcome: str
    ticks: int
    steps: int
    plants: int
    auto_collected: int


def policy_observation(observation):
    return {
        "global": dict(observation.global_state),
        **{name: [dict(item) for item in getattr(observation, name)]
           for name in ("cards", "plants", "zombies", "dropped_cards", "grid_items",
                        "ice_trails", "mowers", "grid", "zombie_health")},
    }


def run_scripted_battle(backend, policy, *, ticks_per_step=30,
                        max_ticks=DEFAULT_MAX_EPISODE_TICKS, headless=False,
                        report=print):
    """Run one battle to win/loss/tick budget; never reset or retry the level."""
    if ticks_per_step <= 0 or max_ticks < 0:
        raise ValueError("ticks_per_step must be positive; max_ticks must be nonnegative")
    facts = backend.wait_for_battle(deterministic_pause=True)
    if facts.status != GameStatus.PLAYING or facts.observation is None:
        raise RuntimeError("battle is not ready")
    observation = facts.observation
    start_clock = clock = facts.game_clock
    steps = plants = collected = 0
    next_report = 5000
    active_headless = False
    try:
        if headless:
            active_headless = backend.begin_headless_mode()
            if not active_headless:
                raise RuntimeError("cannot enter headless simulation")
        while not max_ticks or clock - start_clock < max_ticks:
            mask = backend.get_action_mask()
            if len(mask) != ACTION_SPACE_SIZE:
                raise RuntimeError("invalid battle action mask length")
            choice = policy(list(mask), policy_observation(observation))
            action_id = int(choice[0]) if choice is not None else 0
            if not 0 <= action_id < len(mask) or not mask[action_id]:
                raise RuntimeError(f"script proposed illegal action {action_id}")
            ticks = ticks_per_step if not max_ticks else min(
                ticks_per_step, max_ticks - (clock - start_clock))
            action = decode_action(action_id)
            result = (backend.step_headless(action, ticks) if headless
                      else backend.step_continuous(action, ticks))
            steps += 1
            collected += result.automation_collected
            plants += int(action.type == "plant" and result.action.success)
            if result.status in (GameStatus.WON, GameStatus.LOST):
                return BattleSummary(result.status.name,
                                     max(clock, result.game_clock) - start_clock,
                                     steps, plants, collected)
            if not result.action.success:
                raise RuntimeError(f"native action failed: {result.action.message}")
            if result.status != GameStatus.PLAYING or result.game_clock <= clock:
                raise RuntimeError("battle left PLAYING or failed to advance its game clock")
            clock = result.game_clock
            observation = result.observation
            if clock - start_clock >= next_report:
                report(f"战斗中：ticks={clock-start_clock}, 种植={plants}, 自动收集={collected}")
                next_report += 5000
        return BattleSummary("TICK_LIMIT", clock - start_clock, steps, plants, collected)
    finally:
        if active_headless:
            backend.end_headless_mode()
