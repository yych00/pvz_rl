"""Recover LIVE terminal results when a fast native loop outruns observation reads.

At high animated speeds PvZ can leave BATTLE and advance to the next Adventure
level between the final tick and the workbench's post-tick observation read.
That is a valid terminal transition, not an observation/state failure.
"""

from __future__ import annotations

from typing import Any


_RECOVERABLE_OUTCOMES = {"OBS_ERROR", "STATE_ERROR", "CLOCK_ERROR"}


def classify_terminal_transition(state: dict, expected_level_id: int) -> str:
    """Return WON/LOST when runtime facts prove the previous battle ended."""
    from . import early_levels as _early

    outcome = _early._terminal_outcome(state)
    if outcome:
        return outcome

    try:
        current_level = int(state.get("level_id", 0) or 0)
    except (TypeError, ValueError, OverflowError):
        current_level = 0

    # During one synchronous LIVE episode no caller selects another level until
    # run_early_level_e2e() returns. Therefore seeing a different non-zero
    # Adventure level after battle progress means PvZ itself completed and
    # advanced the just-finished level. This commonly happens at frame=1 ms.
    if expected_level_id > 0 and current_level > 0 and current_level != expected_level_id:
        return "WON"

    return ""


def install_terminal_transition_guard() -> None:
    from . import early_levels as _early

    if getattr(_early, "_terminal_transition_guard_installed", False):
        return

    original = _early.run_early_level_e2e

    def guarded_run_early_level_e2e(client: Any, spec: Any, *args: Any, **kwargs: Any):
        result = original(client, spec, *args, **kwargs)

        # Only reinterpret failures that can be caused by the battle ending
        # between two reads. Setup/action/path failures remain hard failures.
        if (
            result.success
            or result.outcome not in _RECOVERABLE_OUTCOMES
            or int(getattr(result, "ticks_advanced", 0) or 0) <= 0
            or int(getattr(result, "plants_executed", 0) or 0) <= 0
        ):
            return result

        state = _early._runtime(client)
        if state.get("error"):
            return result

        outcome = classify_terminal_transition(state, int(spec.level_id))
        if not outcome:
            return result

        phase = str(state.get("phase_name", "UNKNOWN"))
        try:
            current_level = int(state.get("level_id", 0) or 0)
        except (TypeError, ValueError, OverflowError):
            current_level = 0

        previous_outcome = result.outcome
        terminal_message = (
            f"{outcome}: recovered from {previous_outcome}; "
            f"phase={phase}, level_id={current_level}"
        )
        result.success = True
        result.outcome = outcome
        result.message = (
            f"{spec.label} {result.actual_path} LIVE PASS: terminal={outcome}; "
            f"recovered from {previous_outcome} during fast terminal transition "
            f"(phase={phase}, level_id={current_level})"
        )
        result.phases.append({"phase": "TERMINAL", "message": terminal_message})

        # The original E2E call has already returned, so appending to phases
        # alone does not update the Workbench console. Re-emit the recovered
        # terminal explicitly so stress runs show why the episode counted.
        progress = kwargs.get("progress")
        if callable(progress):
            progress("TERMINAL", terminal_message)

        return result

    _early.run_early_level_e2e = guarded_run_early_level_e2e
    _early._terminal_transition_guard_installed = True


__all__ = ["classify_terminal_transition", "install_terminal_transition_guard"]
