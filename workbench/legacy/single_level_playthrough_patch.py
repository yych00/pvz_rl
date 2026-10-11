# -*- coding: utf-8 -*-
"""Make FIRST_RUN/REPLAY an explicit per-episode single-level configuration.

The generic early-level harness already loops by repeatedly calling
``run_early_level_e2e``.  Patching that one function is enough to guarantee
that every episode in an x5 stress run re-applies the requested Adventure pass
before ``select_level`` creates the new Board.
"""

from __future__ import annotations

from typing import Any

from . import early_levels_live_tests as _early


def install_single_level_playthrough_patch() -> None:
    if getattr(_early, "_single_level_playthrough_patch_installed", False):
        return

    original_run_e2e = _early.run_early_level_e2e

    def run_early_level_e2e_with_forced_playthrough(
        client: Any,
        spec: Any,
        *args,
        **kwargs,
    ):
        requested = kwargs.get("expected_playthrough")
        if requested is None:
            requested = kwargs.get("expected_path", _early.PLAYTHROUGH_AUTO)
        requested = _early._normalize_playthrough(requested)

        if requested in (_early.PLAYTHROUGH_1, _early.PLAYTHROUGH_2):
            playthrough = 1 if requested == _early.PLAYTHROUGH_1 else 2
            setter = getattr(client, "set_adventure_playthrough", None)
            if setter is None:
                return _early.EarlyLevelE2EResult(
                    success=False,
                    level=spec.label,
                    expected_path=requested,
                    actual_path="UNKNOWN",
                    outcome="PLAYTHROUGH_SET_ERROR",
                    message="client has no set_adventure_playthrough(); rebuild/reload unit-test package",
                    elapsed_ms=0.0,
                )

            code, message = setter(playthrough)
            if int(code) != 0:
                return _early.EarlyLevelE2EResult(
                    success=False,
                    level=spec.label,
                    expected_path=requested,
                    actual_path="UNKNOWN",
                    outcome="PLAYTHROUGH_SET_ERROR",
                    message=(
                        f"failed to force {requested} before {spec.label}: "
                        f"code={code}, {message}"
                    ),
                    elapsed_ms=0.0,
                )

        # 1-4 is the first materially long 10-wave Gym validation level. The
        # generic 60k budget is intentionally raised here so both GUI E2E and
        # stress episodes exercise the real terminal path instead of timing out
        # in the harness. Stress passes 60k explicitly into this wrapper, hence
        # the max() rather than only filling a missing keyword.
        if int(getattr(spec, "level_id", -1)) == 4:
            requested_ticks = int(kwargs.get("max_ticks", 0) or 0)
            kwargs["max_ticks"] = max(90000, requested_ticks)

        # Important: run_early_level_stress() resolves this module-global name
        # on every episode, so FIRST_RUN x5 writes 0 five times and REPLAY x5
        # writes 1 five times. No episode is allowed to inherit a changed mode
        # from the previous run.
        result = original_run_e2e(client, spec, *args, **kwargs)

        # BoardResult becomes WON as the original game spawns its level award.
        # The generic battle loop returns as soon as it sees that terminal fact,
        # so a reward created on the same tick (notably 1-4 COIN_SHOVEL) may not
        # receive another automation_tick(). Give every won episode one direct
        # native collection pass before returning to the caller. We intentionally
        # call automation_collect(), not automation_tick(), because the original
        # E2E function has already restored the caller's automation configuration.
        if getattr(result, "outcome", "") == "WON":
            collect = getattr(client, "automation_collect", None)
            if collect is not None:
                try:
                    collected = int(collect())
                    if collected > 0:
                        result.collected_items += collected
                        result.phases.append(
                            {
                                "phase": "TERMINAL_COLLECT",
                                "message": f"post-terminal collected={collected}",
                            }
                        )
                except Exception:
                    # Collection failure must not rewrite an otherwise valid
                    # environment terminal result; the LIVE log still exposes
                    # the missing progression item for diagnosis.
                    pass

        return result

    _early.run_early_level_e2e = run_early_level_e2e_with_forced_playthrough
    _early._single_level_playthrough_patch_installed = True
