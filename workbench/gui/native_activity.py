"""Keep Tk's polling callbacks out of native calls owned by workers."""


def native_worker_busy(gui):
    states = getattr(gui, "_early_level_live_states", {})
    early_busy = isinstance(states, dict) and any(
        isinstance(state, dict) and bool(state.get("running"))
        for state in states.values()
    )
    return early_busy or any(
        bool(getattr(gui, name, False))
        for name in (
            "_live_running",
            "_level_1_1_live_running",
            "_training_running",
            "_speed_benchmark_running",
            "_static_test_running",
            "_prebattle_advancing",
            "_preparing_training_save",
        )
    )
