"""Interactive PvZ Gym developer workbench.

The importable RL library lives in :mod:`pvz_gym`. This package owns the
native-debug GUI, destructive LIVE checks, and legacy GUI patches only.
"""

from .native.c_interface import (
    PvzGymNativeDll,
    install_game_exit_guard,
    global_restore_all_games,
)
from .native.runtime_phase_client import install_runtime_phase_client
from .native.playthrough_mode_client import install_playthrough_mode_client
from .native.dialog_cleanup_policy import install_dialog_cleanup_policy

# Preserve the original native-client patch order before any client instance is
# created by the GUI.
install_runtime_phase_client(PvzGymNativeDll)
install_playthrough_mode_client(PvzGymNativeDll)
install_dialog_cleanup_policy(PvzGymNativeDll)

from .native_tests.cases import NativeUnitTestSuite, TestCaseResult
from .native_tests.battle_actions import install_battle_action_tests
from .native_tests.static_policy import install_static_unit_test_policy

install_battle_action_tests(PvzGymNativeDll, NativeUnitTestSuite, TestCaseResult)
install_static_unit_test_policy(NativeUnitTestSuite, TestCaseResult)

# Re-apply the requested Adventure playthrough before every LIVE episode.
from .legacy.single_level_playthrough_patch import install_single_level_playthrough_patch
install_single_level_playthrough_patch()

from .gui import main_window as _gui
from .legacy.live_gui_constructor_patch import install_live_gui_constructor_patch
from .gui.battle_action_manual import install_battle_action_manual_gui
from .gui.battle_action_summary import install_battle_action_summary_gui
from .gui.plant_validation import install_plant_dual_validation_gui
from .gui.runtime_phase import install_runtime_phase_gui
from .gui.speed_benchmark import install_speed_benchmark_gui
from .gui.speed_benchmark_clicks import install_speed_benchmark_clicks
from .legacy.gui_legacy_tools_cleanup import install_gui_legacy_tools_cleanup
from .gui.scrollbar import install_gui_scrollbar_style_patch
from .gui.left_tabs import install_gui_left_tabs
from .gui.training import install_training_gui
from .gui.dqn_headless import install_dqn_headless_gui
from .gui.resume_run import install_resume_run_gui
from .gui.checkpoint_load_mode import install_checkpoint_load_mode_gui
from .gui.reward_scheme import install_reward_scheme_gui
from .gui.training_controller_lifecycle import install_training_controller_lifecycle
from .gui.training_smoke import install_training_smoke_gui
from .gui.training_validation_ui import install_training_validation_ui
from .live.terminal_transition_guard import install_terminal_transition_guard

# High-speed animated mode can advance out of BATTLE before the post-tick
# observation read. Install the recovery guard before importing LIVE wrappers so
# every early-level runner shares the same terminal-transition semantics.
install_terminal_transition_guard()

from .live.level_1_1_playthrough import (
    run_level_1_1_e2e,
    run_level_1_1_adventure1_e2e,
    run_level_1_1_adventure2_e2e,
    run_level_1_1_adventure1_stress,
    run_level_1_1_adventure2_stress,
    run_level_1_1_first_run_e2e,
    run_level_1_1_replay_e2e,
    run_level_1_1_stress,
)
from .live.early_levels import (
    run_level_1_2_e2e,
    run_level_1_2_adventure1_e2e,
    run_level_1_2_adventure2_e2e,
    run_level_1_2_adventure1_stress,
    run_level_1_2_adventure2_stress,
    run_level_1_2_first_run_e2e,
    run_level_1_2_replay_e2e,
    run_level_1_2_stress,
    run_level_1_3_e2e,
    run_level_1_3_adventure1_e2e,
    run_level_1_3_adventure2_e2e,
    run_level_1_3_adventure1_stress,
    run_level_1_3_adventure2_stress,
    run_level_1_3_first_run_e2e,
    run_level_1_3_replay_e2e,
    run_level_1_3_stress,
)
from .live.level_1_4 import (
    run_level_1_4_e2e,
    run_level_1_4_adventure1_e2e,
    run_level_1_4_adventure2_e2e,
    run_level_1_4_adventure1_stress,
    run_level_1_4_adventure2_stress,
    run_level_1_4_first_run_e2e,
    run_level_1_4_replay_e2e,
    run_level_1_4_stress,
)

PvzGymUnitTestGUI = _gui.PvzGymUnitTestGUI
launch_gui = _gui.launch_gui

# Preserve the original GUI patch order. Several patches wrap methods installed
# by the previous patch, so reordering these calls can silently change behavior.
install_live_gui_constructor_patch(PvzGymUnitTestGUI)
install_runtime_phase_gui(PvzGymUnitTestGUI, _gui)
install_gui_scrollbar_style_patch(PvzGymUnitTestGUI, _gui)
install_battle_action_manual_gui(PvzGymUnitTestGUI, _gui)
install_battle_action_summary_gui(PvzGymUnitTestGUI, _gui)
install_plant_dual_validation_gui(PvzGymUnitTestGUI, _gui)
install_gui_legacy_tools_cleanup(PvzGymUnitTestGUI)
install_speed_benchmark_gui(PvzGymUnitTestGUI, _gui)
install_speed_benchmark_clicks(PvzGymUnitTestGUI, _gui)
install_training_gui(PvzGymUnitTestGUI, _gui)
install_dqn_headless_gui(PvzGymUnitTestGUI, _gui)
install_resume_run_gui(PvzGymUnitTestGUI, _gui)
install_reward_scheme_gui(PvzGymUnitTestGUI, _gui)
install_training_controller_lifecycle(PvzGymUnitTestGUI)
install_checkpoint_load_mode_gui(PvzGymUnitTestGUI, _gui)
install_training_smoke_gui(PvzGymUnitTestGUI, _gui)
install_training_validation_ui(PvzGymUnitTestGUI, _gui)
install_gui_left_tabs(PvzGymUnitTestGUI, _gui)

__all__ = [
    "PvzGymNativeDll",
    "install_game_exit_guard",
    "global_restore_all_games",
    "NativeUnitTestSuite",
    "TestCaseResult",
    "PvzGymUnitTestGUI",
    "launch_gui",
    "run_level_1_1_e2e",
    "run_level_1_1_adventure1_e2e",
    "run_level_1_1_adventure2_e2e",
    "run_level_1_1_adventure1_stress",
    "run_level_1_1_adventure2_stress",
    "run_level_1_1_first_run_e2e",
    "run_level_1_1_replay_e2e",
    "run_level_1_1_stress",
    "run_level_1_2_e2e",
    "run_level_1_2_adventure1_e2e",
    "run_level_1_2_adventure2_e2e",
    "run_level_1_2_adventure1_stress",
    "run_level_1_2_adventure2_stress",
    "run_level_1_2_first_run_e2e",
    "run_level_1_2_replay_e2e",
    "run_level_1_2_stress",
    "run_level_1_3_e2e",
    "run_level_1_3_adventure1_e2e",
    "run_level_1_3_adventure2_e2e",
    "run_level_1_3_adventure1_stress",
    "run_level_1_3_adventure2_stress",
    "run_level_1_3_first_run_e2e",
    "run_level_1_3_replay_e2e",
    "run_level_1_3_stress",
    "run_level_1_4_e2e",
    "run_level_1_4_adventure1_e2e",
    "run_level_1_4_adventure2_e2e",
    "run_level_1_4_adventure1_stress",
    "run_level_1_4_adventure2_stress",
    "run_level_1_4_first_run_e2e",
    "run_level_1_4_replay_e2e",
    "run_level_1_4_stress",
]
