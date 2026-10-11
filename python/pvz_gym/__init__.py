"""Importable PvZ Gym environment library.

Developer GUIs, scripted agents, evaluation runners, and training entrypoints
live in the repository's ``workbench``, ``agents``, ``eval`` and ``train`` packages.
"""

from .bridge import GameFactsBackend, NativeCapabilityError, NativePvzBackend
from .action_space import ACTION_SPACE_SIZE, decode_action, encode_action
from .client import PvZClient
from .curriculum import ResetConfig
from .env import PvZEnv, PvzGymEnv
from .envs import BattleEnv, SeedSelectionEnv
from .observation import encode_observation
from .rewards import (
    calculate_reward,
    deck_outcome_reward,
    seed_selection_action_reward,
    survival_reward,
)
from .seed_types import (
    SeedOptionState,
    SeedSelectionActionType,
    SeedSelectionResult,
    SeedSelectionState,
    SeedWarning,
    decode_seed_action,
    encode_seed_action,
)
from .session import PvZSession
from .runtime import RuntimeCapabilities, RuntimeCapabilityError, RuntimeConfig, TurboConfig
from .env_file import load_dotenv, get_game_executable, set_game_executable
from .types import (
    Action,
    ActionResult,
    GameStatus,
    Observation,
    ResetFacts,
    RngSeedSupport,
    StepEvents,
    StepFacts,
)

__all__ = [
    "Action",
    "ACTION_SPACE_SIZE",
    "ActionResult",
    "BattleEnv",
    "GameFactsBackend",
    "GameStatus",
    "NativeCapabilityError",
    "NativePvzBackend",
    "Observation",
    "PvZClient",
    "PvZEnv",
    "PvZSession",
    "PvzGymEnv",
    "ResetConfig",
    "ResetFacts",
    "RngSeedSupport",
    "RuntimeCapabilities",
    "RuntimeCapabilityError",
    "RuntimeConfig",
    "TurboConfig",
    "SeedOptionState",
    "SeedSelectionActionType",
    "SeedSelectionEnv",
    "SeedSelectionResult",
    "SeedSelectionState",
    "SeedWarning",
    "decode_seed_action",
    "encode_seed_action",
    "StepEvents",
    "StepFacts",
    "calculate_reward",
    "deck_outcome_reward",
    "decode_action",
    "encode_action",
    "encode_observation",
    "seed_selection_action_reward",
    "survival_reward",
]
