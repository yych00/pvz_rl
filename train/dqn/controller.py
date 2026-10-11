"""Public Double DQN controller entrypoint.

The unchanged training loop lives in :mod:`base_controller`; the exported
controller adds runtime-only performance policy from :mod:`optimized_controller`.
Keeping this module as the public import makes CLI, Workbench, and tests use the
same controller implementation.
"""

from .base_controller import DoubleDQNSummary, EventCallback
from .optimized_controller import DoubleDQNTrainerController

__all__ = ["DoubleDQNSummary", "DoubleDQNTrainerController", "EventCallback"]
