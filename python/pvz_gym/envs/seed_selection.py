"""Gym environment for the SeedChooser phase only."""

from __future__ import annotations

from typing import Any, Mapping

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:  # pragma: no cover
    class _Env:
        metadata = {}
        def reset(self, *, seed=None): pass
    class _Discrete:
        def __init__(self, n): self.n = n
        def contains(self, value): return isinstance(value, int) and 0 <= value < self.n
    class _Dict:
        def __init__(self, mapping): self.spaces = mapping
    class _MultiBinary:
        def __init__(self, n): self.n = n
    class _GymCompat:
        Env = _Env
    class _SpacesCompat:
        Discrete = _Discrete
        Dict = _Dict
        MultiBinary = _MultiBinary
    gym = _GymCompat()
    spaces = _SpacesCompat()

from ..bridge.seed_selection import SeedSelectionNativeBackend
from ..rewards.seed_selection import deck_outcome_reward, seed_selection_action_reward
from ..seed_types import (
    SeedSelectionActionType,
    SeedSelectionState,
    decode_seed_action,
    seed_action_count,
)


class SeedSelectionEnv(gym.Env):
    """SeedChooser view that can share the exact same PvZSession as BattleEnv."""

    metadata = {"render_modes": []}

    def __init__(self, backend=None, *, session=None, max_episode_steps: int = 32):
        if backend is not None and session is not None:
            raise ValueError("pass backend or session, not both")
        self.session = session
        if session is not None:
            self.backend = session
            self._owns_backend = False
        elif backend is not None:
            self.backend = backend
            self._owns_backend = False
        else:
            self.backend = SeedSelectionNativeBackend()
            self._owns_backend = True
        self.seed_type_capacity = int(self.backend.get_seed_type_capacity())
        self.max_episode_steps = max_episode_steps
        self.action_space = spaces.Discrete(seed_action_count(self.seed_type_capacity))
        self.observation_space = spaces.Dict({
            "present": spaces.Discrete(2),
            "required_count": spaces.Discrete(self.seed_type_capacity + 1),
            "selected_count": spaces.Discrete(self.seed_type_capacity + 1),
            "can_confirm": spaces.Discrete(2),
            "seeds_in_flight": spaces.Discrete(2),
            "warning_flags": spaces.Discrete(256),
            "imitater_target": spaces.Discrete(self.seed_type_capacity + 1),
            "available": spaces.MultiBinary(self.seed_type_capacity),
            "unlocked": spaces.MultiBinary(self.seed_type_capacity),
            "selected": spaces.MultiBinary(self.seed_type_capacity),
            "disabled": spaces.MultiBinary(self.seed_type_capacity),
        })
        self._steps = 0

    def _project(self, state: SeedSelectionState) -> dict[str, Any]:
        available = [0] * self.seed_type_capacity
        unlocked = [0] * self.seed_type_capacity
        selected = [0] * self.seed_type_capacity
        disabled = [0] * self.seed_type_capacity
        for option in state.options:
            if 0 <= option.plant_type < self.seed_type_capacity:
                i = option.plant_type
                available[i] = int(option.available)
                unlocked[i] = int(option.unlocked)
                selected[i] = int(option.selected)
                disabled[i] = int(option.disabled)
        return {
            "present": int(state.present),
            "required_count": state.required_count,
            "selected_count": state.selected_count,
            "can_confirm": int(state.can_confirm),
            "seeds_in_flight": int(state.seeds_in_flight),
            "warning_flags": int(state.warning_flags),
            "imitater_target": state.imitater_target + 1,
            "available": available,
            "unlocked": unlocked,
            "selected": selected,
            "disabled": disabled,
        }

    def reset(self, *, seed: int | None = None, options: Mapping[str, Any] | None = None):
        super().reset(seed=seed)
        self._steps = 0
        state = self.backend.get_seed_selection()
        if not state.present:
            raise RuntimeError("SeedSelectionEnv requires the shared session to be at SeedChooser")
        info = {
            "action_mask": self.backend.get_seed_selection_mask(),
            "selected_seeds": list(state.selected_seeds),
            "imitater_target": state.imitater_target,
            "warning_flags": int(state.warning_flags),
            "deck_reward_pending": False,
        }
        return self._project(state), info

    def step(self, action: int):
        if not self.action_space.contains(action):
            raise ValueError(f"action outside SeedSelectionEnv.action_space: {action}")

        before = self.backend.get_seed_selection()
        action_type, plant_type = decode_seed_action(int(action), self.seed_type_capacity)
        if action_type == SeedSelectionActionType.SELECT:
            result = self.backend.select_seed(int(plant_type))
        elif action_type == SeedSelectionActionType.SELECT_IMITATER:
            result = self.backend.select_imitater(int(plant_type))
        elif action_type == SeedSelectionActionType.DESELECT:
            result = self.backend.deselect_seed(int(plant_type))
        else:
            result = self.backend.confirm_seeds()

        self._steps += 1
        confirmed = action_type == SeedSelectionActionType.CONFIRM and result.success
        state = before if confirmed else self.backend.get_seed_selection()
        reward = seed_selection_action_reward(result.success)
        terminated = confirmed
        truncated = self._steps >= self.max_episode_steps and not terminated
        info = {
            "action_result": result,
            "action_mask": [] if confirmed else self.backend.get_seed_selection_mask(),
            "selected_seeds": list(state.selected_seeds),
            "imitater_target": state.imitater_target,
            "warning_flags": int(state.warning_flags),
            "deck_reward_pending": confirmed,
        }
        return self._project(state), reward, terminated, truncated, info

    @staticmethod
    def score_battle_outcome(*, won: bool, lost: bool) -> float:
        return deck_outcome_reward(won=won, lost=lost)

    def close(self):
        if self._owns_backend:
            self.backend.close()
