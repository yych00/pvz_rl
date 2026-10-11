"""Simulation mode registration and the separate Turbo policy contract."""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Protocol

from ..types import Action, StepFacts
from .contracts import (
    PhaseAdvanceRequest, PhaseAdvanceResult, RuntimeCapabilities,
    RuntimeCapabilityError, RuntimeConfig,
)


@dataclass(frozen=True)
class SimulationModeSpec:
    key: str
    implemented: bool


MODE_SPECS = (
    SimulationModeSpec("stepped", True),
    SimulationModeSpec("headless", True),
    SimulationModeSpec("continuous", True),
    SimulationModeSpec("turbo", True),
)
SIMULATION_MODES = tuple(spec.key for spec in MODE_SPECS)


def normalize_simulation_mode(value: str | None) -> str:
    mode = str(value or "stepped").strip().lower()
    aliases = {
        "step": "stepped", "sync": "stepped", "renderless": "headless",
        "realtime": "continuous", "real-time": "continuous",
    }
    mode = aliases.get(mode, mode)
    if mode not in SIMULATION_MODES:
        raise ValueError(f"simulation_mode must be one of {SIMULATION_MODES}")
    return mode


def require_runtime_implementation(mode: str, config: RuntimeConfig, backend=None) -> None:
    """Check mode dispatch and backend capabilities before changing game state."""
    key = normalize_simulation_mode(mode)
    spec = next(spec for spec in MODE_SPECS if spec.key == key)
    if not spec.implemented:
        raise RuntimeCapabilityError(f"simulation mode '{key}' has interfaces only; implementation is pending")
    if backend is not None:
        query = getattr(backend, "get_runtime_capabilities", None)
        if callable(query):
            caps = query()
            caps.require_mode(key)
            caps.require_config(config)
        elif key == "turbo" or config != RuntimeConfig():
            raise RuntimeCapabilityError("backend does not advertise the requested runtime capabilities")
        if config != RuntimeConfig() and not callable(getattr(backend, "configure_runtime", None)):
            raise RuntimeCapabilityError("backend cannot configure the requested runtime")
        if key == "turbo" and not all(callable(getattr(backend, n, None)) for n in
                ("begin_turbo_mode", "step_turbo", "end_turbo_mode")):
            raise RuntimeCapabilityError("Turbo lifecycle is unavailable on this backend")


@dataclass(frozen=True)
class TurboConfig:
    """Mode policy only; environment counts and transports belong elsewhere."""

    suppress_render: bool = True
    service_messages: bool = True
    fast_intro: bool = True
    fast_terminal: bool = True
    mute_audio: bool = True

    def __post_init__(self) -> None:
        for item in fields(self):
            if type(getattr(self, item.name)) is not bool:
                raise ValueError(f"{item.name} must be a boolean")

    @property
    def native_flags(self) -> int:
        # Stable ABI bits match pvz/runtime.h; do not depend on field ordering.
        return (int(self.suppress_render) | int(self.service_messages) << 1
                | int(self.fast_intro) << 2 | int(self.fast_terminal) << 3
                | int(self.mute_audio) << 4)


class TurboPolicy(Protocol):
    """Enter before reset so the same owner covers intro, seeds and battle.

    Leave restores runtime state and is idempotent. An unavailable policy must
    fail without installing hooks, resetting a game or advancing its clock.
    """

    def begin_turbo_mode(self, config: TurboConfig) -> bool: ...
    def advance_phase(self, request: PhaseAdvanceRequest) -> PhaseAdvanceResult: ...
    def step_turbo(self, action: Action, ticks: int) -> StepFacts: ...
    def end_turbo_mode(self) -> bool: ...
