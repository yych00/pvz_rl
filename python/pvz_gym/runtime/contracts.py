"""Mode-independent runtime contracts. No game hooks or worker threads live here."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Protocol, TypeVar

from ..types import Action, Observation, ResetFacts, StepFacts


RUNTIME_INTERFACE_VERSION = 1


class RuntimeCapabilityError(RuntimeError):
    """A requested runtime capability has no implemented backend."""


_Config = TypeVar("_Config")


def coerce_config(cls: type[_Config], value: object) -> _Config:
    """Restore validated nested dataclasses from JSON/checkpoint mappings."""
    if value is None:
        return cls()
    if isinstance(value, cls):
        return value
    if isinstance(value, Mapping):
        return cls(**dict(value))
    raise TypeError(f"{cls.__name__} must be a configuration instance or mapping")


@dataclass(frozen=True)
class RuntimeConfig:
    """Shared mechanisms, configured independently of the simulation mode.

    None means existing automatic attachment. Explicit PID attachment, resident
    execution and alternate transports require the corresponding capabilities.
    """

    process_id: int | None = None
    executor: str = "legacy"
    transport: str = "json"
    snapshot: str = "legacy"

    def __post_init__(self) -> None:
        if self.process_id is not None and (
            type(self.process_id) is not int or not 0 < self.process_id <= 0xFFFFFFFF
        ):
            raise ValueError("process_id must be a positive uint32 or None")
        for name, choices in (
            ("executor", ("legacy", "resident")),
            ("transport", ("json", "binary", "shared_memory")),
            ("snapshot", ("legacy", "shared")),
        ):
            if getattr(self, name) not in choices:
                raise ValueError(f"{name} must be one of {choices}")

    @property
    def required_features(self) -> frozenset[str]:
        features = set()
        if self.process_id is not None:
            features.add("attach_pid")
        if self.executor == "resident":
            features.add("resident_executor")
        if self.transport != "json":
            features.add(f"{self.transport}_transport")
        if self.snapshot == "shared":
            features.add("shared_snapshot")
        return frozenset(features)


@dataclass(frozen=True)
class RuntimeCapabilities:
    interface_version: int = RUNTIME_INTERFACE_VERSION
    modes: frozenset[str] = field(default_factory=frozenset)
    features: frozenset[str] = field(default_factory=frozenset)

    @classmethod
    def from_mapping(cls, payload: Mapping[str, object]) -> RuntimeCapabilities:
        version = payload.get("interface_version")
        if type(version) is not int or version != RUNTIME_INTERFACE_VERSION:
            raise RuntimeCapabilityError(f"unsupported runtime interface version: {version!r}")
        sets = {}
        for name in ("modes", "features"):
            values = payload.get(name)
            if not isinstance(values, (list, tuple)) or any(type(v) is not str for v in values):
                raise RuntimeCapabilityError(f"invalid runtime capabilities: {name}")
            sets[name] = frozenset(values)
        return cls(interface_version=version, **sets)

    def require_mode(self, mode: str) -> None:
        if mode not in self.modes:
            raise RuntimeCapabilityError(f"simulation mode '{mode}' is not implemented by this backend")

    def require_config(self, config: RuntimeConfig) -> None:
        missing = config.required_features - self.features
        if missing:
            raise RuntimeCapabilityError("runtime capabilities are not implemented: " + ", ".join(sorted(missing)))


class RuntimePhase(IntEnum):
    """Values match the authoritative native RuntimePhase, not raw GameUI."""

    UNKNOWN = 0
    OFFLINE = 1
    LOADING = 2
    MENU = 3
    LEVEL_INTRO = 4
    DIALOGUE = 5
    SEED_SELECTION = 6
    BATTLE = 7
    LOST = 8
    AWARD = 9
    CREDIT = 10
    CHALLENGE = 11
    LEVEL_COMPLETE = 12


@dataclass(frozen=True)
class PhaseAdvanceRequest:
    expected_phase: RuntimePhase
    updates: int

    def __post_init__(self) -> None:
        if not isinstance(self.expected_phase, RuntimePhase):
            raise TypeError("expected_phase must be RuntimePhase")
        if type(self.updates) is not int or self.updates <= 0:
            raise ValueError("updates must be a positive integer")


@dataclass(frozen=True)
class PhaseAdvanceResult:
    phase_before: RuntimePhase
    phase_after: RuntimePhase
    updates_executed: int
    # Intro update counts must never be used as battle clock increments.
    game_ticks_advanced: int | None = None


@dataclass(frozen=True)
class SnapshotIdentity:
    process_id: int
    episode_id: int
    board_id: int
    revision: int


@dataclass(frozen=True)
class RuntimeSnapshot:
    identity: SnapshotIdentity
    observation: Observation
    action_mask: tuple[bool, ...]


class RuntimeBackend(Protocol):
    """Shared backend surface; capability checks precede mutating operations."""

    def get_runtime_capabilities(self) -> RuntimeCapabilities: ...
    def configure_runtime(self, config: RuntimeConfig) -> None: ...
    def reset(self, config: Mapping[str, object]) -> ResetFacts: ...
    def wait_for_battle(self, *, timeout_ms: int, deterministic_pause: bool) -> ResetFacts: ...
    def advance_phase(self, request: PhaseAdvanceRequest) -> PhaseAdvanceResult: ...
    def read_snapshot(self) -> RuntimeSnapshot: ...
    def step(self, action: Action, ticks: int) -> StepFacts: ...
    def close(self) -> None: ...
