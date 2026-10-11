"""Lightweight .env file loader and manager for PvZ Gym configurations."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


ENV_VARIABLE_NAMES = (
    "PVZ_GAME_EXECUTABLE",
    "GAME_EXECUTABLE",
    "PVZ_GAME_EXE",
)


@dataclass(frozen=True)
class GamePaths:
    executable: Path
    save_template: Path
    instance_root: Path
    repository_root: Path


def repository_root(start_path: str | Path | None = None) -> Path:
    """Resolve repository-relative settings independently of the launch cwd."""
    current = Path(start_path).resolve() if start_path is not None else Path(__file__).resolve()
    if current.is_file():
        current = current.parent
    for directory in (current, *current.parents):
        if (directory / "pyproject.toml").is_file() or (directory / ".git").exists():
            return directory
    return current


def _configured_path(names: tuple[str, ...], env_path: Path | None) -> str | None:
    for name in names:
        value = os.environ.get(name, "").strip()
        if value:
            return value
    parsed = parse_env_file(env_path) if env_path is not None else {}
    for name in names:
        value = parsed.get(name, "").strip()
        if value:
            return value
    return None


def resolve_game_paths(game_executable: str | Path | None = None,
                       save_template: str | Path | None = None,
                       instance_root: str | Path | None = None, *,
                       config_root: str | Path | None = None,
                       env_path: str | Path | None = None) -> GamePaths:
    """Return explicit absolute paths without modifying .env or the source game.

    Relative paths use config_root (the repository by default), including values
    from the canonical .env. Existence is checked by instance prepare.
    """
    base = Path(config_root).resolve() if config_root is not None else repository_root()
    env = Path(env_path).resolve() if env_path is not None else find_env_file(base)

    def resolve(value, names, default):
        raw = value or _configured_path(names, env) or default
        path = Path(os.path.expandvars(os.path.expanduser(str(raw))))
        return (path if path.is_absolute() else base / path).resolve()

    return GamePaths(
        resolve(game_executable, ENV_VARIABLE_NAMES,
                "game/pvz-1.2.0.1065/PlantsVsZombies.exe"),
        resolve(save_template, ("PVZ_TRAINING_SAVE_TEMPLATE",), "local/save_template"),
        resolve(instance_root, ("PVZ_INSTANCE_ROOT",), "runtime/instances"),
        base,
    )


def find_env_file(start_path: str | Path | None = None) -> Path | None:
    """Find the nearest .env file starting from a directory up to project root."""
    if start_path is not None:
        start = Path(start_path).resolve()
        if start.is_file():
            start = start.parent
    else:
        start = repository_root()

    current = start
    for _ in range(8):
        candidate = current / ".env"
        if candidate.is_file():
            return candidate
        if (current / ".git").exists() or (current / "pyproject.toml").is_file():
            break
        parent = current.parent
        if parent == current:
            break
        current = parent

    return None


def default_env_path() -> Path:
    """Return the repository .env independently of the launch cwd."""
    return repository_root() / ".env"


def parse_env_file(path: str | Path) -> dict[str, str]:
    """Parse key-value pairs from a simple .env file."""
    env_vars: dict[str, str] = {}
    path = Path(path)
    if not path.is_file():
        return env_vars

    try:
        content = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        content = path.read_text(encoding="gbk", errors="replace")

    for raw_line in content.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if (value.startswith('"') and value.endswith('"')) or (value.startswith("'") and value.endswith("'")):
            value = value[1:-1]
        env_vars[key] = value

    return env_vars


def load_dotenv(path: str | Path | None = None, override: bool = False) -> Path | None:
    """Load variables from .env into os.environ."""
    target_path = Path(path) if path is not None else find_env_file()
    if target_path is None or not target_path.is_file():
        return None

    vars_dict = parse_env_file(target_path)
    for k, v in vars_dict.items():
        if override or k not in os.environ:
            os.environ[k] = v
    return target_path


def get_game_executable() -> str:
    """Retrieve the configured game executable path from env vars or .env file."""
    configured = _configured_path(ENV_VARIABLE_NAMES, find_env_file())
    if configured:
        return str(resolve_game_paths(game_executable=configured).executable)
    default = resolve_game_paths().executable
    return str(default) if default.is_file() else ""


def set_game_executable(exe_path: str, env_path: str | Path | None = None) -> Path:
    """Persist the game executable path into .env file, updating or creating the entry."""
    target = Path(env_path) if env_path is not None else (find_env_file() or default_env_path())
    target.parent.mkdir(parents=True, exist_ok=True)

    normalized_exe = str(exe_path).strip()
    primary_key = "PVZ_GAME_EXECUTABLE"

    lines: list[str] = []
    updated = False
    if target.is_file():
        try:
            content = target.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            content = target.read_text(encoding="gbk", errors="replace")
        for raw_line in content.splitlines():
            stripped = raw_line.strip()
            if stripped and not stripped.startswith("#") and "=" in stripped:
                k, _ = stripped.split("=", 1)
                if k.strip() in ENV_VARIABLE_NAMES:
                    lines.append(f'{primary_key}="{normalized_exe}"')
                    updated = True
                    continue
            lines.append(raw_line)

    if not updated:
        if lines and lines[-1].strip():
            lines.append("")
        lines.append("# 植物大战僵尸游戏可执行文件路径（多开与常驻执行器使用）")
        lines.append(f'{primary_key}="{normalized_exe}"')

    target.write_text("\n".join(lines) + "\n", encoding="utf-8")
    os.environ[primary_key] = normalized_exe
    return target
