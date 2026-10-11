"""Profile-scoped, recoverable adventure-save cleanup.

No search result is a write authorization. Callers must bind an actual save
directory; an omitted profile is accepted only for a uniquely registered user.
"""
from __future__ import annotations

import json
import logging
import os
import re
import struct
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

from .env_file import resolve_game_paths

logger = logging.getLogger("pvz_gym.save_cleaner")
_ADVENTURE_SAVE_PATTERN = re.compile(r"^game\d+_0\.dat$", re.IGNORECASE)


class SaveScopeError(RuntimeError):
    """The current game/profile/save directory cannot be identified safely."""


@dataclass(frozen=True)
class SaveProfile:
    profile_id: int
    name: str
    use_sequence: int
    user_file: Path


def read_save_profiles(directory: str | Path) -> tuple[SaveProfile, ...]:
    """Read retail users.dat v14; orphan user files are not active profiles.

    PlayerInfo::SyncSummary uses a length-prefixed name, useSeq and profile id.
    Names are retained for private CurUser registry binding, without changing
    original bytes or drawing conclusions about unlocks/achievements.
    """
    root = Path(directory).resolve(strict=True)
    if not root.is_dir():
        raise SaveScopeError(f"save directory is not a directory: {root}")
    indexes = [entry for entry in root.iterdir() if entry.name.lower() == "users.dat"]
    if len(indexes) != 1 or not indexes[0].is_file() or indexes[0].is_symlink():
        raise SaveScopeError(f"expected one regular users.dat in {root}")
    data = indexes[0].read_bytes()
    offset = 0

    def take(size):
        nonlocal offset
        if offset + size > len(data):
            raise SaveScopeError(f"truncated users.dat in {root}")
        value = data[offset:offset + size]
        offset += size
        return value

    version, = struct.unpack("<I", take(4))
    count, = struct.unpack("<H", take(2))
    if version != 14 or not 0 < count <= 100:
        raise SaveScopeError(f"unsupported/empty users.dat version={version}, profiles={count}")
    profiles = []
    for _ in range(count):
        length, = struct.unpack("<H", take(2))
        if not 0 < length <= 1024:
            raise SaveScopeError("invalid profile name length")
        raw_name = take(length)
        try:
            name = raw_name.decode("utf-8")
        except UnicodeDecodeError:
            try:
                name = raw_name.decode("gbk")
            except UnicodeDecodeError as exc:
                raise SaveScopeError("profile name encoding cannot be identified") from exc
        sequence, profile_id = struct.unpack("<II", take(8))
        if not 0 < profile_id < 2**31 or "\x00" in name:
            raise SaveScopeError("invalid profile identity")
        files = [entry for entry in root.iterdir() if entry.name.lower() == f"user{profile_id}.dat"]
        if len(files) != 1 or not files[0].is_file() or files[0].is_symlink():
            raise SaveScopeError(f"profile {profile_id} has no unique regular user file in {root}")
        if files[0].resolve().parent != root:
            raise SaveScopeError("profile file escaped its save directory")
        user_data = files[0].read_bytes()
        if len(user_data) < 820 or struct.unpack_from("<I", user_data)[0] != 12:
            raise SaveScopeError(f"profile {profile_id} is not a supported v12 user file")
        plant_count = struct.unpack_from("<I", user_data, 816)[0]
        if plant_count > 200 or len(user_data) < 820 + plant_count * 88:
            raise SaveScopeError(f"profile {profile_id} has truncated/invalid Zen Garden records")
        profiles.append(SaveProfile(profile_id, name, sequence, files[0]))
    if offset != len(data) or len({p.profile_id for p in profiles}) != count or len({p.name for p in profiles}) != count:
        raise SaveScopeError("users.dat contains trailing data or duplicate profile identities")
    return tuple(profiles)


def resolve_save_profile(directory: str | Path, profile_id: int | None = None) -> SaveProfile:
    profiles = read_save_profiles(directory)
    if profile_id is None:
        if len(profiles) != 1:
            raise SaveScopeError("multiple profiles registered; specify the verified active profile_id")
        return profiles[0]
    if type(profile_id) is not int or profile_id <= 0:
        raise SaveScopeError("profile_id must be a positive integer")
    matches = [profile for profile in profiles if profile.profile_id == profile_id]
    if len(matches) != 1:
        raise SaveScopeError(f"profile {profile_id} is not registered in {directory}")
    return matches[0]


def _writable_directory(target_dir, template_dir=None) -> Path:
    if not target_dir:
        raise SaveScopeError("save cleanup requires an explicitly bound save_dir; automatic directory search is read-only")
    raw = Path(target_dir)
    if raw.is_symlink():
        raise SaveScopeError("save cleanup refuses a symlink directory")
    root = raw.resolve(strict=True)
    if not root.is_dir():
        raise SaveScopeError(f"save directory does not exist: {root}")
    protected = resolve_game_paths(save_template=template_dir).save_template
    if root == protected or protected in root.parents or root in protected.parents:
        raise SaveScopeError(f"training save template is read-only: {protected}")
    return root


def get_pvz_userdata_dirs() -> list[str]:
    """Read-only legacy discovery; never authorizes cleanup."""
    candidate = Path(os.environ.get("ProgramData", r"C:\ProgramData")) / "PopCap Games" / "PlantsVsZombies" / "userdata"
    return [str(candidate)] if candidate.is_dir() else []


def clean_adventure_save(target_dir: str | Path | None = None, *, profile_id: int | None = None,
                         instance_id: str | None = None, template_dir: str | Path | None = None) -> list[str]:
    """Quarantine only game<active id>_0.dat, returning its original path.

    Ambiguous scope fails before native Reset/SelectLevel and never falls back
    to deleting every profile in a system save directory.
    """
    root = _writable_directory(target_dir, template_dir)
    profile = resolve_save_profile(root, profile_id)
    filename = f"game{profile.profile_id}_0.dat"
    matches = [entry for entry in root.iterdir() if entry.name.lower() == filename]
    if not matches:
        return []
    if len(matches) != 1 or not matches[0].is_file() or matches[0].is_symlink() or matches[0].resolve().parent != root:
        raise SaveScopeError(f"adventure save is not a unique regular file: {filename}")
    source = matches[0]
    quarantine = root / ".quarantine"
    if quarantine.exists() and (quarantine.is_symlink() or quarantine.resolve().parent != root):
        raise SaveScopeError("save quarantine escaped its bound save directory")
    destination = quarantine / f"adventure-{time.time_ns()}-{uuid.uuid4().hex}"
    destination.mkdir(parents=True, exist_ok=False)
    receipt = {"version": 1, "instance_id": instance_id, "save_dir": str(root),
               "profile_id": profile.profile_id, "original_name": source.name,
               "quarantined_at_ns": time.time_ns()}
    (destination / "receipt.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(source, destination / source.name)
    logger.info("冒险残局已隔离 instance=%s profile=%s: %s -> %s", instance_id, profile.profile_id, source, destination)
    return [str(source)]


def restore_quarantined_save(quarantined_file: str | Path, target_dir: str | Path, *,
                             profile_id: int, template_dir: str | Path | None = None) -> Path:
    """Restore a known quarantine without overwriting a newly written save."""
    root = _writable_directory(target_dir, template_dir)
    profile = resolve_save_profile(root, profile_id)
    raw_source = Path(quarantined_file)
    if raw_source.is_symlink():
        raise SaveScopeError("quarantined save cannot be a symlink")
    source = raw_source.resolve(strict=True)
    quarantine = root / ".quarantine"
    if quarantine.is_symlink() or quarantine.resolve().parent != root:
        raise SaveScopeError("save quarantine escaped its bound save directory")
    if quarantine.resolve() not in source.parents or source.name.lower() != f"game{profile.profile_id}_0.dat":
        raise SaveScopeError("quarantined save does not belong to this directory/profile")
    destination = root / source.name
    # Exclusive publication prevents overwriting a concurrently created save.
    os.link(source, destination)
    source.unlink()
    return destination
