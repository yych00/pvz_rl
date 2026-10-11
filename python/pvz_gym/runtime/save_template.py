"""Generate a dedicated retail profile without importing personal data.

Serialization references and the target layout are recorded in docs/game_setup.md.
"""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import struct
import tempfile
from ..env_file import resolve_game_paths
from ..save_cleaner import read_save_profiles, SaveScopeError
from .game_validation import validate_game

PROFILE_NAME = "PvZRL"
PROFILE_ID = 1
TEMPLATE_VERSION = 1

def training_profile_bytes() -> tuple[bytes, bytes]:
    name = PROFILE_NAME.encode("ascii")
    index = struct.pack("<IHH", 14, 1, len(name)) + name + struct.pack("<II", 1, PROFILE_ID)
    # Retail v12: 4 header longs, 100 records, 80 purchases, 20 stats, plant count.
    # No GOTY achievements or Zombatar trailer.
    values = [0] * 205
    values[:4] = [12, 1, 0, 2]
    for item in range(9):  # eight shop upgrade plants and Imitater
        values[104 + item] = 1
    values[104 + 21] = 4  # ten seed slots
    values[104 + 22] = 1  # pool cleaner
    values[104 + 23] = 1  # roof cleaner
    return index, struct.pack("<205I", *values)

def prepare_training_save(executable: str | Path | None = None,
                          output: str | Path | None = None) -> Path:
    paths = resolve_game_paths(executable, output)
    validate_game(paths.executable)
    target = paths.save_template
    personal = (Path(os.environ.get("ProgramData", r"C:\ProgramData")) /
                "PopCap Games" / "PlantsVsZombies" / "userdata").resolve()
    for protected in (paths.executable.parent, personal, paths.instance_root):
        if target == protected or protected in target.parents or target in protected.parents:
            raise SaveScopeError("Template must be separate from the game, personal saves and instance pool")
    if any(p.is_symlink() or (hasattr(p, "is_junction") and p.is_junction()) for p in (target, *target.parents)):
        raise SaveScopeError("Template cannot follow symlinks or junctions")
    index, profile = training_profile_bytes()
    files = {"users.dat": index, "user1.dat": profile}
    metadata = {"template_version": TEMPLATE_VERSION, "game_version": "1.2.0.1065",
                "profile_id": PROFILE_ID, "profile_name": PROFILE_NAME,
                "sha256": {n: hashlib.sha256(b).hexdigest() for n, b in files.items()}}
    if target.exists() and any(target.iterdir()):
        if not (target / "template.json").is_file():
            raise SaveScopeError("Refusing to replace existing saves; choose an empty template directory")
        try:
            existing = json.loads((target / "template.json").read_text(encoding="utf-8"))
            valid = existing == metadata and all((target / n).read_bytes() == b for n, b in files.items())
        except (OSError, ValueError):
            valid = False
        if not valid:
            raise SaveScopeError("Template is changed or corrupt; choose a new empty directory. Existing files are retained.")
        read_save_profiles(target)
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".pvzrl-template-", dir=target.parent) as temporary:
        staging = Path(temporary) / "save"
        staging.mkdir()
        for name, data in files.items():
            (staging / name).write_bytes(data)
        (staging / "template.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        read_save_profiles(staging)
        if target.exists():
            target.rmdir()
        os.rename(staging, target)
    return target
