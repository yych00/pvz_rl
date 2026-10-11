"""Read-only checks for the public training target: retail 1.2.0.1065."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import struct

SUPPORTED_TIMESTAMP = 0x4A37D6AF
SUPPORTED_VERSION = "1.2.0.1065"
# Independently verified file bytes; native runtime also detects version_id.
LAYOUT_SIGNATURES = {
    0x00415D50: bytes.fromhex("6aff68a500650064a1000000"),
    0x0041BB00: bytes.fromhex("535533db33c05783cdff8d9b"),
    0x00445690: bytes.fromhex("53555633db5733f63bf38b44"),
}

class GameCompatibilityError(RuntimeError):
    pass

@dataclass(frozen=True)
class PEImage:
    machine: int
    timestamp: int
    image_base: int
    sections: tuple[tuple[int, int, int], ...]
    data: bytes

    def at_address(self, address: int, size: int) -> bytes:
        rva = address - self.image_base
        for virtual_address, raw_size, raw_offset in self.sections:
            if virtual_address <= rva and rva + size <= virtual_address + raw_size:
                offset = raw_offset + rva - virtual_address
                if offset + size <= len(self.data):
                    return self.data[offset:offset + size]
        raise GameCompatibilityError(f"EXE has no file-backed code at 0x{address:08X}")

def read_pe(path: str | Path) -> PEImage:
    path = Path(path)
    try:
        data = path.read_bytes()
    except OSError as exc:
        raise GameCompatibilityError(f"File unavailable: {path}. Supply the complete game directory.") from exc
    try:
        if len(data) < 64 or data[:2] != b"MZ":
            raise ValueError("missing MZ header")
        nt = struct.unpack_from("<I", data, 60)[0]
        if data[nt:nt + 4] != b"PE\0\0":
            raise ValueError("missing PE header")
        machine, count, timestamp = struct.unpack_from("<HHI", data, nt + 4)
        optional_size = struct.unpack_from("<H", data, nt + 20)[0]
        optional = nt + 24
        magic = struct.unpack_from("<H", data, optional)[0]
        if magic not in (0x10B, 0x20B) or not 0 < count <= 96:
            raise ValueError("invalid optional header/sections")
        base = struct.unpack_from("<I" if magic == 0x10B else "<Q", data,
                                  optional + (28 if magic == 0x10B else 24))[0]
        sections = []
        for index in range(count):
            _name, _vsize, va, size, offset = struct.unpack_from("<8sIIII", data, optional + optional_size + 40 * index)
            if offset + size > len(data):
                raise ValueError("truncated section")
            sections.append((va, size, offset))
        return PEImage(machine, timestamp, base, tuple(sections), data)
    except (ValueError, struct.error, OverflowError) as exc:
        raise GameCompatibilityError(f"File is not a supported PE image: {path}: {exc}") from exc

def validate_game(executable: str | Path, *, check_resources: bool = True) -> PEImage:
    path = Path(executable).resolve()
    pe = read_pe(path)
    if pe.machine != 0x14C or pe.image_base != 0x400000 or pe.timestamp != SUPPORTED_TIMESTAMP:
        raise GameCompatibilityError(
            f"Unsupported game: expected retail {SUPPORTED_VERSION} x86 layout; "
            f"machine=0x{pe.machine:X}, timestamp=0x{pe.timestamp:X}. "
            "1.0.0.1051, GOTY and unknown layouts are outside the public training target.")
    for address, expected in LAYOUT_SIGNATURES.items():
        if pe.at_address(address, len(expected)) != expected:
            raise GameCompatibilityError(f"Game layout signature mismatch at 0x{address:08X}; EXE may be modified.")
    if b"PlantsVsZombies" not in pe.data:
        raise GameCompatibilityError("Game product/mutex name is missing; supply an original executable.")
    if check_resources:
        pak = path.parent / "main.pak"
        loose = all((path.parent / n).is_dir() and any(p.is_file() for p in (path.parent / n).rglob("*"))
                    for n in ("images", "sounds", "reanim"))
        if not (pak.is_file() and pak.stat().st_size > 0) and not loose:
            raise GameCompatibilityError("Missing main.pak or extracted images/sounds/reanim resources. Copy the entire game directory.")
    return pe
