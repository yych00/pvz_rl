"""Persistent disk slots with OS-held exclusive leases (never time-expiring)."""
from __future__ import annotations

import ctypes
import hashlib
import json
import os
import time
import uuid
from ctypes import wintypes
from pathlib import Path


class InstanceBusyError(RuntimeError):
    pass


def atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def process_is_alive(pid: int | None) -> bool:
    """Return true when liveness is unknown; never reclaim by elapsed timeout."""
    if type(pid) is not int or pid <= 0:
        return False
    if os.name != "nt":
        try:
            os.kill(pid, 0)
            return True
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
    api = ctypes.WinDLL("kernel32", use_last_error=True)
    api.OpenProcess.argtypes, api.OpenProcess.restype = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD], wintypes.HANDLE
    api.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
    api.CloseHandle.argtypes = [wintypes.HANDLE]
    handle = api.OpenProcess(0x1000, False, pid)
    if not handle:
        return ctypes.get_last_error() != 87  # INVALID_PARAMETER: no such PID
    try:
        code = wintypes.DWORD()
        return not api.GetExitCodeProcess(handle, ctypes.byref(code)) or code.value == 259
    finally:
        api.CloseHandle(handle)


def process_creation_time(pid: int | None) -> int | None:
    """Windows FILETIME identity, retained alongside PIDs to diagnose PID reuse."""
    if os.name != "nt" or type(pid) is not int or pid <= 0:
        return None
    api = ctypes.WinDLL("kernel32", use_last_error=True)
    api.OpenProcess.argtypes, api.OpenProcess.restype = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD], wintypes.HANDLE
    api.GetProcessTimes.argtypes = [wintypes.HANDLE, *([ctypes.POINTER(wintypes.FILETIME)] * 4)]
    api.CloseHandle.argtypes = [wintypes.HANDLE]
    handle = api.OpenProcess(0x1000, False, pid)
    if not handle:
        return None
    try:
        values = [wintypes.FILETIME() for _ in range(4)]
        if not api.GetProcessTimes(handle, *(ctypes.byref(value) for value in values)):
            return None
        return (values[0].dwHighDateTime << 32) | values[0].dwLowDateTime
    finally:
        api.CloseHandle(handle)


def processes_for_executable(executable: Path) -> tuple[int, ...]:
    """Catch an orphan created before its game PID could be written to disk."""
    if os.name != "nt" or not executable.is_file():
        return ()

    class Entry(ctypes.Structure):
        _fields_ = [("size", wintypes.DWORD), ("usage", wintypes.DWORD),
                    ("pid", wintypes.DWORD), ("heap", ctypes.c_size_t),
                    ("module", wintypes.DWORD), ("threads", wintypes.DWORD),
                    ("parent", wintypes.DWORD), ("priority", wintypes.LONG),
                    ("flags", wintypes.DWORD), ("name", wintypes.WCHAR * 260)]

    api = ctypes.WinDLL("kernel32", use_last_error=True)
    api.CreateToolhelp32Snapshot.argtypes, api.CreateToolhelp32Snapshot.restype = [wintypes.DWORD, wintypes.DWORD], wintypes.HANDLE
    api.Process32FirstW.argtypes = api.Process32NextW.argtypes = [wintypes.HANDLE, ctypes.POINTER(Entry)]
    api.OpenProcess.argtypes, api.OpenProcess.restype = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD], wintypes.HANDLE
    api.QueryFullProcessImageNameW.argtypes = [wintypes.HANDLE, wintypes.DWORD, wintypes.LPWSTR, ctypes.POINTER(wintypes.DWORD)]
    api.CloseHandle.argtypes = [wintypes.HANDLE]
    snapshot = api.CreateToolhelp32Snapshot(2, 0)
    if snapshot == ctypes.c_void_p(-1).value:
        raise InstanceBusyError("cannot verify that the old private game exited")
    found = []
    expected = os.path.normcase(str(executable.resolve()))
    try:
        entry = Entry(); entry.size = ctypes.sizeof(Entry)
        ready = api.Process32FirstW(snapshot, ctypes.byref(entry))
        while ready:
            if entry.name.lower() == executable.name.lower():
                handle = api.OpenProcess(0x1000, False, entry.pid)
                if not handle:
                    raise InstanceBusyError(f"cannot verify save ownership for game PID {entry.pid}")
                try:
                    size = wintypes.DWORD(32768)
                    path = ctypes.create_unicode_buffer(size.value)
                    if not api.QueryFullProcessImageNameW(handle, 0, path, ctypes.byref(size)):
                        raise InstanceBusyError(f"cannot verify image for game PID {entry.pid}")
                    if os.path.normcase(str(Path(path.value).resolve())) == expected:
                        found.append(entry.pid)
                finally:
                    api.CloseHandle(handle)
            ready = api.Process32NextW(snapshot, ctypes.byref(entry))
        if ctypes.get_last_error() != 18:  # ERROR_NO_MORE_FILES
            raise InstanceBusyError("process enumeration failed while checking the old game")
    finally:
        api.CloseHandle(snapshot)
    return tuple(found)


def _lock(handle) -> None:
    handle.seek(0)
    if os.name == "nt":
        import msvcrt
        msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
    else:
        import fcntl
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)


def _unlock(handle) -> None:
    handle.seek(0)
    if os.name == "nt":
        import msvcrt
        msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
    else:
        import fcntl
        fcntl.flock(handle, fcntl.LOCK_UN)


class InstanceLease:
    def __init__(self, pool, slot, handle):
        self.pool, self.slot, self._handle = pool, slot, handle
        self.directory = pool.root / f"env-{slot:02d}"
        self.instance_id = f"{pool.pool_id}/env-{slot:02d}"
        # Retail product/mutex strings must retain the exact 15-byte length.
        identity = hashlib.sha256(f"{pool.pool_id}:{slot}".encode()).hexdigest()[:8]
        self.product = f"PvZGym-{identity}"
        self.path = pool.root / f"env-{slot:02d}.lease.json"
        self.record = {"version": 1, "instance_id": self.instance_id, "slot": slot,
                       "token": uuid.uuid4().hex, "owner_pid": os.getpid(),
                       "owner_creation_time_100ns": process_creation_time(os.getpid()),
                       "game_pid": None, "leased_at_ns": time.time_ns(), "state": "leased"}
        atomic_json(self.path, self.record)

    def update(self, **fields):
        if self._handle is None:
            raise RuntimeError("instance lease is closed")
        self.record.update(fields)
        atomic_json(self.path, self.record)

    def release(self):
        if self._handle is None:
            return
        if self.pool.process_probe(self.record.get("game_pid")):
            raise InstanceBusyError("cannot release a slot while its game process is alive")
        self.update(state="released", owner_pid=None, game_pid=None, released_at_ns=time.time_ns())
        _unlock(self._handle)
        self._handle.close()
        self._handle = None


class InstancePool:
    """Runs reuse disk slots; each lease still owns and closes its game process."""
    def __init__(self, root: str | Path, *, max_slots: int = 64,
                 process_probe=None, executable_probe=None):
        if type(max_slots) is not int or max_slots <= 0:
            raise ValueError("max_slots must be a positive integer")
        raw_root = Path(root)
        if raw_root.is_symlink():
            raise ValueError("instance root cannot be a symlink")
        self.root = raw_root.resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.max_slots = max_slots
        self.pool_id = hashlib.sha256(os.path.normcase(str(self.root)).encode()).hexdigest()[:16]
        self.process_probe = process_probe or process_is_alive
        self.executable_probe = executable_probe or processes_for_executable

    def acquire(self, preferred_slot: int = 0, *, executable_name: str | None = None,
                allow_invalid_manifest: bool = False) -> InstanceLease:
        if type(preferred_slot) is not int or not 0 <= preferred_slot < self.max_slots:
            raise ValueError("preferred_slot is outside the instance pool")
        for slot in (*range(preferred_slot, self.max_slots), *range(preferred_slot)):
            lockfile = self.root / f"env-{slot:02d}.lock"
            if lockfile.is_symlink():
                raise InstanceBusyError("instance lock path cannot be a symlink")
            handle = lockfile.open("a+b")
            try:
                # A non-empty first byte also works with Windows byte-range locks.
                if lockfile.stat().st_size == 0:
                    handle.write(b"\0"); handle.flush()
                try:
                    _lock(handle)
                except OSError:
                    continue
                record_path = self.root / f"env-{slot:02d}.lease.json"
                if record_path.exists():
                    try:
                        old = json.loads(record_path.read_text(encoding="utf-8"))
                    except (OSError, ValueError) as exc:
                        raise InstanceBusyError(f"cannot validate old lease for slot {slot}") from exc
                    if self.process_probe(old.get("game_pid")) or (old.get("state") == "leased" and self.process_probe(old.get("owner_pid"))):
                        _unlock(handle)
                        continue
                manifest_path = self.root / f"env-{slot:02d}" / "instance.json"
                if manifest_path.exists():
                    try:
                        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                    except (OSError, ValueError) as exc:
                        if not allow_invalid_manifest or not executable_name:
                            raise InstanceBusyError(f"cannot verify old game PID from slot {slot} manifest") from exc
                        manifest = {}
                    if self.process_probe(manifest.get("pid")):
                        _unlock(handle)
                        continue
                if executable_name and self.executable_probe(self.root / f"env-{slot:02d}" / executable_name):
                    _unlock(handle)
                    continue
                lease = InstanceLease(self, slot, handle)
                handle = None  # Lease owns the OS lock until process cleanup.
                return lease
            finally:
                if handle is not None:
                    handle.close()
        raise InstanceBusyError("all private game slots are in use or awaiting process exit")
