"""Share source SHA256 results across workers under an OS-held file lock."""
from __future__ import annotations

from contextlib import contextmanager
import errno
import hashlib
import json
import os
from pathlib import Path
import time
from collections.abc import Mapping

from .instance_pool import _lock, _unlock, atomic_json

CACHE_VERSION = 1


def _signature(path: Path) -> list[int]:
    if path.is_symlink():
        raise RuntimeError(f"source fingerprint cannot follow a symlink: {path}")
    stat = path.stat()
    return [stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns, stat.st_dev, stat.st_ino]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@contextmanager
def _cache_lock(path: Path):
    # Unlike a PID/timestamp lock, the OS releases this on worker death.
    with path.open("a+b") as handle:
        if path.stat().st_size == 0:
            handle.write(b"\0")
            handle.flush()
        deadline = time.monotonic() + 60.0
        while True:
            try:
                _lock(handle)
                break
            except OSError as exc:
                if exc.errno not in (errno.EACCES, errno.EAGAIN, errno.EDEADLK):
                    raise
                if time.monotonic() >= deadline:
                    raise TimeoutError("shared source fingerprint lock timed out") from exc
                time.sleep(0.02)
        try:
            yield
        finally:
            _unlock(handle)


def cached_file_hashes(paths: Mapping[str, Path], cache_path: Path, *, force: bool = False) -> dict[str, str]:
    """Rehash changed files; explicit Rebuild bypasses every cached digest.

    Normal startup trusts unchanged path, size, timestamps and file identity.
    New/deleted files are reflected in the caller's complete path inventory.
    """
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with _cache_lock(cache_path.with_suffix(".lock")):
        try:
            previous = json.loads(cache_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            previous = None
        cached = previous.get("files", {}) if isinstance(previous, dict) and previous.get("version") == CACHE_VERSION else {}
        if not isinstance(cached, dict):
            cached = {}
        entries = {}
        hashes = {}
        for name, path in paths.items():
            identity = os.path.normcase(str(path.absolute()))
            signature = _signature(path)
            entry = cached.get(name)
            digest = entry.get("sha256") if isinstance(entry, dict) else None
            valid_digest = (isinstance(digest, str) and len(digest) == 64
                            and all(char in "0123456789abcdef" for char in digest))
            if force or not valid_digest or entry.get("path") != identity or entry.get("signature") != signature:
                # Do not publish a digest paired with metadata from a different
                # version of a file modified during hashing.
                for _ in range(3):
                    signature = _signature(path)
                    digest = _sha256(path)
                    if _signature(path) == signature:
                        break
                else:
                    raise RuntimeError(f"source changed repeatedly during SHA256 verification: {path}")
            hashes[name] = digest
            entries[name] = {"path": identity, "signature": signature, "sha256": digest}
        payload = {"version": CACHE_VERSION, "files": entries}
        if payload != previous:
            atomic_json(cache_path, payload)
        return hashes
