"""Own and launch private PvZ copies with distinct product/mutex/save names."""
from __future__ import annotations
import os
import hashlib
import json
import logging
import shutil
import struct
import subprocess
import threading
import time
import winreg
import ctypes
import uuid
from ctypes import wintypes
from pathlib import Path

from ..env_file import resolve_game_paths
from ..save_cleaner import SaveScopeError, read_save_profiles, resolve_save_profile
from .instance_pool import InstancePool, atomic_json, process_creation_time
from .source_fingerprint import cached_file_hashes
from .game_validation import validate_game

logger = logging.getLogger("pvz_gym.instances")
MANIFEST_VERSION = 1
PREPARATION_VERSION = 2


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _assert_child(path: Path, root: Path) -> None:
    resolved = path.resolve()
    if resolved == root.resolve() or root.resolve() not in resolved.parents or path.is_symlink():
        raise RuntimeError(f"private instance path escaped its verified root: {path}")



class _Limits(ctypes.Structure):
    _fields_ = [("process_time", ctypes.c_int64), ("job_time", ctypes.c_int64),
                ("flags", wintypes.DWORD), ("min_working_set", ctypes.c_size_t),
                ("max_working_set", ctypes.c_size_t), ("processes", wintypes.DWORD),
                ("affinity", ctypes.c_size_t), ("priority", wintypes.DWORD), ("scheduling", wintypes.DWORD)]


class _IoCounters(ctypes.Structure):
    _fields_ = [(name, ctypes.c_uint64) for name in ("read", "write", "other", "read_bytes", "write_bytes", "other_bytes")]


class _JobLimits(ctypes.Structure):
    _fields_ = [("basic", _Limits), ("io", _IoCounters), ("process_memory", ctypes.c_size_t),
                ("job_memory", ctypes.c_size_t), ("peak_process", ctypes.c_size_t), ("peak_job", ctypes.c_size_t)]


class _OwnedJob:
    """Windows closes this handle on worker death and kills only its game."""
    def __init__(self, process_handle):
        self.api = ctypes.WinDLL("kernel32", use_last_error=True)
        self.api.CreateJobObjectW.argtypes, self.api.CreateJobObjectW.restype = [ctypes.c_void_p, wintypes.LPCWSTR], wintypes.HANDLE
        self.api.SetInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
        self.api.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
        self.api.CloseHandle.argtypes = [wintypes.HANDLE]
        self.handle = self.api.CreateJobObjectW(None, None)
        if not self.handle: raise ctypes.WinError(ctypes.get_last_error())
        limits = _JobLimits(); limits.basic.flags = 0x2000  # KILL_ON_JOB_CLOSE
        if not self.api.SetInformationJobObject(self.handle, 9, ctypes.byref(limits), ctypes.sizeof(limits)) or not self.api.AssignProcessToJobObject(self.handle, int(process_handle)):
            error = ctypes.get_last_error(); self.close(); raise ctypes.WinError(error)

    def close(self):
        if self.handle: self.api.CloseHandle(self.handle); self.handle = None


class GameInstance:
    """Lease a stable disk slot; game processes are reclaimed at each Run end."""
    def __init__(self, executable=None, root=None, environment_id: int = 0, *,
                 save_template=None, profile_id: int | None = None, pool: InstancePool | None = None):
        self.paths = resolve_game_paths(executable, save_template, root)
        self.template_dir = self.paths.save_template
        self.source = self.paths.executable.resolve(strict=True)
        if not self.source.is_file():
            raise RuntimeError(f"game executable is not a file: {self.source}")
        for protected in (self.source.parent, self.paths.save_template):
            if self.paths.instance_root == protected or protected in self.paths.instance_root.parents or self.paths.instance_root in protected.parents:
                raise RuntimeError("instance_root must be separate from the original game and save template")
        self.environment_id = environment_id
        self.requested_profile_id = profile_id
        self.pool = pool or InstancePool(self.paths.instance_root)
        if self.pool.root != self.paths.instance_root:
            raise ValueError("supplied instance pool does not match instance_root")
        self.lease = None
        self.directory = self.pool.root / f"env-{environment_id:02d}"
        self.process = None
        self.job = None
        self.product = None
        self.save_dir = None
        self.instance_id = None
        self.active_profile_id = None
        self.active_profile_name = None
        self.manifest = None

    def _acquire(self, *, rebuild=False):
        if self.lease is not None:
            return
        self.lease = self.pool.acquire(self.environment_id, executable_name=self.source.name, allow_invalid_manifest=rebuild)
        self.directory, self.product, self.instance_id = self.lease.directory, self.lease.product, self.lease.instance_id
        try:
            popcap = (Path(os.environ.get("ProgramData", r"C:\ProgramData")) / "PopCap Games").resolve()
            product_dir = popcap / self.product
            self.save_dir = product_dir / "userdata"
            if product_dir.resolve() != product_dir or self.save_dir.resolve() != self.save_dir:
                raise SaveScopeError("private product/save directory is redirected by a symlink or junction")
            for protected in (self.source.parent, self.paths.save_template):
                if self.save_dir == protected or protected in self.save_dir.parents or self.save_dir in protected.parents:
                    raise SaveScopeError("private saves overlap the original game or save template")
            self.lease.update(save_dir=str(self.save_dir), executable=str(self.directory / self.source.name))
        except BaseException:
            self.lease.release()
            self.lease = None
            raise

    def _source_description(self, *, force_verify=False):
        validate_game(self.source)
        template = self.paths.save_template.resolve(strict=True)
        profiles = read_save_profiles(template)
        resolve_save_profile(template, self.requested_profile_id)
        resources = {}
        for path in sorted(self.source.parent.rglob("*")):
            if path.is_symlink():
                raise RuntimeError(f"original game resource is a symlink: {path}")
            relative = path.relative_to(self.source.parent)
            if relative.parts[0].lower() == "userdata" or path == self.source or path.suffix.lower() in (".exe", ".bak"):
                continue
            if path.is_file():
                resources[relative.as_posix()] = path
        template_paths = {path.name: path for path in sorted(template.iterdir()) if path.is_file()}
        paths = {"executable": self.source,
                 **{f"resource:{name}": path for name, path in resources.items()},
                 **{f"template:{name}": path for name, path in template_paths.items()}}
        identity = json.dumps([os.path.normcase(str(self.source)), os.path.normcase(str(template))])
        cache_key = hashlib.sha256(identity.encode()).hexdigest()
        hashes = cached_file_hashes(paths, self.pool.root / ".source-fingerprints" / f"{cache_key}.json",
                                   force=force_verify)
        resources = {name: hashes[f"resource:{name}"] for name in resources}
        template_files = {name: hashes[f"template:{name}"] for name in template_paths}
        return {"source_executable": str(self.source), "source_sha256": hashes["executable"],
                "resources": resources, "save_template": str(template),
                "template_files": template_files,
                "template_sha256": hashlib.sha256(json.dumps(template_files, sort_keys=True).encode()).hexdigest(),
                "profile_ids": [profile.profile_id for profile in profiles]}

    def _seed_saves(self):
        """Replace private saves recoverably, never move or write the template."""
        profiles = read_save_profiles(self.paths.save_template)
        parent = self.save_dir.parent
        if parent.is_symlink() or self.save_dir.is_symlink():
            raise SaveScopeError("private save directory cannot be a symlink")
        parent.mkdir(parents=True, exist_ok=True)
        incoming = parent / f".userdata-new-{uuid.uuid4().hex}"
        _assert_child(incoming, parent)
        incoming.mkdir()
        index = next(path for path in self.paths.save_template.iterdir() if path.name.lower() == "users.dat")
        for source in (index, *(profile.user_file for profile in profiles)):
            shutil.copy2(source, incoming / source.name)
        resolve_save_profile(incoming, self.requested_profile_id)
        backup = None
        if self.save_dir.exists():
            _assert_child(self.save_dir, parent)
            backup = parent / f"userdata.backup-{time.time_ns()}-{uuid.uuid4().hex}"
            _assert_child(backup, parent)
            os.replace(self.save_dir, backup)
        try:
            os.replace(incoming, self.save_dir)
        except BaseException:
            if backup is not None and not self.save_dir.exists():
                os.replace(backup, self.save_dir)
            raise
        return str(backup) if backup is not None else None

    def _archive_instance(self):
        if self.directory.exists():
            _assert_child(self.directory, self.pool.root)
            archive = self.pool.root / ".archive"
            if archive.is_symlink():
                raise RuntimeError("instance archive cannot be a symlink")
            archive.mkdir(exist_ok=True)
            destination = archive / f"{self.directory.name}-{time.time_ns()}-{uuid.uuid4().hex}"
            _assert_child(destination, self.pool.root)
            os.replace(self.directory, destination)

    def _bind_profile(self):
        profile = resolve_save_profile(self.save_dir, self.requested_profile_id)
        self.active_profile_id, self.active_profile_name = profile.profile_id, profile.name
        self.lease.update(profile_id=profile.profile_id)

    def prepare(self, *, rebuild: bool = False):
        """Reuse a verified manifest or prepare once under the exclusive lease."""
        self._acquire(rebuild=rebuild)
        try:
            return self._prepare_locked(rebuild=rebuild)
        except BaseException:
            if self.process is None:
                self.lease.release()
                self.lease = None
            raise

    def _prepare_locked(self, *, rebuild=False):
        description = self._source_description(force_verify=rebuild)
        manifest_path = self.directory / "instance.json"
        previous = None
        if manifest_path.is_file() and not rebuild:
            try:
                previous = json.loads(manifest_path.read_text(encoding="utf-8"))
            except (ValueError, OSError) as exc:
                raise RuntimeError("private instance manifest is unreadable; use Rebuild Instance after stopping its game") from exc
            if previous.get("manifest_version") != MANIFEST_VERSION or previous.get("instance_id") != self.instance_id or previous.get("product") != self.product or previous.get("save_dir") != str(self.save_dir):
                raise RuntimeError("private instance manifest identity/version mismatch; use Rebuild Instance")
        target = self.directory / self.source.name
        if (previous is not None and previous.get("preparation_version") == PREPARATION_VERSION
                and previous.get("source") == description and target.is_file()
                and previous.get("private_executable_sha256") == _sha256(target)
                and self.save_dir.is_dir()):
            self.manifest = previous
            if "source_pe_timestamp" not in previous:
                source_image = self.source.read_bytes()
                nt = struct.unpack_from("<I", source_image, 60)[0]
                timestamp = struct.unpack_from("<I", source_image, nt + 8)[0]
                previous.update(source_pe_timestamp=timestamp,
                                game_version="1.2.0.1065" if timestamp == 0x4a37d6af else "1.0.0.1051")
                atomic_json(manifest_path, previous)
            self._bind_profile()
            self._apply_zero_volume_registry()
            logger.info("复用实例 %s exe=%s save_dir=%s profile=%s", self.instance_id, target, self.save_dir, self.active_profile_id)
            return target
        if self.process is not None and self.process.poll() is None:
            raise RuntimeError("stop the private game before preparing/rebuilding an instance")
        image = self.source.read_bytes()
        if len(image) < 64 or image[:2] != b"MZ":
            raise RuntimeError("game executable is not a supported PE image")
        nt = struct.unpack_from("<I", image, 60)[0]
        if nt + 12 > len(image) or image[nt:nt + 4] != b"PE\0\0":
            raise RuntimeError("game executable has an invalid PE header")
        timestamp = struct.unpack_from("<I", image, nt + 8)[0]
        if timestamp not in (0x49ecf563, 0x4a37d6af):
            raise RuntimeError("isolated launch currently supports retail 1.0.0.1051/1.2.0.1065 only")
        needle = b"PlantsVsZombies"
        if needle not in image: raise RuntimeError("game product/mutex name could not be verified")
        if len(self.product.encode("ascii")) != len(needle):
            raise RuntimeError("private game product name must preserve the binary image size")
        image = image.replace(needle, self.product.encode("ascii"))
        title = b"Plants vs. Zombies"
        private_title = self.product.encode("ascii").ljust(len(title), b" ")
        image = image.replace(title, private_title)
        self._archive_instance()
        backup = self._seed_saves()
        self.directory.mkdir(parents=True, exist_ok=True)
        target = self.directory / self.source.name
        target.write_bytes(image)
        # Large resources are immutable and shared by hardlinks. Writable game
        # caches, properties and saves get private copies.
        for child in self.source.parent.iterdir():
            if child == self.source or child.name.lower() == "userdata" or child.suffix.lower() in (".exe", ".bak"): continue
            dest = self.directory / child.name
            if dest.exists(): continue
            if child.is_dir():
                shutil.copytree(child, dest)
            elif child.name.lower() == "main.pak":
                try: os.link(child, dest)
                except OSError: shutil.copy2(child, dest)
            else: shutil.copy2(child, dest)
        # The optional, signed vendor partner file overrides the product name.
        # A private training copy uses the executable defaults. Preserve the
        # original signed file in the source game directory.
        partner_target = self.directory / "properties" / "partner.xml"
        if partner_target.is_file():
            if self.directory.resolve() not in partner_target.resolve().parents:
                raise RuntimeError("private partner file escaped the instance directory")
            partner_target.unlink()
        self._bind_profile()
        self.manifest = {"manifest_version": MANIFEST_VERSION, "preparation_version": PREPARATION_VERSION,
                         "instance_id": self.instance_id, "slot": self.lease.slot, "product": self.product,
                         "source_pe_timestamp": timestamp,
                         "game_version": "1.2.0.1065" if timestamp == 0x4a37d6af else "1.0.0.1051",
                         "source": description, "private_executable_sha256": _sha256(target),
                         "save_dir": str(self.save_dir), "active_profile_id": self.active_profile_id,
                         "created_at_ns": time.time_ns(), "pid": None, "last_save_backup": backup}
        atomic_json(manifest_path, self.manifest)
        self._apply_zero_volume_registry()
        logger.info("准备实例 %s exe=%s template=%s save_dir=%s profile=%s", self.instance_id, target, self.paths.save_template, self.save_dir, self.active_profile_id)
        return target

    def _apply_zero_volume_registry(self) -> None:
        """预设私有实例注册表音频配置为零音量并静音，避免游戏启动瞬间发出声音"""
        if self.product is None or self.lease is None:
            raise RuntimeError("private registry configuration requires an owned instance lease")
        sub_key = rf"Software\PopCap\{self.product}"
        try:
            with winreg.CreateKeyEx(winreg.HKEY_CURRENT_USER, sub_key, 0, winreg.KEY_SET_VALUE) as key:
                winreg.SetValueEx(key, "MusicVolume", 0, winreg.REG_DWORD, 0)
                winreg.SetValueEx(key, "SfxVolume", 0, winreg.REG_DWORD, 0)
                winreg.SetValueEx(key, "Muted", 0, winreg.REG_DWORD, 1)
                winreg.SetValueEx(key, "ScreenMode", 0, winreg.REG_DWORD, 0)
                winreg.SetValueEx(key, "WaitForVSync", 0, winreg.REG_DWORD, 0)
                if self.active_profile_name is not None:
                    winreg.SetValueEx(key, "CurUser", 0, winreg.REG_SZ, self.active_profile_name)
        except OSError as exc:
            raise SaveScopeError(f"cannot bind private instance CurUser registry profile: {self.instance_id}") from exc

    def _cleanup_registry(self) -> None:
        """清理私有实例在注册表中的临时配置"""
        if self.product is None:
            return
        sub_key = rf"Software\PopCap\{self.product}"
        try:
            winreg.DeleteKey(winreg.HKEY_CURRENT_USER, sub_key)
        except OSError:
            pass

    def _start_audio_mute_worker(self, pid: int) -> None:
        """子进程启动后即时执行 WASAPI 静音，并在早期启动阶段持续重试以捕获新创建的音频会话"""
        try:
            from pvz_gym.bridge.native import set_process_audio_muted
            set_process_audio_muted(pid, True)
        except Exception:
            set_process_audio_muted = None

        if set_process_audio_muted is None:
            return

        def _worker():
            deadline = time.monotonic() + 5.0
            muted_count = 0
            while time.monotonic() < deadline:
                if self.process is None or self.process.poll() is not None:
                    break
                try:
                    if set_process_audio_muted(pid, True):
                        muted_count += 1
                        if muted_count >= 2:
                            break
                except Exception:
                    pass
                time.sleep(0.5)

        threading.Thread(target=_worker, daemon=True).start()

    def start(self):
        if self.process is not None and self.process.poll() is None:
            return self.process.pid
        executable = self.prepare()
        startup = subprocess.STARTUPINFO()
        startup.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        startup.wShowWindow = 4  # SW_SHOWNOACTIVATE, no focus stealing
        try:
            # Bind its PID and kill-on-close Job before any game code writes a save.
            self.process = subprocess.Popen([str(executable)], cwd=self.directory,
                                            startupinfo=startup, creationflags=0x4)  # CREATE_SUSPENDED
            creation_time = process_creation_time(self.process.pid)
            self.lease.update(game_pid=self.process.pid, game_creation_time_100ns=creation_time)
            self.job = _OwnedJob(self.process._handle)
            self.manifest.update(pid=self.process.pid, game_creation_time_100ns=creation_time, last_started_at_ns=time.time_ns())
            atomic_json(self.directory / "instance.json", self.manifest)
            api = ctypes.WinDLL("ntdll")
            api.NtResumeProcess.argtypes, api.NtResumeProcess.restype = [wintypes.HANDLE], ctypes.c_long
            status = api.NtResumeProcess(int(self.process._handle))
            if status < 0:
                raise RuntimeError(f"cannot resume private game: NTSTATUS={status:#x}")
        except BaseException:
            self.close()
            raise
        self._start_audio_mute_worker(self.process.pid)
        return self.process.pid

    def close(self, *, release_lease=True):
        pid = self.process.pid if self.process else None
        if self.job:
            self.job.close(); self.job = None
            if self.process:
                try: self.process.wait(timeout=10)
                except subprocess.TimeoutExpired: pass
        if self.process and self.process.poll() is None:
            self.process.terminate()
            try: self.process.wait(timeout=10)
            except subprocess.TimeoutExpired: self.process.kill(); self.process.wait(timeout=5)
        self.process = None
        if self.lease is not None:
            self.lease.update(game_pid=None)
            if self.manifest is not None and self.directory.is_dir():
                self.manifest.update(pid=None, last_stopped_at_ns=time.time_ns())
                atomic_json(self.directory / "instance.json", self.manifest)
        if pid:
            try:
                from pvz_gym.bridge.native import set_process_audio_muted
                set_process_audio_muted(pid, False)
            except Exception:
                pass
        self._cleanup_registry()
        if release_lease and self.lease is not None:
            self.lease.release()
            self.lease = None

    def reset_game(self):
        """Restart the owned process while retaining the private save/profile data."""
        self.close(release_lease=False)
        return self.start()

    def reset_save(self):
        """Restore template profiles after stopping the game; archive current saves."""
        if self.process is not None and self.process.poll() is None:
            raise RuntimeError("stop the private game before Reset Save")
        self._acquire()
        manifest_path = self.directory / "instance.json"
        if not manifest_path.is_file():
            self.prepare()
        else:
            try:
                self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            except (ValueError, OSError) as exc:
                raise RuntimeError("private instance manifest is unreadable; use Rebuild Instance") from exc
            if (self.manifest.get("manifest_version") != MANIFEST_VERSION
                    or self.manifest.get("instance_id") != self.instance_id
                    or self.manifest.get("product") != self.product
                    or self.manifest.get("save_dir") != str(self.save_dir)):
                raise RuntimeError("private instance manifest identity/version mismatch; use Rebuild Instance")
        backup = self._seed_saves()
        self._bind_profile()
        self.manifest.update(active_profile_id=self.active_profile_id, last_save_backup=backup)
        atomic_json(self.directory / "instance.json", self.manifest)
        return backup

    def rebuild(self):
        """Recreate resources and template saves; retain recoverable old directories."""
        if self.process is not None and self.process.poll() is None:
            raise RuntimeError("stop the private game before Rebuild Instance")
        return self.prepare(rebuild=True)

    def __enter__(self): self.start(); return self
    def __exit__(self, *_): self.close()
