"""Read-only diagnosis; never attaches to a personal game."""
import argparse
from dataclasses import asdict
import json
import platform
import struct
import sys
from pathlib import Path
from pvz_gym.env_file import repository_root, resolve_game_paths
from pvz_gym.runtime.game_validation import validate_game, read_pe
from pvz_gym.save_cleaner import read_save_profiles

def diagnose(game_executable=None, dll_path=None, *, require_save=True):
    checks = []
    paths = resolve_game_paths(game_executable)
    def check(name, fn):
        try:
            detail = fn()
            checks.append({"name": name, "ok": True, "detail": str(detail)})
        except Exception as exc:
            checks.append({"name": name, "ok": False, "detail": str(exc)})
    def python_check():
        if platform.system() != "Windows" or struct.calcsize("P") != 8 or not (3, 10) <= sys.version_info[:2] < (3, 13):
            raise ValueError("Windows x64 and 64-bit Python 3.10-3.12 are required")
        return f"{sys.version.split()[0]} x64 on Windows"
    check("python", python_check)
    check("game", lambda: f"retail 1.2.0.1065 x86; timestamp 0x{validate_game(paths.executable).timestamp:X}; code signatures passed")
    dll = Path(dll_path).resolve() if dll_path else repository_root() / "bin/pvz_gym_native.dll"
    def dll_check():
        if read_pe(dll).machine != 0x8664:
            raise ValueError("Native DLL must be x64; the game itself is x86")
        from pvz_gym.bridge.native import NativePvzBackend
        caps = NativePvzBackend(dll).get_runtime_capabilities()
        caps.require_mode("turbo")
        return f"{dll}; native interface {caps.interface_version}; modes={sorted(caps.modes)}"
    check("native", dll_check)
    if require_save:
        check("save_template", lambda: [(p.profile_id, p.name) for p in read_save_profiles(paths.save_template)])
    def torch_check():
        import torch
        available = torch.cuda.is_available()
        gpu = torch.cuda.get_device_name(0) if available else "CPU only; full training not recommended"
        return f"{torch.__version__}; CUDA={torch.version.cuda}; available={available}; {gpu}"
    check("torch", torch_check)
    return {"ok": all(c["ok"] for c in checks), "paths": {k: str(v) for k, v in asdict(paths).items()}, "checks": checks}

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--game-exe")
    parser.add_argument("--dll")
    parser.add_argument("--skip-save", action="store_true", help="first-time check before generating the template")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    report = diagnose(args.game_exe, args.dll, require_save=not args.skip_save)
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        for c in report["checks"]:
            print(f"[{'OK' if c['ok'] else 'FAIL'}] {c['name']}: {c['detail']}")
        print("No game was launched. Native runtime repeats version/execution checks at private-instance startup.")
    raise SystemExit(0 if report["ok"] else 1)

if __name__ == "__main__":
    main()
