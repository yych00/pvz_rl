"""Exercise actual exports and all wire backends while bound to a nonexistent PID."""
from __future__ import annotations
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "python")]

from pvz_gym.bridge.native import NativePvzBackend
from pvz_gym.bridge.wire import decode_payload
from pvz_gym.runtime import RuntimeConfig, TurboConfig


def main():
    if len(sys.argv) != 2:
        raise SystemExit("usage: verify_runtime_framework_abi.py path/to/pvz_gym_native.dll")
    backend = NativePvzBackend(sys.argv[1])
    caps = backend.get_runtime_capabilities()
    assert caps.modes == frozenset({"stepped", "headless", "continuous", "turbo"})
    assert {"attach_pid", "resident_executor", "binary_transport", "shared_memory_transport",
            "shared_snapshot", "phase_advance", "result_requests"} <= caps.features
    dll = backend._dll
    decoded = []
    for transport in ("json", "binary", "shared_memory"):
        backend.configure_runtime(RuntimeConfig(process_id=0xffffffff, executor="resident",
                                               transport=transport, snapshot="shared"))
        assert dll.pvz_gym_begin_turbo_mode(TurboConfig().native_flags) == 3
        assert dll.pvz_gym_end_turbo_mode() == 0
        ret, payload = backend._call_transition(
            lambda buf, size: dll.pvz_gym_env_step_turbo(1, 0, 0, 0, 0, 0, 0, 150, buf, size), initial_size=2)
        assert ret == len(payload)
        result = decode_payload(payload)
        assert not result["success"] and not result["action"]["success"] and result["game_clock"] == 0
        decoded.append(result)
    assert decoded[0] == decoded[1] == decoded[2]
    assert dll.pvz_gym_close_runtime() == 0
    print("Runtime ABI passed: PID binding, capabilities, three wire backends, cached result resize, cleanup")

if __name__ == "__main__": main()
