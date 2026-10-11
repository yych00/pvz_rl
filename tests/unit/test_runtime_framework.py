import json
import unittest
from dataclasses import asdict
from types import SimpleNamespace
from unittest.mock import Mock, patch

from pvz_gym.bridge.native import NativeCapabilityError, NativePvzBackend
from pvz_gym.client import PvZClient
from pvz_gym.env import PvZEnv, normalize_simulation_mode
from pvz_gym.runtime import (
    MODE_SPECS, RuntimeCapabilities, RuntimeCapabilityError, RuntimeConfig, TurboConfig,
)
from tests.unit.test_env_headless import HeadlessFactClient
from train.common import EnvConfig
from pvz_gym.types import Action


def capabilities_dll(*, modes=("stepped", "headless", "continuous"), features=()):
    payload = json.dumps({"interface_version": 1, "modes": modes, "features": features}).encode()

    def query(buffer, capacity):
        if capacity <= len(payload):
            return -(len(payload) + 1)
        buffer.value = payload
        return len(payload)

    return SimpleNamespace(pvz_gym_get_runtime_capabilities_json=query)


class RuntimeFrameworkTest(unittest.TestCase):
    def test_turbo_is_registered_and_old_backend_rejected_before_mutation(self):
        self.assertEqual(normalize_simulation_mode("TURBO"), "turbo")
        self.assertTrue(next(s for s in MODE_SPECS if s.key == "turbo").implemented)
        self.assertEqual(EnvConfig(simulation_mode="turbo").runtime, RuntimeConfig())
        client = HeadlessFactClient()
        with self.assertRaises(RuntimeCapabilityError):
            PvZEnv(client, simulation_mode="turbo")
        self.assertEqual(client.headless_calls, [])

    def test_failed_switch_preserves_live_headless_owner(self):
        client = HeadlessFactClient()
        env = PvZEnv(client, simulation_mode="headless")
        try:
            env.reset()
            calls = list(client.headless_calls)
            with self.assertRaises(RuntimeCapabilityError):
                env.set_simulation_mode("turbo")
            self.assertEqual(env.simulation_mode, "headless")
            self.assertTrue(client.headless_active)
            self.assertEqual(client.headless_calls, calls)
            env.step(0)
        finally:
            env.close()

    def test_explicit_default_runtime_preserves_legacy_backend(self):
        client = HeadlessFactClient()
        env = PvZEnv(client, simulation_mode="headless", runtime_config=RuntimeConfig())
        try:
            env.reset()
            self.assertTrue(client.headless_active)
            env.step(0)
        finally:
            env.close()

    def test_advertised_features_require_configuration_before_mutation(self):
        client = HeadlessFactClient()
        client.get_runtime_capabilities = lambda: RuntimeCapabilities(
            modes=frozenset({"headless"}), features=frozenset({"resident_executor"}))
        with self.assertRaisesRegex(RuntimeCapabilityError, "cannot configure"):
            PvZEnv(client, simulation_mode="headless", runtime_config=RuntimeConfig(executor="resident"))
        self.assertEqual(client.headless_calls, [])

    def test_session_backend_receives_explicit_runtime_configuration(self):
        from pvz_gym.session import PvZSession
        client = HeadlessFactClient()
        client.get_runtime_capabilities = lambda: RuntimeCapabilities(
            modes=frozenset({"headless"}), features=frozenset({"attach_pid"}))
        client.configure_runtime = Mock()
        session = PvZSession(client)
        config = RuntimeConfig(process_id=42)
        try:
            env = PvZEnv(session=session, simulation_mode="headless", runtime_config=config)
            client.configure_runtime.assert_called_once_with(config)
            env.reset()
            env.close()
        finally:
            session.close()

    def test_unavailable_runtime_rejected_before_backend_configuration(self):
        for config in (RuntimeConfig(process_id=123), RuntimeConfig(executor="resident"),
                       RuntimeConfig(transport="shared_memory"), RuntimeConfig(snapshot="shared")):
            with self.subTest(config=config):
                client = HeadlessFactClient()
                with self.assertRaises(RuntimeCapabilityError):
                    PvZEnv(client, runtime_config=config)
                self.assertEqual(client.headless_calls, [])

    def test_json_restores_distinct_validated_runtime_and_policy_configs(self):
        config = EnvConfig(simulation_mode="turbo", runtime=RuntimeConfig(transport="binary"),
                           turbo=TurboConfig(mute_audio=False))
        restored = EnvConfig(**json.loads(json.dumps(asdict(config))))
        self.assertEqual(restored, config)
        self.assertIsInstance(restored.runtime, RuntimeConfig)
        self.assertIsInstance(restored.turbo, TurboConfig)
        self.assertEqual(EnvConfig().runtime, RuntimeConfig())
        with self.assertRaises(ValueError):
            EnvConfig(runtime={"process_id": True})
        with self.assertRaises(ValueError):
            EnvConfig(turbo={"suppress_render": "false"})

    def test_old_dll_probe_requires_complete_exports_and_never_guesses_turbo(self):
        client = object.__new__(NativePvzBackend)
        client._dll = SimpleNamespace(
            pvz_gym_env_step=lambda: None,
            pvz_gym_begin_headless_mode=lambda: None,
            pvz_gym_env_step_headless=lambda: None,
            pvz_gym_env_step_turbo=lambda: None,
        )
        caps = client.get_runtime_capabilities()
        self.assertEqual(caps.modes, frozenset({"stepped"}))
        self.assertEqual(caps.features, frozenset())
        client.configure_runtime(RuntimeConfig())
        with self.assertRaises(RuntimeCapabilityError):
            client.configure_runtime(RuntimeConfig(process_id=123))

    def test_export_presence_is_not_turbo_support(self):
        client = object.__new__(PvZClient)
        client._dll = capabilities_dll()
        client._dll.pvz_gym_begin_turbo_mode = Mock()
        client._dll.pvz_gym_env_step_turbo = Mock()
        with self.assertRaises(RuntimeCapabilityError):
            client.begin_turbo_mode()
        client._dll.pvz_gym_begin_turbo_mode.assert_not_called()
        self.assertTrue(client.end_turbo_mode())

    def test_capability_resize_is_read_only_and_unknown_schema_fails(self):
        client = object.__new__(NativePvzBackend)
        client._dll = capabilities_dll(features=("long_feature_" + "x" * 3000,))
        self.assertEqual(client.get_runtime_capabilities().modes, frozenset({"stepped", "headless", "continuous"}))
        for payload in ({"interface_version": 2, "modes": [], "features": []},
                        {"interface_version": 1, "modes": [], "features": "turbo"}):
            with self.assertRaises(RuntimeCapabilityError):
                RuntimeCapabilities.from_mapping(payload)

    def test_advertised_runtime_features_use_stable_native_option_codes(self):
        client = object.__new__(NativePvzBackend)
        client._dll = capabilities_dll(features=("attach_pid", "resident_executor", "shared_memory_transport", "shared_snapshot"))
        configure = client._dll.pvz_gym_configure_runtime = Mock(return_value=0)
        client.configure_runtime(RuntimeConfig(process_id=42, executor="resident", transport="shared_memory", snapshot="shared"))
        configure.assert_called_once_with(42, 1, 2, 1)

    def test_native_unsupported_begin_does_not_mark_policy_active(self):
        client = object.__new__(PvZClient)
        client._dll = capabilities_dll(modes=("turbo",))
        client._dll.pvz_gym_begin_turbo_mode = Mock(return_value=1)
        with self.assertRaisesRegex(RuntimeCapabilityError, "not implemented"):
            client.begin_turbo_mode(TurboConfig(mute_audio=False))
        client._dll.pvz_gym_begin_turbo_mode.assert_called_once_with(15)
        self.assertFalse(getattr(client, "_turbo_active", False))

    def test_failed_turbo_begin_preserves_native_handoff_diagnostics(self):
        client = object.__new__(PvZClient)
        client._dll = capabilities_dll(modes=("turbo",))
        client._dll.pvz_gym_begin_turbo_mode = Mock(return_value=3)
        client._dll.pvz_gym_get_headless_diagnostics_json = Mock()
        payload = json.dumps({'barrier_failure_count': 1, 'barrier_last_instruction': 123}).encode()
        with patch('pvz_gym.client._call_with_resizable_buffer', return_value=(len(payload), payload)):
            with self.assertRaisesRegex(RuntimeCapabilityError, 'invalid state.*barrier_failure_count'):
                client.begin_turbo_mode()
        self.assertFalse(getattr(client, '_turbo_active', False))

    def test_failed_turbo_begin_keeps_original_error_if_diagnostics_fail(self):
        client = object.__new__(PvZClient)
        client._dll = capabilities_dll(modes=("turbo",))
        client._dll.pvz_gym_begin_turbo_mode = Mock(return_value=3)
        client._dll.pvz_gym_get_headless_diagnostics_json = Mock()
        with patch('pvz_gym.client._call_with_resizable_buffer', side_effect=OSError('read failed')):
            with self.assertRaisesRegex(RuntimeCapabilityError, 'begin_turbo_mode: invalid state'):
                client.begin_turbo_mode()
        self.assertFalse(getattr(client, '_turbo_active', False))

    def test_close_restores_legacy_state_even_if_turbo_exit_fails(self):
        client = object.__new__(PvZClient)
        client._turbo_active = True
        client._dll = SimpleNamespace(
            pvz_gym_end_turbo_mode=lambda: 3,
            pvz_gym_is_paused=lambda: True,
            pvz_gym_resume=Mock(return_value=True),
        )
        with self.assertRaisesRegex(RuntimeCapabilityError, "invalid state"):
            client.close()
        client._dll.pvz_gym_resume.assert_called_once()

    def test_failed_turbo_step_preserves_native_exception_without_retrying_action(self):
        client = object.__new__(PvZClient)
        client._turbo_active = True
        client._dll = capabilities_dll(modes=("turbo",))
        calls = []

        def step(*args):
            calls.append(args)
            return -1

        error = {"operation": "step_turbo", "kind": "native_operation", "code": 1024,
                 "message": "invalid mower array", "executor": {"process_id": 42, "ack": 3}}
        payload = json.dumps(error).encode()

        def diagnostics(buffer, capacity):
            if capacity <= len(payload):
                return -(len(payload) + 1)
            buffer.value = payload
            return len(payload)

        client._dll.pvz_gym_env_step_turbo = step
        client._dll.pvz_gym_get_runtime_error_json = diagnostics
        client._dll.pvz_gym_begin_result_request = lambda: 1
        read_result = client._dll.pvz_gym_read_result_request = Mock()
        with self.assertRaisesRegex(RuntimeCapabilityError, "failed: -1.*invalid mower array.*process_id"):
            client.step_turbo(Action(), 150)
        self.assertEqual(len(calls), 1)
        read_result.assert_not_called()

    def test_turbo_failure_diagnostics_resize_and_ignore_stale_operation(self):
        client = object.__new__(PvZClient)
        payload = json.dumps({"operation": "step_turbo", "message": "x" * 5000}).encode()
        capacities = []

        def diagnostics(buffer, capacity):
            capacities.append(capacity)
            if capacity <= len(payload):
                return -(len(payload) + 1)
            buffer.value = payload
            return len(payload)

        client._dll = SimpleNamespace(pvz_gym_get_runtime_error_json=diagnostics)
        self.assertEqual(client._turbo_failure_details("step_turbo")["runtime_error"]["message"], "x" * 5000)
        self.assertEqual(capacities, [4096, len(payload) + 1])
        self.assertEqual(client._turbo_failure_details("begin_turbo_mode"), {})

    def test_failed_turbo_step_keeps_original_error_with_old_or_broken_diagnostics(self):
        for failure in (None, b"[]", b"invalid JSON", OSError("read failed"),
                        NativeCapabilityError("invalid resize")):
            with self.subTest(failure=failure):
                client = object.__new__(PvZClient)
                client._turbo_active = True
                client._dll = capabilities_dll(modes=("turbo",))
                client._dll.pvz_gym_env_step_turbo = Mock()
                if failure is not None:
                    client._dll.pvz_gym_get_runtime_error_json = Mock()
                client._call_transition = Mock(return_value=(-1, b""))
                kwargs = ({"side_effect": failure} if isinstance(failure, Exception) else
                          {"return_value": (len(failure or b""), failure or b"")})
                with patch("pvz_gym.client._call_with_resizable_buffer", **kwargs):
                    with self.assertRaisesRegex(RuntimeCapabilityError, "^pvz_gym_env_step_turbo failed: -1$"):
                        client.step_turbo(Action(), 150)
                client._call_transition.assert_called_once()


if __name__ == "__main__":
    unittest.main()
