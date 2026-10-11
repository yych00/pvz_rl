from __future__ import annotations
import json
import mmap
import struct
import unittest
import uuid
from dataclasses import replace
from pvz_gym.bridge.wire import decode_payload
from pvz_gym.env import PvZEnv
from pvz_gym.runtime import RuntimeCapabilities, RuntimeConfig
from tests.unit.test_env_headless import HeadlessFactClient
from tests.fake_backend import FactClient
from pvz_gym.types import GameStatus


def binary(value):
    if value is None: return b"\0"
    if isinstance(value, bool): return bytes([2 if value else 1])
    if isinstance(value, int): return b"\3"+struct.pack("<q", value)
    if isinstance(value, float): return b"\4"+struct.pack("<d", value)
    if isinstance(value, str):
        data = value.encode(); return b"\5"+struct.pack("<I", len(data))+data
    if isinstance(value, list): return b"\6"+struct.pack("<I",len(value))+b"".join(binary(v) for v in value)
    return b"\7"+struct.pack("<I",len(value))+b"".join(binary(k)+binary(v) for k,v in value.items())


class TurboFactClient(HeadlessFactClient):
    def __init__(self): super().__init__(); self.turbo_calls = []; self.active = False
    def get_runtime_capabilities(self):
        return RuntimeCapabilities(modes=frozenset({"stepped","headless","continuous","turbo"}),
            features=frozenset({"attach_pid","resident_executor","binary_transport","shared_snapshot"}))
    def configure_runtime(self, options): self.options = options
    def begin_turbo_mode(self, options=None): self.active = True; self.turbo_calls.append("begin"); return True
    def end_turbo_mode(self):
        if self.active: self.turbo_calls.append("end")
        self.active = False; return True
    def reset(self, options):
        if options.get("simulation_mode") == "turbo": assert self.active
        self.turbo_calls.append("reset")
        return super().reset(options)
    def step_turbo(self, action, ticks):
        assert self.active
        self.turbo_calls.append(("step", ticks))
        return FactClient.step(self, action, ticks)


class TurboExecutionTest(unittest.TestCase):
    def test_owner_covers_reset_terminal_and_next_episode_until_exit(self):
        client = TurboFactClient()
        env = PvZEnv(client, simulation_mode="turbo", ticks_per_step=30)
        env.reset(seed=42)
        self.assertEqual(client.turbo_calls[:2], ["begin","reset"])
        _, _, ended, _, info = env.step(0)
        self.assertFalse(ended); self.assertEqual(info["ticks_advanced"], 30)
        _, _, ended, _, info = env.step(0)
        self.assertTrue(ended); self.assertTrue(client.active)
        env.reset(seed=43)
        self.assertEqual(client.turbo_calls.count("begin"), 1)
        env.close(); env.close()
        self.assertEqual(client.turbo_calls.count("end"), 1)

    def test_switch_to_headless_releases_turbo_before_acquiring_next_owner(self):
        client = TurboFactClient(); env = PvZEnv(client, simulation_mode="turbo")
        env.reset(); env.set_simulation_mode("headless")
        self.assertFalse(client.active); self.assertTrue(client.headless_active)
        env.step(0); env.close()

    def test_binary_preserves_nuls_nested_facts_and_numbers(self):
        value = {"message":"中文\0ok", "facts":[None,True,False,-5,1.5,{"mask":[True,False]}]}
        data = b"PVZB\1"+binary(value)
        self.assertEqual(decode_payload(data), value)
        self.assertEqual(decode_payload(json.dumps(value).encode()), value)
        for broken in (data[:-1], data+b"0", b"PVZB\1\xff", b"PVZB\1\6\xff\xff\xff\xff"):
            with self.assertRaises(ValueError): decode_payload(broken)

    @unittest.skipUnless(__import__('sys').platform == 'win32', "Windows shared memory")
    def test_shared_results_own_bytes_and_reject_stale_revision(self):
        data = b"PVZB\1"+binary({"n":7})
        name = "Local\\pvz-gym-test-"+uuid.uuid4().hex
        with mmap.mmap(-1, 65536, tagname=name) as shared:
            shared[:12] = struct.pack("<QI", 4, len(data)); shared[12:12+len(data)] = data
            descriptor = {"name":name,"size":65536,"length":len(data),"revision":4}
            self.assertEqual(decode_payload(b"PVZM"+json.dumps(descriptor).encode()), {"n":7})
            descriptor["revision"] = 3
            with self.assertRaisesRegex(ValueError, "changed"): decode_payload(b"PVZM"+json.dumps(descriptor).encode())
