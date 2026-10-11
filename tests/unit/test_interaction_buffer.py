import ctypes
import unittest
from types import SimpleNamespace
from unittest.mock import patch
from pvz_gym.bridge.native import (NativeCapabilityError, NativePvzBackend,
    _call_transition_with_resizable_buffer)
from pvz_gym.profiling import TimingRecorder


class InteractionBufferTest(unittest.TestCase):
    def test_resize_executes_once_even_if_getter_needs_another_resize(self):
        calls = []
        payload = b"original transition"
        def execute(buf, size):
            calls.append("execute")
            return -8
        def read(buf, size):
            calls.append("read")
            if size <= len(payload):
                return -(len(payload) + 1)
            buf.value = payload
            return len(payload)
        ret, value = _call_transition_with_resizable_buffer(execute, read, initial_size=2)
        self.assertEqual(value, payload)
        self.assertEqual(ret, len(payload))
        self.assertEqual(calls, ["execute", "read", "read"])

    def test_success_and_failure_do_not_fetch(self):
        for ret in (0, -1):
            calls = []
            def execute(buf, size):
                calls.append("execute")
                return ret
            _call_transition_with_resizable_buffer(execute, lambda *a: self.fail("unexpected fetch"), initial_size=2)
            self.assertEqual(calls, ["execute"])

    def test_old_dll_fails_before_mutation(self):
        client = object.__new__(NativePvzBackend)
        client._dll = SimpleNamespace()
        with self.assertRaises(NativeCapabilityError):
            client._call_transition(lambda *a: self.fail("mutation"), initial_size=2)

    def test_request_id_is_used_for_fetch(self):
        client = object.__new__(NativePvzBackend)
        ids = []
        def read(request, buf, size):
            ids.append(request)
            buf.value = b"done"
            return 4
        client._dll = SimpleNamespace(pvz_gym_begin_result_request=lambda: 123,
                                      pvz_gym_read_result_request=read)
        self.assertEqual(client._call_transition(lambda *a: -5, initial_size=2), (4, b"done"))
        self.assertEqual(ids, [123])

    def test_disabled_profiling_has_no_clock_reads(self):
        recorder = TimingRecorder(False)
        with patch("pvz_gym.profiling.perf_counter", side_effect=AssertionError):
            self.assertEqual(recorder.measure("test", lambda: 7), 7)
        self.assertEqual(recorder.snapshot(), {})

    def test_profile_counts_and_bounded_percentiles(self):
        recorder = TimingRecorder(True)
        for i in range(5000):
            recorder.add("test", i)
        item = recorder.snapshot()["test"]
        self.assertEqual(item["count"], 5000)
        self.assertEqual(item["percentile_window"], 4096)
        self.assertEqual(item["mean_ms"], 2499.5)


if __name__ == "__main__":
    unittest.main()
