import json
import unittest
from unittest.mock import patch

from pvz_gym.bridge.native import _parse_step_json, _parse_reset_result
from pvz_gym.bridge.native import NativeCapabilityError


class NativeReadFailureGuardTest(unittest.TestCase):
    def test_failed_native_facts_are_rejected_before_observation_parsing(self):
        for parse in (_parse_step_json, _parse_reset_result):
            for code, message in ((1099, "required native memory read failed at 0x1234"),
                                  (1019, "native injection failed (Win32 error 1460)")):
                with self.subTest(parse=parse.__name__, code=code), patch(
                    "pvz_gym.bridge.native._parse_observation_data"
                ) as read_observation:
                    with self.assertRaisesRegex(NativeCapabilityError, message.replace("(", r"\(").replace(")", r"\)")):
                        parse(json.dumps({"success": False, "error": code,
                                          "message": message, "observation": {}}))
                    read_observation.assert_not_called()
