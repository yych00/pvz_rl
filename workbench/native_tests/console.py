# -*- coding: utf-8 -*-
"""Console entrypoint for the native workbench test suite."""

from __future__ import annotations

import sys

from .c_interface import PvzGymNativeDll
from .cases import NativeUnitTestSuite


def main() -> int:
    client = PvzGymNativeDll()
    suite = NativeUnitTestSuite(client)
    tests = suite.get_all_tests()

    print("=========================================================")
    print(f"  PvZ Gym 单元测试执行 (共 {len(tests)} 项)")
    print("=========================================================")

    passed = failed = skipped = 0
    for idx, test in enumerate(tests, 1):
        result = suite.run_single(test)
        if result.status == "PASS":
            passed += 1
        elif result.status == "FAIL":
            failed += 1
        else:
            skipped += 1
        print(f"[{idx:02d}/{len(tests):02d}] [{result.status}] {result.case_id} {result.name}")
        print(f"       -> {result.message} ({result.elapsed_ms:.2f}ms)")

    print("=========================================================")
    print(f"测试完成: 通过={passed} | 失败={failed} | 跳过={skipped}")
    print("=========================================================")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
