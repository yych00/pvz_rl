# -*- coding: utf-8 -*-
"""Manual native observation latency benchmark."""

from __future__ import annotations

import ctypes
import sys
import time

import numpy as np

from .c_interface import PvzGymNativeDll


def run_benchmark(num_iterations: int = 100):
    print("=========================================================")
    print(f"  PvZ Gym Native Observation 读取性能基准测试 (×{num_iterations})")
    print("=========================================================")

    client = PvzGymNativeDll()
    if not client.is_loaded:
        print(f"[ERROR] DLL 加载失败: {client.error_msg}")
        return 1

    attached, msg = client.attach()
    print(f"[*] 进程附加: {attached} ({msg})")
    if not attached:
        print("[ERROR] 无法附加游戏进程！")
        return 1

    ui, ui_desc = client.get_ui()
    print(f"[*] 当前 UI 状态: {ui} ({ui_desc})")
    if ui != 3:
        print("[ERROR] 游戏未处于战斗就绪状态 (UI != 3)")
        return 1

    client.pause()
    print("[*] 游戏已暂停，开始预热 5 次...")
    buf_size = 65536
    buf = ctypes.create_string_buffer(buf_size)
    for _ in range(5):
        n = client.dll.pvz_gym_get_observation_json(buf, buf_size)
        if n <= 0:
            print(f"[ERROR] 预热读取失败: {n}")
            return 1

    latencies = []
    total_bytes = 0
    for i in range(num_iterations):
        t0 = time.perf_counter()
        n = client.dll.pvz_gym_get_observation_json(buf, buf_size)
        t1 = time.perf_counter()
        if n <= 0:
            print(f"[ERROR] 第 {i+1} 次读取失败 (code: {n})")
            return 1
        latencies.append((t1 - t0) * 1000.0)
        total_bytes += n

    lat_arr = np.array(latencies)
    mean_ms = float(np.mean(lat_arr))
    print("---------------------------------------------------------")
    print(f"  样本总数 (Runs)   : {num_iterations}")
    print(f"  平均单次大小      : {total_bytes / num_iterations:.1f} bytes")
    print(f"  平均耗时 (Mean)   : {mean_ms:.4f} ms")
    print(f"  中位数   (Median) : {float(np.median(lat_arr)):.4f} ms")
    print(f"  95分位   (P95)    : {float(np.percentile(lat_arr, 95)):.4f} ms")
    print(f"  最低耗时 (Min)    : {float(np.min(lat_arr)):.4f} ms")
    print(f"  最高耗时 (Max)    : {float(np.max(lat_arr)):.4f} ms")
    print("---------------------------------------------------------")
    print("[PASS] 平均延迟低于 2.0 ms" if mean_ms < 2.0 else "[WARN] 平均延迟偏高")
    return 0


if __name__ == "__main__":
    sys.exit(run_benchmark(100))
