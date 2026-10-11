# -*- coding: utf-8 -*-
"""Manual native simulation-clock probe."""

from .c_interface import PvzGymNativeDll


def main():
    print("=== PvzGym Native Clock Test ===")
    client = PvzGymNativeDll()
    if not client.is_loaded:
        print(f"[FAIL] DLL 加载失败: {client.error_msg}")
        return

    print(f"[OK] DLL 已加载: {client.dll_path}")
    attached, msg = client.attach()
    print(f"[*] 进程附加状态: {attached} ({msg})")
    on, on_msg = client.is_game_on()
    print(f"[*] 游戏在线状态: {on} ({on_msg})")
    ui, ui_desc = client.get_ui()
    print(f"[*] 当前 UI 状态: {ui} ({ui_desc})")
    print(f"[*] 当前逻辑时钟: {client.get_game_clock()}")
    print(f"[*] 是否暂停: {client.is_paused()}")

    symbols = client.get_exported_symbols()
    print("[*] 关键导出符号检查:")
    for symbol in [
        "pvz_gym_pause",
        "pvz_gym_resume",
        "pvz_gym_is_paused",
        "pvz_gym_advance_ticks",
        "pvz_gym_get_game_clock",
    ]:
        print(f"    - {symbol}: {'[OK]' if symbols.get(symbol) else '[MISSING]'}")
    print("=== 测试结束 ===")


if __name__ == "__main__":
    main()
