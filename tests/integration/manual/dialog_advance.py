import ctypes
import time
from ctypes import wintypes


def main():
    user32 = ctypes.WinDLL("user32", use_last_error=True)
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)

    # 切换到 Default 桌面
    hDesk = user32.OpenDesktopW("Default", 0, False, 0x01FF)
    if hDesk:
        user32.SetThreadDesktop(hDesk)

    hwnd = user32.FindWindowW("MainWindow", "Plants vs. Zombies")
    print("Found hwnd:", hex(hwnd) if hwnd else None)
    if not hwnd:
        if hDesk:
            user32.CloseDesktop(hDesk)
        return

    pid = wintypes.DWORD()
    user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
    hProc = kernel32.OpenProcess(0x1F0FFF, False, pid.value)
    if not hProc:
        if hDesk:
            user32.CloseDesktop(hDesk)
        return

    def r32(addr):
        v = ctypes.c_uint32()
        kernel32.ReadProcessMemory(hProc, ctypes.c_void_p(addr), ctypes.byref(v), 4, None)
        return v.value

    try:
        lawn = r32(0x006A9EC0)
        board = r32(lawn + 0x768)
        ui = r32(lawn + 0x7FC)
        print(f"LawnApp: {hex(lawn)}, Board: {hex(board)}, UI: {ui}")

        # 检查当前处于戴夫对话的特征
        # 看看 board + 0x5568 (时钟)
        clock = r32(board + 0x5568)
        print(f"Game Clock: {clock}")

        # 看看 CutScene
        cut_scene = r32(board + 0x15C)
        print(f"CutScene: {hex(cut_scene)}")
        if cut_scene:
            print(f"  CutScene+0x08 (time): {r32(cut_scene + 0x08)}")
            print(f"  CutScene+0x10 (dave): {r32(cut_scene + 0x10)}")
            print(f"  CutScene+0x14: {r32(cut_scene + 0x14)}")
            print(f"  CutScene+0x18: {r32(cut_scene + 0x18)}")
            print(f"  CutScene+0x20: {r32(cut_scene + 0x20)}")

        # 尝试推进！
        # 激活窗口
        user32.ShowWindow(hwnd, 5)
        user32.SetForegroundWindow(hwnd)
        time.sleep(0.1)

        print("\n--- Sending Clicks ---")
        for i in range(15):
            # 发送鼠标左键点击 (屏幕中心偏下，气泡位置)
            lp = (300 << 16) | 400
            user32.SendMessageW(hwnd, 0x0201, 1, lp)  # WM_LBUTTONDOWN
            time.sleep(0.02)
            user32.SendMessageW(hwnd, 0x0202, 0, lp)  # WM_LBUTTONUP
            time.sleep(0.05)
            # 也发送空格
            user32.SendMessageW(hwnd, 0x0100, 0x20, 0)
            time.sleep(0.02)
            user32.SendMessageW(hwnd, 0x0101, 0x20, 0)
            time.sleep(0.1)

            cur_ui = r32(lawn + 0x7FC)
            cur_cut = r32(board + 0x15C)
            cut_t = r32(cur_cut + 0x08) if cur_cut else 0
            print(f"Step {i+1}: UI={cur_ui}, CutScene={hex(cur_cut)}, cut_t={cut_t}")
            if cur_ui == 3:
                print("Successfully reached Battle (UI=3)!")
                break
    finally:
        if hDesk:
            user32.CloseDesktop(hDesk)
        if hProc:
            kernel32.CloseHandle(hProc)


if __name__ == "__main__":
    main()
