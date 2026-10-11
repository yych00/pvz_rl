# -*- coding: utf-8 -*-
"""PvZ Gym Q值热力图与NOOP显示 Overlay 模块。

核心功能：
1. Win32 原生透明穿透 Overlay 窗口（基于 Owned Window 机制，随游戏窗口一起置顶/置底，绝不遮挡其他软件）。
2. 全场景精确坐标网格映射（适配白天、黑夜、泳池、浓雾、屋顶前5列斜坡）。
3. 草地落点 Top-5 Q值热力图彩色边框，黑色大字分行显示卡牌名称与 Q 值。
4. 游戏窗口正中间底部纯文本打印 NOOP Q 值（无卡片背景框）。
"""

from __future__ import annotations

import ctypes
from ctypes import wintypes
import logging
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("pvz_gym.q_overlay")

# -----------------------------------------------------------------------------
# Win32 API 常量与结构体定义
# -----------------------------------------------------------------------------
user32 = ctypes.WinDLL("user32", use_last_error=True)
gdi32 = ctypes.WinDLL("gdi32", use_last_error=True)
kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)

LRESULT = ctypes.c_ssize_t
WPARAM = wintypes.WPARAM
LPARAM = wintypes.LPARAM
HWND = wintypes.HWND
UINT = wintypes.UINT

# ctypes otherwise assumes c_int arguments/returns, truncating 64-bit handles
# and passing Python strings incorrectly to Unicode Win32 entry points.
def _bind(dll, name, result, *arguments):
    function = getattr(dll, name)
    function.restype = result
    function.argtypes = list(arguments)


_bind(kernel32, "GetModuleHandleW", wintypes.HMODULE, wintypes.LPCWSTR)
_bind(user32, "FindWindowW", HWND, wintypes.LPCWSTR, wintypes.LPCWSTR)
_bind(user32, "RegisterClassExW", wintypes.ATOM, ctypes.c_void_p)
_bind(user32, "UnregisterClassW", wintypes.BOOL, wintypes.LPCWSTR, wintypes.HINSTANCE)
_bind(user32, "CreateWindowExW", HWND, wintypes.DWORD, wintypes.LPCWSTR,
      wintypes.LPCWSTR, wintypes.DWORD, ctypes.c_int, ctypes.c_int,
      ctypes.c_int, ctypes.c_int, HWND, wintypes.HMENU, wintypes.HINSTANCE,
      ctypes.c_void_p)
for _name in ("IsWindow", "IsIconic", "IsWindowVisible", "DestroyWindow", "UpdateWindow"):
    _bind(user32, _name, wintypes.BOOL, HWND)
_bind(user32, "GetForegroundWindow", HWND)
_bind(user32, "GetWindow", HWND, HWND, UINT)
_bind(user32, "GetWindowLongPtrW", ctypes.c_ssize_t, HWND, ctypes.c_int)
_bind(user32, "ShowWindow", wintypes.BOOL, HWND, ctypes.c_int)
_bind(user32, "SetLayeredWindowAttributes", wintypes.BOOL, HWND,
      wintypes.COLORREF, wintypes.BYTE, wintypes.DWORD)
_bind(user32, "GetClientRect", wintypes.BOOL, HWND, ctypes.POINTER(wintypes.RECT))
_bind(user32, "ClientToScreen", wintypes.BOOL, HWND, ctypes.POINTER(wintypes.POINT))
_bind(user32, "GetWindowThreadProcessId", wintypes.DWORD, HWND, ctypes.POINTER(wintypes.DWORD))
WNDENUMPROC = ctypes.WINFUNCTYPE(wintypes.BOOL, HWND, wintypes.LPARAM)
_bind(user32, "EnumWindows", wintypes.BOOL, WNDENUMPROC, wintypes.LPARAM)
_bind(user32, "GetClassNameW", ctypes.c_int, HWND, wintypes.LPWSTR, ctypes.c_int)
_bind(user32, "GetDC", wintypes.HDC, HWND)
_bind(user32, "ReleaseDC", ctypes.c_int, HWND, wintypes.HDC)
_bind(user32, "FillRect", ctypes.c_int, wintypes.HDC, ctypes.POINTER(wintypes.RECT), wintypes.HBRUSH)
_bind(user32, "PeekMessageW", wintypes.BOOL, ctypes.POINTER(wintypes.MSG), HWND, UINT, UINT, UINT)
_bind(user32, "TranslateMessage", wintypes.BOOL, ctypes.POINTER(wintypes.MSG))
_bind(user32, "DispatchMessageW", LRESULT, ctypes.POINTER(wintypes.MSG))
_bind(user32, "OpenDesktopW", wintypes.HANDLE, wintypes.LPCWSTR, wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
_bind(user32, "SetThreadDesktop", wintypes.BOOL, wintypes.HANDLE)
_bind(user32, "CloseDesktop", wintypes.BOOL, wintypes.HANDLE)
_bind(user32, "GetThreadDesktop", wintypes.HANDLE, wintypes.DWORD)
_bind(kernel32, "GetCurrentThreadId", wintypes.DWORD)
_bind(gdi32, "GetStockObject", wintypes.HGDIOBJ, ctypes.c_int)
_bind(gdi32, "CreateSolidBrush", wintypes.HBRUSH, wintypes.COLORREF)
_bind(gdi32, "CreateCompatibleDC", wintypes.HDC, wintypes.HDC)
_bind(gdi32, "CreateCompatibleBitmap", wintypes.HBITMAP, wintypes.HDC, ctypes.c_int, ctypes.c_int)
_bind(gdi32, "SelectObject", wintypes.HGDIOBJ, wintypes.HDC, wintypes.HGDIOBJ)
_bind(gdi32, "DeleteObject", wintypes.BOOL, wintypes.HGDIOBJ)
_bind(gdi32, "DeleteDC", wintypes.BOOL, wintypes.HDC)
_bind(gdi32, "CreatePen", wintypes.HPEN, ctypes.c_int, ctypes.c_int, wintypes.COLORREF)
_bind(gdi32, "CreateFontW", wintypes.HFONT, *([ctypes.c_int] * 5),
      *([wintypes.DWORD] * 8), wintypes.LPCWSTR)
_bind(gdi32, "SetBkMode", ctypes.c_int, wintypes.HDC, ctypes.c_int)
_bind(gdi32, "SetTextColor", wintypes.COLORREF, wintypes.HDC, wintypes.COLORREF)
_bind(gdi32, "Rectangle", wintypes.BOOL, wintypes.HDC, *([ctypes.c_int] * 4))
_bind(gdi32, "BitBlt", wintypes.BOOL, wintypes.HDC, *([ctypes.c_int] * 4),
      wintypes.HDC, ctypes.c_int, ctypes.c_int, wintypes.DWORD)

user32.DefWindowProcW.argtypes = [HWND, UINT, WPARAM, LPARAM]
user32.DefWindowProcW.restype = LRESULT

user32.SetWindowPos.argtypes = [HWND, HWND, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, UINT]
user32.SetWindowPos.restype = wintypes.BOOL

user32.DrawTextW.argtypes = [wintypes.HDC, wintypes.LPCWSTR, ctypes.c_int, ctypes.POINTER(wintypes.RECT), UINT]
user32.DrawTextW.restype = ctypes.c_int

user32.SetWindowLongPtrW.argtypes = [HWND, ctypes.c_int, ctypes.c_ssize_t]
user32.SetWindowLongPtrW.restype = ctypes.c_ssize_t

gdi32.TextOutW.argtypes = [wintypes.HDC, ctypes.c_int, ctypes.c_int, wintypes.LPCWSTR, ctypes.c_int]
gdi32.TextOutW.restype = wintypes.BOOL

try:
    msimg32 = ctypes.windll.msimg32
except Exception:
    msimg32 = None

WS_POPUP = 0x80000000
WS_VISIBLE = 0x10000000
WS_EX_LAYERED = 0x00080000
WS_EX_TRANSPARENT = 0x00000020
WS_EX_TOOLWINDOW = 0x00000080
WS_EX_NOACTIVATE = 0x08000000
WS_EX_TOPMOST = 0x00000008

LWA_COLORKEY = 0x00000001
SW_HIDE = 0
SW_SHOWNOACTIVATE = 4

SWP_NOSIZE = 0x0001
SWP_NOMOVE = 0x0002
SWP_NOZORDER = 0x0004
SWP_NOACTIVATE = 0x0010
SWP_SHOWWINDOW = 0x0040
SWP_NOOWNERZORDER = 0x0200

HWND_TOP = 0
HWND_TOPMOST = -1
HWND_NOTOPMOST = -2
GW_HWNDPREV = 3
GWL_EXSTYLE = -20

GWLP_HWNDPARENT = -8

DT_CENTER = 0x00000001
DT_VCENTER = 0x00000004
DT_SINGLELINE = 0x00000020
DT_NOCLIP = 0x00000100
DT_WORDBREAK = 0x00000010
DT_CALCRECT = 0x00000400

SRCCOPY = 0x00CC0020
TRANSPARENT = 1
OPAQUE = 2

# AlphaBlend 结构体
class BLENDFUNCTION(ctypes.Structure):
    _fields_ = [
        ("BlendOp", ctypes.c_byte),
        ("BlendFlags", ctypes.c_byte),
        ("SourceConstantAlpha", ctypes.c_byte),
        ("AlphaFormat", ctypes.c_byte),
    ]

# 统一热力图色阶 (Top 1 ~ Top 5)
HEATMAP_COLORS = [
    (255, 50, 50),    # Rank 1: 亮红色 (最优)
    (255, 140, 0),   # Rank 2: 金橙色
    (255, 215, 0),   # Rank 3: 明黄色
    (100, 220, 50),  # Rank 4: 黄绿色
    (0, 190, 255),   # Rank 5: 冰蓝色
]


def rgb(r: int, g: int, b: int) -> int:
    return (b << 16) | (g << 8) | r


# Black is visible text, so it must never be the transparent color key.
TRANSPARENT_COLOR = rgb(255, 0, 255)
TEXT_COLOR = rgb(0, 0, 0)
ACTION_FONT_HEIGHT = 20
NOOP_FONT_HEIGHT = 22


def _action_label(item):
    return f"#{item['rank']} {item.get('desc', '')}\nQ:{item.get('q_value', 0.0):+.2f}"


def _selected_label(action_id, top5):
    from pvz_gym.action_space import decode_action
    action = decode_action(int(action_id))
    if action.type == "noop":
        return "不操作"
    if action.type == "plant":
        name = next((item.get("desc") for item in top5 if item.get("action_id") == action_id), None)
        name = name or f"种第 {action.card_slot + 1} 张卡"
        row, col = action.row, action.col
    elif action.type == "shovel":
        name = "铲南瓜" if action.shovel_target == "pumpkin" else "铲植物"
        row, col = action.row, action.col
    elif action.type == "cob_fire":
        name = "玉米炮攻击"
        row, col = action.target_row, action.target_col
    else:
        name, row, col = "点击格子", action.row, action.col
    return f"{name} · 第 {row + 1} 行 {col + 1} 列"


def grid_to_pixel(scene_id: int, row: int, col: int) -> Tuple[int, int, int, int]:
    """计算 800x600 基准分辨率下的网格像素坐标 (x, y, w, h)。

    参数定义严格对齐 tools_perception 与 tools_vision_generator 原生渲染器公式：
    - 列宽固定 80 px，起始 x = 40
    - 白天/黑夜 (Scene 0, 1): 5行，起始 y = 80，格高 100 px
    - 泳池/浓雾 (Scene 2, 3): 6行，起始 y = 80，格高 85 px
    - 屋顶/夜间屋顶 (Scene 4, 5): 5行，起始 y = 70，格高 85 px，前5列每列抬高 (5 - col) * 20 px
    """
    x = 40 + col * 80
    w = 80
    if scene_id in (4, 5):  # 屋顶场景
        h = 85
        slope = max(0, 5 - col) * 20
        y = 70 + row * 85 + slope
    elif scene_id in (2, 3):  # 泳池/浓雾场景
        h = 85
        y = 80 + row * 85
    else:  # 白天/黑夜普通草地场景
        h = 100
        y = 80 + row * 100
    return x, y, w, h


def ensure_thread_desktop():
    """确保当前线程附加到交互式 Default 桌面，以便正常检索和显示原生窗口。"""
    try:
        h_desk = user32.OpenDesktopW("Default", 0, False, 0x01FF)
        if h_desk:
            if user32.SetThreadDesktop(h_desk):
                return h_desk
            user32.CloseDesktop(h_desk)
    except Exception:
        pass


# -------------------------------------------------------------
# Q-Value Heatmap Overlay 控制器
# -------------------------------------------------------------
class QHeatmapOverlay:
    """Win32 穿透透明置顶与置底同步 Overlay 渲染器。"""

    def __init__(self):
        self._lock = threading.Lock()
        self._running = False
        self._enabled = False
        self._visible = False
        self._thread: Optional[threading.Thread] = None

        self._hwnd: Optional[int] = None
        self._target_hwnd: Optional[int] = None
        self._bound_pid: Optional[int] = None
        self._bound_hwnd: Optional[int] = None
        self._binding_revision = 0
        self._topmost_requested = False
        self._promoted_window = None
        self._frame_key = None
        self._selected_action = None
        self._attach_candidate = None
        self._next_attach_attempt = 0.0
        self._last_attach_error = None

        # 待渲染数据快照
        self._top5_actions: List[Dict[str, Any]] = []
        self._noop_q: Optional[float] = None
        self._scene_id: int = 0
        self._has_new_data = False

    def is_enabled(self) -> bool:
        """检查热力图是否已启用。"""
        with self._lock:
            return self._enabled

    def set_enabled(self, enabled: bool):
        """开启或禁用热力图显示。"""
        with self._lock:
            self._enabled = bool(enabled)
            self._visible = bool(enabled)
            # A display toggle must retain the latest snapshot: the trainer may
            # be paused or between decisions when the user turns it back on.
            # Episode completion, mode changes and owner replacement use clear().
            self._has_new_data = True
        if enabled and self._bound_pid:
            self.start()
        elif not enabled and self._thread is not None and self._thread is not threading.current_thread():
            self.stop()

    def bind_target(self, process_id: Optional[int], window_handle: Optional[int] = None, *, topmost=False):
        """Only the explicitly bound process may own the overlay."""
        pid = int(process_id) if process_id else None
        hwnd = int(window_handle) if window_handle else None
        if hwnd and (not pid or self._window_pid(hwnd) != pid):
            raise ValueError("Q overlay HWND does not belong to the bound process")
        with self._lock:
            changed = (pid, hwnd) != (self._bound_pid, self._bound_hwnd)
            if changed:
                self._bound_pid, self._bound_hwnd = pid, hwnd
                self._binding_revision += 1
                self._top5_actions = []
                self._noop_q = None
                self._selected_action = None
                self._frame_key = None
                self._has_new_data = True
            self._topmost_requested = bool(topmost)
            enabled = self._enabled
        if pid and enabled:
            self.start()

    @staticmethod
    def _window_pid(hwnd) -> int:
        pid = wintypes.DWORD()
        if hwnd and user32.IsWindow(hwnd):
            user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
        return int(pid.value)

    def update_frame(self, frame):
        """Reject stale frames, including old episodes and replaced workers."""
        key = (int(frame.get("generation", 0)), int(frame.get("worker_generation", 0)),
               int(frame.get("episode_id", 0)), int(frame.get("sequence_id", -1)))
        with self._lock:
            if not self._enabled or int(frame.get("process_id") or 0) != self._bound_pid:
                return False
            if self._frame_key is not None and key < self._frame_key:
                return False
            self._frame_key = key
            self._top5_actions = [dict(action) for action in frame.get("top5", ())]
            self._noop_q = float(frame["noop_q"]) if frame.get("noop_q") is not None else None
            self._scene_id = int(frame.get("scene_id", 0))
            self._selected_action = frame.get("selected_action")
            self._visible = True
            self._has_new_data = True
            return True

    def start(self):
        """启动后台渲染线程。"""
        with self._lock:
            if self._thread and self._thread.is_alive():
                return
            self._running = True
            self._thread = threading.Thread(target=self._overlay_worker, daemon=True)
            self._thread.start()

    def stop(self):
        """停止并销毁 Overlay 窗口。"""
        with self._lock:
            self._running = False
            self._visible = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=3.0)
            if not self._thread.is_alive():
                self._thread = None
            else:
                raise RuntimeError("Q overlay renderer did not stop")

    def set_visible(self, visible: bool):
        """开关 Overlay 可视状态。"""
        self.set_enabled(visible)

    def clear(self):
        """清空当前热力图与 Q 值显示。"""
        with self._lock:
            self._top5_actions = []
            self._noop_q = None
            self._selected_action = None
            self._has_new_data = True

    def update_data(
        self,
        top5_actions: List[Dict[str, Any]],
        noop_q: Optional[float] = None,
        scene_id: int = 0,
    ):
        """更新最新的 Top-5 落点与 NOOP Q 值。线程安全。"""
        with self._lock:
            if not self._enabled:
                return
            self._top5_actions = list(top5_actions)
            self._noop_q = float(noop_q) if noop_q is not None else None
            self._scene_id = int(scene_id)
            self._has_new_data = True
            self._visible = True

    def _find_game_hwnd(self) -> int:
        """Resolve one visible MainWindow in the configured process; no fallback."""
        with self._lock:
            pid, explicit = self._bound_pid, self._bound_hwnd
        if not pid:
            return 0
        if explicit:
            return explicit if self._window_pid(explicit) == pid and user32.IsWindowVisible(explicit) else 0
        found = []

        def enum_window(hwnd, _):
            if self._window_pid(hwnd) == pid and user32.IsWindowVisible(hwnd):
                class_name = ctypes.create_unicode_buffer(256)
                if user32.GetClassNameW(hwnd, class_name, len(class_name)) and class_name.value == "MainWindow":
                    found.append(int(hwnd))
            return True

        user32.EnumWindows(WNDENUMPROC(enum_window), 0)
        return found[0] if len(found) == 1 else 0

    def _create_window(self, owner_hwnd: int) -> int:
        """注册并创建透明穿透的 Owned Window。"""
        if owner_hwnd and not user32.IsWindow(owner_hwnd):
            return 0
        wnd_class_name = f"PvZ_Q_Overlay_{id(self)}_{time.time_ns()}"
        h_inst = kernel32.GetModuleHandleW(None)

        WNDPROC = ctypes.WINFUNCTYPE(LRESULT, HWND, UINT, WPARAM, LPARAM)

        def py_wnd_proc(hwnd, msg, wparam, lparam):
            return user32.DefWindowProcW(hwnd, msg, wparam, lparam)

        self._wnd_proc_cb = WNDPROC(py_wnd_proc)

        class WNDCLASSEXW(ctypes.Structure):
            _fields_ = [
                ("cbSize", ctypes.c_uint),
                ("style", ctypes.c_uint),
                ("lpfnWndProc", WNDPROC),
                ("cbClsExtra", ctypes.c_int),
                ("cbWndExtra", ctypes.c_int),
                ("hInstance", wintypes.HINSTANCE),
                ("hIcon", wintypes.HICON),
                ("hCursor", wintypes.HANDLE),
                ("hbrBackground", wintypes.HBRUSH),
                ("lpszMenuName", wintypes.LPCWSTR),
                ("lpszClassName", wintypes.LPCWSTR),
                ("hIconSm", wintypes.HICON),
            ]

        wcex = WNDCLASSEXW()
        wcex.cbSize = ctypes.sizeof(WNDCLASSEXW)
        wcex.style = 0
        wcex.lpfnWndProc = self._wnd_proc_cb
        wcex.cbClsExtra = 0
        wcex.cbWndExtra = 0
        wcex.hInstance = h_inst
        wcex.hIcon = None
        wcex.hCursor = None
        self._window_background_brush = gdi32.CreateSolidBrush(TRANSPARENT_COLOR)
        wcex.hbrBackground = self._window_background_brush
        wcex.lpszMenuName = None
        wcex.lpszClassName = wnd_class_name
        wcex.hIconSm = None

        if not user32.RegisterClassExW(ctypes.byref(wcex)):
            self._log_attach_error("register overlay window class", owner_hwnd)
            gdi32.DeleteObject(self._window_background_brush)
            self._window_background_brush = None
            return 0
        self._window_class = (wnd_class_name, h_inst)

        ex_style = WS_EX_LAYERED | WS_EX_TRANSPARENT | WS_EX_TOOLWINDOW | WS_EX_NOACTIVATE
        # The game may already be topmost before the overlay is created.
        # Start in its current band; subsequent owner toggles move both windows.
        if owner_hwnd and user32.GetWindowLongPtrW(owner_hwnd, -20) & WS_EX_TOPMOST:
            ex_style |= WS_EX_TOPMOST
        style = WS_POPUP

        hwnd = user32.CreateWindowExW(
            ex_style,
            wnd_class_name,
            "PvZ_Q_Heatmap_Overlay",
            style,
            0, 0, 800, 600,
            owner_hwnd or 0,
            0,
            h_inst,
            0,
        )

        if not hwnd:
            error = ctypes.get_last_error()
            # The game can close between detection and window creation.
            if not owner_hwnd or user32.IsWindow(owner_hwnd):
                self._log_attach_error("create overlay window", owner_hwnd, error)
            user32.UnregisterClassW(wnd_class_name, h_inst)
            self._window_class = None
            gdi32.DeleteObject(self._window_background_brush)
            self._window_background_brush = None
            return 0

        user32.SetLayeredWindowAttributes(hwnd, TRANSPARENT_COLOR, 255, LWA_COLORKEY)
        self._last_attach_error = None
        return hwnd

    def _log_attach_error(self, operation, owner_hwnd, error=None):
        if error is None:
            error = ctypes.get_last_error()
        failure = (operation, owner_hwnd, error)
        if failure != self._last_attach_error:
            logger.error("Failed to %s: %s; will retry attaching", operation, ctypes.WinError(error))
            self._last_attach_error = failure

    def _destroy_window(self, hwnd):
        """释放当前窗口和窗口类，允许游戏重启后重新创建所属窗口。"""
        if hwnd and user32.IsWindow(hwnd):
            user32.DestroyWindow(hwnd)
        window_class = getattr(self, "_window_class", None)
        if window_class:
            user32.UnregisterClassW(*window_class)
            self._window_class = None
        background_brush = getattr(self, "_window_background_brush", None)
        if background_brush:
            gdi32.DeleteObject(background_brush)
            self._window_background_brush = None

    def _ensure_game_window(self, overlay_hwnd, game_hwnd):
        # Windows can destroy an owned overlay when the game window is destroyed.
        # Recreate with the new owner instead of drawing to the stale HWND.
        if not game_hwnd or not user32.IsWindow(game_hwnd):
            self._destroy_window(overlay_hwnd)
            if self._target_hwnd:
                self.clear()
            self._hwnd = self._target_hwnd = None
            self._attach_candidate = None
            self._next_attach_attempt = 0.0
            self._last_attach_error = None
            return 0
        if game_hwnd == self._target_hwnd and overlay_hwnd and user32.IsWindow(overlay_hwnd):
            return overlay_hwnd

        self._destroy_window(overlay_hwnd)
        self._hwnd = None
        if self._target_hwnd and game_hwnd != self._target_hwnd:
            self.clear()
        self._target_hwnd = None
        now = time.monotonic()
        if game_hwnd == self._attach_candidate and now < self._next_attach_attempt:
            return 0
        self._attach_candidate = game_hwnd
        self._next_attach_attempt = now + 1.0
        overlay_hwnd = self._create_window(game_hwnd)
        if overlay_hwnd:
            self._hwnd = overlay_hwnd
            self._target_hwnd = game_hwnd
            self._next_attach_attempt = 0.0
        return overlay_hwnd or 0

    def _sync_game_z_order(self, overlay_hwnd, game_hwnd):
        """跟随游戏实际置顶状态，并放在游戏紧邻上方，不提升游戏层级。"""
        flags = SWP_NOMOVE | SWP_NOSIZE | SWP_NOACTIVATE | SWP_NOOWNERZORDER
        game_topmost = bool(user32.GetWindowLongPtrW(game_hwnd, GWL_EXSTYLE) & WS_EX_TOPMOST)
        overlay_topmost = bool(user32.GetWindowLongPtrW(overlay_hwnd, GWL_EXSTYLE) & WS_EX_TOPMOST)
        if game_topmost != overlay_topmost:
            band = HWND_TOPMOST if game_topmost else HWND_NOTOPMOST
            if not user32.SetWindowPos(overlay_hwnd, band, 0, 0, 0, 0, flags):
                logger.warning("Q overlay topmost sync failed: %s", ctypes.WinError(ctypes.get_last_error()))
                return

        previous = user32.GetWindow(game_hwnd, GW_HWNDPREV)
        if previous == overlay_hwnd:
            return
        # Inserting after a topmost window would promote an ordinary overlay.
        # At the band boundary use HWND_TOP, which retains the current band.
        if not previous or (not game_topmost and
                user32.GetWindowLongPtrW(previous, GWL_EXSTYLE) & WS_EX_TOPMOST):
            previous = HWND_TOP
        if not user32.SetWindowPos(overlay_hwnd, previous, 0, 0, 0, 0, flags):
            logger.warning("Q overlay z-order sync failed: %s", ctypes.WinError(ctypes.get_last_error()))

    def _apply_topmost(self, game_hwnd):
        with self._lock:
            requested = self._topmost_requested
        if self._promoted_window and (not requested or self._promoted_window[0] != game_hwnd):
            previous, was_topmost = self._promoted_window
            if user32.IsWindow(previous):
                user32.SetWindowPos(previous, HWND_TOPMOST if was_topmost else HWND_NOTOPMOST,
                                     0, 0, 0, 0, SWP_NOMOVE | SWP_NOSIZE | SWP_NOACTIVATE)
            self._promoted_window = None
        if requested and game_hwnd and self._promoted_window is None:
            was_topmost = bool(user32.GetWindowLongPtrW(game_hwnd, GWL_EXSTYLE) & WS_EX_TOPMOST)
            if user32.SetWindowPos(game_hwnd, HWND_TOPMOST, 0, 0, 0, 0,
                                   SWP_NOMOVE | SWP_NOSIZE | SWP_NOACTIVATE):
                self._promoted_window = (game_hwnd, was_topmost)

    def _overlay_worker(self):
        """后台独立渲染主循环。"""
        # 必须显式将后台工作线程附加到交互式 Default 桌面，否则 FindWindowW 在线程中可能返回 0
        previous_desktop = user32.GetThreadDesktop(kernel32.GetCurrentThreadId())
        h_desk = ensure_thread_desktop()

        game_hwnd = overlay_hwnd = 0
        binding_revision = -1
        next_game_search = 0.0

        # 创建通用字体与画刷
        h_font_bold = gdi32.CreateFontW(
            ACTION_FONT_HEIGHT, 0, 0, 0, 700, 0, 0, 0,
            1, 0, 0, 3, 0, "Microsoft YaHei"
        )
        h_font_noop = gdi32.CreateFontW(
            NOOP_FONT_HEIGHT, 0, 0, 0, 700, 0, 0, 0,
            1, 0, 0, 3, 0, "Microsoft YaHei"
        )
        transparent_brush = gdi32.CreateSolidBrush(TRANSPARENT_COLOR)
        last_x = last_y = last_w = last_h = -9999

        try:
            while self._running:
                # 处理 Windows 消息队列
                msg = wintypes.MSG()
                while user32.PeekMessageW(ctypes.byref(msg), 0, 0, 0, 1):
                    user32.TranslateMessage(ctypes.byref(msg))
                    user32.DispatchMessageW(ctypes.byref(msg))

                # 检查游戏窗口状态
                with self._lock:
                    current_binding_revision = self._binding_revision
                if binding_revision != current_binding_revision:
                    binding_revision = current_binding_revision
                    game_hwnd = 0
                    next_game_search = 0.0
                if not game_hwnd or not user32.IsWindow(game_hwnd) or not user32.IsWindowVisible(game_hwnd):
                    game_hwnd = 0
                    now = time.monotonic()
                    if now >= next_game_search:
                        game_hwnd = self._find_game_hwnd()
                        next_game_search = now + 0.5

                refreshed_hwnd = self._ensure_game_window(overlay_hwnd, game_hwnd)
                if not refreshed_hwnd:
                    overlay_hwnd = 0
                    last_x = last_y = last_w = last_h = -9999
                    time.sleep(0.05)
                    continue
                if refreshed_hwnd != overlay_hwnd:
                    last_x = last_y = last_w = last_h = -9999
                overlay_hwnd = refreshed_hwnd

                game_valid = (
                    game_hwnd
                    and user32.IsWindow(game_hwnd)
                    and not user32.IsIconic(game_hwnd)
                    and user32.IsWindowVisible(game_hwnd)
                )

                with self._lock:
                    visible = self._visible
                    top5 = list(self._top5_actions)
                    noop_q = self._noop_q
                    scene_id = self._scene_id
                    selected_action = self._selected_action

                # 若不可见或游戏未激活/最小化，清空画布并隐藏
                if not visible or not game_valid or (not top5 and noop_q is None):
                    if user32.IsWindowVisible(overlay_hwnd):
                        hdc = user32.GetDC(overlay_hwnd)
                        rc = wintypes.RECT()
                        user32.GetClientRect(overlay_hwnd, ctypes.byref(rc))
                        user32.FillRect(hdc, ctypes.byref(rc), transparent_brush)
                        user32.ReleaseDC(overlay_hwnd, hdc)
                        user32.ShowWindow(overlay_hwnd, SW_HIDE)
                    time.sleep(0.05)
                    continue

                if not user32.IsWindowVisible(overlay_hwnd):
                    user32.ShowWindow(overlay_hwnd, SW_SHOWNOACTIVATE)

                # 获取游戏客户区实际尺寸与屏幕坐标
                client_rc = wintypes.RECT()
                user32.GetClientRect(game_hwnd, ctypes.byref(client_rc))
                cw = client_rc.right - client_rc.left
                ch = client_rc.bottom - client_rc.top
                if cw <= 0 or ch <= 0:
                    time.sleep(0.03)
                    continue

                pt = wintypes.POINT(0, 0)
                user32.ClientToScreen(game_hwnd, ctypes.byref(pt))

                # 保持 Overlay 与游戏客户区对齐
                if pt.x != last_x or pt.y != last_y or cw != last_w or ch != last_h:
                    user32.SetWindowPos(
                        overlay_hwnd,
                        0,
                        pt.x, pt.y, cw, ch,
                        SWP_NOACTIVATE | SWP_SHOWWINDOW | SWP_NOZORDER
                    )
                    last_x, last_y, last_w, last_h = pt.x, pt.y, cw, ch
                self._sync_game_z_order(overlay_hwnd, game_hwnd)
                self._apply_topmost(game_hwnd)

                # 双缓冲极速 GDI 绘制
                hdc = user32.GetDC(overlay_hwnd)
                mem_dc = gdi32.CreateCompatibleDC(hdc)
                mem_bmp = gdi32.CreateCompatibleBitmap(hdc, cw, ch)
                old_bmp = gdi32.SelectObject(mem_dc, mem_bmp)

                # 透明背景使用独立色键，保留纯黑文字。
                full_rc = wintypes.RECT(0, 0, cw, ch)
                user32.FillRect(mem_dc, ctypes.byref(full_rc), transparent_brush)

                gdi32.SetBkMode(mem_dc, TRANSPARENT)

                # 比例缩放参数 (针对拉伸自适应)
                scale_x = float(cw) / 800.0
                scale_y = float(ch) / 600.0

                # -------------------------------------------------------------
                # 1. 草地网格热力图 (聚合同一落点多动作)
                # -------------------------------------------------------------
                grouped_actions: Dict[Tuple[int, int], List[Dict[str, Any]]] = {}
                for rank_idx, act in enumerate(top5):
                    r = act.get("row", -1)
                    c = act.get("col", -1)
                    if r >= 0 and c >= 0:
                        act_entry = dict(act)
                        act_entry["rank"] = rank_idx + 1
                        if selected_action is not None and act.get("action_id") == selected_action:
                            act_entry["desc"] = f"{act_entry.get('desc', '')}（实际）"
                        grouped_actions.setdefault((r, c), []).append(act_entry)

                old_font = gdi32.SelectObject(mem_dc, h_font_bold)

                for (r, c), act_list in grouped_actions.items():
                    gx, gy, gw, gh = grid_to_pixel(scene_id, r, c)
                    draw_x = int((gx + 2) * scale_x)
                    draw_y = int((gy + 2) * scale_y)
                    draw_w = int((gw - 4) * scale_x)
                    draw_h = int((gh - 4) * scale_y)

                    # 取该落点中最优排名的热力颜色
                    best_rank = min(item["rank"] for item in act_list)
                    color_tuple = HEATMAP_COLORS[min(best_rank - 1, len(HEATMAP_COLORS) - 1)]
                    color_ref = rgb(*color_tuple)

                    # 绘制 2px 彩色外边框 (中间完全透明透出草坪，不添加任何填充)
                    pen_border = gdi32.CreatePen(0, 2, color_ref)
                    old_pen = gdi32.SelectObject(mem_dc, pen_border)
                    hollow_brush = gdi32.GetStockObject(5)  # HOLLOW_BRUSH
                    old_brush = gdi32.SelectObject(mem_dc, hollow_brush)

                    gdi32.Rectangle(mem_dc, draw_x, draw_y, draw_x + draw_w, draw_y + draw_h)

                    gdi32.SelectObject(mem_dc, old_brush)
                    gdi32.SelectObject(mem_dc, old_pen)
                    gdi32.DeleteObject(pen_border)

                    # 动作名称与 Q 值分行；长中文名称按实际字宽换行。
                    labels = []
                    text_width = max(1, draw_w - 8)
                    for item in act_list:
                        label = _action_label(item)
                        measured = wintypes.RECT(0, 0, text_width, 0)
                        user32.DrawTextW(mem_dc, label, -1, ctypes.byref(measured),
                                         DT_WORDBREAK | DT_CALCRECT)
                        labels.append((label, measured.bottom + 4))
                    total_height = sum(height for _, height in labels)
                    text_y = max(0, min(draw_y + 4, ch - total_height - 4))
                    gdi32.SetTextColor(mem_dc, TEXT_COLOR)
                    for label, height in labels:
                        text_rc = wintypes.RECT(draw_x + 4, text_y,
                                                draw_x + 4 + text_width, text_y + height)
                        user32.DrawTextW(mem_dc, label, -1, ctypes.byref(text_rc), DT_WORDBREAK)
                        text_y += height

                # -------------------------------------------------------------
                # 2. 窗口正中间底部：直接居中打印 NOOP 纯文本 (无卡片背景框)
                # -------------------------------------------------------------
                if noop_q is not None:
                    gdi32.SelectObject(mem_dc, h_font_noop)
                    noop_text = f"NOOP（不操作）\nQ:{noop_q:+.2f}"

                    text_y = max(0, ch - NOOP_FONT_HEIGHT * 2 - 8)
                    rc_noop = wintypes.RECT(0, text_y, cw, ch)
                    gdi32.SetTextColor(mem_dc, TEXT_COLOR)
                    user32.DrawTextW(mem_dc, noop_text, -1, ctypes.byref(rc_noop), DT_CENTER)

                if selected_action is not None:
                    gdi32.SelectObject(mem_dc, h_font_bold)
                    decision_text = f"Q 值前 5 · 本次实际动作：{_selected_label(selected_action, top5)}"
                    decision_rc = wintypes.RECT(0, 0, cw, ACTION_FONT_HEIGHT * 2)
                    user32.DrawTextW(mem_dc, decision_text, -1, ctypes.byref(decision_rc), DT_CENTER)

                # 将内存缓冲区渲染至 Overlay 窗口
                gdi32.BitBlt(hdc, 0, 0, cw, ch, mem_dc, 0, 0, SRCCOPY)

                # 清理本次 GDI 绘制资源
                gdi32.SelectObject(mem_dc, old_font)
                gdi32.SelectObject(mem_dc, old_bmp)
                gdi32.DeleteObject(mem_bmp)
                gdi32.DeleteDC(mem_dc)
                user32.ReleaseDC(overlay_hwnd, hdc)

                time.sleep(0.033)  # ~30 FPS

        finally:
            if self._promoted_window:
                with self._lock:
                    self._topmost_requested = False
                self._apply_topmost(0)
            gdi32.DeleteObject(h_font_bold)
            gdi32.DeleteObject(h_font_noop)
            gdi32.DeleteObject(transparent_brush)
            self._destroy_window(overlay_hwnd)
            if h_desk:
                user32.SetThreadDesktop(previous_desktop)
                user32.CloseDesktop(h_desk)
            self._hwnd = None
            self._target_hwnd = None
            self._running = False


# 全局单例管理器
_GLOBAL_OVERLAY: Optional[QHeatmapOverlay] = None
_GLOBAL_OVERLAY_LOCK = threading.Lock()


def get_q_heatmap_overlay() -> QHeatmapOverlay:
    """获取或初始化全局 Q 值热力图 Overlay 实例。"""
    global _GLOBAL_OVERLAY
    with _GLOBAL_OVERLAY_LOCK:
        if _GLOBAL_OVERLAY is None:
            _GLOBAL_OVERLAY = QHeatmapOverlay()
        return _GLOBAL_OVERLAY
