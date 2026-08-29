from __future__ import annotations

import ctypes
import os
import sys
from ctypes import wintypes


class ForegroundWindowRestorer:
    """Remember and restore the last foreground window outside this process."""

    def __init__(self) -> None:
        self._previous_hwnd: int | None = None
        self._user32 = ctypes.windll.user32 if sys.platform == "win32" else None
        if self._user32 is not None:
            self._configure_api()

    @property
    def previous_hwnd(self) -> int | None:
        return self._previous_hwnd

    def remember_foreground(self) -> int | None:
        if self._user32 is None:
            return None
        hwnd = int(self._user32.GetForegroundWindow() or 0)
        if self._is_candidate(hwnd):
            self._previous_hwnd = hwnd
        return self._previous_hwnd

    def restore_previous(self) -> bool:
        if self._user32 is None:
            return False
        hwnd = self._previous_hwnd
        if not self._is_candidate(hwnd):
            hwnd = self._top_visible_window()
        if hwnd is None:
            return False
        return bool(self._user32.SetForegroundWindow(hwnd))

    def _configure_api(self) -> None:
        assert self._user32 is not None
        self._user32.GetForegroundWindow.restype = wintypes.HWND
        self._user32.IsWindow.argtypes = (wintypes.HWND,)
        self._user32.IsWindowVisible.argtypes = (wintypes.HWND,)
        self._user32.IsIconic.argtypes = (wintypes.HWND,)
        self._user32.SetForegroundWindow.argtypes = (wintypes.HWND,)
        self._user32.GetWindowThreadProcessId.argtypes = (
            wintypes.HWND,
            ctypes.POINTER(wintypes.DWORD),
        )
        self._user32.GetClassNameW.argtypes = (
            wintypes.HWND,
            wintypes.LPWSTR,
            ctypes.c_int,
        )

    def _is_candidate(self, hwnd: int | None) -> bool:
        if self._user32 is None or not hwnd:
            return False
        if not self._user32.IsWindow(hwnd):
            return False
        if not self._user32.IsWindowVisible(hwnd):
            return False
        if self._user32.IsIconic(hwnd):
            return False

        process_id = wintypes.DWORD()
        self._user32.GetWindowThreadProcessId(
            wintypes.HWND(hwnd),
            ctypes.byref(process_id),
        )
        if int(process_id.value) == os.getpid():
            return False

        class_name = ctypes.create_unicode_buffer(256)
        self._user32.GetClassNameW(hwnd, class_name, len(class_name))
        return class_name.value not in {
            "Progman",
            "WorkerW",
            "Shell_TrayWnd",
            "Shell_SecondaryTrayWnd",
        }

    def _top_visible_window(self) -> int | None:
        if self._user32 is None:
            return None
        windows: list[int] = []
        callback_type = ctypes.WINFUNCTYPE(
            wintypes.BOOL,
            wintypes.HWND,
            wintypes.LPARAM,
        )

        @callback_type
        def collect(hwnd: int, _lparam: int) -> bool:
            candidate = int(hwnd or 0)
            if self._is_candidate(candidate):
                windows.append(candidate)
                return False
            return True

        self._user32.EnumWindows(collect, 0)
        return windows[0] if windows else None
