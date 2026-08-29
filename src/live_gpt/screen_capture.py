from __future__ import annotations

import ctypes
import sys
from ctypes import wintypes
from dataclasses import dataclass

from PySide6.QtCore import QBuffer, QIODevice, Qt
from PySide6.QtGui import QImage, QImageWriter

from .logger import Logger


logger = Logger.get_logger(__name__)

# Adapted from ok-script's BitBlt capture method. Window capture always uses
# the equivalent of bitblt.render_full = True.
PW_RENDERFULLCONTENT = 0x00000002
SRCCOPY = 0x00CC0020
DIB_RGB_COLORS = 0
BI_RGB = 0
DWMWA_CLOAKED = 14
MONITOR_DEFAULTTONEAREST = 2
render_full = True


@dataclass(frozen=True)
class CaptureSource:
    key: str
    label: str
    kind: str
    left: int
    top: int
    width: int
    height: int
    hwnd: int = 0


class _BitmapInfoHeader(ctypes.Structure):
    _fields_ = [
        ("biSize", wintypes.DWORD),
        ("biWidth", wintypes.LONG),
        ("biHeight", wintypes.LONG),
        ("biPlanes", wintypes.WORD),
        ("biBitCount", wintypes.WORD),
        ("biCompression", wintypes.DWORD),
        ("biSizeImage", wintypes.DWORD),
        ("biXPelsPerMeter", wintypes.LONG),
        ("biYPelsPerMeter", wintypes.LONG),
        ("biClrUsed", wintypes.DWORD),
        ("biClrImportant", wintypes.DWORD),
    ]


class _BitmapInfo(ctypes.Structure):
    _fields_ = [
        ("bmiHeader", _BitmapInfoHeader),
        ("bmiColors", wintypes.DWORD * 3),
    ]


class _MonitorInfo(ctypes.Structure):
    _fields_ = [
        ("cbSize", wintypes.DWORD),
        ("rcMonitor", wintypes.RECT),
        ("rcWork", wintypes.RECT),
        ("dwFlags", wintypes.DWORD),
    ]


if sys.platform == "win32":
    user32 = ctypes.WinDLL("user32", use_last_error=True)
    gdi32 = ctypes.WinDLL("gdi32", use_last_error=True)
    dwmapi = ctypes.WinDLL("dwmapi", use_last_error=True)

    try:
        user32.SetProcessDpiAwarenessContext(ctypes.c_void_p(-4))
    except Exception:
        pass

    user32.EnumWindows.argtypes = [ctypes.c_void_p, wintypes.LPARAM]
    user32.EnumWindows.restype = wintypes.BOOL
    user32.EnumDisplayMonitors.argtypes = [
        wintypes.HDC,
        ctypes.POINTER(wintypes.RECT),
        ctypes.c_void_p,
        wintypes.LPARAM,
    ]
    user32.EnumDisplayMonitors.restype = wintypes.BOOL
    user32.GetWindowRect.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.RECT)]
    user32.GetWindowRect.restype = wintypes.BOOL
    user32.IsWindowVisible.argtypes = [wintypes.HWND]
    user32.IsWindowVisible.restype = wintypes.BOOL
    user32.IsIconic.argtypes = [wintypes.HWND]
    user32.IsIconic.restype = wintypes.BOOL
    user32.GetWindowTextLengthW.argtypes = [wintypes.HWND]
    user32.GetWindowTextLengthW.restype = ctypes.c_int
    user32.GetWindowTextW.argtypes = [
        wintypes.HWND,
        wintypes.LPWSTR,
        ctypes.c_int,
    ]
    user32.GetWindowTextW.restype = ctypes.c_int
    user32.GetClassNameW.argtypes = [
        wintypes.HWND,
        wintypes.LPWSTR,
        ctypes.c_int,
    ]
    user32.GetClassNameW.restype = ctypes.c_int
    user32.GetMonitorInfoW.argtypes = [wintypes.HMONITOR, ctypes.POINTER(_MonitorInfo)]
    user32.GetMonitorInfoW.restype = wintypes.BOOL
    user32.MonitorFromWindow.argtypes = [wintypes.HWND, wintypes.DWORD]
    user32.MonitorFromWindow.restype = wintypes.HMONITOR
    user32.GetWindowDC.argtypes = [wintypes.HWND]
    user32.GetWindowDC.restype = wintypes.HDC
    user32.GetDC.argtypes = [wintypes.HWND]
    user32.GetDC.restype = wintypes.HDC
    user32.ReleaseDC.argtypes = [wintypes.HWND, wintypes.HDC]
    user32.ReleaseDC.restype = ctypes.c_int
    user32.PrintWindow.argtypes = [wintypes.HWND, wintypes.HDC, wintypes.UINT]
    user32.PrintWindow.restype = wintypes.BOOL

    gdi32.CreateCompatibleDC.argtypes = [wintypes.HDC]
    gdi32.CreateCompatibleDC.restype = wintypes.HDC
    gdi32.CreateCompatibleBitmap.argtypes = [wintypes.HDC, ctypes.c_int, ctypes.c_int]
    gdi32.CreateCompatibleBitmap.restype = wintypes.HBITMAP
    gdi32.SelectObject.argtypes = [wintypes.HDC, wintypes.HGDIOBJ]
    gdi32.SelectObject.restype = wintypes.HGDIOBJ
    gdi32.BitBlt.argtypes = [
        wintypes.HDC,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        wintypes.HDC,
        ctypes.c_int,
        ctypes.c_int,
        wintypes.DWORD,
    ]
    gdi32.BitBlt.restype = wintypes.BOOL
    gdi32.GetDIBits.argtypes = [
        wintypes.HDC,
        wintypes.HBITMAP,
        wintypes.UINT,
        wintypes.UINT,
        ctypes.c_void_p,
        ctypes.POINTER(_BitmapInfo),
        wintypes.UINT,
    ]
    gdi32.GetDIBits.restype = ctypes.c_int
    gdi32.DeleteObject.argtypes = [wintypes.HGDIOBJ]
    gdi32.DeleteObject.restype = wintypes.BOOL
    gdi32.DeleteDC.argtypes = [wintypes.HDC]
    gdi32.DeleteDC.restype = wintypes.BOOL
    dwmapi.DwmGetWindowAttribute.argtypes = [
        wintypes.HWND,
        wintypes.DWORD,
        ctypes.c_void_p,
        wintypes.DWORD,
    ]
    dwmapi.DwmGetWindowAttribute.restype = ctypes.c_long


def list_capture_sources(
    excluded_hwnds: set[int] | None = None,
) -> list[CaptureSource]:
    if sys.platform != "win32":
        return []

    excluded = excluded_hwnds or set()
    displays = _enumerate_displays()
    display_sources = [
        CaptureSource(
            key=f"display:{handle}",
            label=(
                "Screenshot desktop"
                if len(displays) == 1
                else f"Screenshot desktop {index} ({width}×{height})"
            ),
            kind="display",
            left=left,
            top=top,
            width=width,
            height=height,
        )
        for index, (handle, left, top, width, height) in enumerate(
            displays,
            start=1,
        )
    ]

    windows = _enumerate_visible_windows(excluded)
    windows.sort(key=lambda source: source.width * source.height, reverse=True)
    return display_sources + windows


def capture_webp(source: CaptureSource) -> bytes:
    if sys.platform != "win32":
        raise RuntimeError("Screen capture is only available on Windows")
    if source.kind == "display":
        image = _capture_bitmap(
            hwnd=0,
            left=source.left,
            top=source.top,
            width=source.width,
            height=source.height,
            render_full=False,
        )
    elif source.kind == "window":
        rect = wintypes.RECT()
        if not user32.GetWindowRect(source.hwnd, ctypes.byref(rect)):
            raise RuntimeError("The selected window is no longer available")
        image = _capture_bitmap(
            hwnd=source.hwnd,
            left=0,
            top=0,
            width=rect.right - rect.left,
            height=rect.bottom - rect.top,
            render_full=render_full,
        )
    else:
        raise ValueError(f"Unsupported capture source: {source.kind}")

    return _encode_lossless_webp(_resize_for_upload(image))


def _resize_for_upload(image: QImage, longest_edge: int = 1920) -> QImage:
    if max(image.width(), image.height()) <= longest_edge:
        return image
    return image.scaled(
        longest_edge,
        longest_edge,
        Qt.AspectRatioMode.KeepAspectRatio,
        Qt.TransformationMode.SmoothTransformation,
    )


def _encode_lossless_webp(image: QImage) -> bytes:
    output = QBuffer()
    if not output.open(QIODevice.OpenModeFlag.WriteOnly):
        raise RuntimeError("Could not allocate screenshot buffer")
    writer = QImageWriter(output, b"WEBP")
    writer.setQuality(100)
    writer.setOptimizedWrite(True)
    if not writer.write(image):
        raise RuntimeError(
            f"Could not encode lossless WebP: {writer.errorString()}"
        )
    return bytes(output.data())


def _enumerate_displays() -> list[tuple[int, int, int, int, int]]:
    displays: list[tuple[int, int, int, int, int]] = []
    callback_type = ctypes.WINFUNCTYPE(
        wintypes.BOOL,
        wintypes.HMONITOR,
        wintypes.HDC,
        ctypes.POINTER(wintypes.RECT),
        wintypes.LPARAM,
    )

    def callback(monitor, _dc, rect_pointer, _data):
        rect = rect_pointer.contents
        displays.append(
            (
                int(monitor),
                rect.left,
                rect.top,
                rect.right - rect.left,
                rect.bottom - rect.top,
            )
        )
        return True

    callback_reference = callback_type(callback)
    if not user32.EnumDisplayMonitors(0, None, callback_reference, 0):
        raise ctypes.WinError(ctypes.get_last_error())
    displays.sort(key=lambda display: (display[2], display[1]))
    return displays


def _enumerate_visible_windows(excluded: set[int]) -> list[CaptureSource]:
    windows: list[CaptureSource] = []
    callback_errors: list[Exception] = []
    callback_type = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)

    def callback(hwnd, _data):
        try:
            handle = int(hwnd)
            if handle in excluded or not user32.IsWindowVisible(hwnd):
                return True
            if user32.IsIconic(hwnd) or _is_cloaked(hwnd):
                return True
            class_buffer = ctypes.create_unicode_buffer(256)
            user32.GetClassNameW(hwnd, class_buffer, len(class_buffer))
            if class_buffer.value in {
                "Progman",
                "WorkerW",
                "Shell_TrayWnd",
                "Shell_SecondaryTrayWnd",
            }:
                return True

            title_length = user32.GetWindowTextLengthW(hwnd)
            if title_length <= 0:
                return True
            title_buffer = ctypes.create_unicode_buffer(title_length + 1)
            user32.GetWindowTextW(hwnd, title_buffer, len(title_buffer))
            title = title_buffer.value.strip()
            if not title:
                return True

            rect = wintypes.RECT()
            if not user32.GetWindowRect(hwnd, ctypes.byref(rect)):
                return True
            width = rect.right - rect.left
            height = rect.bottom - rect.top
            if width <= 0 or height <= 0:
                return True

            monitor = user32.MonitorFromWindow(hwnd, MONITOR_DEFAULTTONEAREST)
            monitor_info = _MonitorInfo(cbSize=ctypes.sizeof(_MonitorInfo))
            if not monitor or not user32.GetMonitorInfoW(
                monitor,
                ctypes.byref(monitor_info),
            ):
                return True
            monitor_rect = monitor_info.rcMonitor
            screen_width = monitor_rect.right - monitor_rect.left
            screen_height = monitor_rect.bottom - monitor_rect.top
            if not _is_valid_window_size(
                width,
                height,
                screen_width,
                screen_height,
            ):
                return True

            windows.append(
                CaptureSource(
                    key=f"window:{handle}",
                    label=f"{title} ({width}×{height})",
                    kind="window",
                    left=rect.left,
                    top=rect.top,
                    width=width,
                    height=height,
                    hwnd=handle,
                )
            )
            return True
        except Exception as error:
            callback_errors.append(error)
            return False

    callback_reference = callback_type(callback)
    if not user32.EnumWindows(callback_reference, 0):
        if callback_errors:
            raise callback_errors[0]
        error_code = ctypes.get_last_error()
        if error_code:
            raise ctypes.WinError(error_code)
    return windows


def _is_cloaked(hwnd: int) -> bool:
    cloaked = wintypes.DWORD()
    result = dwmapi.DwmGetWindowAttribute(
        wintypes.HWND(hwnd),
        DWMWA_CLOAKED,
        ctypes.byref(cloaked),
        ctypes.sizeof(cloaked),
    )
    return result == 0 and bool(cloaked.value)


def _is_valid_window_size(
    width: int,
    height: int,
    screen_width: int,
    screen_height: int,
) -> bool:
    return width * height > screen_width * screen_height / 8


def _capture_bitmap(
    hwnd: int,
    left: int,
    top: int,
    width: int,
    height: int,
    render_full: bool,
) -> QImage:
    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid capture size: {width}×{height}")

    source_dc = user32.GetWindowDC(hwnd) if hwnd else user32.GetDC(0)
    if not source_dc:
        raise ctypes.WinError(ctypes.get_last_error())
    memory_dc = gdi32.CreateCompatibleDC(source_dc)
    bitmap = gdi32.CreateCompatibleBitmap(source_dc, width, height)
    previous_bitmap = None
    try:
        if not memory_dc or not bitmap:
            raise ctypes.WinError(ctypes.get_last_error())
        previous_bitmap = gdi32.SelectObject(memory_dc, bitmap)

        captured = False
        if render_full:
            captured = bool(
                user32.PrintWindow(hwnd, memory_dc, PW_RENDERFULLCONTENT)
            )
        if not captured:
            captured = bool(
                gdi32.BitBlt(
                    memory_dc,
                    0,
                    0,
                    width,
                    height,
                    source_dc,
                    left,
                    top,
                    SRCCOPY,
                )
            )
        if not captured:
            raise ctypes.WinError(ctypes.get_last_error())

        bitmap_info = _BitmapInfo()
        bitmap_info.bmiHeader = _BitmapInfoHeader(
            biSize=ctypes.sizeof(_BitmapInfoHeader),
            biWidth=width,
            biHeight=-height,
            biPlanes=1,
            biBitCount=32,
            biCompression=BI_RGB,
            biSizeImage=width * height * 4,
        )
        pixels = ctypes.create_string_buffer(width * height * 4)
        scan_lines = gdi32.GetDIBits(
            memory_dc,
            bitmap,
            0,
            height,
            pixels,
            ctypes.byref(bitmap_info),
            DIB_RGB_COLORS,
        )
        if scan_lines != height:
            raise ctypes.WinError(ctypes.get_last_error())
        return QImage(
            pixels.raw,
            width,
            height,
            width * 4,
            QImage.Format.Format_RGB32,
        ).copy()
    finally:
        if previous_bitmap and memory_dc:
            gdi32.SelectObject(memory_dc, previous_bitmap)
        if bitmap:
            gdi32.DeleteObject(bitmap)
        if memory_dc:
            gdi32.DeleteDC(memory_dc)
        user32.ReleaseDC(hwnd, source_dc)


__all__ = ["CaptureSource", "capture_webp", "list_capture_sources"]
