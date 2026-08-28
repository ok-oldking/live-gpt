from __future__ import annotations

import os
import re
import subprocess
import sys
import time
from pathlib import Path

from .browser_discovery import (
    active_port_endpoints,
    browser_user_data_directories,
)
from .logger import Logger


logger = Logger.get_logger(__name__)


def open_remote_debugging_settings() -> str:
    """Open and enable the default browser's remote-debugging settings."""
    executable = windows_default_browser_executable()
    if executable is None:
        raise RuntimeError("Could not find the default browser executable")
    settings_url = _browser_settings_url(executable)

    window_handle = _windows_browser_window(executable, timeout=0.25)
    create_new_tab = window_handle is not None
    if window_handle is None:
        subprocess.Popen(
            [str(executable)],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=subprocess.CREATE_NO_WINDOW,
        )
        window_handle = _windows_browser_window(executable)
    if window_handle is None:
        raise RuntimeError("Could not find the browser window")

    _navigate_browser_window(
        window_handle,
        settings_url,
        create_new_tab=create_new_tab,
    )
    _enable_remote_debugging(window_handle)
    marker_endpoint = _wait_for_remote_debugging_marker(executable)
    if marker_endpoint is None:
        raise RuntimeError(
            "Remote debugging is enabled, but its marker was not created"
        )

    logger.info(
        "Opened browser remote debugging settings "
        f"executable={str(executable)!r} url={settings_url!r} "
        f"marker_endpoint={marker_endpoint!r}"
    )
    return settings_url


def windows_default_browser_executable() -> Path | None:
    """Resolve the executable registered as the current HTTPS browser."""
    if sys.platform != "win32":
        return None

    import winreg

    prog_id: str | None = None
    try:
        with winreg.OpenKey(
            winreg.HKEY_CURRENT_USER,
            (
                r"Software\Microsoft\Windows\Shell\Associations"
                r"\UrlAssociations\https\UserChoice"
            ),
        ) as key:
            prog_id = str(winreg.QueryValueEx(key, "ProgId")[0])
    except OSError:
        pass

    locations: list[tuple[int, str]] = []
    if prog_id:
        locations.extend(
            (
                (
                    winreg.HKEY_CURRENT_USER,
                    rf"Software\Classes\{prog_id}\shell\open\command",
                ),
                (
                    winreg.HKEY_CLASSES_ROOT,
                    rf"{prog_id}\shell\open\command",
                ),
            )
        )
    locations.append(
        (winreg.HKEY_CLASSES_ROOT, r"https\shell\open\command")
    )
    for hive, key_path in locations:
        try:
            with winreg.OpenKey(hive, key_path) as key:
                command = str(winreg.QueryValueEx(key, None)[0])
        except OSError:
            continue
        executable = _extract_windows_executable(command)
        if executable is not None:
            return executable
    return None


def _browser_settings_url(executable: Path) -> str:
    browser_name = executable.name.casefold()
    if "msedge" in browser_name:
        return "edge://inspect/#remote-debugging"
    if any(
        name in browser_name
        for name in ("chrome", "chromium", "brave", "vivaldi", "opera")
    ):
        return "chrome://inspect/#remote-debugging"
    raise RuntimeError(
        f"The default browser ({executable.name}) does not support CDP"
    )


def _extract_windows_executable(command: str) -> Path | None:
    command = os.path.expandvars(command.strip())
    quoted = re.match(r'^"([^"]+\.exe)"', command, flags=re.IGNORECASE)
    unquoted = re.match(r"^([^\r\n]+?\.exe)(?:\s|$)", command, re.IGNORECASE)
    match = quoted or unquoted
    if match is None:
        return None
    executable = Path(match.group(1).strip())
    return executable if executable.is_file() else None


def _windows_browser_window(executable: Path, timeout: float = 4.0) -> int | None:
    """Find the front-most visible window owned by the browser executable."""
    if sys.platform != "win32":
        return None

    import ctypes
    from ctypes import wintypes

    user32 = ctypes.WinDLL("user32", use_last_error=True)
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    process_query_limited_information = 0x1000
    enum_callback = ctypes.WINFUNCTYPE(
        wintypes.BOOL,
        wintypes.HWND,
        wintypes.LPARAM,
    )

    kernel32.OpenProcess.argtypes = [
        wintypes.DWORD,
        wintypes.BOOL,
        wintypes.DWORD,
    ]
    kernel32.OpenProcess.restype = wintypes.HANDLE
    kernel32.QueryFullProcessImageNameW.argtypes = [
        wintypes.HANDLE,
        wintypes.DWORD,
        wintypes.LPWSTR,
        ctypes.POINTER(wintypes.DWORD),
    ]
    kernel32.QueryFullProcessImageNameW.restype = wintypes.BOOL
    kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    user32.EnumWindows.argtypes = [enum_callback, wintypes.LPARAM]
    user32.EnumWindows.restype = wintypes.BOOL
    user32.IsWindowVisible.argtypes = [wintypes.HWND]
    user32.IsWindowVisible.restype = wintypes.BOOL
    user32.GetWindowTextLengthW.argtypes = [wintypes.HWND]
    user32.GetWindowTextLengthW.restype = ctypes.c_int
    user32.GetWindowThreadProcessId.argtypes = [
        wintypes.HWND,
        ctypes.POINTER(wintypes.DWORD),
    ]
    user32.GetWindowThreadProcessId.restype = wintypes.DWORD

    expected_name = executable.name.casefold()
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        matches: list[int] = []

        @enum_callback
        def collect_window(hwnd: int, parameter: int) -> bool:
            del parameter
            if not user32.IsWindowVisible(hwnd):
                return True
            if user32.GetWindowTextLengthW(hwnd) <= 0:
                return True

            process_id = wintypes.DWORD()
            user32.GetWindowThreadProcessId(hwnd, ctypes.byref(process_id))
            process_handle = kernel32.OpenProcess(
                process_query_limited_information,
                False,
                process_id.value,
            )
            if not process_handle:
                return True
            try:
                path_buffer = ctypes.create_unicode_buffer(32_768)
                path_length = wintypes.DWORD(len(path_buffer))
                if not kernel32.QueryFullProcessImageNameW(
                    process_handle,
                    0,
                    path_buffer,
                    ctypes.byref(path_length),
                ):
                    return True
                if Path(path_buffer.value).name.casefold() == expected_name:
                    matches.append(int(hwnd))
            finally:
                kernel32.CloseHandle(process_handle)
            return True

        user32.EnumWindows(collect_window, 0)
        if matches:
            return matches[0]
        time.sleep(0.1)
    return None


def _navigate_browser_window(
    window_handle: int,
    url: str,
    create_new_tab: bool,
) -> None:
    """Navigate using Chromium's Windows accessibility controls."""
    create_new_tab_value = "$true" if create_new_tab else "$false"
    script = rf"""
Add-Type -AssemblyName UIAutomationClient
Add-Type -TypeDefinition @'
using System;
using System.Runtime.InteropServices;
public static class LiveGptWindowMessages {{
    [DllImport("user32.dll", SetLastError=true)]
    public static extern bool PostMessage(
        IntPtr window, uint message, IntPtr parameter, IntPtr extra
    );
}}
'@
$root = [System.Windows.Automation.AutomationElement]::FromHandle(
    [System.IntPtr]::new({window_handle})
)
if ({create_new_tab_value}) {{
    $newTabCondition = New-Object `
        System.Windows.Automation.PropertyCondition(
            [System.Windows.Automation.AutomationElement]::AutomationIdProperty,
            'view_28'
        )
    $newTab = $root.FindFirst(
        [System.Windows.Automation.TreeScope]::Descendants,
        $newTabCondition
    )
    if (-not $newTab) {{
        $nameCondition = New-Object `
            System.Windows.Automation.PropertyCondition(
                [System.Windows.Automation.AutomationElement]::NameProperty,
                'New Tab'
            )
        $newTab = $root.FindFirst(
            [System.Windows.Automation.TreeScope]::Descendants,
            $nameCondition
        )
    }}
    if ($newTab) {{
        $invoke = $newTab.GetCurrentPattern(
            [System.Windows.Automation.InvokePattern]::Pattern
        )
        $invoke.Invoke()
    }} else {{
        [void][LiveGptWindowMessages]::PostMessage(
            [System.IntPtr]::new({window_handle}),
            0x0100,
            [System.IntPtr]::new(0x11),
            [System.IntPtr]::Zero
        )
        [void][LiveGptWindowMessages]::PostMessage(
            [System.IntPtr]::new({window_handle}),
            0x0100,
            [System.IntPtr]::new(0x54),
            [System.IntPtr]::Zero
        )
        [void][LiveGptWindowMessages]::PostMessage(
            [System.IntPtr]::new({window_handle}),
            0x0101,
            [System.IntPtr]::new(0x54),
            [System.IntPtr]::Zero
        )
        [void][LiveGptWindowMessages]::PostMessage(
            [System.IntPtr]::new({window_handle}),
            0x0101,
            [System.IntPtr]::new(0x11),
            [System.IntPtr]::Zero
        )
    }}
    Start-Sleep -Milliseconds 250
}}
$addressCondition = New-Object System.Windows.Automation.PropertyCondition(
    [System.Windows.Automation.AutomationElement]::AutomationIdProperty,
    'view_1021'
)
$deadline = [DateTime]::UtcNow.AddSeconds(3)
$addressBar = $null
do {{
    $addressBar = $root.FindFirst(
        [System.Windows.Automation.TreeScope]::Descendants,
        $addressCondition
    )
    if (-not $addressBar) {{ Start-Sleep -Milliseconds 100 }}
}} while (-not $addressBar -and [DateTime]::UtcNow -lt $deadline)
if (-not $addressBar) {{ throw 'Address bar not found' }}
$value = $addressBar.GetCurrentPattern(
    [System.Windows.Automation.ValuePattern]::Pattern
)
$value.SetValue('{url}')
$addressBar.SetFocus()
[void][LiveGptWindowMessages]::PostMessage(
    [System.IntPtr]::new({window_handle}),
    0x0100,
    [System.IntPtr]::new(0x0D),
    [System.IntPtr]::Zero
)
[void][LiveGptWindowMessages]::PostMessage(
    [System.IntPtr]::new({window_handle}),
    0x0101,
    [System.IntPtr]::new(0x0D),
    [System.IntPtr]::Zero
)
Write-Output 'Navigated'
"""
    output = _run_powershell(script, timeout=7)
    if output != "Navigated":
        raise RuntimeError(f"Could not navigate the browser: {output}")


def _enable_remote_debugging(window_handle: int) -> None:
    script = rf"""
Add-Type -AssemblyName UIAutomationClient
$root = [System.Windows.Automation.AutomationElement]::FromHandle(
    [System.IntPtr]::new({window_handle})
)
$idCondition = New-Object System.Windows.Automation.PropertyCondition(
    [System.Windows.Automation.AutomationElement]::AutomationIdProperty,
    'remote-debugging-enabled'
)
$nameCondition = New-Object System.Windows.Automation.PropertyCondition(
    [System.Windows.Automation.AutomationElement]::NameProperty,
    'Allow remote debugging for this browser instance'
)
$deadline = [DateTime]::UtcNow.AddSeconds(5)
$checkbox = $null
do {{
    $checkbox = $root.FindFirst(
        [System.Windows.Automation.TreeScope]::Descendants,
        $idCondition
    )
    if (-not $checkbox) {{
        $checkbox = $root.FindFirst(
            [System.Windows.Automation.TreeScope]::Descendants,
            $nameCondition
        )
    }}
    if (-not $checkbox) {{ Start-Sleep -Milliseconds 150 }}
}} while (-not $checkbox -and [DateTime]::UtcNow -lt $deadline)
if (-not $checkbox) {{ throw 'Remote debugging checkbox not found' }}
$toggle = $checkbox.GetCurrentPattern(
    [System.Windows.Automation.TogglePattern]::Pattern
)
if ($toggle.Current.ToggleState -eq `
        [System.Windows.Automation.ToggleState]::Off) {{
    $toggle.Toggle()
    Start-Sleep -Milliseconds 300
}}
Write-Output $toggle.Current.ToggleState
"""
    state = _run_powershell(script, timeout=7)
    if state != "On":
        raise RuntimeError(f"Could not enable remote debugging: {state}")


def _wait_for_remote_debugging_marker(
    executable: Path,
    timeout: float = 4.0,
) -> str | None:
    directories = browser_user_data_directories(executable)
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        endpoints = active_port_endpoints(directories)
        if endpoints:
            return endpoints[0]
        time.sleep(0.1)
    return None


def _run_powershell(script: str, timeout: float) -> str:
    result = subprocess.run(
        [
            "powershell.exe",
            "-NoLogo",
            "-NoProfile",
            "-NonInteractive",
            "-Command",
            script,
        ],
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
        creationflags=subprocess.CREATE_NO_WINDOW,
    )
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip() or "unknown error"
        raise RuntimeError(detail)
    return result.stdout.strip()
