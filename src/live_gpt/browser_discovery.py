from __future__ import annotations

import json
import os
import re
import socket
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path
from urllib.parse import urlparse


DEFAULT_DEBUG_PORTS = tuple(range(9222, 9233))


def is_chatgpt_url(url: str) -> bool:
    """Return whether a browser page belongs to ChatGPT."""
    try:
        parsed = urlparse(url)
    except ValueError:
        return False
    return (
        parsed.scheme == "https"
        and parsed.hostname in {"chatgpt.com", "www.chatgpt.com"}
    )


def active_port_endpoint(active_port_file: Path) -> str | None:
    """Resolve a live HTTP or WebSocket endpoint from DevToolsActivePort."""
    try:
        lines = active_port_file.read_text(encoding="utf-8").splitlines()
        port = int(lines[0])
    except (OSError, ValueError, IndexError):
        return None
    if not 0 < port < 65_536 or not _port_is_open(port):
        return None

    websocket_path = lines[1].strip() if len(lines) > 1 else ""
    if websocket_path.startswith("/devtools/browser/"):
        return f"ws://127.0.0.1:{port}{websocket_path}"
    return f"http://127.0.0.1:{port}"


def current_user_data_directories() -> list[Path]:
    """Return known per-user Chromium data roots for the current platform."""
    if sys.platform == "win32":
        local = Path(os.environ.get("LOCALAPPDATA", Path.home()))
        roaming = Path(os.environ.get("APPDATA", Path.home()))
        return [
            local / "Microsoft/Edge/User Data",
            local / "Microsoft/Edge Beta/User Data",
            local / "Microsoft/Edge Dev/User Data",
            local / "Microsoft/Edge SxS/User Data",
            local / "Google/Chrome/User Data",
            local / "Google/Chrome Beta/User Data",
            local / "Google/Chrome SxS/User Data",
            local / "Chromium/User Data",
            local / "BraveSoftware/Brave-Browser/User Data",
            local / "Vivaldi/User Data",
            roaming / "Opera Software/Opera Stable",
        ]
    if sys.platform == "darwin":
        application_support = Path.home() / "Library/Application Support"
        return [
            application_support / "Microsoft Edge",
            application_support / "Google/Chrome",
            application_support / "Chromium",
            application_support / "BraveSoftware/Brave-Browser",
            application_support / "Vivaldi",
        ]
    config = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config"))
    return [
        config / "microsoft-edge",
        config / "google-chrome",
        config / "chromium",
        config / "BraveSoftware/Brave-Browser",
        config / "vivaldi",
    ]


def browser_user_data_directories(executable: Path) -> list[Path]:
    """Filter known user-data roots to the given browser family."""
    browser_name = executable.name.casefold()
    if "msedge" in browser_name:
        path_markers = ("microsoft/edge",)
    elif "brave" in browser_name:
        path_markers = ("bravesoftware/brave-browser",)
    elif "vivaldi" in browser_name:
        path_markers = ("vivaldi",)
    elif "opera" in browser_name:
        path_markers = ("opera software",)
    elif "chromium" in browser_name:
        path_markers = ("chromium",)
    else:
        path_markers = ("google/chrome",)

    return [
        directory
        for directory in current_user_data_directories()
        if any(
            marker in directory.as_posix().casefold()
            for marker in path_markers
        )
    ]


def active_port_endpoints(
    user_data_directories: list[Path] | None = None,
) -> list[str]:
    """Return live DevToolsActivePort endpoints from known profile roots."""
    directories = user_data_directories or current_user_data_directories()
    endpoints: list[str] = []
    for directory in directories:
        endpoint = active_port_endpoint(directory / "DevToolsActivePort")
        if endpoint is not None:
            endpoints.append(endpoint)
    return endpoints


def discover_cdp_endpoint() -> str | None:
    """Find a locally running Chromium browser with remote debugging enabled."""
    configured_endpoint = os.environ.get("LIVE_GPT_CDP_ENDPOINT", "").strip()
    candidates = [configured_endpoint] if configured_endpoint else []
    candidates.extend(active_port_endpoints())
    for endpoint in dict.fromkeys(candidates):
        if _endpoint_is_available(endpoint):
            return endpoint

    ports = _windows_remote_debug_ports()
    ports.update(DEFAULT_DEBUG_PORTS)
    for port in sorted(ports):
        endpoint = f"http://127.0.0.1:{port}"
        if _endpoint_is_available(endpoint):
            return endpoint
    return None


def _debug_port_from_command_line(command_line: str) -> int | None:
    match = re.search(
        r"(?:^|\s)--remote-debugging-port(?:=|\s+)(\d+)(?:\s|$)",
        command_line,
        flags=re.IGNORECASE,
    )
    if match is None:
        return None
    port = int(match.group(1))
    return port if 0 < port < 65_536 else None


def _windows_remote_debug_ports() -> set[int]:
    if sys.platform != "win32":
        return set()

    script = (
        "Get-CimInstance Win32_Process | "
        "Where-Object { $_.CommandLine -like '*--remote-debugging-port*' } | "
        "Select-Object -ExpandProperty CommandLine | ConvertTo-Json -Compress"
    )
    startup_info = subprocess.STARTUPINFO()
    startup_info.dwFlags |= subprocess.STARTF_USESHOWWINDOW
    try:
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
            timeout=4,
            check=False,
            startupinfo=startup_info,
            creationflags=subprocess.CREATE_NO_WINDOW,
        )
    except (OSError, subprocess.SubprocessError):
        return set()

    if result.returncode != 0 or not result.stdout.strip():
        return set()
    try:
        command_lines = json.loads(result.stdout)
    except json.JSONDecodeError:
        return set()
    if isinstance(command_lines, str):
        command_lines = [command_lines]
    if not isinstance(command_lines, list):
        return set()

    ports: set[int] = set()
    for command_line in command_lines:
        if not isinstance(command_line, str):
            continue
        port = _debug_port_from_command_line(command_line)
        if port is not None:
            ports.add(port)
    return ports


def _port_is_open(port: int, timeout: float = 0.2) -> bool:
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=timeout):
            return True
    except OSError:
        return False


def _endpoint_is_available(endpoint: str, timeout: float = 0.4) -> bool:
    parsed = urlparse(endpoint)
    if parsed.scheme in {"ws", "wss"}:
        if parsed.hostname not in {"127.0.0.1", "localhost", "::1"}:
            return False
        return parsed.port is not None and _port_is_open(parsed.port, timeout)

    version_url = f"{endpoint.rstrip('/')}/json/version"
    try:
        with urllib.request.urlopen(version_url, timeout=timeout) as response:
            payload = json.load(response)
    except (OSError, ValueError, urllib.error.URLError):
        return False
    return bool(payload.get("webSocketDebuggerUrl"))
