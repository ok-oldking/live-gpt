from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from PySide6.QtCore import QThread, Signal

from .browser_discovery import (
    DEFAULT_DEBUG_PORTS,
    active_port_endpoint,
    active_port_endpoints,
    browser_user_data_directories,
    current_user_data_directories,
    discover_cdp_endpoint,
    is_chatgpt_url,
)
from .browser_windows import (
    open_remote_debugging_settings,
    windows_default_browser_executable,
)
from .logger import Logger


logger = Logger.get_logger(__name__)


@dataclass
class _MonitorState:
    browser: Any | None = None
    settings_opened: bool = False
    next_settings_attempt: float = 0.0
    rejected_endpoint: str | None = None
    last_tabs: list[dict[str, str]] | None = None
    media_reset_pages: set[str] = field(default_factory=set)


class BrowserMonitor(QThread):
    """Own Playwright on a worker thread and publish live ChatGPT tab state."""

    tabs_changed = Signal(object)
    status_changed = Signal(str)

    def __init__(self) -> None:
        super().__init__()
        self._stop_requested = False
        self._retry_connection_requested = False
        self._last_status = ""

    def request_stop(self) -> None:
        self._stop_requested = True

    def request_retry_connection(self) -> None:
        self._retry_connection_requested = True

    def run(self) -> None:
        try:
            from playwright.sync_api import Error as PlaywrightError
            from playwright.sync_api import sync_playwright
        except ImportError:
            self._set_status("Playwright is not installed")
            self.tabs_changed.emit([])
            return

        state = _MonitorState()
        with sync_playwright() as playwright:
            while not self._stop_requested:
                if not self._is_connected(state.browser):
                    self._try_connect(playwright, PlaywrightError, state)

                if self._is_connected(state.browser):
                    tabs = self._collect_chatgpt_tabs(
                        state.browser,
                        PlaywrightError,
                        state.media_reset_pages,
                    )
                    if tabs is None:
                        self._handle_disconnect(state)
                    elif tabs != state.last_tabs:
                        state.last_tabs = tabs
                        self.tabs_changed.emit(tabs)
                elif state.last_tabs:
                    state.last_tabs = []
                    self.tabs_changed.emit([])

                self.msleep(1_500)

        logger.info("Browser monitor stopped")

    def _try_connect(
        self,
        playwright: Any,
        playwright_error: type[Exception],
        state: _MonitorState,
    ) -> None:
        state.browser = None
        endpoint = discover_cdp_endpoint()
        if self._stop_requested:
            return

        if self._retry_connection_requested:
            self._retry_connection_requested = False
            state.rejected_endpoint = None

        if endpoint is None:
            self._prepare_remote_debugging(state)
            return

        if endpoint == state.rejected_endpoint:
            self._set_status(
                "Approval declined; click Enable Debugging to retry"
            )
            return

        self._set_status("Connecting to browser…")
        try:
            state.browser = playwright.chromium.connect_over_cdp(
                endpoint,
                timeout=15_000,
            )
        except playwright_error as error:
            logger.warning(
                "Unable to connect to remote-debug browser "
                f"endpoint={endpoint!r}: {error}"
            )
            state.rejected_endpoint = endpoint
            self._set_status(
                "Approval declined; click Enable Debugging to retry"
            )
            return

        state.rejected_endpoint = None
        state.media_reset_pages.clear()
        state.settings_opened = False
        self._set_status("Browser connected")
        logger.info(f"Connected to browser endpoint={endpoint!r}")

    def _prepare_remote_debugging(self, state: _MonitorState) -> None:
        if (
            not state.settings_opened
            and time.monotonic() >= state.next_settings_attempt
        ):
            self._set_status("Opening remote debugging settings…")
            try:
                open_remote_debugging_settings()
                state.settings_opened = True
            except Exception as error:
                logger.error("Unable to open remote debugging settings", error)
                self._set_status(str(error))
            state.next_settings_attempt = time.monotonic() + 30

        if state.settings_opened:
            self._set_status("Enable remote debugging in the browser")

    @staticmethod
    def _collect_chatgpt_tabs(
        browser: Any,
        playwright_error: type[Exception],
        media_reset_pages: set[str],
    ) -> list[dict[str, str]] | None:
        try:
            contexts = list(browser.contexts)
        except playwright_error:
            return None

        tabs: list[dict[str, str]] = []
        for context in contexts:
            try:
                pages = list(context.pages)
            except playwright_error:
                continue
            for page in pages:
                try:
                    if page.is_closed() or not is_chatgpt_url(page.url):
                        continue
                    page_id = str(id(page))
                    if page_id not in media_reset_pages:
                        page.emulate_media(color_scheme="null")
                        media_reset_pages.add(page_id)
                    title = page.title().strip() or "ChatGPT"
                    tabs.append(
                        {"id": page_id, "title": title, "url": page.url}
                    )
                except playwright_error:
                    continue

        media_reset_pages.intersection_update(tab["id"] for tab in tabs)
        tabs.sort(key=lambda tab: tab["title"].casefold())
        return tabs

    def _handle_disconnect(self, state: _MonitorState) -> None:
        state.browser = None
        state.media_reset_pages.clear()
        self._set_status("Browser disconnected")
        if state.last_tabs != []:
            state.last_tabs = []
            self.tabs_changed.emit([])

    def _set_status(self, status: str) -> None:
        if status == self._last_status:
            return
        self._last_status = status
        self.status_changed.emit(status)

    @staticmethod
    def _is_connected(browser: Any | None) -> bool:
        return browser is not None and browser.is_connected()


__all__ = [
    "BrowserMonitor",
    "DEFAULT_DEBUG_PORTS",
    "active_port_endpoint",
    "active_port_endpoints",
    "browser_user_data_directories",
    "current_user_data_directories",
    "discover_cdp_endpoint",
    "is_chatgpt_url",
    "open_remote_debugging_settings",
    "windows_default_browser_executable",
]
