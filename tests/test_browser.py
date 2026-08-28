from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from live_gpt.browser import BrowserMonitor, _MonitorState
from live_gpt.browser_discovery import is_chatgpt_url
from live_gpt import browser_windows


class FakePlaywrightError(Exception):
    pass


class BrowserDiscoveryTests(unittest.TestCase):
    def test_chatgpt_url_requires_exact_https_host(self) -> None:
        self.assertTrue(is_chatgpt_url("https://chatgpt.com/c/conversation"))
        self.assertTrue(is_chatgpt_url("https://www.chatgpt.com/"))
        self.assertFalse(is_chatgpt_url("http://chatgpt.com/"))
        self.assertFalse(is_chatgpt_url("https://chatgpt.com.example.org/"))


class BrowserMonitorTests(unittest.TestCase):
    @patch("live_gpt.browser.discover_cdp_endpoint")
    def test_declined_endpoint_is_not_retried_until_requested(
        self,
        discover_endpoint: Mock,
    ) -> None:
        discover_endpoint.return_value = "http://127.0.0.1:9222"
        chromium = Mock()
        chromium.connect_over_cdp.side_effect = FakePlaywrightError("declined")
        playwright = Mock(chromium=chromium)
        monitor = BrowserMonitor()
        state = _MonitorState()

        monitor._try_connect(playwright, FakePlaywrightError, state)
        monitor._try_connect(playwright, FakePlaywrightError, state)
        self.assertEqual(chromium.connect_over_cdp.call_count, 1)

        monitor.request_retry_connection()
        monitor._try_connect(playwright, FakePlaywrightError, state)
        self.assertEqual(chromium.connect_over_cdp.call_count, 2)

    def test_native_color_scheme_is_restored_once_per_page(self) -> None:
        page = Mock()
        page.is_closed.return_value = False
        page.url = "https://chatgpt.com/c/conversation"
        page.title.return_value = "Conversation"
        browser = Mock()
        browser.contexts = [Mock(pages=[page])]
        reset_pages: set[str] = set()

        first = BrowserMonitor._collect_chatgpt_tabs(
            browser,
            FakePlaywrightError,
            reset_pages,
        )
        second = BrowserMonitor._collect_chatgpt_tabs(
            browser,
            FakePlaywrightError,
            reset_pages,
        )

        self.assertEqual(first, second)
        page.emulate_media.assert_called_once_with(color_scheme="null")


class BrowserWindowsTests(unittest.TestCase):
    @patch("live_gpt.browser_windows._wait_for_remote_debugging_marker")
    @patch("live_gpt.browser_windows._enable_remote_debugging")
    @patch("live_gpt.browser_windows._navigate_browser_window")
    @patch("live_gpt.browser_windows._windows_browser_window")
    @patch("live_gpt.browser_windows.windows_default_browser_executable")
    def test_existing_browser_gets_new_settings_tab(
        self,
        default_executable: Mock,
        browser_window: Mock,
        navigate: Mock,
        enable_debugging: Mock,
        wait_for_marker: Mock,
    ) -> None:
        default_executable.return_value = Path("C:/Program Files/Edge/msedge.exe")
        browser_window.return_value = 1234
        wait_for_marker.return_value = "ws://127.0.0.1:9222/devtools/browser/id"

        url = browser_windows.open_remote_debugging_settings()

        self.assertEqual(url, "edge://inspect/#remote-debugging")
        navigate.assert_called_once_with(
            1234,
            "edge://inspect/#remote-debugging",
            create_new_tab=True,
        )
        enable_debugging.assert_called_once_with(1234)


if __name__ == "__main__":
    unittest.main()
