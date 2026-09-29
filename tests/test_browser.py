from __future__ import annotations

import time
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from live_gpt.browser import (
    ASSISTANT_TURN_SELECTOR,
    CHATGPT_COMPOSER_SELECTOR,
    DICTATION_CANCEL_SELECTORS,
    DICTATION_RESULT_POLL_COUNT,
    DICTATION_RESULT_TIMEOUT_MS,
    _ActiveReading,
    BrowserMonitor,
    _ActiveResponse,
    _MonitorState,
    _ResponseSnapshot,
)
from live_gpt.browser_discovery import is_chatgpt_url
from live_gpt import browser_windows


class FakePlaywrightError(Exception):
    pass


class BrowserDiscoveryTests(unittest.TestCase):
    @patch.dict("os.environ", {"LIVE_GPT_CDP_ENDPOINT": ""})
    @patch("live_gpt.browser_discovery.sys.platform", "win32")
    @patch("live_gpt.browser_windows._preferred_open_browser", return_value=(Path("C:/Chrome/chrome.exe"), 1234))
    @patch("live_gpt.browser_discovery.browser_user_data_directories", return_value=[Path("chrome-profile")])
    @patch("live_gpt.browser_discovery.active_port_endpoints")
    @patch("live_gpt.browser_discovery._endpoint_is_available", return_value=True)
    def test_discovery_uses_open_chrome_marker_instead_of_old_edge_marker(
        self, available, markers, directories, selected,
    ):
        from live_gpt.browser_discovery import discover_cdp_endpoint
        chrome = "ws://127.0.0.1:4422/devtools/browser/chrome"
        edge = "ws://127.0.0.1:9222/devtools/browser/edge"
        markers.side_effect = lambda roots=None: [chrome] if roots == [Path("chrome-profile")] else [edge, chrome]
        self.assertEqual(discover_cdp_endpoint(), chrome)
        available.assert_called_once_with(chrome)
        markers.assert_called_once_with([Path("chrome-profile")])

    @patch.dict("os.environ", {"LIVE_GPT_CDP_ENDPOINT": ""})
    @patch("live_gpt.browser_discovery.sys.platform", "win32")
    @patch("live_gpt.browser_windows._preferred_open_browser", return_value=(Path("C:/Chrome/chrome.exe"), 1234))
    @patch("live_gpt.browser_discovery.browser_user_data_directories", return_value=[Path("chrome-profile")])
    @patch("live_gpt.browser_discovery.active_port_endpoints", return_value=[])
    @patch("live_gpt.browser_discovery._windows_remote_debug_ports")
    def test_disabled_open_browser_requires_setup_not_other_browser_fallback(
        self, ports, markers, directories, selected,
    ):
        from live_gpt.browser_discovery import discover_cdp_endpoint
        self.assertIsNone(discover_cdp_endpoint())
        ports.assert_not_called()

    @patch.dict("os.environ", {"LIVE_GPT_CDP_ENDPOINT": "http://localhost:5555"})
    @patch("live_gpt.browser_discovery._endpoint_is_available", return_value=True)
    @patch("live_gpt.browser_windows._preferred_open_browser")
    def test_explicit_endpoint_still_overrides_browser_selection(self, selected, available):
        from live_gpt.browser_discovery import discover_cdp_endpoint
        self.assertEqual(discover_cdp_endpoint(), "http://localhost:5555")
        selected.assert_not_called()

    def test_chatgpt_url_requires_exact_https_host(self) -> None:
        self.assertTrue(is_chatgpt_url("https://chatgpt.com/c/conversation"))
        self.assertTrue(is_chatgpt_url("https://www.chatgpt.com/"))
        self.assertFalse(is_chatgpt_url("http://chatgpt.com/"))
        self.assertFalse(is_chatgpt_url("https://chatgpt.com.example.org/"))


class BrowserMonitorTests(unittest.TestCase):
    def test_stalled_browser_operation_interrupts_driver(self):
        monitor = BrowserMonitor()
        loop, stop_transport = Mock(), Mock()
        loop.call_soon_threadsafe.side_effect = lambda callback: callback()
        monitor._playwright_cancellation = (loop, stop_transport)
        with patch("live_gpt.browser.threading.Timer") as timer:
            with self.assertRaisesRegex(TimeoutError, "reconnecting"):
                with monitor._browser_operation(10):
                    timer.call_args.args[1]()
        stop_transport.assert_called_once()
        timer.return_value.cancel.assert_called_once()

    def test_late_watchdog_callback_does_not_stop_healthy_driver(self):
        monitor = BrowserMonitor()
        loop, stop_transport = Mock(), Mock()
        monitor._playwright_cancellation = (loop, stop_transport)
        with patch("live_gpt.browser.threading.Timer") as timer:
            with monitor._browser_operation(10):
                timer.call_args.args[1]()
            loop.call_soon_threadsafe.call_args.args[0]()
        stop_transport.assert_not_called()

    def test_send_during_browser_approval_fails_without_queueing(self):
        monitor = BrowserMonitor()
        results = []
        monitor.send_finished.connect(lambda *args: results.append(args))
        monitor._connection_pending.set()
        monitor.request_send("old-window", "Keep my text")
        self.assertTrue(monitor._send_requests.empty())
        self.assertEqual(results, [(False, "Keep my text",
                                   "Approve remote debugging in the browser before sending")])

    def test_monitor_restarts_after_stalled_session_and_fails_queued_send(self):
        monitor = BrowserMonitor()
        results, tabs = [], []
        monitor.send_finished.connect(lambda *args: results.append(args))
        monitor.tabs_changed.connect(tabs.append)
        monitor._wake_event = Mock()
        new_page = {"id": "new-window", "title": "ChatGPT", "url": "https://chatgpt.com/"}

        def session(_factory, _error, state):
            if runner.call_count == 1:
                state.browser = Mock()
                state.last_tabs = [{"id": "old-window"}]
                monitor.request_send("old-window", "Keep my text")
                raise TimeoutError("stale connection")
            self.assertIsNone(state.browser)
            self.assertEqual(state.last_tabs, [])
            monitor.tabs_changed.emit([new_page])
            monitor.request_stop()

        with patch.object(monitor, "_run_session", side_effect=session) as runner:
            monitor.run()
        self.assertEqual(runner.call_count, 2)
        self.assertEqual(results, [(False, "Keep my text", "Browser is not connected")])
        self.assertIn([new_page], tabs)

    def test_queued_send_fails_before_waiting_for_browser_approval(self):
        monitor = BrowserMonitor()
        results = []
        monitor.send_finished.connect(lambda *args: results.append(args))
        monitor.request_send("old-window", "Keep my text")
        playwright = Mock()

        def connect(*args, **kwargs):
            self.assertEqual(results, [(False, "Keep my text", "Browser is not connected")])
            return Mock()

        playwright.chromium.connect_over_cdp.side_effect = connect
        with patch("live_gpt.browser.discover_cdp_endpoint", return_value="ws://localhost:9222"):
            monitor._try_connect(playwright, FakePlaywrightError, _MonitorState())
        playwright.chromium.connect_over_cdp.assert_called_once()

    def test_stale_paused_media_uses_subtitle_timing_fallback(self):
        monitor = BrowserMonitor()
        page = Mock()
        page.evaluate.return_value = {"playCount": 0, "paused": True, "ended": True, "currentTime": 0, "duration": 100}
        reading = _ActiveReading(page=page, full_text="The final answer.", subtitles=(), started_at=0)
        state = _MonitorState(active_reading=reading)
        updates = []
        monitor.reading_changed.connect(updates.append)
        with patch("live_gpt.browser.time.monotonic", return_value=9) as clock:
            monitor._poll_active_reading(state)
            self.assertEqual(updates[-1], {"text": "The final answer.", "fraction": 0.0})
            clock.return_value = 10
            monitor._poll_active_reading(state)
            self.assertGreater(updates[-1]["fraction"], 0)
            clock.return_value = 1000
            monitor._poll_active_reading(state)
            self.assertIsNone(state.active_reading)

    def test_response_timeout_retries_then_completes_after_stability_window(self):
        from playwright.sync_api import TimeoutError as PlaywrightTimeoutError

        for browser_voice in (True, False):
            with self.subTest(browser_voice=browser_voice):
                monitor = BrowserMonitor()
                monitor.set_use_browser_voice(browser_voice)
                response = _ActiveResponse(
                    page=Mock(), turn_marker_before="old", started_at=90,
                    last_text="Reply", last_text_changed_at=95,
                    completion_candidate_at=95,
                )
                state = _MonitorState(active_response=response)
                finished, voice_updates = [], []
                monitor.response_finished.connect(lambda *args: finished.append(args))
                monitor.local_voice_updated.connect(lambda *args: voice_updates.append(args))
                snapshot = _ResponseSnapshot(True, False, True, "Reply", "Finishing reply…")
                with (
                    patch.object(monitor, "_response_snapshot", side_effect=[
                        PlaywrightTimeoutError("Locator.inner_text: Timeout 1000ms exceeded"),
                        snapshot, snapshot,
                    ]),
                    patch.object(monitor, "_click_read_aloud", return_value=True) as read,
                    patch("live_gpt.browser.time.monotonic", return_value=100) as clock,
                ):
                    monitor._poll_active_response(state)
                    self.assertIs(state.active_response, response)
                    self.assertEqual(response.last_text, "Reply")
                    self.assertIsNone(response.completion_candidate_at)
                    self.assertEqual(finished, [])
                    self.assertEqual(voice_updates, [])
                    clock.return_value = 101
                    monitor._poll_active_response(state)
                    self.assertIs(state.active_response, response)
                    read.assert_not_called()
                    clock.return_value = 103.1
                    monitor._poll_active_response(state)
                self.assertIsNone(state.active_response)
                self.assertEqual(read.call_count, int(browser_voice))
                if not browser_voice:
                    self.assertEqual(voice_updates, [("Reply", True)])

    def test_missing_turn_preserves_text_and_response_deadline(self):
        monitor = BrowserMonitor()
        monitor.set_use_browser_voice(False)
        response = _ActiveResponse(
            page=Mock(), turn_marker_before="old", started_at=0,
            last_text="Partial reply", completion_candidate_at=10,
        )
        state = _MonitorState(active_response=response)
        finished, changes = [], []
        monitor.response_finished.connect(lambda *args: finished.append(args))
        monitor.response_changed.connect(lambda *args: changes.append(args))
        with (
            patch.object(monitor, "_response_snapshot", return_value=
                         _ResponseSnapshot(False, False, False, "", "Waiting")) as snapshot,
            patch("live_gpt.browser.time.monotonic", return_value=599) as clock,
        ):
            monitor._poll_active_response(state)
            self.assertEqual(response.last_text, "Partial reply")
            self.assertEqual(changes, [])
            self.assertIsNone(response.completion_candidate_at)
            clock.return_value = 600
            monitor._poll_active_response(state)
            snapshot.assert_called_once()
        self.assertIsNone(state.active_response)
        self.assertEqual(finished, [(False, "Timed out while waiting for ChatGPT's reply")])

    def test_non_timeout_response_error_still_fails(self):
        monitor = BrowserMonitor()
        state = _MonitorState(active_response=_ActiveResponse(page=Mock(), turn_marker_before=None))
        finished = []
        monitor.response_finished.connect(lambda *args: finished.append(args))
        with patch.object(monitor, "_response_snapshot", side_effect=RuntimeError("Page closed")):
            monitor._poll_active_response(state)
        self.assertIsNone(state.active_response)
        self.assertEqual(finished, [(False, "Could not read ChatGPT's reply: Page closed")])

    def test_response_snapshot_keeps_named_http_links_without_changing_text(self):
        turn = Mock()
        turn.get_attribute.return_value = "new-turn"
        turn.inner_text.return_value = "Read the documentation"
        anchors = Mock()
        anchors.evaluate_all.return_value = [
            ["documentation", "https://example.com/docs"],
            ["unsafe", "javascript:alert(1)"],
        ]
        turn.locator.side_effect = lambda selector: (
            anchors if selector == "a[href]" else Mock(last=Mock(count=lambda: 0), count=lambda: 1)
        )
        page = Mock()
        page.locator.side_effect = lambda selector: (
            Mock(count=lambda: 1, last=turn) if 'data-turn="assistant"' in selector
            else Mock(first=Mock(is_visible=lambda: False))
        )
        snapshot = BrowserMonitor._response_snapshot(page, "old-turn")
        self.assertEqual(snapshot.text, "Read the documentation")
        self.assertEqual(snapshot.links, (("documentation", "https://example.com/docs"),))

    def test_refresh_discovers_new_tabs_and_navigation_after_empty_connection(self):
        monitor = BrowserMonitor()
        updates = []
        monitor.tabs_changed.connect(updates.append)
        page = Mock(url="about:blank")
        page.is_closed.return_value = False
        page.title.return_value = "New conversation"
        context = Mock(pages=[])
        browser = Mock(contexts=[context])
        state = _MonitorState(browser=browser)
        session = browser.new_browser_cdp_session.return_value
        monitor._refresh_tabs(state, FakePlaywrightError)
        self.assertEqual(updates, [[]])
        # Model events becoming visible only when a sync API call pumps them.
        session.send.side_effect = lambda _: context.pages.append(page)
        monitor._refresh_tabs(state, FakePlaywrightError)
        self.assertEqual(updates, [[]])
        session.send.side_effect = lambda _: setattr(page, "url", "https://chatgpt.com/")
        monitor._refresh_tabs(state, FakePlaywrightError)
        self.assertEqual(updates[-1], [{"id": str(id(page)), "title": "New conversation", "url": "https://chatgpt.com/"}])
        monitor._refresh_tabs(state, FakePlaywrightError)
        self.assertEqual(len(updates), 2)
        browser.new_browser_cdp_session.assert_called_once()
        session.send.assert_called_with("Target.getTargets")
        session.send.side_effect = lambda _: context.pages.clear()
        monitor._refresh_tabs(state, FakePlaywrightError)
        self.assertEqual(updates[-1], [])

    def test_discovery_failure_clears_connection_and_session(self):
        monitor = BrowserMonitor()
        connections = []
        monitor.debug_connection_changed.connect(connections.append)
        browser = Mock()
        browser.new_browser_cdp_session.return_value.send.side_effect = FakePlaywrightError("Disconnected")
        state = _MonitorState(browser=browser)
        monitor._refresh_tabs(state, FakePlaywrightError)
        self.assertIsNone(state.browser)
        self.assertIsNone(state.discovery_session)
        self.assertEqual(connections, [False])

    def test_composer_text_uses_visible_contenteditable_match(self) -> None:
        hidden_composer = Mock()
        hidden_composer.evaluate.return_value = "Hidden composer"
        visible_composer = Mock()
        visible_composer.evaluate.return_value = "Visible dictated text"
        page = Mock()

        def locate(selector: str) -> Mock:
            if selector == "#prompt-textarea":
                return Mock(first=hidden_composer)
            if ":visible" in selector and "contenteditable" in selector:
                return Mock(first=visible_composer)
            raise AssertionError(f"Unexpected selector: {selector}")

        page.locator.side_effect = locate

        text = BrowserMonitor._read_composer_text(page)

        self.assertEqual(text, "Visible dictated text")
        visible_composer.wait_for.assert_called_once_with(
            state="visible",
            timeout=DICTATION_RESULT_TIMEOUT_MS,
        )
        hidden_composer.wait_for.assert_not_called()

    @patch("live_gpt.browser.discover_cdp_endpoint")
    def test_declined_endpoint_waits_for_explicit_retry(
        self,
        discover_endpoint: Mock,
    ) -> None:
        discover_endpoint.return_value = "http://127.0.0.1:9222"
        chromium = Mock()
        chromium.connect_over_cdp.side_effect = FakePlaywrightError("declined")
        playwright = Mock(chromium=chromium)
        monitor = BrowserMonitor()
        state = _MonitorState(settings_opened=True)

        monitor._try_connect(playwright, FakePlaywrightError, state)
        monitor._try_connect(playwright, FakePlaywrightError, state)
        self.assertEqual(chromium.connect_over_cdp.call_count, 1)

        monitor.request_retry_connection()
        monitor._try_connect(playwright, FakePlaywrightError, state)
        self.assertEqual(chromium.connect_over_cdp.call_count, 2)
        chromium.connect_over_cdp.assert_called_with(
            "http://127.0.0.1:9222",
            timeout=0,
        )

    @patch("live_gpt.browser.discover_cdp_endpoint")
    def test_requested_retry_retries_declined_endpoint(
        self,
        discover_endpoint: Mock,
    ) -> None:
        discover_endpoint.return_value = "http://127.0.0.1:9222"
        chromium = Mock()
        chromium.connect_over_cdp.side_effect = FakePlaywrightError("declined")
        playwright = Mock(chromium=chromium)
        monitor = BrowserMonitor()
        state = _MonitorState(settings_opened=True)

        with patch("live_gpt.browser.time.monotonic", return_value=100.0):
            monitor._try_connect(playwright, FakePlaywrightError, state)
            monitor.request_retry_connection()
            monitor._try_connect(playwright, FakePlaywrightError, state)

        self.assertEqual(chromium.connect_over_cdp.call_count, 2)

    @patch("live_gpt.browser.open_remote_debugging_settings")
    @patch("live_gpt.browser.discover_cdp_endpoint", return_value="ws://127.0.0.1:9222/devtools/browser/id")
    def test_rejected_endpoint_opens_settings_and_retries_once(self, discover, open_settings):
        monitor = BrowserMonitor()
        state = _MonitorState()
        playwright = Mock()
        playwright.chromium.connect_over_cdp.side_effect = FakePlaywrightError("403 Forbidden")
        monitor._try_connect(playwright, FakePlaywrightError, state)
        open_settings.assert_called_once()
        self.assertTrue(state.settings_opened)
        self.assertIsNone(state.retry_endpoint)
        monitor._try_connect(playwright, FakePlaywrightError, state)
        monitor._try_connect(playwright, FakePlaywrightError, state)
        self.assertEqual(playwright.chromium.connect_over_cdp.call_count, 2)
        open_settings.assert_called_once()
        monitor.request_retry_connection()
        playwright.chromium.connect_over_cdp.side_effect = None
        monitor._try_connect(playwright, FakePlaywrightError, state)
        self.assertIsNotNone(state.browser)
        self.assertIsNone(state.retry_endpoint)

    @patch("live_gpt.browser.open_remote_debugging_settings", side_effect=[RuntimeError("Window unavailable"), "chrome://inspect/#remote-debugging"])
    @patch("live_gpt.browser.discover_cdp_endpoint", return_value="ws://127.0.0.1:9222/devtools/browser/id")
    def test_rejected_endpoint_retries_failed_settings_after_cooldown(self, discover, open_settings):
        monitor = BrowserMonitor()
        state = _MonitorState()
        playwright = Mock()
        playwright.chromium.connect_over_cdp.side_effect = FakePlaywrightError("403 Forbidden")
        with patch("live_gpt.browser.time.monotonic", return_value=100) as clock:
            monitor._try_connect(playwright, FakePlaywrightError, state)
            monitor._try_connect(playwright, FakePlaywrightError, state)
            open_settings.assert_called_once()
            self.assertEqual(playwright.chromium.connect_over_cdp.call_count, 1)
            clock.return_value = 131
            monitor._try_connect(playwright, FakePlaywrightError, state)
        self.assertEqual(open_settings.call_count, 2)
        self.assertIsNone(state.retry_endpoint)
        self.assertTrue(state.settings_opened)

    def test_stop_interrupts_pending_browser_approval(self) -> None:
        monitor = BrowserMonitor()
        loop = Mock()
        stop_transport = Mock()
        monitor._playwright_cancellation = (loop, stop_transport)
        monitor._connection_pending.set()

        monitor.request_stop()

        loop.call_soon_threadsafe.assert_called_once_with(stop_transport)

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

    def test_send_request_targets_selected_chatgpt_page(self) -> None:
        composer = Mock()
        composer_locator = Mock(first=composer)
        send_button = Mock()
        send_button_locator = Mock(first=send_button)
        composer.locator.return_value.locator.return_value = send_button_locator
        attachment_locator = Mock()
        attachment_locator.count.return_value = 0
        previous_turn = Mock()
        previous_turn.get_attribute.side_effect = (
            lambda attribute: "previous-turn"
            if attribute == "data-turn-id"
            else "conversation-turn-4"
        )
        assistant_turns = Mock(last=previous_turn)
        assistant_turns.count.return_value = 2

        page = Mock()
        page.is_closed.return_value = False
        page.url = "https://chatgpt.com/c/conversation"
        def locate(selector: str) -> Mock:
            if selector == CHATGPT_COMPOSER_SELECTOR:
                return composer_locator
            if selector == ASSISTANT_TURN_SELECTOR:
                return assistant_turns
            if selector.startswith('button[aria-label="Remove file"]'):
                return attachment_locator
            return send_button_locator

        page.locator.side_effect = locate
        browser = Mock(contexts=[Mock(pages=[page])])
        monitor = BrowserMonitor()
        state = _MonitorState()
        results: list[tuple[bool, str, str]] = []
        monitor.send_finished.connect(
            lambda success, text, message: results.append(
                (success, text, message)
            )
        )

        monitor.request_send(str(id(page)), "Hello from Live GPT")
        monitor._process_send_requests(
            browser,
            FakePlaywrightError,
            state,
        )

        page.bring_to_front.assert_not_called()
        composer.wait_for.assert_called_once_with(
            state="visible",
            timeout=5_000,
        )
        composer.fill.assert_called_once_with("Hello from Live GPT")
        send_button.click.assert_called_once_with(timeout=15_000)
        self.assertEqual(
            results,
            [(True, "Hello from Live GPT", "Sent to ChatGPT")],
        )
        self.assertIsNotNone(state.active_response)

    def test_send_replaces_existing_attachment_with_screenshot(self) -> None:
        composer = Mock()
        remove_button = Mock()
        remove_button.is_visible.return_value = True
        remove_buttons = Mock(last=remove_button)
        remove_buttons.count.return_value = 1
        file_input = Mock()
        file_inputs = Mock(last=file_input)
        file_inputs.count.return_value = 1
        send_button = Mock()

        page = Mock()

        composer.locator.return_value.locator.return_value = Mock(first=send_button)

        def locate(selector: str) -> Mock:
            if selector == CHATGPT_COMPOSER_SELECTOR:
                return Mock(first=composer)
            if selector.startswith('button[aria-label="Remove file"]'):
                return remove_buttons
            if selector.startswith('input[type="file"]'):
                return file_inputs
            return Mock(first=send_button)

        page.locator.side_effect = locate

        BrowserMonitor._send_to_chatgpt_page(
            page,
            "Describe this screenshot",
            b"png bytes",
        )

        remove_button.click.assert_called_once_with(timeout=5_000)
        composer.fill.assert_called_once_with("Describe this screenshot")
        file_input.set_input_files.assert_called_once_with(
            {
                "name": "live-gpt-screenshot.webp",
                "mimeType": "image/webp",
                "buffer": b"png bytes",
            },
            timeout=10_000,
        )
        send_button.click.assert_called_once_with(timeout=15_000)

    def test_preuploaded_attachment_is_preserved_when_sending(self) -> None:
        composer = Mock()
        send_button = Mock()
        composer.locator.return_value.locator.return_value = Mock(first=send_button)
        page = Mock()

        def locate(selector: str) -> Mock:
            if selector == CHATGPT_COMPOSER_SELECTOR:
                return Mock(first=composer)
            return Mock(first=send_button)

        page.locator.side_effect = locate

        with patch.object(BrowserMonitor, "_clear_chatgpt_attachments") as clear:
            BrowserMonitor._send_to_chatgpt_page(
                page,
                "Dictated prompt",
                preserve_attachments=True,
            )

        clear.assert_not_called()
        composer.fill.assert_called_once_with("Dictated prompt")
        send_button.click.assert_called_once_with(timeout=15_000)

    def test_attachment_request_replaces_pending_chatgpt_image(self) -> None:
        page = Mock()
        page.is_closed.return_value = False
        page.url = "https://chatgpt.com/c/conversation"
        browser = Mock(contexts=[Mock(pages=[page])])
        monitor = BrowserMonitor()

        with (
            patch.object(BrowserMonitor, "_clear_chatgpt_attachments") as clear,
            patch.object(BrowserMonitor, "_paste_screenshot") as paste,
        ):
            monitor.request_replace_attachment(str(id(page)), b"webp bytes")
            monitor._process_attachment_requests(browser, FakePlaywrightError)

        clear.assert_called_once_with(page)
        paste.assert_called_once_with(page, b"webp bytes")

    def test_recording_request_stops_active_browser_playback(self) -> None:
        page = Mock()
        monitor = BrowserMonitor()
        state = _MonitorState(
            active_reading=_ActiveReading(
                page=page,
                full_text="Reply",
                subtitles=("Reply",),
            )
        )
        results: list[tuple[bool, str]] = []
        monitor.reading_finished.connect(
            lambda success, message: results.append((success, message))
        )

        monitor.request_stop_reading()
        monitor._stop_active_reading(state)

        page.evaluate.assert_called_once()
        self.assertIsNone(state.active_reading)
        self.assertEqual(results, [(False, "Playback stopped for recording")])

    def test_recording_cancels_pending_reply_before_it_can_read_aloud(self) -> None:
        monitor = BrowserMonitor()
        state = _MonitorState(
            active_response=_ActiveResponse(page=Mock(), turn_marker_before=None)
        )

        monitor.request_stop_reading()
        monitor._stop_active_reading(state)
        with patch.object(monitor, "_click_read_aloud") as read:
            monitor._poll_active_response(state)

        self.assertIsNone(state.active_response)
        read.assert_not_called()

    def test_dictation_press_clicks_chatgpt_microphone(self) -> None:
        composer = Mock()
        composer.evaluate.return_value = "Existing text"
        microphone = Mock()
        microphone.is_visible.return_value = True

        page = Mock()
        page.is_closed.return_value = False
        page.url = "https://chatgpt.com/c/conversation"

        def locate(selector: str) -> Mock:
            if selector == CHATGPT_COMPOSER_SELECTOR:
                return Mock(first=composer)
            if selector in DICTATION_CANCEL_SELECTORS:
                unavailable = Mock()
                unavailable.is_visible.return_value = False
                return Mock(last=unavailable)
            return Mock(last=microphone)

        page.locator.side_effect = locate
        browser = Mock(contexts=[Mock(pages=[page])])
        monitor = BrowserMonitor()
        results: list[tuple[bool, str]] = []
        monitor.dictation_started.connect(
            lambda success, message: results.append((success, message))
        )

        monitor.request_start_dictation(str(id(page)))
        monitor._process_dictation_requests(browser, FakePlaywrightError)

        page.bring_to_front.assert_not_called()
        microphone.click.assert_called_once_with(timeout=5_000)
        self.assertEqual(
            monitor._dictation_initial_text[str(id(page))],
            "Existing text",
        )
        self.assertEqual(results, [(True, "Browser dictation is listening")])

    def test_start_clears_stale_dictation_before_clicking_microphone(self) -> None:
        cancel = Mock()
        cancel.is_visible.side_effect = [True, False]
        microphone = Mock()
        microphone.is_visible.return_value = True
        end_button = Mock()
        end_button.is_visible.side_effect = [False, True]
        unavailable = Mock()
        unavailable.is_visible.return_value = False
        page = Mock()

        def locate(selector: str) -> Mock:
            if selector == DICTATION_CANCEL_SELECTORS[0]:
                return Mock(last=cancel)
            if selector == 'button[aria-label="Start dictation"]':
                return Mock(last=microphone)
            if selector == 'button[aria-label="Submit dictation"]':
                return Mock(last=end_button)
            return Mock(last=unavailable)

        page.locator.side_effect = locate
        monitor = BrowserMonitor()

        monitor._cancel_existing_dictation(page)
        monitor._start_browser_dictation(page)

        cancel.click.assert_called_once_with(timeout=5_000)
        microphone.click.assert_called_once_with(timeout=5_000)

    def test_cancel_dictation_restores_original_composer_text(self) -> None:
        composer = Mock()
        cancel = Mock()
        cancel.is_visible.side_effect = [True, False]
        unavailable = Mock()
        unavailable.is_visible.return_value = False
        page = Mock()

        def locate(selector: str) -> Mock:
            if selector == CHATGPT_COMPOSER_SELECTOR:
                return Mock(first=composer)
            if selector == DICTATION_CANCEL_SELECTORS[0]:
                return Mock(last=cancel)
            return Mock(last=unavailable)

        page.locator.side_effect = locate
        monitor = BrowserMonitor()

        text = monitor._cancel_browser_dictation(page, "Original text")

        cancel.click.assert_called_once_with(timeout=5_000)
        composer.fill.assert_called_once_with("Original text")
        self.assertEqual(text, "Original text")

    def test_clear_request_empties_background_chatgpt_composer(self) -> None:
        composer = Mock()
        page = Mock()
        page.is_closed.return_value = False
        page.url = "https://chatgpt.com/c/conversation"
        page.locator.return_value = Mock(first=composer)
        browser = Mock(contexts=[Mock(pages=[page])])
        monitor = BrowserMonitor()
        results: list[tuple[bool, str]] = []
        monitor.clear_finished.connect(
            lambda success, message: results.append((success, message))
        )

        monitor.request_clear(str(id(page)))
        monitor._process_clear_requests(browser, FakePlaywrightError)

        composer.wait_for.assert_called_once_with(
            state="visible",
            timeout=5_000,
        )
        composer.fill.assert_called_once_with("")
        page.bring_to_front.assert_not_called()
        self.assertEqual(results, [(True, "Text cleared")])

    def test_dictation_release_clicks_done_and_returns_composer_text(self) -> None:
        composer = Mock()
        composer.evaluate.return_value = "Dictated in ChatGPT"
        done = Mock()
        done.is_visible.return_value = True

        page = Mock()
        page.is_closed.return_value = False
        page.url = "https://chatgpt.com/c/conversation"

        unavailable = Mock()
        unavailable.is_visible.return_value = False

        def locate(selector: str) -> Mock:
            if selector == CHATGPT_COMPOSER_SELECTOR:
                return Mock(first=composer)
            if selector == 'button[aria-label="Submit dictation"]':
                return Mock(last=done)
            return Mock(last=unavailable)

        page.locator.side_effect = locate
        browser = Mock(contexts=[Mock(pages=[page])])
        monitor = BrowserMonitor()
        monitor._dictation_initial_text[str(id(page))] = ""
        results: list[tuple[bool, str, str]] = []
        monitor.dictation_finished.connect(
            lambda success, text, message: results.append(
                (success, text, message)
            )
        )

        monitor.request_finish_dictation(str(id(page)))
        monitor._process_dictation_requests(browser, FakePlaywrightError)

        done.click.assert_called_once_with(timeout=5_000)
        page.locator.assert_any_call(
            'button[aria-label="Submit dictation"]'
        )
        page.wait_for_function.assert_not_called()
        self.assertEqual(
            results,
            [
                (
                    True,
                    "Dictated in ChatGPT",
                    "Dictation copied from ChatGPT",
                )
            ],
        )

    def test_unchanged_dictation_text_finishes_without_timeout_error(self) -> None:
        composer = Mock()
        composer.evaluate.return_value = "Existing text"
        done = Mock()
        done.is_visible.side_effect = [True] + [False] * 50
        unavailable = Mock()
        unavailable.is_visible.return_value = False
        page = Mock()

        def locate(selector: str) -> Mock:
            if selector == CHATGPT_COMPOSER_SELECTOR:
                return Mock(first=composer)
            if selector == 'button[aria-label="Submit dictation"]':
                return Mock(last=done)
            return Mock(last=unavailable)

        page.locator.side_effect = locate
        monitor = BrowserMonitor()

        text = monitor._finish_browser_dictation(page, "Existing text")

        self.assertEqual(text, "Existing text")
        done.click.assert_called_once_with(timeout=5_000)
        composer.wait_for.assert_called_with(
            state="visible",
            timeout=DICTATION_RESULT_TIMEOUT_MS,
        )
        self.assertEqual(
            page.wait_for_timeout.call_count,
            DICTATION_RESULT_POLL_COUNT,
        )

    def test_stopping_monitor_cancels_dictation_wait(self) -> None:
        monitor = BrowserMonitor()
        monitor.request_stop()
        page = Mock()

        text = monitor._finish_browser_dictation(page, "Existing text")

        self.assertEqual(text, "Existing text")
        page.locator.assert_not_called()

    def test_completed_response_is_read_aloud(self) -> None:
        page = Mock()
        monitor = BrowserMonitor()
        response = _ActiveResponse(
            page=page,
            turn_marker_before="previous-turn",
            started_at=90.0,
            last_text="The completed reply",
            last_text_changed_at=97.0,
            completion_candidate_at=97.0,
        )
        state = _MonitorState(
            active_response=response
        )
        snapshot = _ResponseSnapshot(
            has_new_turn=True,
            is_generating=False,
            has_completion_controls=True,
            text="The completed reply",
            status="Finishing reply…",
        )
        changes: list[tuple[str, str]] = []
        finished: list[tuple[bool, str]] = []
        reading_started: list[str] = []
        monitor.response_changed.connect(
            lambda status, text: changes.append((status, text))
        )
        monitor.response_finished.connect(
            lambda success, message: finished.append((success, message))
        )
        monitor.reading_started.connect(reading_started.append)

        with (
            patch.object(monitor, "_response_snapshot", return_value=snapshot),
            patch.object(monitor, "_click_read_aloud", return_value=True) as read,
            patch("live_gpt.browser.time.monotonic", return_value=100.0),
        ):
            monitor._poll_active_response(state)

        read.assert_called_once_with(page)
        self.assertEqual(changes[0], ("Finishing reply…", "The completed reply"))
        self.assertEqual(finished, [])
        self.assertEqual(reading_started, ["Preparing Read aloud…"])
        self.assertIsNone(state.active_response)
        self.assertIsNotNone(state.active_reading)

    def test_completed_response_uses_local_voice_when_selected(self) -> None:
        page = Mock()
        monitor = BrowserMonitor()
        monitor.set_use_browser_voice(False)
        state = _MonitorState(
            active_response=_ActiveResponse(
                page=page,
                turn_marker_before="previous-turn",
                started_at=90.0,
                last_text="Local reply",
                last_text_changed_at=97.0,
                completion_candidate_at=97.0,
            )
        )
        snapshot = _ResponseSnapshot(
            has_new_turn=True,
            is_generating=False,
            has_completion_controls=True,
            text="Local reply",
            status="Finishing reply…",
        )
        requested: list[str] = []
        finished: list[tuple[bool, str]] = []
        monitor.local_voice_requested.connect(requested.append)
        monitor.response_finished.connect(
            lambda success, message: finished.append((success, message))
        )

        with (
            patch.object(monitor, "_response_snapshot", return_value=snapshot),
            patch.object(monitor, "_click_read_aloud") as read,
            patch("live_gpt.browser.time.monotonic", return_value=100.0),
        ):
            monitor._poll_active_response(state)

        read.assert_not_called()
        self.assertEqual(requested, ["Local reply"])
        self.assertEqual(
            finished,
            [(True, "Reply complete · Local voice queued")],
        )
        self.assertIsNone(state.active_response)
        self.assertIsNone(state.active_reading)

    def test_growing_local_response_is_emitted_before_completion(self) -> None:
        monitor = BrowserMonitor()
        monitor.set_use_browser_voice(False)
        state = _MonitorState(
            active_response=_ActiveResponse(
                page=Mock(),
                turn_marker_before="previous-turn",
            )
        )
        snapshot = _ResponseSnapshot(
            has_new_turn=True,
            is_generating=True,
            has_completion_controls=False,
            text="The first sentence. The second is growing",
            status="ChatGPT is responding…",
        )
        updates: list[tuple[str, bool]] = []
        monitor.local_voice_updated.connect(
            lambda text, final: updates.append((text, final))
        )

        with patch.object(monitor, "_response_snapshot", return_value=snapshot):
            monitor._poll_active_response(state)

        self.assertEqual(updates, [(snapshot.text, False)])
        self.assertIsNotNone(state.active_response)

    def test_late_trailing_text_resets_completion_stability_window(self) -> None:
        page = Mock()
        monitor = BrowserMonitor()
        state = _MonitorState(
            active_response=_ActiveResponse(
                page=page,
                turn_marker_before="previous-turn",
                started_at=90.0,
                last_text="Almost complete",
                last_status="Finishing reply…",
                last_text_changed_at=95.0,
                completion_candidate_at=95.0,
            )
        )
        snapshot = _ResponseSnapshot(
            has_new_turn=True,
            is_generating=False,
            has_completion_controls=True,
            text="Almost complete!",
            status="Finishing reply…",
        )

        with (
            patch.object(monitor, "_response_snapshot", return_value=snapshot),
            patch.object(monitor, "_click_read_aloud", return_value=True) as read,
            patch(
                "live_gpt.browser.time.monotonic",
                side_effect=[100.0, 101.0, 102.1],
            ),
        ):
            monitor._poll_active_response(state)
            monitor._poll_active_response(state)
            read.assert_not_called()
            monitor._poll_active_response(state)

        read.assert_called_once_with(page)
        assert state.active_reading is not None
        self.assertEqual(state.active_reading.full_text, "Almost complete!")

    def test_reading_progress_emits_subtitles_and_finishes(self) -> None:
        page = Mock()
        page.evaluate.side_effect = [
            {
                "playCount": 1,
                "currentTime": 0,
                "duration": 10,
                "paused": False,
                "ended": False,
            },
            {
                "playCount": 1,
                "currentTime": 6,
                "duration": 10,
                "paused": False,
                "ended": False,
            },
            {
                "playCount": 1,
                "currentTime": 10,
                "duration": 10,
                "paused": True,
                "ended": True,
            },
        ]
        monitor = BrowserMonitor()
        state = _MonitorState(
            active_reading=_ActiveReading(
                page=page,
                full_text="First subtitle. Second subtitle.",
                subtitles=("First subtitle.", "Second subtitle."),
            )
        )
        subtitles: list[object] = []
        finished: list[tuple[bool, str]] = []
        monitor.reading_changed.connect(subtitles.append)
        monitor.reading_finished.connect(
            lambda success, message: finished.append((success, message))
        )

        monitor._poll_active_reading(state)
        monitor._poll_active_reading(state)
        assert state.active_reading is not None
        state.active_reading.quiet_since = time.monotonic() - 5
        monitor._poll_active_reading(state)

        self.assertEqual(
            subtitles,
            [
                {
                    "text": "First subtitle. Second subtitle.",
                    "fraction": 0.0,
                },
                {
                    "text": "First subtitle. Second subtitle.",
                    "fraction": 0.6,
                },
                {
                    "text": "First subtitle. Second subtitle.",
                    "fraction": 1.0,
                },
            ],
        )
        self.assertEqual(finished, [(True, "Read aloud complete")])
        self.assertIsNone(state.active_reading)

    def test_subtitles_wait_for_playback_to_begin(self) -> None:
        page = Mock()
        page.evaluate.return_value = None
        monitor = BrowserMonitor()
        state = _MonitorState(
            active_reading=_ActiveReading(
                page=page,
                full_text="The reply has not started playing yet.",
                subtitles=("The reply has not started playing yet.",),
            )
        )
        subtitles: list[str] = []
        monitor.reading_changed.connect(subtitles.append)

        monitor._poll_active_reading(state)

        self.assertEqual(subtitles, [])
        self.assertIsNotNone(state.active_reading)

    def test_subtitle_rolls_forward_one_line_at_a_time(self) -> None:
        text = " ".join(f"word{index}" for index in range(40)) + "."

        lines = BrowserMonitor._subtitle_segments(text)
        weights = [max(len(line), 12) for line in lines]
        first = BrowserMonitor._subtitle_at_progress(lines, 0.0)
        after_first_line = BrowserMonitor._subtitle_at_progress(
            lines,
            (weights[0] + 0.1) / sum(weights),
        )

        first_lines = first.splitlines()
        next_lines = after_first_line.splitlines()
        self.assertGreaterEqual(len(first_lines), 2)
        self.assertGreaterEqual(len(next_lines), 2)
        self.assertEqual(first_lines[1], next_lines[0])
        self.assertLessEqual(max(map(len, lines)), 54)

    def test_read_aloud_uses_more_actions_menu(self) -> None:
        turn = Mock()
        turn.get_attribute.side_effect = (
            lambda attribute: "turn-id"
            if attribute == "data-turn-id"
            else "conversation-turn-8"
        )
        turns = Mock(last=turn)
        turns.count.return_value = 1
        direct_button = Mock()
        direct_buttons = Mock(last=direct_button)
        direct_buttons.count.return_value = 0
        more_actions = Mock()
        more_buttons = Mock(last=more_actions)
        more_buttons.count.return_value = 1
        read_aloud = Mock()

        def locate_in_turn(selector: str) -> Mock:
            if "voice-play-turn-action-button" in selector:
                return direct_buttons
            return more_buttons

        turn.locator.side_effect = locate_in_turn

        page = Mock()

        def locate(selector: str) -> Mock:
            if selector == ASSISTANT_TURN_SELECTOR:
                return turns
            return Mock(last=read_aloud)

        page.locator.side_effect = locate
        direct_button.is_visible.return_value = False

        clicked = BrowserMonitor._click_read_aloud(page)

        self.assertTrue(clicked)
        turn.hover.assert_not_called()
        more_actions.click.assert_called_once_with(timeout=3_000, force=True)
        read_aloud.click.assert_called_once_with(timeout=3_000, force=True)

    def test_read_aloud_uses_dom_click_when_pointer_click_times_out(self) -> None:
        direct_button = Mock()
        direct_button.is_visible.return_value = True
        direct_button.click.side_effect = RuntimeError("pointer timeout")
        direct_buttons = Mock(last=direct_button)
        direct_buttons.count.return_value = 1
        turn = Mock()
        turn.get_attribute.return_value = "turn-id"
        turn.locator.return_value = direct_buttons
        turns = Mock(last=turn)
        turns.count.return_value = 1
        page = Mock()
        page.locator.return_value = turns

        clicked = BrowserMonitor._click_read_aloud(page)

        self.assertTrue(clicked)
        direct_button.evaluate.assert_called_once_with(
            "element => element.click()"
        )

    def test_response_snapshot_reads_current_assistant_section(self) -> None:
        markdown = Mock()
        markdown.count.return_value = 1
        markdown.evaluate.return_value = "Visible assistant reply"
        markdown.inner_text.return_value = "Visible assistant reply"
        completion_controls = Mock()
        completion_controls.count.return_value = 0

        turn = Mock()
        turn.inner_text.return_value = "Visible assistant reply"
        turn.get_attribute.side_effect = (
            lambda attribute, **kwargs: "new-turn"
            if attribute == "data-turn-id"
            else "conversation-turn-6"
        )

        def locate_in_turn(selector: str) -> Mock:
            if selector.startswith(".markdown"):
                return Mock(last=markdown)
            return completion_controls

        turn.locator.side_effect = locate_in_turn
        turns = Mock(last=turn)
        turns.count.return_value = 4
        stop_button = Mock()
        stop_button.is_visible.return_value = True

        page = Mock()

        def locate_in_page(selector: str) -> Mock:
            if selector.startswith('[data-testid^="conversation-turn-"]'):
                return turns
            return Mock(first=stop_button)

        page.locator.side_effect = locate_in_page

        snapshot = BrowserMonitor._response_snapshot(page, "previous-turn")

        self.assertTrue(snapshot.has_new_turn)
        self.assertTrue(snapshot.is_generating)
        self.assertEqual(snapshot.text, "Visible assistant reply")
        self.assertEqual(snapshot.status, "ChatGPT is responding…")


class BrowserWindowsTests(unittest.TestCase):
    @patch("live_gpt.browser_windows._run_powershell")
    def test_navigation_has_keyboard_new_tab_fallback(
        self,
        run_powershell: Mock,
    ) -> None:
        run_powershell.return_value = "Navigated"

        browser_windows._navigate_browser_window(
            1234,
            "edge://inspect/#remote-debugging",
            create_new_tab=True,
        )

        script = run_powershell.call_args.args[0]
        self.assertIn("0x54", script)
        self.assertNotIn("New Tab button not found", script)
        self.assertIn("AttachThreadInput", script)
        self.assertEqual(
            script.count("[LiveGptWindowMessages]::ActivateWindow($window)"),
            2,
        )

    @patch("live_gpt.browser_windows._wait_for_remote_debugging_marker")
    @patch("live_gpt.browser_windows._enable_remote_debugging")
    @patch("live_gpt.browser_windows._navigate_browser_window")
    @patch("live_gpt.browser_windows._windows_browser_windows")
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
        browser_window.return_value = [(default_executable.return_value, 1234)]
        wait_for_marker.return_value = "ws://127.0.0.1:9222/devtools/browser/id"

        url = browser_windows.open_remote_debugging_settings()

        self.assertEqual(url, "edge://inspect/#remote-debugging")
        navigate.assert_called_once_with(
            1234,
            "edge://inspect/#remote-debugging",
            create_new_tab=True,
        )
        enable_debugging.assert_called_once_with(1234)

    @patch("live_gpt.browser_windows._wait_for_remote_debugging_marker", return_value="ws://127.0.0.1:9222/devtools/browser/id")
    @patch("live_gpt.browser_windows._enable_remote_debugging")
    @patch("live_gpt.browser_windows._navigate_browser_window")
    @patch("live_gpt.browser_windows._windows_browser_windows")
    @patch("live_gpt.browser_windows.windows_default_browser_executable")
    @patch("live_gpt.browser_windows.subprocess.Popen")
    def test_only_open_chrome_is_used_even_when_edge_is_default(
        self, launch, default, windows, navigate, enable, marker,
    ):
        default.return_value = Path("C:/Edge/msedge.exe")
        chrome = Path("C:/Chrome/chrome.exe")
        windows.return_value = [(chrome, 1234), (chrome, 5678)]
        self.assertEqual(browser_windows.open_remote_debugging_settings(),
                         "chrome://inspect/#remote-debugging")
        navigate.assert_called_once_with(1234, "chrome://inspect/#remote-debugging", create_new_tab=True)
        enable.assert_called_once_with(1234)
        marker.assert_called_once_with(chrome)
        launch.assert_not_called()

    @patch("live_gpt.browser_windows._wait_for_remote_debugging_marker", return_value="ws://127.0.0.1:9222/devtools/browser/id")
    @patch("live_gpt.browser_windows._enable_remote_debugging")
    @patch("live_gpt.browser_windows._navigate_browser_window")
    @patch("live_gpt.browser_windows._windows_browser_windows")
    @patch("live_gpt.browser_windows.windows_default_browser_executable")
    def test_open_default_is_preferred_when_both_browsers_are_open(
        self, default, windows, navigate, enable, marker,
    ):
        edge = Path("C:/Edge/msedge.exe")
        default.return_value = edge
        windows.return_value = [(Path("C:/Chrome/chrome.exe"), 1234), (edge, 5678)]
        self.assertEqual(browser_windows.open_remote_debugging_settings(),
                         "edge://inspect/#remote-debugging")
        navigate.assert_called_once_with(5678, "edge://inspect/#remote-debugging", create_new_tab=True)
        enable.assert_called_once_with(5678)


if __name__ == "__main__":
    unittest.main()
