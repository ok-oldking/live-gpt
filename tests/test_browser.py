from __future__ import annotations

import time
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from live_gpt.browser import (
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

    def test_send_request_targets_selected_chatgpt_page(self) -> None:
        composer = Mock()
        composer_locator = Mock(first=composer)
        send_button = Mock()
        send_button_locator = Mock(first=send_button)
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
            if selector == "#prompt-textarea":
                return composer_locator
            if selector == (
                '[data-testid^="conversation-turn-"]'
                '[data-turn="assistant"]'
            ):
                return assistant_turns
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
        send_button.click.assert_called_once_with(timeout=5_000)
        self.assertEqual(
            results,
            [(True, "Hello from Live GPT", "Sent to ChatGPT")],
        )
        self.assertIsNotNone(state.active_response)

    def test_dictation_press_clicks_chatgpt_microphone(self) -> None:
        composer = Mock()
        composer.evaluate.return_value = "Existing text"
        microphone = Mock()
        microphone.is_visible.return_value = True

        page = Mock()
        page.is_closed.return_value = False
        page.url = "https://chatgpt.com/c/conversation"

        def locate(selector: str) -> Mock:
            if selector == "#prompt-textarea":
                return Mock(first=composer)
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
            if selector == "#prompt-textarea":
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
        page.wait_for_function.assert_called_once()
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

    def test_completed_response_is_read_aloud(self) -> None:
        page = Mock()
        monitor = BrowserMonitor()
        state = _MonitorState(
            active_response=_ActiveResponse(
                page=page,
                turn_marker_before="previous-turn",
                started_at=time.monotonic() - 2,
            )
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
        ):
            monitor._poll_active_response(state)
            monitor._poll_active_response(state)
            monitor._poll_active_response(state)

        read.assert_called_once_with(page)
        self.assertEqual(changes[0], ("Finishing reply…", "The completed reply"))
        self.assertEqual(finished, [])
        self.assertEqual(reading_started, ["Preparing Read aloud…"])
        self.assertIsNone(state.active_response)
        self.assertIsNotNone(state.active_reading)

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
        subtitles: list[str] = []
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
            ["First subtitle.\nSecond subtitle.", "Second subtitle."],
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
        self.assertEqual(len(first_lines), 2)
        self.assertEqual(len(next_lines), 2)
        self.assertEqual(first_lines[1], next_lines[0])
        self.assertLessEqual(max(map(len, lines)), 54)

    def test_read_aloud_uses_more_actions_menu(self) -> None:
        turn = Mock()
        turns = Mock(last=turn)
        turns.count.return_value = 1
        direct_button = Mock()
        more_actions = Mock()
        read_aloud = Mock()

        page = Mock()

        def locate(selector: str) -> Mock:
            if selector == (
                '[data-testid^="conversation-turn-"]'
                '[data-turn="assistant"]'
            ):
                return turns
            if "voice-play-turn-action-button" in selector:
                return Mock(last=direct_button)
            if selector == 'button[aria-label="More actions"]':
                return Mock(last=more_actions)
            return Mock(last=read_aloud)

        page.locator.side_effect = locate
        direct_button.is_visible.return_value = False

        clicked = BrowserMonitor._click_read_aloud(page)

        self.assertTrue(clicked)
        turn.hover.assert_called_once_with(timeout=2_000)
        more_actions.click.assert_called_once_with(timeout=5_000)
        read_aloud.click.assert_called_once_with(timeout=5_000)

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
            lambda attribute: "new-turn"
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
