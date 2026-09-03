from __future__ import annotations

import time
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from live_gpt.browser import (
    CHATGPT_COMPOSER_SELECTOR,
    DICTATION_CANCEL_SELECTORS,
    DICTATION_RESULT_POLL_COUNT,
    DICTATION_RESULT_TIMEOUT_MS,
    REMOTE_DEBUGGING_RETRY_INTERVAL_SECONDS,
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
    def test_timed_out_endpoint_is_retried_automatically(
        self,
        discover_endpoint: Mock,
    ) -> None:
        discover_endpoint.return_value = "http://127.0.0.1:9222"
        chromium = Mock()
        chromium.connect_over_cdp.side_effect = FakePlaywrightError("declined")
        playwright = Mock(chromium=chromium)
        monitor = BrowserMonitor()
        state = _MonitorState()

        with patch("live_gpt.browser.time.monotonic") as monotonic:
            monotonic.return_value = 100.0
            monitor._try_connect(playwright, FakePlaywrightError, state)
            monitor._try_connect(playwright, FakePlaywrightError, state)
            self.assertEqual(chromium.connect_over_cdp.call_count, 1)

            monotonic.return_value = (
                100.0 + REMOTE_DEBUGGING_RETRY_INTERVAL_SECONDS
            )
            monitor._try_connect(playwright, FakePlaywrightError, state)
        self.assertEqual(chromium.connect_over_cdp.call_count, 2)

    @patch("live_gpt.browser.discover_cdp_endpoint")
    def test_requested_retry_does_not_wait_for_interval(
        self,
        discover_endpoint: Mock,
    ) -> None:
        discover_endpoint.return_value = "http://127.0.0.1:9222"
        chromium = Mock()
        chromium.connect_over_cdp.side_effect = FakePlaywrightError("declined")
        playwright = Mock(chromium=chromium)
        monitor = BrowserMonitor()
        state = _MonitorState()

        with patch("live_gpt.browser.time.monotonic", return_value=100.0):
            monitor._try_connect(playwright, FakePlaywrightError, state)
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
            if selector == (
                '[data-testid^="conversation-turn-"]'
                '[data-turn="assistant"]'
            ):
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
            if selector == (
                '[data-testid^="conversation-turn-"]'
                '[data-turn="assistant"]'
            ):
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
