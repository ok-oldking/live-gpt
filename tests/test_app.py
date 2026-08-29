from __future__ import annotations

import os
import time
import unittest
from unittest.mock import Mock, patch


os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QPoint, QRect, Qt  # noqa: E402
from PySide6.QtGui import QPalette  # noqa: E402
from PySide6.QtTest import QTest  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from live_gpt.app import (  # noqa: E402
    OverlayWindow,
    TranscriptEditor,
    TrayController,
)
from live_gpt.screen_capture import CaptureSource  # noqa: E402


class TranscriptEditorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.application = QApplication.instance() or QApplication([])

    def setUp(self) -> None:
        self.editor = TranscriptEditor()
        self.editor.show()
        self.editor.setFocus()

    def tearDown(self) -> None:
        self.editor.close()

    def test_enter_clicks_send(self) -> None:
        sent: list[bool] = []
        self.editor.send_button.clicked.connect(lambda: sent.append(True))
        self.editor.setPlainText("Send this")

        QTest.keyClick(self.editor, Qt.Key.Key_Return)

        self.assertEqual(sent, [True])
        self.assertEqual(self.editor.toPlainText(), "Send this")

    def test_shift_enter_inserts_newline(self) -> None:
        sent: list[bool] = []
        self.editor.send_button.clicked.connect(lambda: sent.append(True))
        self.editor.setPlainText("First line")
        cursor = self.editor.textCursor()
        cursor.movePosition(cursor.MoveOperation.End)
        self.editor.setTextCursor(cursor)

        QTest.keyClick(
            self.editor,
            Qt.Key.Key_Return,
            Qt.KeyboardModifier.ShiftModifier,
        )

        self.assertEqual(sent, [])
        self.assertEqual(self.editor.toPlainText(), "First line\n")

    def test_reading_restores_full_response(self) -> None:
        full_response = "First sentence. Second sentence. Final sentence."
        self.editor.begin_response()
        self.editor.update_response(full_response)

        self.editor.begin_reading()

        self.assertEqual(self.editor.toPlainText(), "")

        self.editor.finish_reading()

        self.assertEqual(self.editor.toPlainText(), full_response)


class TrayControllerBrowserTests(unittest.TestCase):
    def test_overlay_is_fifty_percent_wider(self) -> None:
        window = OverlayWindow()
        try:
            self.assertEqual(window.width(), 1140)
            self.assertEqual(window.height(), 240)
            self.assertFalse(window.configure_button.icon().isNull())
            window.show()
            QApplication.processEvents()
            self.assertLess(
                window.configure_button.geometry().left(),
                window.lock_button.geometry().left(),
            )
            self.assertLess(
                window.lock_button.geometry().left(),
                window.hide_button.geometry().left(),
            )
        finally:
            window.close()

    def test_overlay_can_resize_and_lock_its_geometry(self) -> None:
        window = OverlayWindow()
        try:
            window.show()
            QApplication.processEvents()

            window.resize(1280, 320)
            QApplication.processEvents()
            self.assertEqual(window.size().width(), 1280)
            self.assertEqual(window.size().height(), 320)

            window.lock_button.click()
            locked_size = window.size()
            window.resize(900, 200)
            QApplication.processEvents()

            self.assertTrue(window.lock_button.isChecked())
            self.assertEqual(window.size(), locked_size)

            window.lock_button.click()
            window.resize(900, 200)
            QApplication.processEvents()

            self.assertFalse(window.lock_button.isChecked())
            self.assertEqual(window.size().width(), 900)
            self.assertEqual(window.size().height(), 200)
        finally:
            window.close()

    def test_every_overlay_border_and_corner_is_resizable(self) -> None:
        window = OverlayWindow()
        try:
            window.show()
            window.setGeometry(100, 100, 1000, 300)
            QApplication.processEvents()
            width = window._resize_surface.width()
            height = window._resize_surface.height()
            self.assertEqual(
                window._resize_edges_at(QPoint(1, height // 2)),
                Qt.Edge.LeftEdge,
            )
            self.assertEqual(
                window._resize_edges_at(QPoint(15, height // 2)),
                Qt.Edge.LeftEdge,
            )
            self.assertEqual(
                window._resize_edges_at(QPoint(width - 2, height // 2)),
                Qt.Edge.RightEdge,
            )
            self.assertEqual(
                window._resize_edges_at(QPoint(width // 2, 1)),
                Qt.Edge.TopEdge,
            )
            self.assertEqual(
                window._resize_edges_at(QPoint(width // 2, height - 2)),
                Qt.Edge.BottomEdge,
            )
            self.assertEqual(
                window._resize_edges_at(QPoint(1, 1)),
                Qt.Edge.TopEdge | Qt.Edge.LeftEdge,
            )

            window._begin_border_resize(
                Qt.Edge.TopEdge | Qt.Edge.LeftEdge,
                QPoint(100, 100),
            )
            window._update_border_resize(QPoint(80, 70))

            self.assertEqual(window.geometry(), QRect(80, 70, 1020, 330))
            window._end_border_resize()
        finally:
            window.close()

    def test_chrome_is_transparent_outside_and_visible_while_resizing(self) -> None:
        window = OverlayWindow()
        try:
            self.assertEqual(window._title_opacity.opacity(), 0.0)
            self.assertEqual(window._microphone_opacity.opacity(), 0.0)
            self.assertFalse(window.panel.property("chromeVisible"))
            self.assertFalse(
                window._resize_surface.property("chromeVisible")
            )

            window._set_chrome_visible(True)

            self.assertEqual(window._title_opacity.opacity(), 1.0)
            self.assertEqual(window._microphone_opacity.opacity(), 1.0)
            self.assertTrue(window.panel.property("chromeVisible"))
            self.assertTrue(
                window._resize_surface.property("chromeVisible")
            )
            window.show()
            QApplication.processEvents()
            corner = window.grab().toImage().pixelColor(1, 1)
            self.assertEqual(corner.alpha(), 0)

            window._begin_border_resize(Qt.Edge.LeftEdge, QPoint(0, 0))
            window._set_chrome_visible(False)

            self.assertEqual(window._title_opacity.opacity(), 1.0)
            self.assertTrue(window.panel.property("chromeVisible"))
        finally:
            window.close()

    def test_status_uses_input_hint_and_errors_are_red(self) -> None:
        window = OverlayWindow()
        try:
            self.assertFalse(hasattr(window, "status_label"))

            window.set_status("Connection failed", error=True)

            self.assertEqual(
                window.transcript_area.placeholderText(),
                "Connection failed",
            )
            self.assertEqual(
                window.transcript_area.palette()
                .color(QPalette.ColorRole.PlaceholderText)
                .name(),
                "#ff667a",
            )
        finally:
            window.close()

    @patch("live_gpt.app.capture_webp", return_value=b"screenshot")
    def test_send_captures_selected_source(self, capture: Mock) -> None:
        source = CaptureSource(
            key="window:123",
            label="Test window",
            kind="window",
            left=0,
            top=0,
            width=1200,
            height=800,
            hwnd=123,
        )
        controller = TrayController.__new__(TrayController)
        controller.window = Mock()
        controller.browser_monitor = Mock()
        controller.selected_chatgpt_tab_id = "selected-tab"

        controller._handle_send_requested("Explain this", source)

        capture.assert_called_once_with(source)
        controller.browser_monitor.request_send.assert_called_once_with(
            "selected-tab",
            "Explain this",
            b"screenshot",
        )
        controller.window.begin_response_display.assert_called_once_with(
            "Explain this"
        )

    def test_clear_queues_selected_browser_composer(self) -> None:
        controller = TrayController.__new__(TrayController)
        controller.window = Mock()
        controller.browser_monitor = Mock()
        controller.selected_chatgpt_tab_id = "selected-tab"

        controller._handle_clear_requested()

        controller.browser_monitor.request_clear.assert_called_once_with(
            "selected-tab"
        )
        controller.window.set_status.assert_called_once_with(
            "Clearing ChatGPT input…"
        )

    def test_reading_uses_two_dedicated_subtitle_labels(self) -> None:
        window = OverlayWindow()
        try:
            window.transcript_area.begin_response()
            window.transcript_area.update_response("The complete response")

            window.begin_reading("Preparing Read aloud…")
            window.set_reading_subtitle("Current subtitle\nNext subtitle")

            self.assertTrue(window.transcript_area.isHidden())
            self.assertFalse(window.subtitle_panel.isHidden())
            self.assertEqual(
                " ".join(
                    filter(
                        None,
                        (
                            window.subtitle_line_one.text(),
                            window.subtitle_line_two.text(),
                        ),
                    )
                ),
                "Current subtitle Next subtitle",
            )

            window.finish_reading(True, "Read aloud complete")

            self.assertTrue(window.transcript_area.isHidden())
            self.assertFalse(window.subtitle_panel.isHidden())

            window.dismiss_subtitle_mode()

            self.assertFalse(window.transcript_area.isHidden())
            self.assertTrue(window.subtitle_panel.isHidden())
            self.assertEqual(window.transcript_area.toPlainText(), "")
        finally:
            window.close()

    def test_subtitle_hover_expands_full_text_and_click_dismisses(self) -> None:
        window = OverlayWindow()
        try:
            window.show()
            window.begin_response_display("Question")
            response = " ".join(
                f"complete-response-word-{index}" for index in range(80)
            )
            window.set_response_update("Writing…", response)
            QApplication.processEvents()
            collapsed_height = window.height()

            window._expand_subtitle()
            QApplication.processEvents()

            self.assertTrue(window.subtitle_line_one.isHidden())
            self.assertFalse(window.subtitle_full_text.isHidden())
            self.assertEqual(
                window.subtitle_full_text.toPlainText(),
                response,
            )
            self.assertGreater(window.height(), collapsed_height)

            QTest.mouseClick(
                window.subtitle_full_text.viewport(),
                Qt.MouseButton.LeftButton,
            )
            QApplication.processEvents()

            self.assertTrue(window.subtitle_panel.isHidden())
            self.assertFalse(window.transcript_area.isHidden())
            self.assertFalse(window.transcript_area.is_showing_response)
        finally:
            window.close()

    def test_subtitle_lines_use_the_available_panel_width(self) -> None:
        window = OverlayWindow()
        try:
            window.show()
            window.begin_reading("Reading aloud…")
            QApplication.processEvents()
            subtitle = " ".join(
                f"subtitle-word-{index}" for index in range(60)
            )

            window.set_reading_subtitle(subtitle)

            available = window.subtitle_panel.width() - 32
            metrics = window.subtitle_line_one.fontMetrics()
            first_width = metrics.horizontalAdvance(
                window.subtitle_line_one.text()
            )
            second_width = metrics.horizontalAdvance(
                window.subtitle_line_two.text()
            )
            self.assertGreater(first_width, available * 0.7)
            self.assertGreater(second_width, available * 0.7)
            self.assertLessEqual(first_width, available)
            self.assertLessEqual(second_width, available)
        finally:
            window.close()

    def test_subtitle_progress_rolls_exactly_one_rendered_line(self) -> None:
        window = OverlayWindow()
        try:
            window.show()
            window.begin_reading("Reading aloud…")
            QApplication.processEvents()
            text = " ".join(f"spoken-word-{index}" for index in range(80))

            window.set_reading_subtitle({"text": text, "fraction": 0.0})
            previous_second_line = window.subtitle_line_two.text()
            lines = window._subtitle_lines()
            weights = [max(len(line), 12) for line in lines]
            after_first_line = (weights[0] + 0.1) / sum(weights)

            window.set_reading_subtitle(
                {"text": text, "fraction": after_first_line}
            )

            self.assertTrue(previous_second_line)
            self.assertEqual(
                window.subtitle_line_one.text(),
                previous_second_line,
            )
            self.assertEqual(window._subtitle_line_index, 1)
        finally:
            window.close()

    def test_capture_selector_keeps_no_screenshot_first(self) -> None:
        window = OverlayWindow()
        source = CaptureSource(
            "display:1",
            "Screenshot desktop",
            "display",
            0,
            0,
            1920,
            1080,
        )
        try:
            window.set_capture_sources([source])

            self.assertEqual(window.capture_source_combo.itemText(0), "No screenshot")
            self.assertIsNone(window.capture_source_combo.itemData(0))
            self.assertEqual(window.capture_source_combo.itemData(1), source)
        finally:
            window.close()

    def test_selected_screenshot_adds_text_only_send_action(self) -> None:
        window = OverlayWindow()
        source = CaptureSource(
            "display:1",
            "Screenshot desktop",
            "display",
            0,
            0,
            1920,
            1080,
        )
        requests: list[tuple[str, object]] = []
        window.send_requested.connect(
            lambda text, selected: requests.append((text, selected))
        )
        try:
            window.set_chatgpt_tabs(
                [{"id": "tab", "title": "ChatGPT", "url": "https://chatgpt.com"}]
            )
            window.set_capture_sources([source])
            window.capture_source_combo.setCurrentIndex(1)
            window.set_transcript("Explain this")

            self.assertFalse(
                window.send_without_screenshot_button.isHidden()
            )
            window.send_without_screenshot_button.click()

            self.assertEqual(requests, [("Explain this", None)])
        finally:
            window.close()

    def test_normal_send_keeps_selected_screenshot(self) -> None:
        window = OverlayWindow()
        source = CaptureSource(
            "display:1",
            "Screenshot desktop",
            "display",
            0,
            0,
            1920,
            1080,
        )
        requests: list[tuple[str, object]] = []
        window.send_requested.connect(
            lambda text, selected: requests.append((text, selected))
        )
        try:
            window.set_chatgpt_tabs(
                [{"id": "tab", "title": "ChatGPT", "url": "https://chatgpt.com"}]
            )
            window.set_capture_sources([source])
            window.capture_source_combo.setCurrentIndex(1)
            window.set_transcript("Explain this")

            window.send_button.click()

            self.assertEqual(requests, [("Explain this", source)])
        finally:
            window.close()

    def test_overlay_send_restores_focus_but_hotkey_send_does_not(self) -> None:
        window = OverlayWindow()
        focus_restorer = Mock()
        focus_restorer.restore_previous.return_value = True
        window._focus_restorer = focus_restorer
        try:
            window.set_chatgpt_tabs(
                [{"id": "tab", "title": "ChatGPT", "url": "https://chatgpt.com"}]
            )
            window.set_transcript("Send from overlay")

            window.send_button.click()
            QApplication.processEvents()

            focus_restorer.restore_previous.assert_called_once_with()

            focus_restorer.reset_mock()
            window.set_transcript("Send from global hotkey")
            window.request_send_from_hotkey(True)
            QApplication.processEvents()

            focus_restorer.restore_previous.assert_not_called()
        finally:
            window.close()

    def test_dictation_states_replace_input_area(self) -> None:
        window = OverlayWindow()
        try:
            window.begin_dictation_waiting()

            self.assertTrue(window.transcript_area.isHidden())
            self.assertFalse(window.dictation_panel.isHidden())
            self.assertEqual(
                window.dictation_state_label.text(),
                "Waiting for the browser to start listening…",
            )

            window.set_dictation_listening()
            self.assertEqual(window.dictation_state_label.text(), "Listening…")

            window.end_dictation_display()
            self.assertFalse(window.transcript_area.isHidden())
            self.assertTrue(window.dictation_panel.isHidden())
        finally:
            window.close()

    def test_microphone_press_and_release_queue_browser_dictation(self) -> None:
        controller = TrayController.__new__(TrayController)
        controller.window = Mock()
        controller.window.transcript_area.is_showing_response = False
        controller.browser_monitor = Mock()
        controller.selected_chatgpt_tab_id = "selected-tab"
        controller.dictation_tab_id = None

        controller.start_dictation()
        controller._on_dictation_started(
            True,
            "Browser dictation is listening",
        )
        controller._dictation_listening_since = time.monotonic() - 0.6
        controller.finish_dictation()

        controller.browser_monitor.request_start_dictation.assert_called_once_with(
            "selected-tab"
        )
        controller.window.dismiss_subtitle_mode.assert_called_once_with()
        controller.browser_monitor.request_finish_dictation.assert_called_once_with(
            "selected-tab"
        )
        controller.window.begin_dictation_waiting.assert_called_once_with()
        controller.window.set_dictation_listening.assert_called_once_with()
        controller.window.set_dictation_finishing.assert_called_once_with()
        self.assertEqual(controller._dictation_state, "finishing")

    def test_short_mouse_dictation_is_cancelled(self) -> None:
        controller = TrayController.__new__(TrayController)
        controller.window = Mock()
        controller.window.transcript_area.is_showing_response = False
        controller.browser_monitor = Mock()
        controller.selected_chatgpt_tab_id = "selected-tab"
        controller.dictation_tab_id = None
        controller._dictation_state = "idle"
        controller._dictation_input_held = False
        controller._dictation_listening_since = None

        controller.start_dictation()
        controller._on_dictation_started(True, "Listening")
        controller.finish_dictation()

        controller.browser_monitor.request_cancel_dictation.assert_called_once_with(
            "selected-tab"
        )
        controller.browser_monitor.request_finish_dictation.assert_not_called()
        self.assertEqual(controller._dictation_state, "cancelling")

    def test_release_before_browser_listens_cancels_when_ready(self) -> None:
        controller = TrayController.__new__(TrayController)
        controller.window = Mock()
        controller.window.transcript_area.is_showing_response = False
        controller.browser_monitor = Mock()
        controller.selected_chatgpt_tab_id = "selected-tab"
        controller.dictation_tab_id = None
        controller._dictation_state = "idle"
        controller._dictation_input_held = False
        controller._dictation_listening_since = None

        controller.start_dictation()
        controller.finish_dictation()
        controller._on_dictation_started(True, "Listening")

        controller.browser_monitor.request_cancel_dictation.assert_called_once_with(
            "selected-tab"
        )

    def test_press_during_cancel_restarts_only_if_still_held(self) -> None:
        controller = TrayController.__new__(TrayController)
        controller.window = Mock()
        controller.window.transcript_area.is_showing_response = False
        controller.browser_monitor = Mock()
        controller.selected_chatgpt_tab_id = "selected-tab"
        controller.dictation_tab_id = None
        controller._dictation_state = "idle"
        controller._dictation_input_held = False
        controller._dictation_listening_since = None

        controller.start_dictation()
        controller._on_dictation_started(True, "Listening")
        controller.finish_dictation()
        controller.start_dictation()
        controller._on_dictation_finished(
            True,
            "Original text",
            "Dictation cancelled",
        )

        self.assertEqual(
            controller.browser_monitor.request_start_dictation.call_count,
            2,
        )
        self.assertEqual(controller._dictation_state, "starting")

    def test_input_and_microphone_require_a_chatgpt_window(self) -> None:
        window = OverlayWindow()
        try:
            self.assertFalse(window.transcript_area.isEnabled())
            self.assertFalse(window.microphone_button.isEnabled())

            window.set_chatgpt_tabs(
                [{"id": "tab", "title": "ChatGPT", "url": "https://chatgpt.com"}]
            )

            self.assertTrue(window.transcript_area.isEnabled())
            self.assertTrue(window.microphone_button.isEnabled())

            window.set_chatgpt_tabs([])

            self.assertFalse(window.transcript_area.isEnabled())
            self.assertFalse(window.microphone_button.isEnabled())
        finally:
            window.close()

    def test_finished_dictation_populates_app_input(self) -> None:
        controller = TrayController.__new__(TrayController)
        controller.window = Mock()

        controller._on_dictation_finished(
            True,
            "Text recognized by ChatGPT",
            "Dictation copied from ChatGPT",
        )

        controller.window.set_transcript.assert_called_once_with(
            "Text recognized by ChatGPT"
        )
        controller.window.end_dictation_display.assert_called_once_with()
        controller.window.set_microphone_state.assert_called_once_with(
            "saved",
            "Dictation copied from ChatGPT",
        )

    @patch("live_gpt.app.open_remote_debugging_settings")
    @patch("live_gpt.app.discover_cdp_endpoint")
    def test_live_endpoint_retries_without_opening_settings(
        self,
        discover_endpoint: Mock,
        open_settings: Mock,
    ) -> None:
        discover_endpoint.return_value = "ws://127.0.0.1:9222/devtools/id"
        controller = TrayController.__new__(TrayController)
        controller.window = Mock()
        controller.browser_monitor = Mock()

        controller._open_remote_debugging_settings()

        open_settings.assert_not_called()
        controller.browser_monitor.request_retry_connection.assert_called_once()
        controller.window.set_browser_status.assert_called_once_with(
            "Retrying connection… approve it in the browser"
        )


if __name__ == "__main__":
    unittest.main()
