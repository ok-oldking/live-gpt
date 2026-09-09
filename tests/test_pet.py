from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QEvent, QPoint, QPointF, QRect, Qt
from PySide6.QtGui import QMouseEvent
from PySide6.QtWidgets import QApplication

from live_gpt.app import OverlayWindow
from live_gpt.app import HotkeyConfigDialog
from live_gpt.config import Config
from PySide6.QtGui import QKeySequence


class PetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.window = OverlayWindow()
        self.window.show()
        QApplication.processEvents()
        self.pet = self.window.pet
        self.pet._timer.stop()

    def tearDown(self):
        self.window.hide()
        self.window.deleteLater()

    def mouse(self, widget, kind, global_position, button=Qt.MouseButton.LeftButton):
        local = widget.mapFromGlobal(global_position)
        event = QMouseEvent(
            kind, QPointF(local), QPointF(global_position), button,
            Qt.MouseButton.NoButton if kind == QEvent.Type.MouseButtonRelease else Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier,
        )
        QApplication.sendEvent(widget, event)

    def test_activity_and_error_transitions(self):
        w = self.window
        self.assertEqual(self.pet.state, "idle")
        w.begin_dictation_waiting()
        self.assertEqual(self.pet.state, "waiting")
        w.set_dictation_listening()
        self.assertEqual(self.pet.state, "waiting")
        w.end_dictation_display()
        self.assertEqual(self.pet.state, "idle")
        w.begin_response_display("hello")
        self.assertEqual(self.pet.state, "running")
        w.begin_reading("Playing")
        self.assertEqual(self.pet.state, "review")
        w.finish_reading(True, "Audio caught up")
        self.assertEqual(self.pet.state, "idle")
        w.set_response_finished(True, "Done")
        self.assertEqual(self.pet.state, "idle")
        w.set_status("Error", error=True)
        self.assertEqual(self.pet.state, "failed")
        w.begin_response_display("retry")
        self.assertEqual(self.pet.state, "running")
        w.set_send_result(False, "retry", "Failed")
        self.assertEqual(self.pet.state, "failed")

    def test_listening_keeps_input_region_visible_outside_overlay(self):
        w = self.window
        w.set_chatgpt_tabs([{"id": "tab", "title": "ChatGPT", "url": "https://chatgpt.com"}])
        outside = w._pointer_hover_bounds().topLeft() - QPoint(100, 100)
        with patch("live_gpt.app.QCursor.pos", return_value=outside):
            w._track_pointer(outside)
            self.assertFalse(w._chrome_visible)
            w.begin_dictation_waiting()
            w.set_microphone_state("recording")
            w.set_dictation_listening()
            w._track_pointer(outside)
            self.assertEqual(self.pet.animation, "waiting")
            self.assertTrue(w.dictation_panel.isVisible())
            self.assertFalse(w._chrome_visible)
            self.assertTrue(w._content_visible)
            self.assertEqual(w.dictation_panel.graphicsEffect().opacity(), 1)
            w.set_dictation_partial("Live transcript")
            w.set_dictation_finishing()
            w._track_pointer(outside)
            self.assertFalse(w._chrome_visible)
            self.assertTrue(w._content_visible)
            w.end_dictation_display()
            w.set_microphone_state("idle")
            self.assertFalse(w._chrome_visible)

    def test_dismissed_subtitle_still_tracks_playback_completion(self):
        w = self.window
        w.begin_response_display()
        w.begin_reading("Playing")
        w.dismiss_subtitle_mode()
        self.assertEqual(self.pet.state, "review")
        w.set_response_finished(True, "Done")
        w.finish_reading(True, "Done")
        self.assertEqual(self.pet.state, "idle")

    def test_response_failure_takes_priority_over_playback(self):
        self.window.begin_response_display()
        self.window.begin_reading("Playing")
        self.window.set_response_finished(False, "Disconnected")
        self.assertEqual(self.pet.state, "failed")

    def test_processing_and_playback_show_content_and_hover_reveals_controls(self):
        w = self.window
        w.set_chatgpt_tabs([{"id": "tab", "title": "ChatGPT", "url": "https://chatgpt.com"}])
        outside = w._pointer_hover_bounds().topLeft() - QPoint(100, 100)
        with patch("live_gpt.app.QCursor.pos", return_value=outside):
            w.begin_response_display("Hello")
            for action in (lambda: None, lambda: w.begin_reading("Playing")):
                action()
                self.assertFalse(w._chrome_visible)
                self.assertTrue(w._content_visible)
                self.assertEqual(w._title_opacity.opacity(), 0)
                self.assertEqual(w._microphone_opacity.opacity(), 0)
                self.assertEqual(w.subtitle_panel.graphicsEffect().opacity(), 1)
                self.assertTrue(w.subtitle_panel.isVisible())
                w._track_pointer(w.frameGeometry().center())
                self.assertTrue(w._chrome_visible)
                self.assertEqual(w._microphone_opacity.opacity(), 1)
                w._track_pointer(outside)
                self.assertFalse(w._chrome_visible)
                self.assertTrue(w._content_visible)
            w.set_response_finished(True, "Done")
            self.assertTrue(w._content_visible)
            w.finish_reading(True, "Done")
            self.assertEqual(self.pet.state, "idle")
            self.assertTrue(w._content_visible)
            self.assertEqual(w._playback_input_timer.interval(), 5000)
            w._finish_playback_input_delay()
            self.assertFalse(w._content_visible)

    @patch("live_gpt.pet.QApplication.screenAt")
    def test_overlay_and_pet_drag_and_restore_current_activity(self, screen_at):
        screen_at.return_value = SimpleNamespace(availableGeometry=lambda: QRect(-4000, -4000, 8000, 8000))
        w = self.window
        origin = QPoint(w.pos())
        point = w.title_bar.mapToGlobal(w.title_bar.rect().center())
        self.mouse(w.title_bar, QEvent.Type.MouseButtonPress, point)
        self.mouse(w.title_bar, QEvent.Type.MouseMove, point + QPoint(50, 0), Qt.MouseButton.NoButton)
        self.mouse(w.title_bar, QEvent.Type.MouseButtonRelease, point + QPoint(50, 0))
        self.assertEqual(w.pos(), origin + QPoint(50, 0))
        origin = QPoint(w.pos())
        point = self.pet.mapToGlobal(self.pet.rect().center())
        self.mouse(self.pet, QEvent.Type.MouseButtonPress, point)
        self.mouse(self.pet, QEvent.Type.MouseMove, point + QPoint(50, 0), Qt.MouseButton.NoButton)
        self.assertEqual(w.pos(), origin + QPoint(50, 0))
        self.assertEqual(self.pet.animation, "running-right")
        self.mouse(self.pet, QEvent.Type.MouseMove, point + QPoint(20, 0), Qt.MouseButton.NoButton)
        self.assertEqual(self.pet.animation, "running-left")
        w.begin_response_display()
        self.assertEqual(self.pet.animation, "running-left")
        self.mouse(self.pet, QEvent.Type.MouseButtonRelease, point + QPoint(20, 0))
        self.assertEqual(self.pet.animation, "running")
        w.lock_button.setChecked(True)
        origin = QPoint(w.pos())
        self.mouse(self.pet, QEvent.Type.MouseButtonPress, point)
        self.mouse(self.pet, QEvent.Type.MouseMove, point + QPoint(100, 0), Qt.MouseButton.NoButton)
        self.assertEqual(w.pos(), origin)

    def test_drag_stays_in_work_area_and_crosses_to_other_monitors(self):
        w = self.window
        w.move(100, 100)
        self.pet.begin_drag(QPoint(150, 150))
        areas = [QRect(0, 0, 1920, 1040), QRect(1920, 0, 1920, 1040),
                 QRect(-1920, 0, 1920, 1040), QRect(0, -1080, 1920, 1080)]
        monitors = [SimpleNamespace(availableGeometry=lambda area=area: area) for area in areas]

        def screen_at(point):
            return next((screen for screen, area in zip(monitors, areas) if area.contains(point)), monitors[0])

        with patch("live_gpt.pet.QApplication.screenAt", side_effect=screen_at):
            for point, area in [(QPoint(0, 0), areas[0]),
                                (QPoint(1919, 1039), areas[0]),
                                (QPoint(1920, 500), areas[1]),
                                (QPoint(3839, 1039), areas[1]),
                                (QPoint(-1, 500), areas[2]),
                                (QPoint(-1920, 0), areas[2]),
                                (QPoint(500, -1), areas[3])]:
                self.pet.drag_to(point)
                self.assertTrue(area.contains(w.frameGeometry()), (point, w.frameGeometry()))
        self.pet._finish_drag()

    def test_v2_all_look_directions_and_idle_deadzone(self):
        import math
        center = self.pet.mapToGlobal(self.pet.rect().center())
        for direction in range(16):
            angle = math.radians(direction * 22.5)
            self.pet.update_look(center + QPoint(round(100 * math.sin(angle)), round(-100 * math.cos(angle))))
            self.assertEqual(self.pet.look_direction, direction)
        for offset in (QPoint(), QPoint(self.pet.screen().geometry().height(), 0)):
            self.pet.update_look(center + offset)
            self.assertIsNone(self.pet.look_direction)
        self.pet.set_state("review")
        self.pet.update_look(center + QPoint(100, 0))
        self.assertIsNone(self.pet.look_direction)

    def test_v1_manifest_and_invalid_sheet(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.pet.sheet.copy(0, 0, 1536, 1872).save(str(root / "sheet.png"))
            for version in (None, 1):
                data = {"spritesheetPath": "sheet.png"}
                if version is not None:
                    data["spriteVersionNumber"] = version
                (root / "pet.json").write_text(json.dumps(data))
                self.pet.load_pet(root)
                self.assertEqual(self.pet.version, 1)
                self.pet.update_look(self.pet.mapToGlobal(QPoint(200, 0)))
                self.assertIsNone(self.pet.look_direction)
            (root / "pet.json").write_text(json.dumps({"spriteVersionNumber": 2, "spritesheetPath": "sheet.png"}))
            with self.assertRaises(ValueError):
                self.pet.load_pet(root)

    def test_pointer_timeout_distance_and_resumed_movement(self):
        center = self.pet.mapToGlobal(self.pet.rect().center())
        radius = self.pet.screen().geometry().height() // 2
        near = center + QPoint(radius - 1, 0)
        with patch("live_gpt.pet.time.monotonic", return_value=10):
            self.pet.update_look(near)
        self.assertEqual(self.pet.look_direction, 4)
        with patch("live_gpt.pet.time.monotonic", return_value=14.99):
            self.pet.update_look(near)
        self.assertEqual(self.pet.look_direction, 4)
        with patch("live_gpt.pet.time.monotonic", return_value=15):
            self.pet.update_look(near)
        self.assertIsNone(self.pet.look_direction)
        with patch("live_gpt.pet.time.monotonic", return_value=16):
            self.pet.update_look(near + QPoint(-1, 0))
        self.assertEqual(self.pet.look_direction, 4)
        self.pet.update_look(center + QPoint(radius, 0))
        self.assertIsNone(self.pet.look_direction)

    def test_hiding_requires_connection_and_disconnect_restores_controls(self):
        w = self.window
        outside = w._pointer_hover_bounds().topLeft() - QPoint(1, 1)
        w._track_pointer(outside)
        self.assertTrue(w._chrome_visible)
        self.assertTrue(all(effect.opacity() == 1 for effect in w._content_opacities))
        w.auto_hide_button.setChecked(True)
        w._hide_for_auto_hide()
        self.assertFalse(w.isHidden())
        self.assertFalse(w._auto_hide_timer.isActive())
        tabs = [{"id": "tab", "title": "ChatGPT", "url": "https://chatgpt.com"}]
        w.set_chatgpt_tabs(tabs)
        w._track_pointer(outside)
        self.assertFalse(w._chrome_visible)
        self.assertTrue(w._auto_hide_timer.isActive())
        w._hide_for_auto_hide()
        self.assertTrue(w.isHidden())
        w.set_chatgpt_tabs([])
        self.assertFalse(w.isHidden())
        self.assertTrue(w._chrome_visible)
        self.assertFalse(w._auto_hide_timer.isActive())
        w._hide_for_auto_hide()
        self.assertFalse(w.isHidden())
        self.assertTrue(w.auto_hide_enabled)
        w.set_chatgpt_tabs(tabs)
        self.assertTrue(w._auto_hide_timer.isActive())

    def test_debug_connected_hint_then_late_tab_is_selected(self):
        w = self.window
        w.set_debug_connection(True)
        w.set_chatgpt_tabs([])
        self.assertIn("Open chatgpt.com", w.transcript_area.placeholderText())
        self.assertTrue(w.remote_debugging_button.isHidden())
        for status in ("Browser connected", "Enable remote debugging in the browser",
                       "Retrying connection… approve it in the browser"):
            w.set_browser_status(status)
            self.assertIn("Debugger connected", w.chatgpt_tab_combo.currentText())
            self.assertIn("Open chatgpt.com", w.transcript_area.placeholderText())
            self.assertIn("Open chatgpt.com", w.chatgpt_tab_combo.toolTip())
        self.assertFalse(w.transcript_area.isEnabled())
        self.assertTrue(w._chrome_visible)
        selections = []
        w.chatgpt_tab_selected.connect(selections.append)
        w.set_chatgpt_tabs([{"id": "new", "title": "ChatGPT", "url": "https://chatgpt.com/"}])
        self.assertEqual(selections, ["new"])
        self.assertTrue(w.transcript_area.isEnabled())
        w.set_chatgpt_tabs([])
        self.assertIn("Open chatgpt.com", w.transcript_area.placeholderText())
        w.set_debug_connection(False)
        self.assertFalse(w.remote_debugging_button.isHidden())

    def test_visible_input_repaints_after_debug_approval(self):
        w = self.window
        w.set_browser_status("Approve remote debugging in the browser…")
        QApplication.processEvents()
        area = w.transcript_area.rect()
        area.moveTopLeft(w.transcript_area.mapTo(w, QPoint()))
        before = w.grab(area).toImage()
        w.set_debug_connection(True)
        w.set_chatgpt_tabs([])
        QApplication.processEvents()
        self.assertFalse(w.transcript_area.graphicsEffect().isEnabled())
        self.assertIn("Open chatgpt.com", w.transcript_area.placeholderText())
        self.assertNotEqual(before, w.grab(area).toImage())
        w.set_chatgpt_tabs([{"id": "tab", "title": "ChatGPT", "url": "https://chatgpt.com"}])
        w._set_chrome_visible(False)
        self.assertTrue(w.transcript_area.graphicsEffect().isEnabled())
        w._set_chrome_visible(True)
        self.assertFalse(w.transcript_area.graphicsEffect().isEnabled())

    def test_hover_fits_editable_content_and_restores_compact_geometry(self):
        w = self.window
        outside = QPoint(-10000, -10000)
        with patch("live_gpt.app.QCursor.pos", return_value=outside):
            w._track_pointer(outside)
            base = QRect(w.geometry())
            w.set_transcript("\n".join(f"Line {i}" for i in range(20)))
            w._track_pointer(w.frameGeometry().center())
            QApplication.processEvents()
            self.assertGreater(w.height(), base.height())
            self.assertEqual(w.width(), base.width())
            self.assertLessEqual(w.height(), w.screen().availableGeometry().height())
            w.set_transcript("\n".join(f"Line {i}" for i in range(200)))
            w._track_pointer(w.frameGeometry().center())
            QApplication.processEvents()
            self.assertLessEqual(w.height(), w.screen().availableGeometry().height())
            w._track_pointer(outside)
            self.assertEqual(w.geometry(), base)
            self.assertIn("Line 199", w.transcript_area.toPlainText())

    def test_screen_tracking_shows_margin_and_hides_everything_except_pet(self):
        w = self.window
        w.set_chatgpt_tabs([{"id": "tab", "title": "ChatGPT", "url": "https://chatgpt.com"}])
        bounds = w._pointer_hover_bounds()
        inside_margin = QPoint(w.frameGeometry().left() - 1, w.frameGeometry().center().y())
        outside = bounds.topLeft() - QPoint(1, 1)
        with patch("live_gpt.pet.QCursor.pos", return_value=inside_margin):
            self.pet._tick()
        self.assertTrue(w._chrome_visible)
        self.assertTrue(all(effect.opacity() == 1 for effect in w._content_opacities))
        with patch("live_gpt.pet.QCursor.pos", return_value=outside):
            self.pet._tick()
        self.assertFalse(w._chrome_visible)
        self.assertTrue(all(effect.opacity() == 0 for effect in w._content_opacities))
        self.assertEqual(w._title_opacity.opacity(), 0)
        self.assertEqual(w._microphone_opacity.opacity(), 0)
        self.assertTrue(self.pet.isVisible())
        self.assertIsNone(self.pet.graphicsEffect())
        w.pet.begin_drag(inside_margin)
        w._track_pointer(outside)
        self.assertTrue(w._chrome_visible)
        w.pet._finish_drag()
        w._track_pointer(outside)
        self.assertFalse(w._chrome_visible)

    def test_idle_timing_skips_unused_cells_and_hidden_timer_stops(self):
        self.pet._last_tick = 0
        for tick in range(1, 101):
            with patch("live_gpt.pet.time.monotonic", return_value=tick / 10):
                self.pet._tick()
            self.assertLess(self.pet.frame, 6)
        self.window.hide()
        self.assertFalse(self.pet._timer.isActive())
        self.window.show()
        self.assertTrue(self.pet._timer.isActive())

    def test_idle_modes_and_timer_restart(self):
        pet = self.pet
        with patch("live_gpt.pet.time.monotonic", return_value=0):
            pet.set_idle_behavior("never")
        pet._last_tick = 0
        with patch("live_gpt.pet.time.monotonic", return_value=0.3):
            pet._tick()
        self.assertEqual(pet.frame, 0)
        with patch("live_gpt.pet.time.monotonic", return_value=1):
            pet.set_idle_behavior("timed", 10)
        pet._last_tick = 1
        with patch("live_gpt.pet.time.monotonic", return_value=1.3):
            pet._tick()
        self.assertEqual(pet.frame, 1)
        with patch("live_gpt.pet.time.monotonic", return_value=11):
            pet._tick()
        self.assertEqual(pet.frame, 0)
        pet.set_state("running")
        with patch("live_gpt.pet.time.monotonic", return_value=20):
            pet.set_state("idle")
        pet._last_tick = 20
        with patch("live_gpt.pet.time.monotonic", return_value=20.3):
            pet._tick()
        self.assertEqual(pet.frame, 1)
        pet.set_idle_behavior("always")
        pet._last_tick = 0
        with patch("live_gpt.pet.time.monotonic", return_value=100.4):
            pet._tick()
        self.assertNotEqual(pet.frame, 0)

    def test_pet_settings_navigation_selection_and_persistence(self):
        with tempfile.TemporaryDirectory() as folder:
            config = Config(Path(folder) / "config.json")
            dialog = HotkeyConfigDialog(QKeySequence("CapsLock"), QKeySequence("Ctrl+S"), QKeySequence("Ctrl+D"), config=config)
            try:
                dialog.pet_nav_button.click()
                self.assertEqual(dialog.settings_pages.currentIndex(), 4)
                self.assertGreater(dialog.pet_list.count(), 0)
                self.assertEqual(dialog.pet_idle_seconds.value(), 10)
                self.assertFalse(dialog.pet_idle_form.isRowVisible(dialog.pet_idle_seconds))
                changes = []
                dialog.pet_settings_changed.connect(lambda *values: changes.append(values))
                dialog.pet_idle_combo.setCurrentIndex(dialog.pet_idle_combo.findData("timed"))
                dialog.pet_idle_seconds.setValue(7)
                self.assertTrue(dialog.pet_idle_seconds.isEnabled())
                self.assertTrue(dialog.pet_idle_form.isRowVisible(dialog.pet_idle_seconds))
                saved = Config(Path(folder) / "config.json")
                self.assertEqual(saved["pet_idle_mode"], "timed")
                self.assertEqual(saved["pet_idle_seconds"], 7)
                self.assertTrue(Path(saved["pet_path"]).is_dir())
                self.assertEqual(changes[-1][1:], ("timed", 7))
                dialog.pet_idle_combo.setCurrentIndex(dialog.pet_idle_combo.findData("never"))
                self.assertFalse(dialog.pet_idle_seconds.isEnabled())
                self.assertFalse(dialog.pet_idle_form.isRowVisible(dialog.pet_idle_seconds))
            finally:
                dialog.deleteLater()


if __name__ == "__main__":
    unittest.main()
