from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QEvent, QPoint, QPointF, Qt
from PySide6.QtGui import QMouseEvent
from PySide6.QtWidgets import QApplication

from live_gpt.app import OverlayWindow


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
        self.assertEqual(self.pet.state, "running")
        w.set_response_finished(True, "Done")
        self.assertEqual(self.pet.state, "idle")
        w.set_status("Error", error=True)
        self.assertEqual(self.pet.state, "failed")
        w.begin_response_display("retry")
        self.assertEqual(self.pet.state, "running")
        w.set_send_result(False, "retry", "Failed")
        self.assertEqual(self.pet.state, "failed")

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

    def test_overlay_and_pet_drag_and_restore_current_activity(self):
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

    def test_screen_tracking_shows_margin_and_hides_everything_except_pet(self):
        w = self.window
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


if __name__ == "__main__":
    unittest.main()
