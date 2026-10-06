from __future__ import annotations

import sys
import unittest
from unittest.mock import Mock, patch

from PySide6.QtGui import QColor, QImage

from live_gpt.screen_capture import (
    CaptureSource,
    _encode_lossless_webp,
    _is_valid_window_size,
    _resize_for_upload,
    list_capture_sources,
)
from live_gpt import screen_capture


class CaptureSourceTests(unittest.TestCase):
    @unittest.skipUnless(sys.platform == "win32", "Windows capture APIs")
    def test_cursor_uses_current_window_origin_and_can_be_disabled(self) -> None:
        source = CaptureSource("window:1", "Window", "window", 0, 0, 20, 20, 1)
        image = QImage(20, 20, QImage.Format.Format_RGB32)
        image.fill(QColor("white"))

        def current_rect(hwnd, pointer):
            rect = pointer._obj
            rect.left, rect.top, rect.right, rect.bottom = -1920, 100, -1900, 120
            return True

        with patch.object(screen_capture.user32, "GetWindowRect", side_effect=current_rect), \
                patch.object(screen_capture, "_capture_bitmap", return_value=image) as capture:
            screen_capture.capture_webp(source)
            self.assertEqual(capture.call_args.kwargs["cursor_origin"], (-1920, 100))
            screen_capture.capture_webp(source, include_cursor=False)
            self.assertIsNone(capture.call_args.kwargs["cursor_origin"])
            desktop = CaptureSource("display:1", "Desktop", "display", -1920, 100, 20, 20)
            screen_capture.capture_webp(desktop)
            self.assertNotIn("cursor_origin", capture.call_args.kwargs)

    def test_cursor_hotspot_and_resource_cleanup(self) -> None:
        def cursor_info(pointer):
            cursor = pointer._obj
            cursor.flags = screen_capture.CURSOR_SHOWING
            cursor.hCursor = 123
            cursor.ptScreenPos.x, cursor.ptScreenPos.y = -1800, 150
            return True

        def icon_info(handle, pointer):
            icon = pointer._obj
            icon.xHotspot, icon.yHotspot = 7, 11
            icon.hbmMask, icon.hbmColor = 31, 32
            return True

        user = Mock()
        user.GetCursorInfo.side_effect = cursor_info
        user.CopyIcon.return_value = 456
        user.GetIconInfo.side_effect = icon_info
        gdi = Mock()
        with patch.object(screen_capture, "user32", user, create=True), \
                patch.object(screen_capture, "gdi32", gdi, create=True):
            screen_capture._draw_cursor(1, -1920, 100)
            user.DrawIconEx.assert_called_once_with(
                1, 113, 39, 456, 0, 0, 0, None, screen_capture.DI_NORMAL,
            )
            self.assertEqual([call.args[0] for call in gdi.DeleteObject.call_args_list], [31, 32])
            user.DestroyCursor.assert_called_once_with(456)
            user.DrawIconEx.side_effect = RuntimeError("draw failed")
            with self.assertRaisesRegex(RuntimeError, "draw failed"):
                screen_capture._draw_cursor(1, -1920, 100)
            self.assertEqual(user.DestroyCursor.call_count, 2)
            self.assertEqual(gdi.DeleteObject.call_count, 4)

    def test_hidden_or_suppressed_cursor_is_omitted(self) -> None:
        for flags in (0, 2):
            def cursor_info(pointer):
                pointer._obj.flags = flags
                pointer._obj.hCursor = 123
                return True

            user = Mock()
            user.GetCursorInfo.side_effect = cursor_info
            with patch.object(screen_capture, "user32", user, create=True):
                screen_capture._draw_cursor(1, 0, 0)
            user.CopyIcon.assert_not_called()
            user.DrawIconEx.assert_not_called()

    def test_large_capture_is_scaled_to_1920_preserving_aspect_ratio(self) -> None:
        image = QImage(3840, 2160, QImage.Format.Format_RGB32)

        resized = _resize_for_upload(image)

        self.assertEqual((resized.width(), resized.height()), (1920, 1080))

    def test_webp_encoding_is_lossless(self) -> None:
        image = QImage(4, 3, QImage.Format.Format_RGB32)
        for y in range(image.height()):
            for x in range(image.width()):
                image.setPixelColor(
                    x,
                    y,
                    QColor(x * 51, y * 73, (x + y) * 31),
                )

        encoded = _encode_lossless_webp(image)
        decoded = QImage.fromData(encoded, "WEBP").convertToFormat(
            QImage.Format.Format_RGB32
        )

        self.assertEqual(encoded[:4], b"RIFF")
        self.assertEqual(encoded[8:12], b"WEBP")
        self.assertEqual(decoded.size(), image.size())
        for y in range(image.height()):
            for x in range(image.width()):
                self.assertEqual(
                    decoded.pixelColor(x, y),
                    image.pixelColor(x, y),
                )

    def test_window_must_be_larger_than_one_eighth_of_screen(self) -> None:
        self.assertFalse(_is_valid_window_size(800, 324, 1920, 1080))
        self.assertTrue(_is_valid_window_size(801, 324, 1920, 1080))

    @patch("live_gpt.screen_capture._enumerate_visible_windows")
    @patch("live_gpt.screen_capture._enumerate_displays")
    def test_displays_precede_windows_sorted_largest_first(
        self,
        displays,
        windows,
    ) -> None:
        displays.return_value = [
            (1, 0, 0, 1920, 1080),
            (2, 1920, 0, 2560, 1440),
        ]
        small = CaptureSource(
            "window:1", "Small", "window", 0, 0, 900, 500, 1
        )
        large = CaptureSource(
            "window:2", "Large", "window", 0, 0, 1600, 900, 2
        )
        windows.return_value = [small, large]

        sources = list_capture_sources()

        self.assertEqual(
            [source.key for source in sources],
            ["display:1", "display:2", "window:2", "window:1"],
        )
        self.assertEqual(sources[0].label, "Screenshot desktop 1 (1920×1080)")


if __name__ == "__main__":
    unittest.main()
