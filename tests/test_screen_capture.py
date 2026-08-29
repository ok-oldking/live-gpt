from __future__ import annotations

import unittest
from unittest.mock import patch

from PySide6.QtGui import QColor, QImage

from live_gpt.screen_capture import (
    CaptureSource,
    _encode_lossless_webp,
    _is_valid_window_size,
    _resize_for_upload,
    list_capture_sources,
)


class CaptureSourceTests(unittest.TestCase):
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
