from __future__ import annotations

import os
import unittest


os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtGui import QKeySequence  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from live_gpt.hotkeys import (  # noqa: E402
    GlobalHotkeyMonitor,
    HotkeyBinding,
    VK_CAPITAL,
    VK_CONTROL,
)


class GlobalHotkeyMonitorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.application = QApplication.instance() or QApplication([])

    def setUp(self) -> None:
        self.pressed: set[int] = set()
        self.now = 10.0
        self.monitor = GlobalHotkeyMonitor(
            HotkeyBinding.from_sequence("CapsLock"),
            HotkeyBinding.from_sequence("Ctrl+S"),
            HotkeyBinding.from_sequence("Ctrl+D"),
            key_state=lambda key: 0x8000 if key in self.pressed else 0,
            clock=lambda: self.now,
        )

    def test_hold_hotkey_emits_press_and_release_edges(self) -> None:
        events: list[str] = []
        self.monitor.hold_pressed.connect(lambda: events.append("pressed"))
        self.monitor.hold_released.connect(lambda: events.append("released"))

        self.monitor.poll_now()
        self.pressed.add(VK_CAPITAL)
        self.monitor.poll_now()
        self.now += 0.31
        self.monitor.poll_now()
        self.pressed.remove(VK_CAPITAL)
        self.monitor.poll_now()

        self.assertEqual(events, ["pressed", "released"])

    def test_short_hold_hotkey_does_not_start_dictation(self) -> None:
        events: list[str] = []
        self.monitor.hold_pressed.connect(lambda: events.append("pressed"))
        self.monitor.hold_released.connect(lambda: events.append("released"))

        self.pressed.add(VK_CAPITAL)
        self.monitor.poll_now()
        self.now += 0.29
        self.monitor.poll_now()
        self.pressed.remove(VK_CAPITAL)
        self.monitor.poll_now()

        self.assertEqual(events, [])

    def test_send_hotkey_only_emits_once_until_released(self) -> None:
        events: list[str] = []
        self.monitor.send_pressed.connect(lambda: events.append("send"))

        self.pressed.update({VK_CONTROL, ord("S")})
        self.monitor.poll_now()
        self.monitor.poll_now()
        self.pressed.remove(ord("S"))
        self.monitor.poll_now()
        self.pressed.add(ord("S"))
        self.monitor.poll_now()

        self.assertEqual(events, ["send", "send"])

    def test_send_without_screenshot_has_its_own_binding(self) -> None:
        events: list[str] = []
        self.monitor.send_without_screenshot_pressed.connect(
            lambda: events.append("without-screenshot")
        )

        self.pressed.update({VK_CONTROL, ord("D")})
        self.monitor.poll_now()

        self.assertEqual(events, ["without-screenshot"])

    def test_binding_uses_portable_key_sequence_text(self) -> None:
        binding = HotkeyBinding.from_sequence(QKeySequence("Ctrl+S"))

        self.assertEqual(binding.text, "Ctrl+S")
        self.assertEqual(binding.virtual_key, ord("S"))
        self.assertTrue(binding.control)


if __name__ == "__main__":
    unittest.main()
