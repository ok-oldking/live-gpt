from __future__ import annotations

import os
import unittest


os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtGui import QKeySequence  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from live_gpt.hotkeys import (  # noqa: E402
    HotkeyEdit,
    SIDED_MODIFIERS,
    GlobalHotkeyMonitor,
    HotkeyBinding,
    VK_CAPITAL,
    VK_CONTROL,
    VK_SHIFT,
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
            hold_without_screenshot=HotkeyBinding.from_sequence("Shift"),
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

    def test_shift_is_an_independent_hold_hotkey(self) -> None:
        events: list[str] = []
        self.monitor.hold_without_screenshot_pressed.connect(
            lambda: events.append("pressed")
        )
        self.monitor.hold_without_screenshot_released.connect(
            lambda: events.append("released")
        )

        self.pressed.add(VK_SHIFT)
        self.monitor.poll_now()
        self.now += 0.31
        self.monitor.poll_now()
        self.pressed.remove(VK_SHIFT)
        self.monitor.poll_now()

        self.assertEqual(events, ["pressed", "released"])

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

    def test_each_modifier_side_is_independent(self) -> None:
        for name, vk in SIDED_MODIFIERS.items():
            with self.subTest(name=name):
                binding = HotkeyBinding.from_sequence(name)
                aggregate = {0xA0: 0x10, 0xA1: 0x10, 0xA2: 0x11,
                             0xA3: 0x11, 0xA4: 0x12, 0xA5: 0x12}[vk]
                down = {vk, aggregate}
                self.assertTrue(binding.is_pressed(lambda key: 0x8000 if key in down else 0))
                down = {vk ^ 1, aggregate}
                self.assertFalse(binding.is_pressed(lambda key: 0x8000 if key in down else 0))

    def test_right_alt_accepts_windows_altgr_control_state(self) -> None:
        down = {0xA5, 0x12, 0x11, 0xA2}
        state = lambda key: 0x8000 if key in down else 0
        self.assertTrue(HotkeyBinding.from_sequence("Right Alt").is_pressed(state))
        self.assertFalse(HotkeyBinding.from_sequence("Right Ctrl").is_pressed(state))

    def test_sided_modifier_combination(self) -> None:
        binding = HotkeyBinding.from_sequence("Right Ctrl+A")
        down = {0xA3, 0x11, ord("A")}
        self.assertTrue(binding.is_pressed(lambda key: 0x8000 if key in down else 0))
        down = {0xA2, 0x11, ord("A")}
        self.assertFalse(binding.is_pressed(lambda key: 0x8000 if key in down else 0))

    def test_editor_captures_native_modifier_sides(self) -> None:
        from PySide6.QtCore import QEvent, Qt
        from PySide6.QtGui import QKeyEvent
        editor = HotkeyEdit("Right Alt")
        for name, vk in SIDED_MODIFIERS.items():
            qt_key = {"Shift": Qt.Key.Key_Shift, "Ctrl": Qt.Key.Key_Control,
                      "Alt": Qt.Key.Key_Alt}[name.split()[1]]
            for event_type in (QEvent.Type.KeyPress, QEvent.Type.KeyRelease):
                event = QKeyEvent(event_type, qt_key, Qt.KeyboardModifier.NoModifier, 0, vk, 0)
                self.application.sendEvent(editor, event)
            self.assertEqual(editor.keySequence(), name)
        editor.close()

    def test_editor_captures_extended_windows_alt_and_control(self) -> None:
        from PySide6.QtCore import QEvent, Qt
        from PySide6.QtGui import QKeyEvent
        for key, vk, scan, native_modifiers, expected in (
            (Qt.Key.Key_Alt, 0x12, 0xE038, 0, "Right Alt"),
            (Qt.Key.Key_Control, 0x11, 0xE01D, 0, "Right Ctrl"),
            (Qt.Key.Key_Alt, 0x12, 0x138, 0, "Right Alt"),
            (Qt.Key.Key_Alt, 0x12, 0x38, 0x01000000, "Right Alt"),
            (Qt.Key.Key_AltGr, 0x12, 0x38, 0, "Right Alt"),
            (Qt.Key.Key_Alt, 0x12, 0x38, 0, "Left Alt"),
            (Qt.Key.Key_Control, 0x11, 0x1D, 0, "Left Ctrl"),
        ):
            with self.subTest(scan=scan, expected=expected):
                editor = HotkeyEdit("")
                event = QKeyEvent(QEvent.Type.KeyPress, key,
                                  Qt.KeyboardModifier.NoModifier, scan, vk, native_modifiers)
                self.application.sendEvent(editor, event)
                self.assertEqual(editor.keySequence(), expected)
                editor.close()

    def test_clear_button_disables_binding_and_allows_reassignment(self) -> None:
        editor = HotkeyEdit("Right Alt")
        changes = []
        editor.keySequenceChanged.connect(changes.append)
        editor.clear_button.click()
        binding = HotkeyBinding.from_sequence(editor.keySequence())
        self.assertFalse(binding.is_pressed(lambda key: 0x8000))
        self.assertEqual(changes, [""])
        self.assertTrue(editor.clear_button.isHidden())
        editor.setKeySequence("Left Alt")
        self.assertEqual(editor.keySequence(), "Left Alt")
        self.assertFalse(editor.clear_button.isHidden())
        editor.close()

    def test_disabled_shortcuts_do_not_trigger_and_can_be_enabled(self) -> None:
        events = []
        self.monitor.hold_pressed.connect(lambda: events.append("hold"))
        self.monitor.hold_without_screenshot_pressed.connect(lambda: events.append("shift"))
        self.monitor.send_pressed.connect(lambda: events.append("send"))
        self.monitor.send_without_screenshot_pressed.connect(lambda: events.append("text"))
        def configure(enabled):
            self.monitor.update_bindings(
                HotkeyBinding.from_sequence("CapsLock"),
                HotkeyBinding.from_sequence("Ctrl+S"),
                HotkeyBinding.from_sequence("Ctrl+D"),
                enabled=enabled,
            )
        configure({"hold": True})
        for keys in ({VK_CAPITAL}, {VK_SHIFT}, {VK_CONTROL, ord("S")}, {VK_CONTROL, ord("D")}):
            self.pressed = keys
            self.monitor.poll_now()
            self.now += 0.4
            self.monitor.poll_now()
            self.pressed = set()
            self.monitor.poll_now()
        self.assertEqual(events, ["hold"])
        configure({"send": True})
        self.pressed = {VK_CONTROL, ord("S")}
        self.monitor.poll_now()
        self.assertEqual(events, ["hold", "send"])


if __name__ == "__main__":
    unittest.main()
