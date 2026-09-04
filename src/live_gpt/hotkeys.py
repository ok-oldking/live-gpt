from __future__ import annotations

import ctypes
import sys
import time
from dataclasses import dataclass
from typing import Callable

from PySide6.QtCore import QObject, QTimer, Signal, Qt
from PySide6.QtGui import QKeySequence


VK_BACK = 0x08
VK_TAB = 0x09
VK_RETURN = 0x0D
VK_PAUSE = 0x13
VK_CAPITAL = 0x14
VK_ESCAPE = 0x1B
VK_SPACE = 0x20
VK_PRIOR = 0x21
VK_NEXT = 0x22
VK_END = 0x23
VK_HOME = 0x24
VK_LEFT = 0x25
VK_UP = 0x26
VK_RIGHT = 0x27
VK_DOWN = 0x28
VK_PRINT = 0x2A
VK_SNAPSHOT = 0x2C
VK_INSERT = 0x2D
VK_DELETE = 0x2E
VK_LWIN = 0x5B
VK_RWIN = 0x5C
VK_NUMLOCK = 0x90
VK_SCROLL = 0x91
VK_SHIFT = 0x10
VK_CONTROL = 0x11
VK_MENU = 0x12
VK_F1 = 0x70


_QT_SPECIAL_KEYS = {
    Qt.Key.Key_Backspace.value: VK_BACK,
    Qt.Key.Key_Tab.value: VK_TAB,
    Qt.Key.Key_Return.value: VK_RETURN,
    Qt.Key.Key_Enter.value: VK_RETURN,
    Qt.Key.Key_Pause.value: VK_PAUSE,
    Qt.Key.Key_CapsLock.value: VK_CAPITAL,
    Qt.Key.Key_Escape.value: VK_ESCAPE,
    Qt.Key.Key_Space.value: VK_SPACE,
    Qt.Key.Key_PageUp.value: VK_PRIOR,
    Qt.Key.Key_PageDown.value: VK_NEXT,
    Qt.Key.Key_End.value: VK_END,
    Qt.Key.Key_Home.value: VK_HOME,
    Qt.Key.Key_Left.value: VK_LEFT,
    Qt.Key.Key_Up.value: VK_UP,
    Qt.Key.Key_Right.value: VK_RIGHT,
    Qt.Key.Key_Down.value: VK_DOWN,
    Qt.Key.Key_Print.value: VK_PRINT,
    Qt.Key.Key_Insert.value: VK_INSERT,
    Qt.Key.Key_Delete.value: VK_DELETE,
    Qt.Key.Key_NumLock.value: VK_NUMLOCK,
    Qt.Key.Key_ScrollLock.value: VK_SCROLL,
    Qt.Key.Key_Shift.value: VK_SHIFT,
    Qt.Key.Key_Control.value: VK_CONTROL,
    Qt.Key.Key_Alt.value: VK_MENU,
}


def _default_key_state(virtual_key: int) -> int:
    if sys.platform != "win32":
        return 0
    return int(ctypes.windll.user32.GetAsyncKeyState(virtual_key))


@dataclass(frozen=True)
class HotkeyBinding:
    """A single Windows key combination sampled without intercepting input."""

    text: str
    virtual_key: int
    control: bool = False
    shift: bool = False
    alt: bool = False
    meta: bool = False

    @classmethod
    def from_sequence(cls, sequence: QKeySequence | str) -> HotkeyBinding:
        if isinstance(sequence, str):
            sequence = QKeySequence(sequence)
        if sequence.isEmpty() or sequence.count() != 1:
            raise ValueError("Choose one key combination")

        combination = sequence[0]
        key = combination.key().value
        modifiers = combination.keyboardModifiers()
        virtual_key = _virtual_key_for_qt_key(key)
        text = sequence.toString(QKeySequence.SequenceFormat.PortableText)
        return cls(
            text=text,
            virtual_key=virtual_key,
            control=bool(modifiers & Qt.KeyboardModifier.ControlModifier),
            shift=bool(modifiers & Qt.KeyboardModifier.ShiftModifier),
            alt=bool(modifiers & Qt.KeyboardModifier.AltModifier),
            meta=bool(modifiers & Qt.KeyboardModifier.MetaModifier),
        )

    def is_pressed(self, key_state: Callable[[int], int]) -> bool:
        is_down = lambda key: bool(key_state(key) & 0x8000)
        meta_down = is_down(VK_LWIN) or is_down(VK_RWIN)
        return (
            is_down(self.virtual_key)
            and (
                self.virtual_key == VK_CONTROL
                or is_down(VK_CONTROL) == self.control
            )
            and (
                self.virtual_key == VK_SHIFT
                or is_down(VK_SHIFT) == self.shift
            )
            and (
                self.virtual_key == VK_MENU
                or is_down(VK_MENU) == self.alt
            )
            and meta_down == self.meta
        )


def _virtual_key_for_qt_key(key: int) -> int:
    if ord("A") <= key <= ord("Z") or ord("0") <= key <= ord("9"):
        return key
    if Qt.Key.Key_F1.value <= key <= Qt.Key.Key_F24.value:
        return VK_F1 + key - Qt.Key.Key_F1.value
    try:
        return _QT_SPECIAL_KEYS[key]
    except KeyError as error:
        raise ValueError("That key is not supported as a global hotkey") from error


class GlobalHotkeyMonitor(QObject):
    """Poll global key state so shortcuts remain visible to other programs."""

    hold_pressed = Signal()
    hold_released = Signal()
    hold_without_screenshot_pressed = Signal()
    hold_without_screenshot_released = Signal()
    send_pressed = Signal()
    send_without_screenshot_pressed = Signal()

    def __init__(
        self,
        hold: HotkeyBinding,
        send: HotkeyBinding,
        send_without_screenshot: HotkeyBinding,
        parent: QObject | None = None,
        *,
        hold_without_screenshot: HotkeyBinding | None = None,
        key_state: Callable[[int], int] | None = None,
        clock: Callable[[], float] | None = None,
        interval_ms: int = 25,
        hold_delay_ms: int = 300,
    ) -> None:
        super().__init__(parent)
        self._key_state = key_state or _default_key_state
        self._clock = clock or time.monotonic
        self._hold_delay = hold_delay_ms / 1_000
        self._hold_started_at: dict[str, float | None] = {
            "hold": None,
            "hold_without_screenshot": None,
        }
        self._hold_emitted = {
            "hold": False,
            "hold_without_screenshot": False,
        }
        self._bindings: dict[str, HotkeyBinding] = {}
        self._pressed = {
            "hold": False,
            "send": False,
            "send_without_screenshot": False,
        }
        self._timer = QTimer(self)
        self._timer.setInterval(interval_ms)
        self._timer.timeout.connect(self.poll_now)
        self.update_bindings(
            hold,
            send,
            send_without_screenshot,
            hold_without_screenshot=hold_without_screenshot,
        )

    def update_bindings(
        self,
        hold: HotkeyBinding,
        send: HotkeyBinding,
        send_without_screenshot: HotkeyBinding,
        *,
        hold_without_screenshot: HotkeyBinding | None = None,
    ) -> None:
        self._bindings = {
            "hold": hold,
            "hold_without_screenshot": (
                hold_without_screenshot
                or HotkeyBinding.from_sequence("Shift")
            ),
            "send": send,
            "send_without_screenshot": send_without_screenshot,
        }
        self._pressed = {name: False for name in self._bindings}
        self._hold_started_at = {
            "hold": None,
            "hold_without_screenshot": None,
        }
        self._hold_emitted = {
            "hold": False,
            "hold_without_screenshot": False,
        }

    def start(self) -> None:
        self._timer.start()

    @property
    def hold_delay_seconds(self) -> float:
        return self._hold_delay

    def stop(self) -> None:
        if self._hold_emitted["hold"]:
            self.hold_released.emit()
        if self._hold_emitted["hold_without_screenshot"]:
            self.hold_without_screenshot_released.emit()
        self._pressed = {name: False for name in self._bindings}
        self._hold_started_at = {
            "hold": None,
            "hold_without_screenshot": None,
        }
        self._hold_emitted = {
            "hold": False,
            "hold_without_screenshot": False,
        }
        self._timer.stop()

    def poll_now(self) -> None:
        current = {
            name: binding.is_pressed(self._key_state)
            for name, binding in self._bindings.items()
        }

        now = self._clock()
        hold_signals = {
            "hold": (self.hold_pressed, self.hold_released),
            "hold_without_screenshot": (
                self.hold_without_screenshot_pressed,
                self.hold_without_screenshot_released,
            ),
        }
        for name, (pressed_signal, released_signal) in hold_signals.items():
            if current[name]:
                if not self._pressed[name]:
                    self._hold_started_at[name] = now
                started_at = self._hold_started_at[name]
                if (
                    not self._hold_emitted[name]
                    and started_at is not None
                    and now - started_at >= self._hold_delay
                ):
                    self._hold_emitted[name] = True
                    pressed_signal.emit()
            else:
                if self._hold_emitted[name]:
                    released_signal.emit()
                self._hold_started_at[name] = None
                self._hold_emitted[name] = False

        if current["send"] and not self._pressed["send"]:
            self.send_pressed.emit()
        if (
            current["send_without_screenshot"]
            and not self._pressed["send_without_screenshot"]
        ):
            self.send_without_screenshot_pressed.emit()

        self._pressed = current
