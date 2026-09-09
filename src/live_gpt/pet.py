"""Codex v1/v2 sprite animation and screen pointer tracking."""
from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

from PySide6.QtCore import QPoint, QRect, QTimer, Qt, Signal
from PySide6.QtGui import QCursor, QPainter, QPixmap
from PySide6.QtWidgets import QApplication, QWidget


ANIMATIONS = {
    "idle": (0, (280, 110, 110, 140, 140, 320)),
    "running-right": (1, (120,) * 7 + (220,)),
    "running-left": (2, (120,) * 7 + (220,)),
    "waving": (3, (140,) * 3 + (280,)),
    "jumping": (4, (140,) * 4 + (280,)),
    "failed": (5, (140,) * 7 + (240,)),
    "waiting": (6, (150,) * 5 + (260,)),
    "running": (7, (120,) * 5 + (220,)),
    "review": (8, (150,) * 5 + (280,)),
}


def default_pet_path() -> Path:
    source = Path(__file__).resolve().parents[2] / "assets/pets/feibi-jiubi"
    if source.is_dir():
        return source
    return Path(sys.prefix) / "share/live-gpt/pets/feibi-jiubi"


def available_pets() -> list[tuple[str, str]]:
    pets = []
    for manifest in sorted(default_pet_path().parent.glob("*/pet.json")):
        try:
            data = json.loads(manifest.read_text(encoding="utf-8-sig"))
            if isinstance(data, dict):
                pets.append((str(data.get("displayName") or manifest.parent.name), str(manifest.parent)))
        except (OSError, ValueError):
            continue
    return pets


class PetWidget(QWidget):
    pointer_tracked = Signal(QPoint)

    def __init__(self, path: str | Path = "", parent: QWidget | None = None):
        super().__init__(parent)
        self.setFixedSize(96, 104)
        self.setAccessibleName("Pet — drag to move window")
        self.setToolTip("Drag the pet to move the overlay")
        self.setCursor(Qt.CursorShape.OpenHandCursor)
        self.state = "idle"
        self.animation = "idle"
        self.frame = 0
        self.look_direction: int | None = None
        self._drag_offset: QPoint | None = None
        self._last_drag_position = QPoint()
        self._locked = False
        self._elapsed = 0
        self._last_tick = time.monotonic()
        self.idle_mode = "always"
        self.idle_seconds = 10
        self._idle_started = self._last_tick
        self._last_pointer_position: QPoint | None = None
        self._last_pointer_movement = self._last_tick
        self.load_pet(path or default_pet_path())
        self._timer = QTimer(self)
        self._timer.setInterval(30)
        self._timer.timeout.connect(self._tick)

    def load_pet(self, path: str | Path) -> None:
        manifest = Path(path).expanduser()
        if manifest.is_dir():
            manifest /= "pet.json"
        data = json.loads(manifest.read_text(encoding="utf-8-sig"))
        if not isinstance(data, dict):
            raise ValueError("Pet manifest must be a JSON object")
        version = data.get("spriteVersionNumber", 1)
        if type(version) is not int or version not in (1, 2):
            raise ValueError("Pet spriteVersionNumber must be 1 or 2")
        sheet = QPixmap(str(manifest.parent / data.get("spritesheetPath", "spritesheet.webp")))
        rows = 11 if version == 2 else 9
        if sheet.isNull() or sheet.width() != 1536 or sheet.height() != rows * 208:
            raise ValueError(f"Pet v{version} requires a 1536×{rows * 208} sprite sheet")
        self.version = version
        self.sheet = sheet
        self.frame = 0
        self.look_direction = None
        self._elapsed = 0
        self._idle_started = time.monotonic()
        self.update()

    def set_idle_behavior(self, mode: str, seconds: int = 10) -> None:
        if mode not in ("always", "never", "timed") or not 1 <= seconds <= 3600:
            raise ValueError("Invalid idle animation settings")
        self.idle_mode = mode
        self.idle_seconds = seconds
        self._idle_started = time.monotonic()
        if self.animation == "idle":
            self.frame = 0
            self._elapsed = 0
        self.update()

    def set_state(self, state: str) -> None:
        if state not in ANIMATIONS:
            raise ValueError(f"Unknown pet animation: {state}")
        self.state = state
        if self._drag_offset is None:
            self._animate(state)

    def _animate(self, animation: str) -> None:
        if animation != self.animation:
            self.animation = animation
            self.frame = 0
            self._elapsed = 0
            if animation == "idle":
                self._idle_started = time.monotonic()
        self.look_direction = None
        self.update()

    def set_locked(self, locked: bool) -> None:
        self._locked = locked
        self._finish_drag()

    def _finish_drag(self) -> None:
        self._drag_offset = None
        self.setCursor(Qt.CursorShape.ArrowCursor if self._locked else Qt.CursorShape.OpenHandCursor)
        self._animate(self.state)

    def update_look(self, global_position: QPoint) -> None:
        now = time.monotonic()
        if global_position != self._last_pointer_position:
            self._last_pointer_position = QPoint(global_position)
            self._last_pointer_movement = now
        self.look_direction = None
        if self.version == 2 and self.animation == "idle" and self._drag_offset is None:
            center = self.mapToGlobal(self.rect().center())
            delta = global_position - center
            distance = math.hypot(delta.x(), delta.y())
            screen = QApplication.screenAt(center) or self.screen()
            if 0 < distance < screen.geometry().height() / 2 and now - self._last_pointer_movement < 5:
                angle = math.degrees(math.atan2(delta.x(), -delta.y())) % 360
                self.look_direction = int(angle / 22.5 + 0.5) % 16
        self.update()

    def _tick(self) -> None:
        now = time.monotonic()
        self._elapsed += (now - self._last_tick) * 1000
        self._last_tick = now
        durations = ANIMATIONS[self.animation][1]
        self._elapsed %= sum(durations)
        while self._elapsed >= durations[self.frame]:
            self._elapsed -= durations[self.frame]
            self.frame = (self.frame + 1) % len(durations)
        if self.animation == "idle" and (
            self.idle_mode == "never" or
            (self.idle_mode == "timed" and now - self._idle_started >= self.idle_seconds)
        ):
            self.frame = 0
            self._elapsed = 0
        position = QCursor.pos()
        self.update_look(position)
        self.pointer_tracked.emit(position)

    def paintEvent(self, event) -> None:  # noqa: N802
        row = ANIMATIONS[self.animation][0]
        column = self.frame
        if self.look_direction is not None:
            row = 9 + self.look_direction // 8
            column = self.look_direction % 8
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        painter.drawPixmap(self.rect(), self.sheet, QRect(column * 192, row * 208, 192, 208))

    def mousePressEvent(self, event) -> None:  # noqa: N802
        if event.button() == Qt.MouseButton.LeftButton:
            self.begin_drag(event.globalPosition().toPoint())
        event.accept()

    def begin_drag(self, position: QPoint) -> None:
        if not self._locked:
            self._last_drag_position = position
            self._drag_offset = self._last_drag_position - self.window().frameGeometry().topLeft()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            self._animate("running-right")

    def mouseMoveEvent(self, event) -> None:  # noqa: N802
        if event.buttons() & Qt.MouseButton.LeftButton:
            self.drag_to(event.globalPosition().toPoint())
        event.accept()

    def drag_to(self, position: QPoint) -> None:
        if self._drag_offset is not None:
            dx = position.x() - self._last_drag_position.x()
            if dx:
                self._animate("running-right" if dx > 0 else "running-left")
            self._last_drag_position = position
            window = self.window()
            # Choose by pointer, not window center, so clamping at a shared
            # edge never prevents a drag from crossing onto another monitor.
            screen = QApplication.screenAt(position) or window.screen()
            bounds = screen.availableGeometry()
            target = position - self._drag_offset
            size = window.frameGeometry().size()
            # If the window is larger than the work area, keep its top-left
            # controls reachable rather than moving them beyond the screen.
            max_x = max(bounds.left(), bounds.right() - size.width() + 1)
            max_y = max(bounds.top(), bounds.bottom() - size.height() + 1)
            target.setX(max(bounds.left(), min(target.x(), max_x)))
            target.setY(max(bounds.top(), min(target.y(), max_y)))
            window.move(target)

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802
        if event.button() == Qt.MouseButton.LeftButton:
            self._finish_drag()
        event.accept()

    def showEvent(self, event) -> None:  # noqa: N802
        self._last_tick = time.monotonic()
        self._timer.start()
        super().showEvent(event)

    def hideEvent(self, event) -> None:  # noqa: N802
        self._timer.stop()
        self._finish_drag()
        super().hideEvent(event)
