"""Asynchronous reply thumbnails and a native enlarged image window."""

from __future__ import annotations

import base64
from collections import OrderedDict
from html import escape
from html.parser import HTMLParser
from pathlib import Path

from PySide6.QtCore import QEvent, QObject, QSize, Qt, QUrl, Signal
from PySide6.QtGui import QColor, QIcon, QImage, QPainter, QPixmap
from PySide6.QtNetwork import QNetworkAccessManager, QNetworkReply, QNetworkRequest
from PySide6.QtWidgets import (
    QDialog, QFrame, QGraphicsPixmapItem, QGraphicsScene, QGraphicsView,
    QHBoxLayout, QLabel, QPushButton, QSizeGrip, QVBoxLayout, QWidget,
)

from .localization import tr


class ReplyImageMarkup(HTMLParser):
    """Give each image its own resource and clickable preview target."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.images: dict[str, tuple[str, str, QSize]] = {}

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag != "img":
            self.parts.append(self.get_starttag_text())
            return
        attributes = dict(attrs)
        source = attributes.get("src") or ""
        if QUrl(source).scheme().lower() not in ("http", "https", "data"):
            return
        name = f"live-gpt-image:{len(self.images)}"
        label = attributes.get("alt") or tr("Image")
        try:
            size = QSize(int(attributes.get("width") or 280), int(attributes.get("height") or 168))
        except ValueError:
            size = QSize(280, 168)
        if size.width() <= 0 or size.height() <= 0:
            size = QSize(280, 168)
        self.images[name] = (source, label, size)
        thumbnail = size.scaled(QSize(280, 190), Qt.AspectRatioMode.KeepAspectRatio)
        self.parts.append(f'<a href="{name}"><img src="{name}" alt="{escape(label, quote=True)}" '
                          f'width="{thumbnail.width()}" height="{thumbnail.height()}"></a>')

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self.handle_starttag(tag, attrs)

    def handle_endtag(self, tag: str) -> None:
        self.parts.append(f"</{tag}>")

    def handle_data(self, data: str) -> None:
        self.parts.append(escape(data))


class ReplyImageLoader(QObject):
    ready = Signal(str)
    MAX_DOWNLOAD = 20 * 1024 * 1024
    MAX_CACHE = 64 * 1024 * 1024

    def __init__(self, parent: QObject) -> None:
        super().__init__(parent)
        self.network = QNetworkAccessManager(self)
        self.cache: OrderedDict[str, QImage] = OrderedDict()
        self.pending: dict[str, QNetworkReply] = {}
        self.failed: set[str] = set()

    def request(self, source: str) -> QImage | None:
        if source in self.cache:
            self.cache.move_to_end(source)
            return self.cache[source]
        if source in self.pending or source in self.failed:
            return None
        url = QUrl(source)
        if url.scheme() == "data":
            try:
                header, encoded = source.split(",", 1)
                data = base64.b64decode(encoded, validate=True) if header.startswith("data:image/") and header.endswith(";base64") else b""
                if len(data) > self.MAX_DOWNLOAD:
                    data = b""
            except ValueError:
                data = b""
            return self._store(source, QImage.fromData(data))
        if url.scheme() not in ("http", "https") or not url.host():
            self.failed.add(source)
            return None
        request = QNetworkRequest(url)
        request.setTransferTimeout(15_000)
        request.setAttribute(QNetworkRequest.Attribute.RedirectPolicyAttribute,
                             QNetworkRequest.RedirectPolicy.NoLessSafeRedirectPolicy)
        reply = self.network.get(request)
        self.pending[source] = reply
        reply.downloadProgress.connect(lambda received, _total: reply.abort() if received > self.MAX_DOWNLOAD else None)
        reply.finished.connect(lambda: self._finished(source, reply))
        return None

    def _store(self, source: str, image: QImage) -> QImage | None:
        if image.isNull() or image.sizeInBytes() > self.MAX_CACHE:
            self.failed.add(source)
            return None
        while self.cache and sum(item.sizeInBytes() for item in self.cache.values()) + image.sizeInBytes() > self.MAX_CACHE:
            self.cache.popitem(last=False)
        self.cache[source] = image
        return image

    def _finished(self, source: str, reply: QNetworkReply) -> None:
        self.pending.pop(source, None)
        image = QImage()
        if reply.error() == QNetworkReply.NetworkError.NoError and reply.bytesAvailable() <= self.MAX_DOWNLOAD:
            image = QImage.fromData(reply.readAll())
        self._store(source, image)
        reply.deleteLater()
        self.ready.emit(source)

    def placeholder(self, source: str, size: QSize) -> QImage:
        image = QImage(size, QImage.Format.Format_ARGB32)
        image.fill(QColor("#17223b"))
        painter = QPainter(image)
        painter.setPen(QColor("#c1cbe0"))
        painter.drawText(image.rect(), Qt.AlignmentFlag.AlignCenter,
                         tr("Image unavailable") if source in self.failed else tr("Loading image…"))
        painter.end()
        return image


class ImageView(QGraphicsView):
    """Fit the original image, then zoom and pan without resampling its source."""

    def __init__(self, parent: QWidget) -> None:
        super().__init__(parent)
        scene = QGraphicsScene(self)
        self.setScene(scene)
        self.image_item = QGraphicsPixmapItem()
        scene.addItem(self.image_item)
        self.image_item.setTransformationMode(Qt.TransformationMode.SmoothTransformation)
        self.setFrameShape(QFrame.Shape.NoFrame)
        self.setBackgroundBrush(QColor("#0c1224"))
        self.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorViewCenter)
        self.setDragMode(QGraphicsView.DragMode.NoDrag)
        self.setMinimumSize(1, 1)
        self.zoom = 1.0
        self._pan_position = None

    def set_image(self, image: QImage) -> None:
        self.image_item.setPixmap(QPixmap.fromImage(image))
        self.setSceneRect(self.image_item.boundingRect())
        self.fit_image()

    def fit_image(self) -> None:
        if not self.image_item.pixmap().isNull():
            self.zoom = 1.0
            self._pan_position = None
            self.viewport().setCursor(Qt.CursorShape.ArrowCursor)
            self.setSceneRect(self.image_item.boundingRect())
            # Hide scrollbars while measuring so previous zoom does not reduce
            # the fit area or make resize events repeatedly change the scale.
            self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
            self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
            self.fitInView(self.image_item.boundingRect(), Qt.AspectRatioMode.KeepAspectRatio)
            self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
            self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
            self.centerOn(self.image_item)

    def wheelEvent(self, event) -> None:  # noqa: N802
        if self.image_item.pixmap().isNull():
            event.ignore()
            return
        delta = event.angleDelta().y() or event.pixelDelta().y()
        target = min(32.0, max(0.1, self.zoom * 1.2 ** (delta / 120)))
        factor = target / self.zoom
        self.zoom = target
        self.scale(factor, factor)
        self._update_pan_area()
        event.accept()

    def _update_pan_area(self) -> None:
        bounds = self.image_item.boundingRect()
        center = self.mapToScene(self.viewport().rect().center())
        if self.zoom > 1.0:
            # Leave room to drag in both directions, including when a wide
            # image is zoomed but its height still fits inside the viewport.
            visible = self.mapToScene(self.viewport().rect()).boundingRect()
            bounds = bounds.adjusted(-visible.width(), -visible.height(),
                                     visible.width(), visible.height())
        self.setSceneRect(bounds)
        self.centerOn(center)
        self.viewport().setCursor(Qt.CursorShape.OpenHandCursor if self.zoom > 1.0
                                  else Qt.CursorShape.ArrowCursor)

    def mousePressEvent(self, event) -> None:  # noqa: N802
        if event.button() == Qt.MouseButton.LeftButton and self.zoom > 1.0:
            self._pan_position = event.position().toPoint()
            self.viewport().setCursor(Qt.CursorShape.ClosedHandCursor)
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:  # noqa: N802
        if self._pan_position is not None:
            position = event.position().toPoint()
            delta = position - self._pan_position
            self._pan_position = position
            horizontal, vertical = self.horizontalScrollBar(), self.verticalScrollBar()
            horizontal.setValue(horizontal.value() - delta.x())
            vertical.setValue(vertical.value() - delta.y())
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802
        if event.button() == Qt.MouseButton.LeftButton and self._pan_position is not None:
            self._pan_position = None
            self.viewport().setCursor(Qt.CursorShape.OpenHandCursor if self.zoom > 1.0
                                      else Qt.CursorShape.ArrowCursor)
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        if hasattr(self, "image_item") and self.zoom == 1.0:
            self.fit_image()
        elif hasattr(self, "image_item") and not self.image_item.pixmap().isNull():
            self._update_pan_area()


class ImagePreviewDialog(QDialog):
    """A borderless, movable image window with zoom and maximize controls."""

    def __init__(self, loader: ReplyImageLoader, source: str, label: str, parent: QWidget) -> None:
        super().__init__(parent, Qt.WindowType.Dialog | Qt.WindowType.WindowStaysOnTopHint
                         | Qt.WindowType.FramelessWindowHint)
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        self.setWindowTitle(label)
        self.setWindowModality(Qt.WindowModality.ApplicationModal)
        self.source = source
        self.loader = loader
        self.image: QImage | None = loader.request(source)
        self._drag_offset = None
        self.setMinimumSize(360, 240)
        self.setStyleSheet("QDialog { background: #0c1224; } QLabel { color: #f5f7ff; }"
                           "QPushButton { color: #f5f7ff; background: #243352; border: none; border-radius: 10px; padding: 10px; }")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 16, 20, 20)
        self.title_bar = QWidget(self)
        self.title_bar.installEventFilter(self)
        toolbar = QHBoxLayout(self.title_bar)
        toolbar.setContentsMargins(0, 0, 0, 0)
        title = QLabel(label)
        title.setTextFormat(Qt.TextFormat.PlainText)
        title.setWordWrap(True)
        title.installEventFilter(self)
        toolbar.addWidget(title, 1)
        self.maximize_button = QPushButton()
        self.maximize_button.setIconSize(QSize(18, 18))
        self.maximize_button.setFixedSize(40, 40)
        self.maximize_button.clicked.connect(self._toggle_maximize)
        toolbar.addWidget(self.maximize_button)
        close = QPushButton()
        close.setIcon(QIcon(str(Path(__file__).parent / "assets" / "exit.svg")))
        close.setIconSize(QSize(18, 18))
        close.setToolTip(tr("Close image"))
        close.setAccessibleName(tr("Close image"))
        close.setFixedSize(40, 40)
        close.clicked.connect(self.close)
        toolbar.addWidget(close)
        layout.addWidget(self.title_bar)
        self.picture = QLabel()
        self.picture.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.picture.setMinimumSize(1, 1)
        self.picture.setScaledContents(False)
        layout.addWidget(self.picture, 1)
        self.view = ImageView(self)
        layout.addWidget(self.view, 1)
        footer = QHBoxLayout()
        hint = QLabel(tr("Scroll to zoom · Drag to pan"))
        footer.addWidget(hint, 1)
        fit_button = QPushButton(tr("Fit image"))
        fit_button.clicked.connect(self.view.fit_image)
        footer.addWidget(fit_button)
        self.size_grip = QSizeGrip(self)
        footer.addWidget(self.size_grip, 0, Qt.AlignmentFlag.AlignBottom)
        layout.addLayout(footer)
        screen = parent.window().screen()
        self.setScreen(screen)
        available = screen.availableGeometry()
        self.resize(min(1100, available.width() - 80), min(800, available.height() - 80))
        self.move(screen.geometry().center() - self.rect().center())
        loader.ready.connect(self._loaded)
        self._update_window_controls()
        self._render()

    def _loaded(self, source: str) -> None:
        if source == self.source:
            self.image = self.loader.cache.get(source)
            self._render()

    def _render(self) -> None:
        if self.image is None:
            self.view.hide()
            self.picture.show()
            self.picture.setText(tr("Image unavailable") if self.source in self.loader.failed else tr("Loading image…"))
        else:
            self.picture.hide()
            self.view.show()
            self.view.set_image(self.image)

    def _toggle_maximize(self) -> None:
        if self.isMaximized():
            self.showNormal()
        else:
            self.showMaximized()
        self._update_window_controls()

    def _update_window_controls(self) -> None:
        maximized = self.isMaximized()
        label = tr("Restore image window") if maximized else tr("Maximize image window")
        self.maximize_button.setToolTip(label)
        self.maximize_button.setAccessibleName(label)
        icon = "restore.svg" if maximized else "maximize.svg"
        self.maximize_button.setIcon(QIcon(str(Path(__file__).parent / "assets" / icon)))
        self.size_grip.setVisible(not maximized)

    def changeEvent(self, event) -> None:  # noqa: N802
        super().changeEvent(event)
        if event.type() == QEvent.Type.WindowStateChange and hasattr(self, "size_grip"):
            self._update_window_controls()

    def eventFilter(self, watched, event) -> bool:  # noqa: N802
        if event.type() == QEvent.Type.MouseButtonDblClick and event.button() == Qt.MouseButton.LeftButton:
            self._toggle_maximize()
            return True
        if event.type() == QEvent.Type.MouseButtonPress and event.button() == Qt.MouseButton.LeftButton:
            if not self.isMaximized():
                handle = self.windowHandle()
                if not handle or not handle.startSystemMove():
                    self._drag_offset = event.globalPosition().toPoint() - self.pos()
            return True
        if event.type() == QEvent.Type.MouseMove and self._drag_offset is not None:
            self.move(event.globalPosition().toPoint() - self._drag_offset)
            return True
        if event.type() == QEvent.Type.MouseButtonRelease:
            self._drag_offset = None
        return super().eventFilter(watched, event)
