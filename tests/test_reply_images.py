from __future__ import annotations

import os
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QBuffer, QIODevice, QPoint, QPointF, Qt, QUrl
from PySide6.QtGui import QColor, QImage, QTextCursor, QTextDocument, QWheelEvent
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from live_gpt.app import OverlayWindow, ReplyDisplay
from live_gpt.reply_images import ReplyImageMarkup


class ReplyImageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.application = QApplication.instance() or QApplication([])
        cls.image = QImage(800, 400, QImage.Format.Format_RGB32)
        cls.image.fill(QColor("#2980b9"))
        buffer = QBuffer()
        buffer.open(QIODevice.OpenModeFlag.WriteOnly)
        cls.image.save(buffer, "PNG")
        cls.png = bytes(buffer.data())
        cls.gate = threading.Event()
        cls.requests = []

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                cls.requests.append(self.path)
                if self.path.startswith("/slow"):
                    cls.gate.wait(3)
                if self.path == "/missing":
                    self.send_error(404)
                    return
                self.send_response(200)
                self.send_header("Content-Type", "image/png")
                self.send_header("Content-Length", str(len(cls.png)))
                self.end_headers()
                self.wfile.write(cls.png)

            def log_message(self, *_args):
                pass

        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.base = f"http://127.0.0.1:{cls.server.server_port}"

    @classmethod
    def tearDownClass(cls):
        cls.gate.set()
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join(2)

    def setUp(self):
        self.display = ReplyDisplay()
        self.display.resize(700, 350)
        self.display.show()
        self.addCleanup(self.display.deleteLater)
        self.addCleanup(self.display.close)

    def wait_for(self, predicate):
        deadline = time.monotonic() + 3
        while not predicate() and time.monotonic() < deadline:
            QTest.qWait(10)
        self.assertTrue(predicate())

    def image_fragment(self):
        block = self.display.document().begin()
        while block.isValid():
            iterator = block.begin()
            while not iterator.atEnd():
                fragment = iterator.fragment()
                if fragment.isValid() and fragment.charFormat().isImageFormat():
                    return fragment
                iterator += 1
            block = block.next()
        self.fail("Image was not rendered")

    @patch("live_gpt.app.webbrowser.open_new_tab")
    def test_remote_thumbnail_loads_and_click_opens_original_in_resizable_window(self, open_tab):
        source = self.base + "/image"
        self.display.setHtml(f'<p>Before</p><p><img src="{source}" alt="Stormwind"></p><p>After</p>')
        self.wait_for(lambda: source in self.display._image_loader.cache)
        fragment = self.image_fragment()
        image_format = fragment.charFormat().toImageFormat()
        self.assertLessEqual(image_format.width(), 280)
        self.assertEqual(image_format.width() / image_format.height(), 2)
        resource = self.display.document().resource(QTextDocument.ResourceType.ImageResource, QUrl(image_format.name()))
        self.assertEqual(resource.size(), self.image.size())
        cursor = QTextCursor(self.display.document())
        cursor.setPosition(fragment.position())
        point = self.display.cursorRect(cursor).topLeft() + QPoint(20, 20)
        QTest.mouseClick(self.display.viewport(), Qt.MouseButton.LeftButton, pos=point)
        preview = self.display._image_preview
        self.assertIsNotNone(preview)
        self.assertTrue(preview.isVisible())
        self.assertEqual(preview.image.size(), self.image.size())
        self.assertGreater(preview.width(), image_format.width())
        preview.resize(600, 400)
        QApplication.processEvents()
        bounds = preview.view.transform().mapRect(preview.view.image_item.boundingRect())
        self.assertLessEqual(bounds.width(), preview.view.viewport().width())
        self.assertLessEqual(bounds.height(), preview.view.viewport().height())
        QTest.keyClick(preview, Qt.Key.Key_Escape)
        self.assertIsNone(self.display._image_preview)
        self.assertTrue(self.display.isVisible())
        open_tab.assert_not_called()

    def open_preview(self):
        source = self.base + "/image"
        self.display.setHtml(f'<img src="{source}">')
        self.wait_for(lambda: source in self.display._image_loader.cache)
        self.display._open_link(QUrl("live-gpt-image:0"))
        preview = self.display._image_preview
        QApplication.processEvents()
        self.addCleanup(preview.close)
        return preview

    def test_borderless_preview_maximizes_and_restores_previous_geometry(self):
        preview = self.open_preview()
        self.assertTrue(preview.windowFlags() & Qt.WindowType.FramelessWindowHint)
        self.assertEqual(preview.geometry().center(), self.display.screen().geometry().center())
        preview.setGeometry(30, 40, 600, 400)
        QApplication.processEvents()
        original = preview.geometry()
        original_icon = preview.maximize_button.icon().cacheKey()
        preview.maximize_button.click()
        QApplication.processEvents()
        self.assertTrue(preview.isMaximized())
        self.assertTrue(preview.size_grip.isHidden())
        self.assertNotEqual(preview.maximize_button.icon().cacheKey(), original_icon)
        preview.maximize_button.click()
        QApplication.processEvents()
        self.assertFalse(preview.isMaximized())
        self.assertEqual(preview.geometry(), original)
        self.assertTrue(preview.size_grip.isVisible())
        QTest.mouseDClick(preview.title_bar, Qt.MouseButton.LeftButton, pos=QPoint(5, 5))
        self.assertTrue(preview.isMaximized())

    def test_wheel_zooms_in_and_out_preserves_zoom_on_resize_and_fit_resets(self):
        preview = self.open_preview()
        view = preview.view
        initial_scale = view.transform().m11()
        point = view.viewport().rect().center()
        QTest.mouseMove(view.viewport(), point)

        def wheel(delta):
            event = QWheelEvent(QPointF(point), QPointF(view.viewport().mapToGlobal(point)),
                QPoint(), QPoint(0, delta), Qt.MouseButton.NoButton, Qt.KeyboardModifier.NoModifier,
                Qt.ScrollPhase.NoScrollPhase, False)
            QApplication.sendEvent(view.viewport(), event)
            QApplication.processEvents()
            self.assertTrue(event.isAccepted())

        wheel(120)
        self.assertAlmostEqual(view.transform().m11(), initial_scale * 1.2)
        self.assertAlmostEqual(view.zoom, 1.2)
        before = view.mapToScene(point)
        QTest.mousePress(view.viewport(), Qt.MouseButton.LeftButton, pos=point)
        self.assertEqual(view.viewport().cursor().shape(), Qt.CursorShape.ClosedHandCursor)
        QTest.mouseMove(view.viewport(), point + QPoint(35, 30))
        QTest.mouseRelease(view.viewport(), Qt.MouseButton.LeftButton, pos=point + QPoint(35, 30))
        after = view.mapToScene(point)
        self.assertLess(after.x(), before.x())
        self.assertLess(after.y(), before.y())
        self.assertAlmostEqual(view.zoom, 1.2)
        self.assertEqual(view.viewport().cursor().shape(), Qt.CursorShape.OpenHandCursor)
        wheel(120)
        self.assertAlmostEqual(view.zoom, 1.44)
        self.assertGreater(view.horizontalScrollBar().maximum(), 0)
        zoomed_scale = view.transform().m11()
        preview.resize(500, 300)
        QApplication.processEvents()
        self.assertAlmostEqual(view.transform().m11(), zoomed_scale)
        wheel(-120)
        self.assertAlmostEqual(view.zoom, 1.2)
        # Zoomed images can be panned rather than clipping inaccessible edges.
        scrollbar = view.horizontalScrollBar()
        before = scrollbar.value()
        QTest.mousePress(view.viewport(), Qt.MouseButton.LeftButton, pos=point)
        QTest.mouseMove(view.viewport(), point + QPoint(-40, 0))
        QTest.mouseRelease(view.viewport(), Qt.MouseButton.LeftButton, pos=point + QPoint(-40, 0))
        self.assertNotEqual(scrollbar.value(), before)
        view.fit_image()
        self.assertEqual(view.zoom, 1)
        bounds = view.transform().mapRect(view.image_item.boundingRect())
        self.assertLessEqual(bounds.width(), view.viewport().width())
        self.assertLessEqual(bounds.height(), view.viewport().height())
        self.assertEqual(view.horizontalScrollBar().maximum(), 0)

    def test_slow_download_keeps_ui_responsive_and_old_reply_completion_is_ignored(self):
        self.gate.clear()
        self.addCleanup(self.gate.set)
        source = self.base + "/slow"
        self.display.setHtml(f'<p><img src="{source}"></p>')
        self.wait_for(lambda: source in self.display._image_loader.pending)
        self.display._open_link(QUrl("live-gpt-image:0"))
        preview = self.display._image_preview
        self.assertIsNotNone(preview)
        self.assertIsNone(preview.image)
        self.display.setHtml("<p>Next reply</p>")
        self.gate.set()
        self.wait_for(lambda: source not in self.display._image_loader.pending)
        self.assertEqual(self.display.toPlainText(), "Next reply")
        self.assertIsNotNone(preview.image)
        preview.close()

    def test_failed_image_shows_unavailable_and_does_not_repeat_requests(self):
        source = self.base + "/missing"
        self.display.setHtml(f'<img src="{source}" alt="Missing">')
        self.wait_for(lambda: source in self.display._image_loader.failed)
        before = self.requests.count("/missing")
        self.display.setHtml(f'<img src="{source}">')
        QTest.qWait(30)
        self.assertEqual(self.requests.count("/missing"), before)
        self.display._open_link(QUrl("live-gpt-image:0"))
        preview = self.display._image_preview
        self.assertTrue(preview.picture.text())
        self.assertIsNone(preview.image)
        preview.close()

    def test_embedded_image_decodes_without_network_and_gallery_fits_narrow_view(self):
        import base64

        source = "data:image/png;base64," + base64.b64encode(self.png).decode()
        self.display.resize(420, 350)
        self.display.setHtml('<table cellspacing="8"><tr>' +
                             f'<td><img src="{source}"></td>' * 3 + '</tr></table>')
        QApplication.processEvents()
        self.assertEqual(self.display._image_loader.cache[source].size(), self.image.size())
        self.assertEqual(self.display._image_loader.pending, {})
        self.assertEqual(self.display.horizontalScrollBar().maximum(), 0)
        self.assertEqual(len(self.display._images), 3)

    def test_new_reply_replaces_cached_thumbnail_with_its_own_image(self):
        import base64

        first = "data:image/png;base64," + base64.b64encode(self.png).decode()
        self.display.setHtml(f'<img src="{first}">')
        QApplication.processEvents()
        other = QImage(80, 160, QImage.Format.Format_RGB32)
        other.fill(QColor("red"))
        buffer = QBuffer()
        buffer.open(QIODevice.OpenModeFlag.WriteOnly)
        other.save(buffer, "PNG")
        second = "data:image/png;base64," + base64.b64encode(bytes(buffer.data())).decode()
        self.display.setHtml(f'<img src="{second}">')
        QApplication.processEvents()
        resource = self.display.document().resource(QTextDocument.ResourceType.ImageResource, QUrl("live-gpt-image:0"))
        self.assertEqual(resource.size(), other.size())
        self.assertEqual(resource.pixelColor(0, 0), QColor("red"))

    def test_preview_keeps_overlay_reply_expanded_when_pointer_leaves(self):
        window = OverlayWindow()
        window.show()
        window.begin_response_display("Question")
        window.set_response_update("Writing…", "Reply")
        window.set_response_html(f'<img src="{self.base}/image">')
        window._expand_subtitle()
        display = window.subtitle_full_text
        display._open_link(QUrl("live-gpt-image:0"))
        with patch.object(window, "_pointer_hover_bounds") as bounds:
            bounds.return_value.contains.return_value = False
            window._collapse_subtitle_if_outside()
        self.assertTrue(window._subtitle_expanded)
        display._image_preview.close()
        window.close()
        window.deleteLater()

    def test_markup_preserves_text_links_and_rejects_local_image_paths(self):
        parser = ReplyImageMarkup()
        parser.feed('<p>A &amp; B <a href="https://example.com">Docs</a></p>'
                    '<img src="file:///private.png"><img src="https://example.com/image" alt="A &amp; B">')
        self.assertEqual(len(parser.images), 1)
        self.assertEqual(parser.images["live-gpt-image:0"][1], "A & B")
        self.assertNotIn("file:", "".join(parser.parts))
        self.assertIn('href="https://example.com"', "".join(parser.parts))
