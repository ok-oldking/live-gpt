from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from string import Formatter
from types import SimpleNamespace
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QCoreApplication, QLocale
from PySide6.QtGui import QAction, QPalette
from PySide6.QtWidgets import QApplication, QMenu

from live_gpt.app import HotkeyConfigDialog, OverlayWindow
from live_gpt.config import Config
from live_gpt.localization import UiTranslations, localization, resolve_language, translate_message, tr
from live_gpt.screen_capture import CaptureSource
from live_gpt.translations_zh import TRANSLATIONS


class LocalizationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.application = QApplication.instance() or QApplication([])

    def setUp(self) -> None:
        localization.set_language("en")

    def tearDown(self) -> None:
        localization.set_language("en")

    def test_system_display_language_and_unsupported_fallback(self) -> None:
        for languages, expected in (
            (["zh-Hans-CN", "en-US"], "zh"),
            (["zh_TW"], "zh"),
            (["en-US", "zh-CN"], "en"),
            (["fr-FR"], "en"),
            ([], "en"),
        ):
            with self.subTest(languages=languages), patch(
                "live_gpt.localization.QLocale.system",
                return_value=SimpleNamespace(uiLanguages=lambda: languages, name=lambda: "en_US"),
            ):
                self.assertEqual(resolve_language(), expected)
                self.assertEqual(resolve_language("en"), "en")
                self.assertEqual(resolve_language("zh"), "zh")

    def test_catalog_preserves_format_fields_and_unknown_diagnostics(self) -> None:
        fields = lambda value: {field for _, field, _, _ in Formatter().parse(value) if field is not None}
        for source, translation in TRANSLATIONS.items():
            with self.subTest(source=source):
                self.assertEqual(fields(source), fields(translation))
        localization.set_language("zh")
        message = "Could not capture screenshot: window 123 unavailable"
        translated = translate_message(message)
        self.assertEqual(translated, "无法截图：window 123 unavailable")
        self.assertEqual(translate_message("Unexpected driver error 0x123"), "Unexpected driver error 0x123")
        self.assertEqual(translate_message("Pet ready and selected: Settings"), "宠物已准备就绪并选中：Settings")
        self.assertEqual(translate_message("Missing GPT-SoVITS file: English"), "缺少 GPT-SoVITS 文件：English")
        localization.set_language("en")
        self.assertEqual(translate_message(translated), message)

    def test_first_launch_selects_and_saves_matching_language(self) -> None:
        for locale, expected in (("zh_CN", "zh"), ("en_US", "en"), ("fr_FR", "en")):
            with self.subTest(locale=locale), tempfile.TemporaryDirectory() as directory, patch(
                "live_gpt.localization.QLocale.system", return_value=QLocale(locale)
            ):
                path = Path(directory) / "config.json"
                config = Config(path)
                self.assertEqual(config["language"], expected)
                self.assertEqual(json.loads(path.read_text(encoding="utf-8"))["language"], expected)
                dialog = HotkeyConfigDialog("Right Alt", config=config)
                try:
                    self.assertEqual(dialog.language_combo.count(), 2)
                    self.assertEqual(dialog.language_combo.findData("system"), -1)
                    self.assertEqual(dialog.language_combo.currentData(), expected)
                    self.assertEqual(dialog.language_combo.currentText(), "简体中文" if expected == "zh" else "English")
                finally:
                    dialog.close()

    def test_empty_missing_and_legacy_language_values_are_detected_and_saved(self) -> None:
        for locale, expected in (("zh_CN", "zh"), ("en_US", "en")):
            for loaded in ({}, {"language": ""}, {"language": "system"}):
                with self.subTest(locale=locale, loaded=loaded), tempfile.TemporaryDirectory() as directory, patch(
                    "live_gpt.localization.QLocale.system", return_value=QLocale(locale)
                ):
                    path = Path(directory) / "config.json"
                    path.write_text(json.dumps(loaded), encoding="utf-8")
                    config = Config(path)
                    self.assertEqual(config["language"], expected)
                    self.assertEqual(json.loads(path.read_text(encoding="utf-8"))["language"], expected)
                    self.assertEqual(config["hotkey_hold"], "Right Alt")

    def test_explicit_choices_survive_restart_with_different_system_language(self) -> None:
        with tempfile.TemporaryDirectory() as directory, patch(
            "live_gpt.localization.QLocale.system", return_value=QLocale("zh_CN")
        ):
            path = Path(directory) / "config.json"
            config = Config(path)
            self.assertEqual(config["language"], "zh")
            dialog = HotkeyConfigDialog("Right Alt", config=config)
            try:
                self.assertEqual(dialog.windowTitle(), "Live GPT 设置")
                self.assertEqual(dialog.language_combo.currentText(), "简体中文")
                dialog.language_combo.setCurrentIndex(dialog.language_combo.findData("en"))
                self.assertEqual(dialog.windowTitle(), "Live GPT settings")
                self.assertEqual(Config(path)["language"], "en")
            finally:
                dialog.close()
            dialog = HotkeyConfigDialog("Right Alt", config=Config(path))
            try:
                self.assertEqual(dialog.windowTitle(), "Live GPT settings")
                dialog.language_combo.setCurrentIndex(dialog.language_combo.findData("zh"))
                self.assertEqual(dialog.windowTitle(), "Live GPT 设置")
                self.assertEqual(json.loads(path.read_text(encoding="utf-8"))["language"], "zh")
            finally:
                dialog.close()
            with patch("live_gpt.localization.QLocale.system", return_value=QLocale("en_US")):
                self.assertEqual(Config(path)["language"], "zh")

    def test_live_switch_preserves_editor_shortcuts_and_selection_data(self) -> None:
        overlay = OverlayWindow()
        menu = QMenu()
        action = QAction(tr("Exit"), menu)
        menu.addAction(action)
        binding = UiTranslations(menu)
        display = CaptureSource("display:1", "Screenshot desktop", "display", 0, 0, 1920, 1080)
        window = CaptureSource("window:1", "Settings", "window", 0, 0, 900, 600)
        overlay.set_capture_sources([display, window])
        overlay.set_chatgpt_tabs([{"title": "Settings", "id": "tab", "url": "https://chatgpt.com/c/1"}])
        overlay.capture_source_combo.setCurrentIndex(2)
        overlay.set_transcript("Settings\n用户输入 {raw}")
        overlay.set_status("Browser connected")
        dialog = HotkeyConfigDialog("Right Alt", language="en")
        dialog.stt_test_result.setText("untranslated transcript")
        dialog.voice_test_text.setText("User's playback text")
        try:
            for language in ("zh", "en", "zh"):
                dialog.language_combo.setCurrentIndex(dialog.language_combo.findData(language))
                self.assertEqual(overlay.transcript_area.toPlainText(), "Settings\n用户输入 {raw}")
                self.assertEqual(overlay.chatgpt_tab_combo.currentText(), "Settings")
                self.assertEqual(overlay.capture_source_combo.currentData(), window)
                self.assertEqual(overlay.capture_source_combo.currentText(), "Settings")
                self.assertEqual(dialog.hold_microphone_edit.keySequence(), "Right Alt")
                self.assertEqual(dialog.stt_test_result.text(), "untranslated transcript")
                self.assertEqual(dialog.voice_test_text.text(), "User's playback text")
                self.assertEqual(dialog.stt_model(), "zh_zipformer_ctc_int8_2025_07_03")
                self.assertEqual(action.text(), "退出" if language == "zh" else "Exit")
                self.assertEqual(overlay.transcript_area.placeholderText(), "浏览器已连接" if language == "zh" else "Browser connected")
                self.assertEqual(overlay.capture_source_combo.itemText(1), "桌面截图" if language == "zh" else "Screenshot desktop")
            overlay.begin_dictation_waiting()
            self.assertEqual(overlay.dictation_state_label.text(), "正在等待浏览器开始聆听…")
            overlay.set_dictation_partial("Settings 用户说的话")
            localization.set_language("en")
            self.assertEqual(overlay.dictation_state_label.text(), "Listening…\n\nSettings 用户说的话")
            overlay.set_status("Could not send to ChatGPT: unavailable", error=True)
            localization.set_language("zh")
            self.assertEqual(overlay.transcript_area.placeholderText(), "无法发送到 ChatGPT：unavailable")
            self.assertEqual(
                overlay.transcript_area.palette().color(QPalette.ColorRole.PlaceholderText).name(),
                "#ff667a",
            )
        finally:
            dialog.close()
            overlay.close()
            menu.close()
            del binding

    def test_native_qt_buttons_follow_language(self) -> None:
        localization.set_language("zh")
        self.assertEqual(QCoreApplication.translate("QPlatformTheme", "Cancel"), "取消")
        localization.set_language("en")
        self.assertEqual(QCoreApplication.translate("QPlatformTheme", "Cancel"), "Cancel")


if __name__ == "__main__":
    unittest.main()
