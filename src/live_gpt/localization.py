"""Interface language selection and live Qt widget translation.

English source strings are the catalog keys. Unknown diagnostics are preserved,
and chat content, file paths, model identifiers, and shortcut values stay intact.
"""
from __future__ import annotations

import re
from string import Formatter

from PySide6.QtCore import QCoreApplication, QLibraryInfo, QObject, QLocale, QTranslator, Signal
from PySide6.QtGui import QAction
from PySide6.QtWidgets import (
    QAbstractButton, QComboBox, QLabel, QLineEdit, QPlainTextEdit, QSpinBox, QWidget,
)

from .translations_zh import TRANSLATIONS


def resolve_language(preference: str = "") -> str:
    if preference in ("en", "zh"):
        return preference
    # uiLanguages reflects the OS display-language preference, including Windows
    # systems whose regional format differs from their interface language.
    languages = QLocale.system().uiLanguages()
    language = languages[0] if languages else QLocale.system().name()
    return "zh" if language.replace("_", "-").lower().split("-")[0] == "zh" else "en"


class Localization(QObject):
    changed = Signal()

    def __init__(self) -> None:
        super().__init__()
        self.language = resolve_language()
        self._qt_translator: QTranslator | None = None
        self._qt_language = ""

    def _translate_qt(self, language: str) -> None:
        application = QCoreApplication.instance()
        if application is None or self._qt_language == language:
            return
        if self._qt_translator is not None:
            application.removeTranslator(self._qt_translator)
            self._qt_translator.deleteLater()
            self._qt_translator = None
        if language == "zh":
            self._qt_translator = QTranslator(self)
            path = QLibraryInfo.path(QLibraryInfo.LibraryPath.TranslationsPath)
            if self._qt_translator.load("qtbase_zh_CN", path):
                application.installTranslator(self._qt_translator)
        self._qt_language = language

    def set_language(self, preference: str) -> None:
        language = resolve_language(preference)
        self._translate_qt(language)
        if self.language != language:
            self.language = language
            self.changed.emit()


localization = Localization()


def tr(source: str, **values: object) -> str:
    translated = TRANSLATIONS.get(source, source) if localization.language == "zh" else source
    return translated.format(**values) if values else translated


_ENGLISH_SOURCES = {translated: source for source, translated in TRANSLATIONS.items()}


def _message_patterns() -> list[tuple[re.Pattern[str], str]]:
    patterns = []
    for source, translated in TRANSLATIONS.items():
        if "{" not in source:
            continue
        for template in (source, translated):
            parts = []
            for literal, field, _spec, _conversion in Formatter().parse(template):
                parts.append(re.escape(literal))
                if field is not None:
                    parts.append(f"(?P<{field}>.+?)")
            patterns.append((re.compile("".join(parts), re.DOTALL), source))
    return patterns


_MESSAGE_PATTERNS = _message_patterns()


def translate_message(message: str) -> str:
    """Translate authored status messages, preserving variable diagnostics."""
    source = message if message in TRANSLATIONS else _ENGLISH_SOURCES.get(message)
    if source is not None:
        return tr(source)
    for pattern, source in _MESSAGE_PATTERNS:
        match = pattern.fullmatch(message)
        if match:
            nested_messages = {"runtime", "model", "label", "mirror", "engine", "mode", "accuracy", "compute", "best_for", "timing"}
            return tr(source, **{
                key: translate_message(value) if key in nested_messages else value
                for key, value in match.groupdict().items()
            })
    return message


class UiTranslations(QObject):
    """Retain source text for UI properties and update them without rebuilding.

    Only catalogued interface strings are bound. Editable text and external
    content are never translated. Combo item data is left untouched, and signals
    are blocked while labels change so autosave cannot alter a preference.
    """

    def __init__(self, root: QWidget) -> None:
        super().__init__(root)
        self.root = root
        localization.changed.connect(self.refresh)
        self.refresh()

    def _property(self, widget: QObject, name: str) -> None:
        current = widget.property(name)
        if not isinstance(current, str):
            return
        key = "_translation_" + name
        previous = widget.property(key)
        source = previous if previous and current == widget.property(key + "_rendered") else None
        if source is None:
            source = current if current in TRANSLATIONS else _ENGLISH_SOURCES.get(current)
        if source is not None:
            widget.setProperty(key, source)
            rendered = tr(source)
            widget.setProperty(key + "_rendered", rendered)
            widget.setProperty(name, rendered)
        elif name in ("text", "toolTip", "placeholderText", "accessibleName"):
            widget.setProperty(name, translate_message(current))

    def refresh(self) -> None:
        for widget in [self.root, *self.root.findChildren(QWidget), *self.root.findChildren(QAction)]:
            for name in ("windowTitle", "toolTip", "accessibleName"):
                self._property(widget, name)
            if isinstance(widget, (QAbstractButton, QAction)) or (
                isinstance(widget, QLabel)
                and widget.objectName() not in ("subtitleLine", "dictationState", "voiceInstallLog")
            ):
                self._property(widget, "text")
            if isinstance(widget, (QLineEdit, QPlainTextEdit)):
                self._property(widget, "placeholderText")
            if isinstance(widget, QSpinBox):
                self._property(widget, "suffix")
            if isinstance(widget, QComboBox):
                blocked = widget.blockSignals(True)
                for index in range(widget.count()):
                    # Keep available interface languages readable in their own
                    # language, even when the current interface is Chinese.
                    if widget.objectName() == "languageCombo" and widget.itemData(index) in ("en", "zh"):
                        continue
                    # Conversation and window names belong to external apps.
                    if widget.objectName() in ("chatgptTabCombo", "captureSourceCombo") and widget.itemData(index) is not None:
                        continue
                    text = widget.itemText(index)
                    source = text if text in TRANSLATIONS else _ENGLISH_SOURCES.get(text)
                    if source is not None:
                        widget.setItemText(index, tr(source))
                    else:
                        widget.setItemText(index, translate_message(text))
                widget.blockSignals(blocked)


__all__ = ["UiTranslations", "localization", "resolve_language", "translate_message", "tr"]
