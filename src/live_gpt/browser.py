from __future__ import annotations

import math
import re
import threading
import time
from dataclasses import dataclass, field
from queue import Empty, Queue
from typing import Any

from PySide6.QtCore import QThread, Signal

from .browser_discovery import (
    DEFAULT_DEBUG_PORTS,
    active_port_endpoint,
    active_port_endpoints,
    browser_user_data_directories,
    current_user_data_directories,
    discover_cdp_endpoint,
    is_chatgpt_url,
)
from .browser_windows import (
    open_remote_debugging_settings,
    windows_default_browser_executable,
)
from .logger import Logger


logger = Logger.get_logger(__name__)
ASSISTANT_TURN_SELECTOR = (
    '[data-testid^="conversation-turn-"][data-turn="assistant"]'
)
CHATGPT_COMPOSER_SELECTOR = (
    '[data-composer-surface="true"]:visible '
    '#prompt-textarea[contenteditable="true"]:visible, '
    '[data-composer-surface="true"]:visible '
    '[contenteditable="true"][role="textbox"]:visible, '
    '#prompt-textarea[contenteditable="true"]:visible, '
    'textarea[name="prompt-textarea"]:visible'
)
DICTATION_RESULT_TIMEOUT_MS = 20_000
DICTATION_RESULT_POLL_INTERVAL_MS = 200
DICTATION_RESULT_POLL_COUNT = (
    DICTATION_RESULT_TIMEOUT_MS // DICTATION_RESULT_POLL_INTERVAL_MS
)
REMOTE_DEBUGGING_RETRY_INTERVAL_SECONDS = 3.0
DICTATION_END_SELECTORS = (
    'button[aria-label="Submit dictation"]',
    'button[aria-label="Done"]',
    'button[aria-label="Stop dictation"]',
    'button[aria-label="Finish dictation"]',
    'button[aria-label="Stop recording"]',
    'button[data-testid="composer-dictation-done-button"]',
    'button[data-testid="dictation-done-button"]',
    'button:text-is("Done")',
)
DICTATION_CANCEL_SELECTORS = (
    'button[aria-label="Cancel dictation"]',
    'button[aria-label="Cancel recording"]',
    'button[aria-label*="cancel" i][aria-label*="dictation" i]',
    'button[aria-label*="cancel" i][aria-label*="record" i]',
    'button[data-testid="composer-dictation-cancel-button"]',
    'button[data-testid="dictation-cancel-button"]',
    'button[data-testid*="dictation"][data-testid*="cancel"]',
    'form button:text-is("Cancel")',
)
_MEDIA_TRACKER_SCRIPT = """
() => {
    if (window.__liveGptMediaTrackerInstalled) return;
    window.__liveGptMediaTrackerInstalled = true;
    window.__liveGptReadAloudTracker = {
        media: null,
        playCount: 0
    };
    const originalPlay = HTMLMediaElement.prototype.play;
    HTMLMediaElement.prototype.play = function(...args) {
        const tracker = window.__liveGptReadAloudTracker;
        tracker.media = this;
        tracker.playCount += 1;
        return originalPlay.apply(this, args);
    };
}
"""
_MEDIA_PROGRESS_SCRIPT = """
() => {
    const tracker = window.__liveGptReadAloudTracker;
    const media = tracker?.media ||
        [...document.querySelectorAll('audio, video')].find(item =>
            !item.paused || item.currentTime > 0
        );
    if (!media) return null;
    return {
        playCount: tracker?.playCount || 0,
        currentTime: Number.isFinite(media.currentTime)
            ? media.currentTime : 0,
        duration: Number.isFinite(media.duration)
            ? media.duration : null,
        paused: media.paused,
        ended: media.ended
    };
}
"""


@dataclass
class _MonitorState:
    browser: Any | None = None
    settings_opened: bool = False
    next_settings_attempt: float = 0.0
    retry_endpoint: str | None = None
    next_connection_attempt: float = 0.0
    last_tabs: list[dict[str, str]] | None = None
    media_reset_pages: set[str] = field(default_factory=set)
    active_response: _ActiveResponse | None = None
    active_reading: _ActiveReading | None = None


@dataclass(frozen=True)
class _SendRequest:
    tab_id: str
    text: str
    screenshot_webp: bytes | None = None
    preserve_attachments: bool = False


@dataclass(frozen=True)
class _AttachmentRequest:
    action: str
    tab_id: str
    screenshot_webp: bytes | None = None


@dataclass(frozen=True)
class _DictationRequest:
    action: str
    tab_id: str


@dataclass(frozen=True)
class _ClearRequest:
    tab_id: str


@dataclass
class _ActiveResponse:
    page: Any
    turn_marker_before: str | None
    started_at: float = field(default_factory=time.monotonic)
    last_text: str = ""
    last_status: str = ""
    last_text_changed_at: float = field(default_factory=time.monotonic)
    completion_candidate_at: float | None = None


@dataclass(frozen=True)
class _ResponseSnapshot:
    has_new_turn: bool
    is_generating: bool
    has_completion_controls: bool
    text: str
    status: str


@dataclass
class _ActiveReading:
    page: Any
    full_text: str
    subtitles: tuple[str, ...]
    started_at: float = field(default_factory=time.monotonic)
    last_subtitle: str = ""
    last_progress_fraction: float | None = None
    audio_seen: bool = False
    playback_started_at: float | None = None
    last_play_count: int = 0
    quiet_since: float | None = None


class BrowserMonitor(QThread):
    """Own Playwright on a worker thread and publish live ChatGPT tab state."""

    tabs_changed = Signal(object)
    status_changed = Signal(str)
    send_finished = Signal(bool, str, str)
    response_changed = Signal(str, str)
    response_finished = Signal(bool, str)
    reading_started = Signal(str)
    reading_changed = Signal(object)
    reading_finished = Signal(bool, str)
    local_voice_requested = Signal(str)
    local_voice_updated = Signal(str, bool)
    dictation_started = Signal(bool, str)
    dictation_finished = Signal(bool, str, str)
    clear_finished = Signal(bool, str)

    def __init__(self) -> None:
        super().__init__()
        self._stop_requested = False
        self._retry_connection_requested = False
        self._send_requests: Queue[_SendRequest] = Queue()
        self._attachment_requests: Queue[_AttachmentRequest] = Queue()
        self._dictation_requests: Queue[_DictationRequest] = Queue()
        self._clear_requests: Queue[_ClearRequest] = Queue()
        self._wake_event = threading.Event()
        self._dictation_initial_text: dict[str, str] = {}
        self._last_status = ""
        self._use_browser_voice = True

    def set_use_browser_voice(self, enabled: bool) -> None:
        self._use_browser_voice = bool(enabled)

    def request_stop(self) -> None:
        self._stop_requested = True
        self._wake_event.set()

    def request_retry_connection(self) -> None:
        self._retry_connection_requested = True
        self._wake_event.set()

    def request_send(
        self,
        tab_id: str,
        text: str,
        screenshot_webp: bytes | None = None,
        *,
        preserve_attachments: bool = False,
    ) -> None:
        """Queue text for the selected ChatGPT page on the worker thread."""
        self._send_requests.put(
            _SendRequest(
                tab_id=tab_id,
                text=text,
                screenshot_webp=screenshot_webp,
                preserve_attachments=preserve_attachments,
            )
        )
        self._wake_event.set()

    def request_replace_attachment(
        self,
        tab_id: str,
        screenshot_webp: bytes,
    ) -> None:
        self._attachment_requests.put(
            _AttachmentRequest("replace", tab_id, screenshot_webp)
        )
        self._wake_event.set()

    def request_clear_attachments(self, tab_id: str) -> None:
        self._attachment_requests.put(_AttachmentRequest("clear", tab_id))
        self._wake_event.set()

    def request_start_dictation(self, tab_id: str) -> None:
        self._dictation_requests.put(
            _DictationRequest(action="start", tab_id=tab_id)
        )
        self._wake_event.set()

    def request_finish_dictation(self, tab_id: str) -> None:
        self._dictation_requests.put(
            _DictationRequest(action="finish", tab_id=tab_id)
        )
        self._wake_event.set()

    def request_cancel_dictation(self, tab_id: str) -> None:
        self._dictation_requests.put(
            _DictationRequest(action="cancel", tab_id=tab_id)
        )
        self._wake_event.set()

    def request_clear(self, tab_id: str) -> None:
        self._clear_requests.put(_ClearRequest(tab_id=tab_id))
        self._wake_event.set()

    def run(self) -> None:
        try:
            from playwright.sync_api import Error as PlaywrightError
            from playwright.sync_api import sync_playwright
        except ImportError:
            self._set_status("Playwright is not installed")
            self.tabs_changed.emit([])
            return

        state = _MonitorState()
        with sync_playwright() as playwright:
            while not self._stop_requested:
                if not self._is_connected(state.browser):
                    self._try_connect(playwright, PlaywrightError, state)

                if self._is_connected(state.browser):
                    tabs = self._collect_chatgpt_tabs(
                        state.browser,
                        PlaywrightError,
                        state.media_reset_pages,
                    )
                    if tabs is None:
                        self._handle_disconnect(state)
                    elif tabs != state.last_tabs:
                        state.last_tabs = tabs
                        self.tabs_changed.emit(tabs)
                elif state.last_tabs:
                    state.last_tabs = []
                    self.tabs_changed.emit([])

                connected_browser = (
                    state.browser
                    if self._is_connected(state.browser)
                    else None
                )
                self._process_dictation_requests(
                    connected_browser,
                    PlaywrightError,
                )
                self._process_attachment_requests(
                    connected_browser,
                    PlaywrightError,
                )
                self._process_clear_requests(
                    connected_browser,
                    PlaywrightError,
                )
                self._process_send_requests(
                    connected_browser,
                    PlaywrightError,
                    state,
                )
                self._poll_active_response(state)
                self._poll_active_reading(state)
                is_busy = (
                    state.active_response is not None
                    or state.active_reading is not None
                )
                self._wake_event.wait(0.25 if is_busy else 1.5)
                self._wake_event.clear()

        logger.info("Browser monitor stopped")

    def _try_connect(
        self,
        playwright: Any,
        playwright_error: type[Exception],
        state: _MonitorState,
    ) -> None:
        state.browser = None
        endpoint = discover_cdp_endpoint()
        if self._stop_requested:
            return

        if self._retry_connection_requested:
            self._retry_connection_requested = False
            state.retry_endpoint = None
            state.next_connection_attempt = 0.0

        if endpoint is None:
            self._prepare_remote_debugging(state)
            return

        if (
            endpoint == state.retry_endpoint
            and time.monotonic() < state.next_connection_attempt
        ):
            self._set_status(
                "Waiting for remote debugging approval; retrying automatically…"
            )
            return

        self._set_status("Approve remote debugging in the browser…")
        try:
            state.browser = playwright.chromium.connect_over_cdp(
                endpoint,
                timeout=15_000,
            )
        except playwright_error as error:
            logger.warning(
                "Unable to connect to remote-debug browser "
                f"endpoint={endpoint!r}: {error}"
            )
            state.retry_endpoint = endpoint
            state.next_connection_attempt = (
                time.monotonic()
                + REMOTE_DEBUGGING_RETRY_INTERVAL_SECONDS
            )
            self._set_status(
                "Waiting for remote debugging approval; retrying automatically…"
            )
            return

        state.retry_endpoint = None
        state.next_connection_attempt = 0.0
        state.media_reset_pages.clear()
        state.settings_opened = False
        self._set_status("Browser connected")
        logger.info(f"Connected to browser endpoint={endpoint!r}")

    def _prepare_remote_debugging(self, state: _MonitorState) -> None:
        if (
            not state.settings_opened
            and time.monotonic() >= state.next_settings_attempt
        ):
            self._set_status("Opening remote debugging settings…")
            try:
                open_remote_debugging_settings()
                state.settings_opened = True
            except Exception as error:
                logger.error("Unable to open remote debugging settings", error)
                self._set_status(str(error))
            state.next_settings_attempt = time.monotonic() + 30

        if state.settings_opened:
            self._set_status("Enable remote debugging in the browser")

    @staticmethod
    def _collect_chatgpt_tabs(
        browser: Any,
        playwright_error: type[Exception],
        media_reset_pages: set[str],
    ) -> list[dict[str, str]] | None:
        try:
            contexts = list(browser.contexts)
        except playwright_error:
            return None

        tabs: list[dict[str, str]] = []
        for context in contexts:
            try:
                pages = list(context.pages)
            except playwright_error:
                continue
            for page in pages:
                try:
                    if page.is_closed() or not is_chatgpt_url(page.url):
                        continue
                    page_id = str(id(page))
                    if page_id not in media_reset_pages:
                        page.emulate_media(color_scheme="null")
                        media_reset_pages.add(page_id)
                    title = page.title().strip() or "ChatGPT"
                    tabs.append(
                        {"id": page_id, "title": title, "url": page.url}
                    )
                except playwright_error:
                    continue

        media_reset_pages.intersection_update(tab["id"] for tab in tabs)
        tabs.sort(key=lambda tab: tab["title"].casefold())
        return tabs

    def _handle_disconnect(self, state: _MonitorState) -> None:
        if state.active_response is not None:
            self.response_finished.emit(False, "Browser disconnected")
            state.active_response = None
        if state.active_reading is not None:
            self.reading_finished.emit(False, "Browser disconnected")
            state.active_reading = None
        state.browser = None
        state.media_reset_pages.clear()
        self._set_status("Browser disconnected")
        if state.last_tabs != []:
            state.last_tabs = []
            self.tabs_changed.emit([])

    def _process_dictation_requests(
        self,
        browser: Any | None,
        playwright_error: type[Exception],
    ) -> None:
        while True:
            try:
                request = self._dictation_requests.get_nowait()
            except Empty:
                return

            if browser is None:
                self._emit_dictation_failure(
                    request.action,
                    "Browser is not connected",
                )
                continue

            page = self._find_chatgpt_page(
                browser,
                request.tab_id,
                playwright_error,
            )
            if page is None:
                self._emit_dictation_failure(
                    request.action,
                    "Selected ChatGPT window is no longer available",
                )
                continue

            try:
                if request.action == "start":
                    self._cancel_existing_dictation(page)
                    initial_text = self._read_composer_text(page)
                    self._start_browser_dictation(page)
                    self._dictation_initial_text[request.tab_id] = initial_text
                    self.dictation_started.emit(
                        True,
                        "Browser dictation is listening",
                    )
                elif request.action == "finish":
                    initial_text = self._dictation_initial_text.pop(
                        request.tab_id,
                        "",
                    )
                    text = self._finish_browser_dictation(
                        page,
                        initial_text,
                    )
                    self.dictation_finished.emit(
                        True,
                        text,
                        "Dictation copied from ChatGPT",
                    )
                else:
                    initial_text = self._dictation_initial_text.pop(
                        request.tab_id,
                        "",
                    )
                    text = self._cancel_browser_dictation(
                        page,
                        initial_text,
                    )
                    self.dictation_finished.emit(
                        True,
                        text,
                        "Dictation cancelled",
                    )
            except Exception as error:
                logger.error(
                    f"Unable to {request.action} browser dictation",
                    error,
                )
                self._emit_dictation_failure(
                    request.action,
                    f"Could not {request.action} browser dictation: {error}",
                )

    def _emit_dictation_failure(self, action: str, message: str) -> None:
        if action == "start":
            self.dictation_started.emit(False, message)
        else:
            self.dictation_finished.emit(False, "", message)

    def _process_attachment_requests(
        self,
        browser: Any | None,
        playwright_error: type[Exception],
    ) -> None:
        while True:
            try:
                request = self._attachment_requests.get_nowait()
            except Empty:
                return

            if browser is None:
                logger.warning(
                    "Unable to update dictation screenshot: browser is not connected"
                )
                continue
            page = self._find_chatgpt_page(
                browser,
                request.tab_id,
                playwright_error,
            )
            if page is None:
                logger.warning(
                    "Unable to update dictation screenshot: ChatGPT tab is gone"
                )
                continue
            try:
                self._clear_chatgpt_attachments(page)
                if request.action == "replace":
                    if request.screenshot_webp is None:
                        raise RuntimeError("The screenshot data is empty")
                    self._paste_screenshot(page, request.screenshot_webp)
                    logger.info(
                        "Pasted dictation screenshot into ChatGPT "
                        f"tab_id={request.tab_id!r}"
                    )
                else:
                    logger.info(
                        "Removed pending dictation screenshot from ChatGPT "
                        f"tab_id={request.tab_id!r}"
                    )
            except Exception as error:
                logger.error("Unable to update dictation screenshot", error)

    def _process_clear_requests(
        self,
        browser: Any | None,
        playwright_error: type[Exception],
    ) -> None:
        while True:
            try:
                request = self._clear_requests.get_nowait()
            except Empty:
                return

            if browser is None:
                self.clear_finished.emit(False, "Browser is not connected")
                continue

            page = self._find_chatgpt_page(
                browser,
                request.tab_id,
                playwright_error,
            )
            if page is None:
                self.clear_finished.emit(
                    False,
                    "Selected ChatGPT window is no longer available",
                )
                continue

            try:
                self._clear_chatgpt_composer(page)
            except Exception as error:
                logger.error("Unable to clear ChatGPT input", error)
                self.clear_finished.emit(
                    False,
                    f"Could not clear ChatGPT input: {error}",
                )
                continue

            self.clear_finished.emit(True, "Text cleared")

    def _process_send_requests(
        self,
        browser: Any | None,
        playwright_error: type[Exception],
        state: _MonitorState,
    ) -> None:
        while True:
            try:
                request = self._send_requests.get_nowait()
            except Empty:
                return

            if browser is None:
                self.send_finished.emit(
                    False,
                    request.text,
                    "Browser is not connected",
                )
                continue

            if (
                state.active_response is not None
                or state.active_reading is not None
            ):
                self.send_finished.emit(
                    False,
                    request.text,
                    "Wait for the current reply or reading to finish",
                )
                continue

            page = self._find_chatgpt_page(
                browser,
                request.tab_id,
                playwright_error,
            )
            if page is None:
                self.send_finished.emit(
                    False,
                    request.text,
                    "Selected ChatGPT window is no longer available",
                )
                continue

            try:
                turn_marker_before = self._assistant_turn_marker(page)
                self._send_to_chatgpt_page(
                    page,
                    request.text,
                    request.screenshot_webp,
                    preserve_attachments=request.preserve_attachments,
                )
            except Exception as error:
                logger.error("Unable to send text to ChatGPT", error)
                self.send_finished.emit(
                    False,
                    request.text,
                    f"Could not send to ChatGPT: {error}",
                )
                continue

            logger.info(
                "Sent text to ChatGPT "
                f"tab_id={request.tab_id!r} characters={len(request.text)}"
            )
            state.active_response = _ActiveResponse(
                page=page,
                turn_marker_before=turn_marker_before,
            )
            self.send_finished.emit(True, request.text, "Sent to ChatGPT")

    def _poll_active_response(self, state: _MonitorState) -> None:
        response = state.active_response
        if response is None:
            return

        try:
            snapshot = self._response_snapshot(
                response.page,
                response.turn_marker_before,
            )
        except Exception as error:
            logger.error("Unable to read the ChatGPT response", error)
            if not self._use_browser_voice and response.last_text:
                self.local_voice_updated.emit(response.last_text, True)
            self.response_finished.emit(
                False,
                f"Could not read ChatGPT's reply: {error}",
            )
            state.active_response = None
            return

        now = time.monotonic()
        text_changed = snapshot.text != response.last_text
        if text_changed:
            logger.debug(
                "ChatGPT response text changed "
                f"characters={len(snapshot.text)} "
                f"generating={snapshot.is_generating} "
                f"completion_controls={snapshot.has_completion_controls}"
            )
            response.last_text = snapshot.text
            response.last_text_changed_at = now
            response.completion_candidate_at = None

        if (
            text_changed
            or snapshot.status != response.last_status
        ):
            response.last_status = snapshot.status
            self.response_changed.emit(snapshot.status, snapshot.text)
        if text_changed and not self._use_browser_voice and snapshot.text:
            self.local_voice_updated.emit(snapshot.text, False)
        completion_candidate = (
            snapshot.has_new_turn
            and bool(snapshot.text)
            and not snapshot.is_generating
            and snapshot.has_completion_controls
        )
        if completion_candidate:
            if response.completion_candidate_at is None:
                response.completion_candidate_at = now
                logger.debug(
                    "ChatGPT response completion candidate started "
                    f"characters={len(snapshot.text)}"
                )
        else:
            response.completion_candidate_at = None

        response_age = now - response.started_at
        text_stable_for = now - response.last_text_changed_at
        controls_stable_for = (
            now - response.completion_candidate_at
            if response.completion_candidate_at is not None
            else 0.0
        )
        is_complete = (
            completion_candidate
            and text_stable_for >= 2.0
            and controls_stable_for >= 2.0
            and response_age >= 2.0
        )
        if not is_complete:
            if response_age >= 600:
                if not self._use_browser_voice and response.last_text:
                    self.local_voice_updated.emit(response.last_text, True)
                self.response_finished.emit(
                    False,
                    "Timed out while waiting for ChatGPT's reply",
                )
                state.active_response = None
            return

        if not self._use_browser_voice:
            logger.info(
                "ChatGPT response is stable; handing reply to local voice "
                f"characters={len(snapshot.text)}"
            )
            self.response_finished.emit(
                True,
                "Reply complete · Local voice queued",
            )
            self.local_voice_updated.emit(snapshot.text, True)
            self.local_voice_requested.emit(snapshot.text)
            state.active_response = None
            return

        logger.info(
            "ChatGPT response is stable; attempting Read aloud "
            f"characters={len(snapshot.text)} "
            f"stable_seconds={text_stable_for:.2f}"
        )
        read_aloud_clicked = self._click_read_aloud(response.page)
        if read_aloud_clicked:
            state.active_reading = _ActiveReading(
                page=response.page,
                full_text=snapshot.text,
                subtitles=self._subtitle_segments(snapshot.text),
            )
            self.reading_started.emit("Preparing Read aloud…")
        else:
            self.response_finished.emit(
                True,
                "Reply complete · Read aloud was unavailable",
            )
        state.active_response = None

    def _poll_active_reading(self, state: _MonitorState) -> None:
        reading = state.active_reading
        if reading is None:
            return

        now = time.monotonic()
        elapsed_since_click = now - reading.started_at
        estimated_duration = self._estimated_reading_duration(
            reading.full_text
        )
        try:
            media = reading.page.evaluate(_MEDIA_PROGRESS_SCRIPT)
        except Exception as error:
            logger.warning(f"Unable to read playback progress: {error}")
            media = None

        fraction: float | None = None
        finished = False
        if isinstance(media, dict):
            play_count = int(media.get("playCount") or 0)
            current_time = self._finite_float(media.get("currentTime")) or 0.0
            duration = self._finite_float(media.get("duration"))
            playback_has_started = play_count > 0 and (
                current_time > 0.02 or not bool(media.get("paused"))
            )
            if playback_has_started:
                reading.audio_seen = True
                if reading.playback_started_at is None:
                    reading.playback_started_at = now

            if play_count != reading.last_play_count:
                reading.last_play_count = play_count
                reading.quiet_since = None

            if reading.playback_started_at is not None:
                playback_elapsed = now - reading.playback_started_at
                is_full_response_audio = (
                    duration is not None
                    and duration >= estimated_duration * 0.35
                )
                if is_full_response_audio:
                    fraction = min(
                        max(current_time / duration, 0.0),
                        1.0,
                    )
                else:
                    fraction = min(
                        playback_elapsed / estimated_duration,
                        0.99,
                    )

            is_playing = (
                reading.audio_seen
                and not bool(media.get("paused"))
                and not bool(media.get("ended"))
            )
            if is_playing:
                reading.quiet_since = None
            elif reading.audio_seen:
                if reading.quiet_since is None:
                    reading.quiet_since = now
                finished = now - reading.quiet_since >= 4.0
        elif reading.audio_seen:
            if reading.quiet_since is None:
                reading.quiet_since = now
            finished = now - reading.quiet_since >= 4.0
        elif elapsed_since_click >= 8.0:
            if reading.playback_started_at is None:
                reading.playback_started_at = now
            playback_elapsed = now - reading.playback_started_at
            fraction = min(playback_elapsed / estimated_duration, 1.0)
            finished = playback_elapsed >= estimated_duration

        if fraction is None:
            return

        if (
            reading.last_progress_fraction is None
            or abs(fraction - reading.last_progress_fraction) >= 0.0001
        ):
            reading.last_progress_fraction = fraction
            self.reading_changed.emit(
                {
                    "text": reading.full_text,
                    "fraction": fraction,
                }
            )

        if not finished:
            return

        logger.info(
            "ChatGPT Read aloud finished "
            f"media_detected={reading.audio_seen}"
        )
        self.reading_finished.emit(True, "Read aloud complete")
        state.active_reading = None

    @staticmethod
    def _find_chatgpt_page(
        browser: Any,
        tab_id: str,
        playwright_error: type[Exception],
    ) -> Any | None:
        try:
            contexts = list(browser.contexts)
        except playwright_error:
            return None

        for context in contexts:
            try:
                pages = list(context.pages)
            except playwright_error:
                continue
            for page in pages:
                try:
                    if (
                        str(id(page)) == tab_id
                        and not page.is_closed()
                        and is_chatgpt_url(page.url)
                    ):
                        return page
                except playwright_error:
                    continue
        return None

    @classmethod
    def _send_to_chatgpt_page(
        cls,
        page: Any,
        text: str,
        screenshot_webp: bytes | None = None,
        *,
        preserve_attachments: bool = False,
    ) -> None:
        composer = page.locator(CHATGPT_COMPOSER_SELECTOR).first
        composer.wait_for(state="visible", timeout=5_000)
        if not preserve_attachments:
            cls._clear_chatgpt_attachments(page)
        composer.fill(text)
        if screenshot_webp is not None:
            cls._paste_screenshot(page, screenshot_webp)

        send_button = page.locator(
            'button[data-testid="send-button"]'
        ).first
        send_button.wait_for(state="visible", timeout=5_000)
        send_button.click(timeout=15_000)

    @classmethod
    def _clear_chatgpt_attachments(cls, page: Any) -> None:
        remove_buttons = page.locator(
            'button[aria-label="Remove file"], '
            'button[aria-label="Remove attachment"], '
            'button[aria-label="Remove image"], '
            'button[data-testid*="remove"][data-testid*="file"], '
            'button[data-testid*="remove"][data-testid*="attachment"]'
        )
        for _ in range(min(int(remove_buttons.count()), 20)):
            button = remove_buttons.last
            if not cls._locator_is_visible(button):
                break
            button.click(timeout=5_000)
            page.wait_for_timeout(100)

    @staticmethod
    def _paste_screenshot(page: Any, screenshot_webp: bytes) -> None:
        file_inputs = page.locator(
            'input[type="file"][accept*="image"], input[type="file"]'
        )
        if int(file_inputs.count()) == 0:
            raise RuntimeError("ChatGPT's image input was not found")
        file_inputs.last.set_input_files(
            {
                "name": "live-gpt-screenshot.webp",
                "mimeType": "image/webp",
                "buffer": screenshot_webp,
            },
            timeout=10_000,
        )
        page.wait_for_timeout(500)

    @staticmethod
    def _clear_chatgpt_composer(page: Any) -> None:
        composer = page.locator(CHATGPT_COMPOSER_SELECTOR).first
        composer.wait_for(state="visible", timeout=5_000)
        composer.fill("")

    @staticmethod
    def _read_composer_text(
        page: Any,
        timeout: int = DICTATION_RESULT_TIMEOUT_MS,
    ) -> str:
        composer = page.locator(CHATGPT_COMPOSER_SELECTOR).first
        composer.wait_for(state="visible", timeout=timeout)
        text = composer.evaluate(
            """
            element => {
                if (typeof element.value === 'string') {
                    return element.value;
                }
                return element.innerText || element.textContent || '';
            }
            """
        )
        return str(text or "").strip()

    def _start_browser_dictation(self, page: Any) -> None:
        button = self._first_visible_locator(
            page,
            (
                'button[aria-label="Start dictation"]',
                'button[data-testid="composer-speech-button"]',
                'button[data-testid="dictation-button"]',
            ),
            timeout=5_000,
        )
        if button is None:
            raise RuntimeError("ChatGPT's dictation microphone was not found")
        button.click(timeout=5_000)

        end_button = self._first_visible_locator(
            page,
            DICTATION_END_SELECTORS,
            timeout=10_000,
        )
        if end_button is None:
            if self._stop_requested:
                raise RuntimeError("Dictation cancelled")
            raise RuntimeError("ChatGPT did not start listening")

    def _cancel_existing_dictation(self, page: Any) -> bool:
        """Return ChatGPT to an idle composer before starting a new session."""
        cancel_button = None
        for selector in DICTATION_CANCEL_SELECTORS:
            candidate = page.locator(selector).last
            if self._locator_is_visible(candidate):
                cancel_button = candidate
                break
        if cancel_button is None:
            return False

        logger.info("Cancelling stale ChatGPT dictation before starting")
        cancel_button.click(timeout=5_000)
        if not self._wait_for_dictation_controls_hidden(page):
            raise RuntimeError("ChatGPT's stale dictation did not cancel")
        return True

    def _cancel_browser_dictation(
        self,
        page: Any,
        initial_text: str,
    ) -> str:
        button = self._first_visible_locator(
            page,
            DICTATION_CANCEL_SELECTORS,
            timeout=2_000,
        )
        if button is None:
            button = self._first_visible_locator(
                page,
                DICTATION_END_SELECTORS,
                timeout=2_000,
            )
        if button is not None:
            button.click(timeout=5_000)
            if not self._wait_for_dictation_controls_hidden(page):
                raise RuntimeError("ChatGPT's dictation did not cancel")
        elif not self._stop_requested:
            raise RuntimeError("ChatGPT's dictation Cancel button was not found")

        composer = page.locator(CHATGPT_COMPOSER_SELECTOR).first
        composer.wait_for(state="visible", timeout=5_000)
        composer.fill(initial_text)
        return initial_text.strip()

    def _wait_for_dictation_controls_hidden(self, page: Any) -> bool:
        for _ in range(50):
            if self._stop_requested:
                return False
            if not self._any_visible_locator(
                page,
                DICTATION_CANCEL_SELECTORS + DICTATION_END_SELECTORS,
            ):
                return True
            page.wait_for_timeout(100)
        return False

    def _finish_browser_dictation(
        self,
        page: Any,
        initial_text: str,
    ) -> str:
        button = self._first_visible_locator(
            page,
            DICTATION_END_SELECTORS,
            timeout=5_000,
        )
        if button is None:
            if self._stop_requested:
                return initial_text
            raise RuntimeError("ChatGPT's dictation Done button was not found")
        button.click(timeout=5_000)

        text = initial_text.strip()
        changed = False
        stable_polls = 0
        end_hidden_polls = 0
        for _ in range(DICTATION_RESULT_POLL_COUNT):
            if self._stop_requested:
                break
            page.wait_for_timeout(DICTATION_RESULT_POLL_INTERVAL_MS)
            current_text = self._read_composer_text(
                page,
                timeout=DICTATION_RESULT_TIMEOUT_MS,
            )
            if current_text == text:
                stable_polls += 1
            else:
                text = current_text
                changed = text != initial_text.strip()
                stable_polls = 0
            if not self._any_visible_locator(page, DICTATION_END_SELECTORS):
                end_hidden_polls += 1
            else:
                end_hidden_polls = 0

            if changed and stable_polls >= 2:
                return text
            if (
                not changed
                and end_hidden_polls >= DICTATION_RESULT_POLL_COUNT
            ):
                return text
        return text

    def _first_visible_locator(
        self,
        page: Any,
        selectors: tuple[str, ...],
        timeout: int,
    ) -> Any | None:
        deadline = time.monotonic() + timeout / 1_000
        while time.monotonic() < deadline and not self._stop_requested:
            for selector in selectors:
                locator = page.locator(selector).last
                if self._locator_is_visible(locator):
                    return locator
            page.wait_for_timeout(100)
        return None

    @classmethod
    def _any_visible_locator(
        cls,
        page: Any,
        selectors: tuple[str, ...],
    ) -> bool:
        return any(
            cls._locator_is_visible(page.locator(selector).last)
            for selector in selectors
        )

    @staticmethod
    def _assistant_turn_marker(page: Any) -> str | None:
        turns = page.locator(ASSISTANT_TURN_SELECTOR)
        if int(turns.count()) == 0:
            return None
        turn = turns.last
        return (
            turn.get_attribute("data-turn-id")
            or turn.get_attribute("data-testid")
        )

    @classmethod
    def _response_snapshot(
        cls,
        page: Any,
        turn_marker_before: str | None,
    ) -> _ResponseSnapshot:
        turns = page.locator(ASSISTANT_TURN_SELECTOR)
        turn_count = int(turns.count())
        stop_button = page.locator(
            'button[data-testid="stop-button"], '
            'button[aria-label="Stop generating"]'
        ).first
        is_generating = cls._locator_is_visible(stop_button)

        if turn_count == 0:
            return _ResponseSnapshot(
                has_new_turn=False,
                is_generating=is_generating,
                has_completion_controls=False,
                text="",
                status="Waiting for ChatGPT…",
            )

        turn = turns.last
        turn_marker = (
            turn.get_attribute("data-turn-id")
            or turn.get_attribute("data-testid")
        )
        if turn_marker == turn_marker_before:
            return _ResponseSnapshot(
                has_new_turn=False,
                is_generating=is_generating,
                has_completion_controls=False,
                text="",
                status="Waiting for ChatGPT…",
            )

        turn_text = turn.inner_text(timeout=1_000).strip()
        markdown = turn.locator(
            '.markdown, [data-message-author-role="assistant"] .prose'
        ).last
        text = (
            cls._clean_markdown_text(markdown)
            if int(markdown.count()) > 0
            else cls._response_text_from_turn(turn_text)
        )
        completion_controls = turn.locator(
            'button[aria-label="Copy response"], '
            'button[aria-label="More actions"]'
        )
        return _ResponseSnapshot(
            has_new_turn=True,
            is_generating=is_generating,
            has_completion_controls=int(completion_controls.count()) > 0,
            text=text,
            status=cls._response_activity_status(turn_text, is_generating),
        )

    @staticmethod
    def _response_text_from_turn(turn_text: str) -> str:
        action_labels = {
            "copy response",
            "rate response",
            "share",
            "switch model",
            "more actions",
        }
        lines = [
            line
            for line in turn_text.splitlines()
            if line.strip().casefold() not in action_labels
        ]
        return "\n".join(lines).strip()

    @staticmethod
    def _clean_markdown_text(markdown: Any) -> str:
        clean_text = markdown.evaluate(
            """
            element => {
                const clone = element.cloneNode(true);
                clone.querySelectorAll([
                    '[data-testid="webpage-citation-pill"]',
                    '[data-testid="webpage-citation-card"]',
                    '[data-content-reference-start]',
                    'button[aria-label="Copy table"]',
                    'svg',
                    '.sr-only'
                ].join(',')).forEach(item => item.remove());
                return clone.innerText || clone.textContent || '';
            }
            """
        )
        if isinstance(clean_text, str):
            return clean_text.strip()
        return markdown.inner_text(timeout=1_000).strip()

    @staticmethod
    def _response_activity_status(turn_text: str, is_generating: bool) -> str:
        if not is_generating:
            return "Finishing reply…"

        activity_words = (
            "searching",
            "browsing",
            "looking up",
            "reading",
            "analyzing",
            "thinking",
        )
        for line in turn_text.splitlines():
            label = line.strip()
            if (
                0 < len(label) <= 100
                and any(word in label.casefold() for word in activity_words)
            ):
                return label
        return "ChatGPT is responding…"

    @staticmethod
    def _subtitle_segments(
        text: str,
        target_length: int = 54,
    ) -> tuple[str, ...]:
        normalized = re.sub(r"\s+", " ", text).strip()
        if not normalized:
            return ()

        units: list[str] = []
        for word in normalized.split():
            while len(word) > target_length:
                units.append(word[:target_length])
                word = word[target_length:]
            if word:
                units.append(word)

        lines: list[str] = []
        current = ""
        for unit in units:
            candidate = f"{current} {unit}".strip()
            if current and len(candidate) > target_length:
                lines.append(current)
                current = unit
            else:
                current = candidate
        if current:
            lines.append(current)
        return tuple(lines)

    @staticmethod
    def _subtitle_at_progress(
        subtitles: tuple[str, ...],
        fraction: float,
    ) -> str:
        if not subtitles:
            return ""
        weights = [max(len(line), 12) for line in subtitles]
        target = min(max(fraction, 0.0), 1.0) * sum(weights)
        cumulative = 0
        for index, weight in enumerate(weights):
            cumulative += weight
            if target < cumulative:
                return "\n".join(subtitles[index:index + 2])
        return subtitles[-1]

    @staticmethod
    def _estimated_reading_duration(text: str) -> float:
        word_count = max(len(text.split()), 1)
        return max(word_count / 1.9, 2.0)

    @staticmethod
    def _finite_float(value: object) -> float | None:
        try:
            number = float(value)
        except (TypeError, ValueError):
            return None
        return number if math.isfinite(number) else None

    @staticmethod
    def _locator_is_visible(locator: Any) -> bool:
        try:
            return bool(locator.is_visible())
        except Exception:
            return False

    @classmethod
    def _click_read_aloud(cls, page: Any) -> bool:
        try:
            page.evaluate(_MEDIA_TRACKER_SCRIPT)
            turns = page.locator(ASSISTANT_TURN_SELECTOR)
            turn_count = int(turns.count())
            if turn_count == 0:
                logger.warning("Read aloud inspection found no assistant turn")
                return False

            turn = turns.last
            turn_marker = (
                turn.get_attribute("data-turn-id")
                or turn.get_attribute("data-testid")
            )
            logger.debug(
                "Read aloud inspection "
                f"turn_count={turn_count} turn_marker={turn_marker!r}"
            )

            direct_buttons = turn.locator(
                'button[data-testid="voice-play-turn-action-button"], '
                'button[aria-label="Read aloud"]'
            )
            direct_count = int(direct_buttons.count())
            direct_button = direct_buttons.last
            direct_visible = (
                direct_count > 0
                and cls._locator_is_visible(direct_button)
            )
            logger.debug(
                "Read aloud direct control "
                f"count={direct_count} visible={direct_visible}"
            )
            if direct_visible:
                cls._click_action_control(
                    page,
                    direct_button,
                    "direct Read aloud",
                )
                click_path = "direct"
            else:
                more_action_buttons = turn.locator(
                    'button[aria-label="More actions"]'
                )
                more_count = int(more_action_buttons.count())
                more_actions = more_action_buttons.last
                more_visible = (
                    more_count > 0
                    and cls._locator_is_visible(more_actions)
                )
                logger.debug(
                    "Read aloud More actions control "
                    f"count={more_count} visible={more_visible}"
                )
                if not more_visible:
                    raise RuntimeError(
                        "The latest assistant turn has no visible More actions button"
                    )
                cls._click_action_control(
                    page,
                    more_actions,
                    "More actions",
                    expanded_control=True,
                )
                read_aloud = page.locator(
                    '[role="menuitem"]:has-text("Read aloud"), '
                    '[role="menuitemradio"]:has-text("Read aloud")'
                ).last
                try:
                    read_aloud.wait_for(state="visible", timeout=5_000)
                except Exception:
                    menu_items = page.locator(
                        '[role="menuitem"], [role="menuitemradio"]'
                    )
                    try:
                        labels = menu_items.all_inner_texts()[:20]
                    except Exception:
                        labels = []
                    logger.debug(
                        "Read aloud menu inspection "
                        f"item_count={int(menu_items.count())} labels={labels!r}"
                    )
                    raise
                logger.debug("Read aloud menu item is visible")
                cls._click_action_control(
                    page,
                    read_aloud,
                    "Read aloud menu item",
                )
                click_path = "More actions menu"
            logger.info(f"Clicked ChatGPT Read aloud path={click_path}")
            return True
        except Exception as error:
            logger.warning(f"Unable to click ChatGPT Read aloud: {error}")
            return False

    @staticmethod
    def _click_action_control(
        page: Any,
        locator: Any,
        description: str,
        expanded_control: bool = False,
    ) -> None:
        try:
            locator.click(timeout=3_000, force=True)
        except Exception as click_error:
            if (
                expanded_control
                and locator.get_attribute("aria-expanded") == "true"
            ):
                logger.debug(
                    f"{description} opened despite Playwright click timeout"
                )
                return
            try:
                page.wait_for_timeout(150)
                media = page.evaluate(_MEDIA_PROGRESS_SCRIPT)
            except Exception:
                media = None
            if isinstance(media, dict) and int(media.get("playCount") or 0) > 0:
                logger.debug(
                    f"{description} started playback despite click timeout"
                )
                return
            logger.debug(
                f"Playwright click failed for {description}; "
                f"using DOM click fallback: {click_error}"
            )
            locator.evaluate("element => element.click()")

    def _set_status(self, status: str) -> None:
        if status == self._last_status:
            return
        self._last_status = status
        self.status_changed.emit(status)

    @staticmethod
    def _is_connected(browser: Any | None) -> bool:
        return browser is not None and browser.is_connected()


__all__ = [
    "BrowserMonitor",
    "DEFAULT_DEBUG_PORTS",
    "active_port_endpoint",
    "active_port_endpoints",
    "browser_user_data_directories",
    "current_user_data_directories",
    "discover_cdp_endpoint",
    "is_chatgpt_url",
    "open_remote_debugging_settings",
    "windows_default_browser_executable",
]
