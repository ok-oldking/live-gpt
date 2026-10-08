from __future__ import annotations

import math
import re
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from queue import Empty, Queue
from typing import Any, Callable, Iterator

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
from .response_content import RESPONSE_CONTENT_SCRIPT


logger = Logger.get_logger(__name__)
ASSISTANT_TURN_SELECTOR = (
    '[data-testid^="conversation-turn-"][data-turn="assistant"], '
    '[data-turn-key]:has([data-chatgpt-search-unit-key$=":assistant"]), '
    '[data-turn-key]:has([data-markdown-text-style="assistant-message"])'
)
CHATGPT_COMPOSER_SELECTOR = (
    'form[data-thread-find-composer="true"]:visible '
    '[data-composer-markdown][contenteditable="true"][role="textbox"]:visible, '
    '[data-composer-surface="true"]:visible '
    '#prompt-textarea[contenteditable="true"]:visible, '
    '[data-composer-surface="true"]:visible '
    '[contenteditable="true"][role="textbox"]:visible, '
    '#prompt-textarea[contenteditable="true"]:visible, '
    'textarea[name="prompt-textarea"]:visible'
)
USAGE_LIMIT_BANNER_SELECTOR = (
    'form[data-thread-find-composer="true"] aside[role="status"]:visible, '
    'form aside[role="status"]:visible, '
    '[data-above-composer-portal] [role="status"]:visible, '
    '[role="alert"]:visible'
)
_USAGE_LIMIT_PATTERN = re.compile(
    r"(?:reached|hit|exceeded)[^.!?\n]{0,120}\blimit\b"
    r"|\b(?:usage|message|rate) limit (?:reached|exceeded)\b"
    r"|\bout of (?:usage|credits)\b"
    r"|\b(?:no|not enough|insufficient) credits\b"
    r"|(?:已达到|已達到|已达|已達|达到|達到|超出|超过|超過).{0,60}(?:上限|限额|限額)"
    r"|(?:额度|額度|次数|次數|点数|點數).{0,20}(?:用尽|用盡|不足)",
    re.IGNORECASE,
)


class _UsageLimitError(RuntimeError):
    """ChatGPT cannot accept the prompt until its usage allowance resets."""


def _label_selectors(element: str, *labels: str) -> str:
    """Match localized controls when ChatGPT exposes no stable test ID."""
    return ", ".join(f'{element}[aria-label="{label}"]' for label in labels)


MORE_ACTIONS_SELECTOR = _label_selectors(
    "button", "More actions", "更多操作"
)
READ_ALOUD_SELECTOR = (
    'button[data-testid="voice-play-turn-action-button"], '
    + _label_selectors("button", "Read aloud", "朗读", "朗讀")
)
READ_ALOUD_MENU_SELECTOR = ", ".join(
    # Menu text lives in nested spans/divs; text-is on the role element
    # misses it. Filter visibility before .last to exclude stale menus.
    f'[role="{role}"]:visible:has-text("{label}")'
    for role in ("menuitem", "menuitemradio")
    for label in ("Read aloud", "朗读", "朗讀")
)
DICTATION_RESULT_TIMEOUT_MS = 20_000
DICTATION_RESULT_POLL_INTERVAL_MS = 200
DICTATION_RESULT_POLL_COUNT = (
    DICTATION_RESULT_TIMEOUT_MS // DICTATION_RESULT_POLL_INTERVAL_MS
)
DICTATION_END_SELECTORS = (
    'button[aria-label="Submit dictation"]',
    'button[aria-label="Done"]',
    'button[aria-label="Stop dictation"]',
    'button[aria-label="Finish dictation"]',
    'button[aria-label="Stop recording"]',
    'button[data-testid="composer-dictation-done-button"]',
    'button[data-testid="dictation-done-button"]',
    'button:text-is("Done")',
    _label_selectors(
        "button", "提交听写", "提交聽寫", "完成", "停止听写", "停止聽寫",
        "结束听写", "結束聽寫", "停止录音", "停止錄音",
    ),
    'form button:text-is("完成")',
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
    _label_selectors(
        "button", "取消听写", "取消聽寫", "取消录音", "取消錄音"
    ),
    'form button:text-is("取消")',
)
_MEDIA_TRACKER_SCRIPT = """
() => {
    // Each Read aloud click starts a fresh session, including on reused audio.
    window.__liveGptReadAloudTracker = {
        media: null,
        playCount: 0,
        mediaCount: 0,
        playedTime: 0,
        lastTime: 0,
        source: ''
    };
    if (window.__liveGptMediaTrackerInstalled === 2) return;
    window.__liveGptMediaTrackerInstalled = 2;
    const sources = new Map();
    window.__liveGptMediaSources = sources;
    const originalCreateURL = URL.createObjectURL;
    URL.createObjectURL = function(object) {
        const url = originalCreateURL.call(this, object);
        if (typeof MediaSource !== 'undefined' && object instanceof MediaSource) {
            sources.set(url, new WeakRef(object));
            if (sources.size > 128) sources.delete(sources.keys().next().value);
        }
        return url;
    };
    const sample = () => {
        const tracker = window.__liveGptReadAloudTracker;
        const media = tracker.media;
        if (!media) return;
        const current = Number.isFinite(media.currentTime) ? media.currentTime : 0;
        const source = media.currentSrc || media.src || '';
        if (source !== tracker.source || (!media.seeking && current < tracker.lastTime - 0.05)) {
            tracker.mediaCount += 1;
            tracker.lastTime = 0;
        }
        // Only consumed audio advances the estimate; pauses and buffering do not.
        if (!media.seeking) tracker.playedTime += Math.max(current - tracker.lastTime, 0);
        tracker.lastTime = current;
        tracker.source = source;
    };
    window.__liveGptSampleMedia = sample;
    const observed = new WeakSet();
    const originalPlay = HTMLMediaElement.prototype.play;
    HTMLMediaElement.prototype.play = function(...args) {
        sample();
        const tracker = window.__liveGptReadAloudTracker;
        if (tracker.media !== this) {
            tracker.media = this;
            tracker.mediaCount += 1;
            tracker.lastTime = Number.isFinite(this.currentTime) ? this.currentTime : 0;
            tracker.source = this.currentSrc || this.src || '';
        }
        tracker.playCount += 1;
        if (!observed.has(this)) {
            observed.add(this);
            for (const event of ['timeupdate', 'pause', 'waiting', 'ended', 'emptied', 'seeking', 'seeked']) {
                this.addEventListener(event, () => {
                    if (window.__liveGptReadAloudTracker.media === this) sample();
                });
            }
        }
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
    window.__liveGptSampleMedia?.();
    const source = window.__liveGptMediaSources?.get(media.currentSrc || media.src)?.deref();
    return {
        playCount: tracker?.playCount || 0,
        mediaCount: tracker?.mediaCount || 0,
        playedTime: tracker?.playedTime || 0,
        // A growing MediaSource duration describes the buffered prefix only.
        durationIsFinal: !source || source.readyState === 'ended',
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
    discovery_session: Any | None = None
    settings_opened: bool = False
    settings_attempted: bool = False
    retry_endpoint: str | None = None
    last_tabs: list[dict[str, str]] | None = None
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
    last_html: str = ""
    last_links: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True)
class _ResponseSnapshot:
    has_new_turn: bool
    is_generating: bool
    has_completion_controls: bool
    text: str
    status: str
    links: tuple[tuple[str, str], ...] = ()
    error_message: str = ""
    html: str = ""


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
    last_media_time: float = 0.0
    completed_media_time: float = 0.0
    quiet_since: float | None = None


class BrowserMonitor(QThread):
    response_links_changed = Signal(object)
    response_html_changed = Signal(str)
    debug_connection_changed = Signal(bool)
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
    local_voice_announcement = Signal(str)
    dictation_started = Signal(bool, str)
    dictation_finished = Signal(bool, str, str)
    clear_finished = Signal(bool, str)

    def __init__(self) -> None:
        super().__init__()
        self._stop_requested = False
        self._retry_connection_requested = False
        self._connection_pending = threading.Event()
        self._playwright_cancellation: (
            tuple[Any, Callable[[], None]] | None
        ) = None
        self._stop_reading_requested = False
        self._stop_reading_message = "Playback stopped for recording"
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
        cancellation = self._playwright_cancellation
        if self._connection_pending.is_set() and cancellation is not None:
            loop, stop_transport = cancellation
            loop.call_soon_threadsafe(stop_transport)

    def request_retry_connection(self) -> None:
        self._retry_connection_requested = True
        self._wake_event.set()

    def request_stop_reading(self, *, message: str = "Playback stopped for recording") -> None:
        self._stop_reading_message = message
        self._stop_reading_requested = True
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
        if self._connection_pending.is_set():
            self.send_finished.emit(
                False, text, "Approve remote debugging in the browser before sending",
            )
            return
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
        while not self._stop_requested:
            try:
                self._run_session(sync_playwright, PlaywrightError, state)
            except Exception as error:
                if not self._stop_requested:
                    logger.error("Browser monitor failed; reconnecting", error)
            finally:
                self._playwright_cancellation = None
                self._connection_pending.clear()
                self._handle_disconnect(state)
                self._fail_pending_requests(PlaywrightError, state)
            if not self._stop_requested:
                self._wake_event.wait(1.5)
                self._wake_event.clear()
        logger.info("Browser monitor stopped")

    def _fail_pending_requests(
        self, playwright_error: type[Exception], state: _MonitorState,
    ) -> None:
        self._process_dictation_requests(None, playwright_error)
        self._process_attachment_requests(None, playwright_error)
        self._process_clear_requests(None, playwright_error)
        self._process_send_requests(None, playwright_error, state)

    @contextmanager
    def _browser_operation(self, timeout: float) -> Iterator[None]:
        """Bound CDP/page calls that have no API timeout.

        Stop only our Playwright driver. The outer loop creates a fresh driver
        and rediscovers the browser without closing the user's windows.
        """
        cancellation = self._playwright_cancellation
        if cancellation is None:
            yield
            return
        loop, stop_transport = cancellation
        expired = threading.Event()
        cancelled = threading.Event()

        def interrupt() -> None:
            if not cancelled.is_set():
                expired.set()
                logger.warning(f"Browser operation exceeded {timeout:g}s; reconnecting")
                stop_transport()

        def schedule_interrupt() -> None:
            try:
                loop.call_soon_threadsafe(interrupt)
            except RuntimeError:
                pass  # The driver has already stopped.

        timer = threading.Timer(timeout, schedule_interrupt)
        timer.daemon = True
        timer.start()
        try:
            yield
        finally:
            cancelled.set()
            timer.cancel()
            if expired.is_set():
                raise TimeoutError("Browser stopped responding; reconnecting")

    def _run_session(
        self, sync_playwright: Callable, PlaywrightError: type[Exception],
        state: _MonitorState,
    ) -> None:
        with sync_playwright() as playwright:
            connection = playwright._impl_obj._connection
            # An unlimited CDP approval wait still needs to be cancellable when
            # the application exits. Playwright exposes no public cancellation
            # token for connect_over_cdp, so terminate its driver safely on the
            # Playwright event loop. The operation watchdog uses this same
            # cancellation path for unresponsive established connections.
            self._playwright_cancellation = (
                connection._loop,
                connection._transport._proc.terminate,
            )
            while not self._stop_requested:
                if not self._is_connected(state.browser):
                    if state.browser is not None:
                        self._handle_disconnect(state)
                    self._try_connect(playwright, PlaywrightError, state)

                if self._is_connected(state.browser):
                    with self._browser_operation(10):
                        self._refresh_tabs(state, PlaywrightError)
                elif state.last_tabs:
                    state.last_tabs = []
                    self.tabs_changed.emit([])

                connected_browser = (
                    state.browser
                    if self._is_connected(state.browser)
                    else None
                )
                with self._browser_operation(60):
                    self._stop_active_reading(state)
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
            self._playwright_cancellation = None

    def _stop_active_reading(self, state: _MonitorState) -> None:
        if not self._stop_reading_requested:
            return
        self._stop_reading_requested = False
        # A dismissed reply or microphone press must not start future playback.
        state.active_response = None
        reading = state.active_reading
        if reading is None:
            return
        try:
            reading.page.evaluate(
                """
                () => {
                    // Read aloud can use an Audio object outside the DOM.
                    const mediaElements = new Set([
                        window.__liveGptReadAloudTracker?.media,
                        ...document.querySelectorAll('audio, video')
                    ]);
                    for (const media of mediaElements) {
                        if (!media) continue;
                        media.pause();
                        try { media.currentTime = 0; } catch (_) {}
                    }
                }
                """
            )
        except Exception as error:
            logger.warning(f"Unable to stop ChatGPT playback cleanly: {error}")
        state.active_reading = None
        self.reading_finished.emit(False, self._stop_reading_message)

    def _try_connect(
        self,
        playwright: Any,
        playwright_error: type[Exception],
        state: _MonitorState,
    ) -> None:
        state.browser = None
        state.discovery_session = None
        endpoint = discover_cdp_endpoint()
        if self._stop_requested:
            return

        if self._retry_connection_requested:
            self._retry_connection_requested = False
            state.retry_endpoint = None
            if not state.settings_opened:
                state.settings_attempted = False

        if endpoint is None:
            self._prepare_remote_debugging(state)
            return

        if endpoint == state.retry_endpoint:
            if self._prepare_remote_debugging(state):
                state.retry_endpoint = None
                return
            self._set_status(
                "Remote debugging was not approved; "
                "click Enable Debugging to retry"
            )
            return

        self._set_status("Approve remote debugging in the browser…")
        self._connection_pending.set()
        self._fail_pending_requests(playwright_error, state)
        try:
            state.browser = playwright.chromium.connect_over_cdp(
                endpoint,
                timeout=0,
                no_defaults=True,
            )
        except Exception as error:
            if self._stop_requested:
                return
            if not isinstance(error, playwright_error):
                raise
            logger.warning(
                "Unable to connect to remote-debug browser "
                f"endpoint={endpoint!r}: {error}"
            )
            state.retry_endpoint = endpoint
            # A live marker/port does not mean Chrome will accept CDP. Open
            # settings even when discovery succeeded, then retry once after
            # setup. Keep subsequent rejections gated on an explicit retry.
            if self._prepare_remote_debugging(state):
                state.retry_endpoint = None
                return
            self._set_status(
                "Remote debugging was not approved; "
                "click Enable Debugging to retry"
            )
            return
        finally:
            self._connection_pending.clear()

        state.retry_endpoint = None
        state.settings_opened = False
        state.settings_attempted = False
        self.debug_connection_changed.emit(True)
        self._set_status("Browser connected")
        logger.info(f"Connected to browser endpoint={endpoint!r}")

    def _prepare_remote_debugging(self, state: _MonitorState) -> bool:
        opened = False
        if (
            not state.settings_opened
            and not state.settings_attempted
        ):
            # Setup can create a tab before a later step fails. Never repeat
            # that side effect in the background; wait for an explicit retry.
            state.settings_attempted = True
            self._set_status("Opening remote debugging settings…")
            try:
                open_remote_debugging_settings()
                state.settings_opened = True
                opened = True
            except Exception as error:
                logger.error("Unable to open remote debugging settings", error)
                self._set_status(str(error))
        if state.settings_opened:
            self._set_status("Enable remote debugging in the browser")
        return opened

    def _refresh_tabs(self, state: _MonitorState, playwright_error: type[Exception]) -> None:
        try:
            if state.discovery_session is None:
                state.discovery_session = state.browser.new_browser_cdp_session()
            # This browser-level call both pumps page/navigation events and
            # supplies titles without evaluating JavaScript in a tab. Frozen
            # or busy renderers must not restart a connection just approved.
            target_snapshot = state.discovery_session.send("Target.getTargets")
        except playwright_error:
            self._handle_disconnect(state)
            return
        target_titles: dict[str, str] = {}
        if isinstance(target_snapshot, dict):
            for target in target_snapshot.get("targetInfos", []):
                if not isinstance(target, dict) or target.get("type") != "page":
                    continue
                url, title = target.get("url"), target.get("title")
                if isinstance(url, str) and isinstance(title, str) and title.strip():
                    target_titles[url] = title.strip()
        tabs = self._collect_chatgpt_tabs(
            state.browser, playwright_error, target_titles, state.last_tabs,
        )
        if tabs is None:
            self._handle_disconnect(state)
        elif tabs != state.last_tabs:
            state.last_tabs = tabs
            self.tabs_changed.emit(tabs)

    @staticmethod
    def _collect_chatgpt_tabs(
        browser: Any,
        playwright_error: type[Exception],
        target_titles: dict[str, str],
        previous_tabs: list[dict[str, str]] | None = None,
    ) -> list[dict[str, str]] | None:
        try:
            contexts = list(browser.contexts)
        except playwright_error:
            return None

        tabs: list[dict[str, str]] = []
        previous = {tab["id"]: tab for tab in previous_tabs or []}
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
                    url = page.url
                    old_tab = previous.get(page_id, {})
                    title = target_titles.get(url) or (
                        old_tab.get("title") if old_tab.get("url") == url else None
                    ) or "ChatGPT"
                    tabs.append(
                        {"id": page_id, "title": title, "url": url}
                    )
                except playwright_error:
                    continue

        tabs.sort(key=lambda tab: tab["title"].casefold())
        return tabs

    def _handle_disconnect(self, state: _MonitorState) -> None:
        if state.active_response is not None:
            if not self._use_browser_voice:
                self.local_voice_announcement.emit("Browser disconnected")
            self.response_finished.emit(False, "Browser disconnected")
            state.active_response = None
        if state.active_reading is not None:
            self.reading_finished.emit(False, "Browser disconnected")
            state.active_reading = None
        state.browser = None
        state.discovery_session = None
        self.debug_connection_changed.emit(False)
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
                    self._ensure_screenshot_uploaded(page, request.screenshot_webp)
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
            except _UsageLimitError as error:
                self.send_finished.emit(False, request.text, str(error))
                continue
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
        from playwright.sync_api import TimeoutError as PlaywrightTimeoutError

        response = state.active_response
        if response is None:
            return

        # Apply the deadline even when every DOM read fails or the virtualized
        # turn stays unmounted. Transient failures must not extend it forever.
        now = time.monotonic()
        if now - response.started_at >= 600:
            if not self._use_browser_voice:
                self.local_voice_announcement.emit("Timed out while waiting for ChatGPT's reply")
            if not self._use_browser_voice and response.last_text:
                self.local_voice_updated.emit(response.last_text, True)
            self.response_finished.emit(False, "Timed out while waiting for ChatGPT's reply")
            state.active_response = None
            return

        try:
            snapshot = self._response_snapshot(
                response.page,
                response.turn_marker_before,
            )
        except PlaywrightTimeoutError:
            # ChatGPT can replace/unmount the turn between locator calls.
            # Keep the last text and retry on the next worker-loop poll.
            response.completion_candidate_at = None
            logger.debug("ChatGPT response DOM read timed out; retrying next poll")
            return
        except Exception as error:
            logger.error("Unable to read the ChatGPT response", error)
            if not self._use_browser_voice:
                self.local_voice_announcement.emit(f"Could not read ChatGPT's reply: {error}")
            if not self._use_browser_voice and response.last_text:
                self.local_voice_updated.emit(response.last_text, True)
            self.response_finished.emit(
                False,
                f"Could not read ChatGPT's reply: {error}",
            )
            state.active_response = None
            return

        if snapshot.error_message:
            if not self._use_browser_voice:
                self.local_voice_announcement.emit(snapshot.error_message)
            if not self._use_browser_voice and response.last_text:
                self.local_voice_updated.emit(response.last_text, True)
            self.response_finished.emit(False, snapshot.error_message)
            state.active_response = None
            return

        if not snapshot.has_new_turn:
            response.completion_candidate_at = None
            if snapshot.status != response.last_status and (
                snapshot.is_generating or self._is_thinking_status(response.last_status)
            ):
                response.last_status = snapshot.status
                self.response_changed.emit(snapshot.status, response.last_text)
            return

        text_changed = snapshot.text != response.last_text
        html_changed = snapshot.html != response.last_html
        links_changed = snapshot.links != response.last_links
        if links_changed:
            response.last_links = snapshot.links
        if html_changed:
            response.last_html = snapshot.html
            if not snapshot.text:
                response.last_text_changed_at = now
                response.completion_candidate_at = None
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
        if links_changed or text_changed:
            self.response_links_changed.emit(snapshot.links)
        if html_changed:
            self.response_html_changed.emit(snapshot.html)
        if text_changed and not self._use_browser_voice and snapshot.text:
            self.local_voice_updated.emit(snapshot.text, False)
        completion_candidate = (
            snapshot.has_new_turn
            and (bool(snapshot.text) or "<img " in snapshot.html)
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
            return

        if not snapshot.text:
            self.response_finished.emit(True, "Reply complete")
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
            self.reading_changed.emit({"text": snapshot.text, "fraction": 0.0})
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
                playback_elapsed = self._finite_float(media.get("playedTime"))
                if playback_elapsed is None:
                    if current_time < reading.last_media_time - 0.05:
                        reading.completed_media_time += reading.last_media_time
                    reading.last_media_time = current_time
                    playback_elapsed = reading.completed_media_time + current_time
                is_full_response_audio = (
                    duration is not None
                    and media.get("durationIsFinal") is not False
                    and int(media.get("mediaCount") or 1) == 1
                    and reading.completed_media_time == 0.0
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
            elif reading.audio_seen and bool(media.get("ended")):
                if reading.quiet_since is None:
                    reading.quiet_since = now
                finished = now - reading.quiet_since >= 4.0
                if finished:
                    fraction = 1.0
            else:
                # Paused or stalled audio is still an active reading session.
                reading.quiet_since = None
        elif reading.audio_seen:
            fraction = reading.last_progress_fraction
            if reading.quiet_since is None:
                reading.quiet_since = now
            finished = now - reading.quiet_since >= 4.0
        if not reading.audio_seen and elapsed_since_click >= 8.0:
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
        usage_error = cls._usage_limit_message(page)
        if usage_error:
            raise _UsageLimitError(usage_error)
        composer = page.locator(CHATGPT_COMPOSER_SELECTOR).first
        composer.wait_for(state="visible", timeout=5_000)
        if not preserve_attachments:
            cls._clear_chatgpt_attachments(page)
        composer.fill(text)
        upload_attempts_left = 3
        if screenshot_webp is not None:
            upload_attempts_left -= cls._ensure_screenshot_uploaded(
                page, screenshot_webp, preserve=preserve_attachments
            )
        elif preserve_attachments and cls._screenshot_upload_state(page) != "missing":
            cls._wait_for_screenshot_upload(page)

        # Project side panes can coexist with a background composer. Resolve
        # the submit control from the form we filled, excluding hidden copies.
        form = composer.locator('xpath=ancestor::form[1]')
        send_button = form.locator(
            'button[data-testid="send-button"]:visible, '
            'button#composer-submit-button[type="submit"]:visible, '
            'button[type="submit"]:visible'
        ).first
        for _ in range(3):
            try:
                send_button.wait_for(state="visible", timeout=5_000)
                send_button.click(timeout=15_000)
                # Clicking a usable button does not prove ChatGPT accepted the prompt.
                page.wait_for_function("""() => {
                    const composer = [...document.querySelectorAll(
                        '[data-composer-markdown], #prompt-textarea, textarea[name="prompt-textarea"], '
                        + '[data-composer-surface="true"] [contenteditable="true"][role="textbox"]'
                    )].find(element => element.getClientRects().length);
                    if (!composer) return false;
                    const text = composer.value ?? composer.innerText ?? '';
                    const attachments = composer.closest('form')?.querySelector(
                        '[data-composer-attachments] [class*="group/composer-attachment"]'
                    );
                    return !text.trim() && !attachments;
                }""", timeout=5_000)
                return
            except Exception:
                # A limit or failed upload can appear while Send becomes ready.
                usage_error = cls._usage_limit_message(page)
                if usage_error:
                    raise _UsageLimitError(usage_error) from None
                if cls._screenshot_upload_state(page) == "failed":
                    if (
                        screenshot_webp is not None
                        and upload_attempts_left > 0
                        and cls._read_composer_text(page) == text.strip()
                    ):
                        cls._clear_chatgpt_attachments(page)
                        upload_attempts_left -= cls._ensure_screenshot_uploaded(
                            page, screenshot_webp, attempts=upload_attempts_left
                        )
                        continue
                    raise RuntimeError("Screenshot upload failed") from None
                raise

    @classmethod
    def _clear_chatgpt_attachments(cls, page: Any) -> None:
        remove_buttons = page.locator(
            'button[aria-label="Remove file"], '
            'button[aria-label="Remove attachment"], '
            'button[aria-label="Remove image"], '
            + _label_selectors(
                "button", "移除文件", "删除文件", "移除附件", "删除附件",
                "移除图片", "删除图片", "移除檔案", "刪除檔案",
                "刪除附件", "移除圖片", "刪除圖片",
            ) + ', '
            'button[data-testid*="remove"][data-testid*="file"], '
            'button[data-testid*="remove"][data-testid*="attachment"], '
            '[data-composer-attachments] button[aria-label^="Remove "], '
            '[data-composer-attachments] button[aria-label^="移除"], '
            '[data-composer-attachments] button[aria-label^="删除"]'
        )
        for _ in range(min(int(remove_buttons.count()), 20)):
            button = remove_buttons.last
            if not cls._locator_is_visible(button):
                break
            # Attachment removal controls can be opacity-zero until hover.
            button.evaluate("element => element.click()")
            page.wait_for_timeout(100)

    @staticmethod
    def _screenshot_upload_state(page: Any) -> str:
        return page.locator(
            '[data-composer-attachments]:visible, '
            'form [class*="group/composer-attachment"]:visible'
        ).evaluate_all("""elements => {
            if (!elements.length) return 'missing';
            const text = elements.map(element => element.textContent || '').join(' ');
            if (/upload failed|failed to upload|上传失败|上傳失敗/i.test(text)) return 'failed';
            if (/uploading|正在上传|正在上傳/i.test(text) || elements.some(element =>
                element.querySelector('[role="progressbar"], [aria-busy="true"]')
            )) return 'pending';
            const images = elements.flatMap(element => [...element.querySelectorAll('img')]);
            const form = elements[0].closest('form');
            const send = form?.querySelector('button[type="submit"], button[data-testid="send-button"]');
            if (send && (send.disabled || send.closest('[aria-disabled="true"]'))) return 'pending';
            return images.some(img => img.complete && img.naturalWidth > 0) ? 'ready' : 'pending';
        }""")

    @classmethod
    def _wait_for_screenshot_upload(cls, page: Any) -> None:
        for _ in range(40):
            usage_error = cls._usage_limit_message(page)
            if usage_error:
                raise _UsageLimitError(usage_error)
            state = cls._screenshot_upload_state(page)
            if state == "failed":
                raise RuntimeError("Screenshot upload failed")
            if state == "ready":
                return
            page.wait_for_timeout(250)
        raise RuntimeError("Screenshot upload timed out")

    @classmethod
    def _ensure_screenshot_uploaded(
        cls, page: Any, screenshot_webp: bytes, *, preserve: bool = False, attempts: int = 3,
    ) -> int:
        for attempt in range(attempts):
            if not preserve or attempt:
                if attempt:
                    cls._clear_chatgpt_attachments(page)
                cls._paste_screenshot(page, screenshot_webp)
            try:
                cls._wait_for_screenshot_upload(page)
                return attempt + 1
            except _UsageLimitError:
                raise
            except RuntimeError:
                if attempt == attempts - 1:
                    raise
                logger.warning("Screenshot upload failed; retrying with the captured image")

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
                _label_selectors("button", "开始听写", "開始聽寫"),
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
            or turn.get_attribute("data-turn-key")
        )

    @classmethod
    def _usage_limit_message(cls, page: Any) -> str:
        messages = page.locator(USAGE_LIMIT_BANNER_SELECTOR).evaluate_all("""elements =>
            elements.map(element => {
                const clone = element.cloneNode(true);
                clone.querySelectorAll('button, [role="button"], [aria-hidden="true"]')
                    .forEach(node => node.remove());
                clone.querySelectorAll('h1, h2, h3, h4, p, div')
                    .forEach(node => node.append(' '));
                return clone.textContent || '';
            })
        """)
        if isinstance(messages, list):
            for message in messages:
                if isinstance(message, str) and _USAGE_LIMIT_PATTERN.search(message):
                    return " ".join(message.split())
        return ""

    @classmethod
    def _response_snapshot(
        cls,
        page: Any,
        turn_marker_before: str | None,
    ) -> _ResponseSnapshot:
        usage_error = cls._usage_limit_message(page)
        if usage_error:
            return _ResponseSnapshot(
                has_new_turn=False,
                is_generating=False,
                has_completion_controls=False,
                text="",
                status=usage_error,
                error_message=usage_error,
            )
        if cls._screenshot_upload_state(page) == "failed":
            return _ResponseSnapshot(False, False, False, "", "Screenshot upload failed", error_message="Screenshot upload failed")
        turns = page.locator(ASSISTANT_TURN_SELECTOR)
        turn_count = int(turns.count())
        stop_button = page.locator(
            'button[data-testid="stop-button"], '
            + _label_selectors("button", "Stop generating", "停止生成", "停止產生")
            + ', ' + _label_selectors(
                'form[data-thread-find-composer="true"] button',
                'Stop', '停止',
            )
        ).first
        is_generating = cls._locator_is_visible(stop_button)
        live_labels = page.locator(
            '[data-request-input-activity-root] [role="status"]:visible'
        ).evaluate_all("elements => elements.map(element => element.textContent || '')")
        live_status = ""
        activity_labels = page.locator('[data-turn-key]').last.locator(
            '[data-d-component="shimmer-text"]:visible, '
            '[class*="cadencedShimmer-"][class*="cadencedShimmerActive-"]:visible'
        ).evaluate_all("""elements => elements.map(element => {
            const clone = element.cloneNode(true);
            // The animated sweep repeats the label for visual effects only.
            clone.querySelectorAll('[aria-hidden="true"]').forEach(node => node.remove());
            return clone.textContent || '';
        })""")
        if isinstance(live_labels, list):
            for label in reversed(live_labels):
                status = cls._response_activity_status(str(label), True)
                if status != "ChatGPT is responding…":
                    live_status = status
                    break
        # These are explicitly rendered activity labels, not arbitrary reply
        # text. Preserve them verbatim across languages and activity types.
        if isinstance(activity_labels, list):
            for label in reversed(activity_labels):
                if isinstance(label, str) and label.strip():
                    live_status = label.strip()
                    break
        is_generating = is_generating or bool(live_status)

        if turn_count == 0:
            return _ResponseSnapshot(
                has_new_turn=False,
                is_generating=is_generating,
                has_completion_controls=False,
                text="",
                status=live_status or ("ChatGPT is responding…" if is_generating else "Waiting for ChatGPT…"),
            )

        turn = turns.last
        turn_marker = (
            turn.get_attribute("data-turn-id", timeout=1_000)
            or turn.get_attribute("data-testid", timeout=1_000)
            or turn.get_attribute("data-turn-key", timeout=1_000)
        )
        if turn_marker == turn_marker_before:
            return _ResponseSnapshot(
                has_new_turn=False,
                is_generating=is_generating,
                has_completion_controls=False,
                text="",
                status=live_status or ("ChatGPT is responding…" if is_generating else "Waiting for ChatGPT…"),
            )

        markdown = turn.locator(
            '.markdown, [data-message-author-role="assistant"] .prose, '
            '[data-markdown-text-style="assistant-message"], '
            '[data-markdown-copy="contents"]:has([data-dil-message-id])'
        ).last
        has_markdown = int(markdown.count()) > 0
        text, reply_html = cls._response_content(markdown if has_markdown else turn)
        if not has_markdown:
            text = cls._response_text_from_turn(text)
        completion_controls = turn.locator(
            'button[data-testid="copy-turn-action-button"], '
            + _label_selectors("button", "Copy response", "复制回复", "複製回覆")
            + ', ' + MORE_ACTIONS_SELECTOR
            + ', ' + _label_selectors(
                '.turn-action-controls button', 'Copy', '复制', '複製',
            )
        )
        anchors = turn.locator("a[href]").evaluate_all(
            "elements => elements.map(a => [a.innerText, a.href])"
        )
        links = tuple(
            (label, url) for label, url in anchors
            if label and isinstance(url, str) and url.startswith(("https://", "http://"))
        ) if isinstance(anchors, list) else ()
        return _ResponseSnapshot(
            has_new_turn=True,
            is_generating=is_generating,
            has_completion_controls=int(completion_controls.count()) > 0,
            text=text,
            status=live_status or ("ChatGPT is responding…" if is_generating else "Finishing reply…"),
            links=links,
            html=reply_html,
        )

    @staticmethod
    def _response_text_from_turn(turn_text: str) -> str:
        action_labels = {
            "copy response",
            "rate response",
            "share",
            "switch model",
            "more actions",
            "chatgpt 说：",
            "chatgpt 說：",
            "复制回复",
            "複製回覆",
            "评价回复",
            "評價回覆",
            "分享",
            "切换模型",
            "切換模型",
            "更多操作",
        }
        lines = [
            line
            for line in turn_text.splitlines()
            if line.strip().casefold() not in action_labels
        ]
        return "\n".join(lines).strip()

    @staticmethod
    def _response_content(markdown: Any) -> tuple[str, str]:
        clean_text = markdown.evaluate(RESPONSE_CONTENT_SCRIPT)
        if isinstance(clean_text, dict):
            return str(clean_text.get("text", "")).strip(), str(clean_text.get("html", ""))
        if isinstance(clean_text, str):
            return clean_text.strip(), ""
        return markdown.inner_text(timeout=1_000).strip(), ""

    @classmethod
    def _clean_markdown_text(cls, markdown: Any) -> str:
        return cls._response_content(markdown)[0]

    @staticmethod
    def _is_thinking_status(status: str) -> bool:
        return status.strip().rstrip(".…。 ").casefold() in {"thinking", "正在思考", "思考中"}

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
            "搜索", "搜尋", "浏览", "瀏覽", "查找",
            "阅读", "閱讀", "分析", "思考",
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
        # CJK speech units do not depend on spaces. A paragraph of Chinese
        # must not be treated as one English word (a two-second response).
        cjk = r"[\u3400-\u9fff\uf900-\ufaff\u3040-\u30ff\uac00-\ud7af]"
        character_count = len(re.findall(cjk, text))
        other_text = re.sub(cjk, " ", text)
        word_count = len(re.findall(r"[^\W_]+(?:['’][^\W_]+)*", other_text))
        pauses = len(re.findall(r"[.!?。！？;；\n]", text)) * 0.3
        pauses += len(re.findall(r"[,，、:：]", text)) * 0.12
        return max(word_count / 1.9 + character_count / 3.5 + pauses, 2.0)

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
                or turn.get_attribute("data-turn-key")
            )
            logger.debug(
                "Read aloud inspection "
                f"turn_count={turn_count} turn_marker={turn_marker!r}"
            )

            direct_buttons = turn.locator(READ_ALOUD_SELECTOR)
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
                more_action_buttons = turn.locator(MORE_ACTIONS_SELECTOR)
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
                read_aloud = page.locator(READ_ALOUD_MENU_SELECTOR).last
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
