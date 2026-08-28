from __future__ import annotations

import argparse
import inspect
import logging
import os
import queue
import re
import sys
import traceback
from logging.handlers import QueueHandler, QueueListener, TimedRotatingFileHandler


_LOG_FORMATTER = logging.Formatter(
    "%(asctime)s %(levelname)s %(threadName)s %(message)s"
)
_LOGGER = logging.getLogger("live-gpt")
_STDOUT_HANDLER_MARKER = "_live_gpt_stdout_handler"
_file_listener: QueueListener | None = None
_file_handler: SafeFileHandler | None = None
_queue_handler: QueueHandler | None = None


def _ensure_default_console_logger() -> None:
    if _LOGGER.handlers:
        return

    stdout_handler = logging.StreamHandler(sys.stdout)
    setattr(stdout_handler, _STDOUT_HANDLER_MARKER, True)
    stdout_handler.setFormatter(_LOG_FORMATTER)
    stdout_handler.setLevel(logging.DEBUG)
    _LOGGER.addHandler(stdout_handler)
    _LOGGER.setLevel(logging.DEBUG)


class Logger:
    def __init__(self, name: str) -> None:
        _ensure_default_console_logger()
        self.logger = _LOGGER
        self.name = name.split(".")[-1]

    def debug(self, message: object) -> None:
        self.logger.debug("%s:%s", self.name, message)

    def info(self, message: object) -> None:
        self.logger.info("%s:%s", self.name, message)

    def warning(self, message: object) -> None:
        self.logger.warning("%s:%s", self.name, message)

    def error(self, message: object, exception: Exception | None = None) -> None:
        stack_trace = self.exception_to_str(exception)
        suffix = f" {stack_trace}" if stack_trace else ""
        self.logger.error("%s:%s%s", self.name, message, suffix)

    def critical(self, message: object) -> None:
        self.logger.critical("%s:%s", self.name, message)

    @staticmethod
    def call_stack() -> str:
        stack = ""
        for frame_info in inspect.stack():
            frame = frame_info.frame
            stack += (
                f"  File: {frame.f_code.co_filename}, "
                f"Line: {frame.f_lineno}, "
                f"Function: {frame.f_code.co_name}\n"
            )
        return stack

    @staticmethod
    def get_logger(name: str) -> Logger:
        return Logger(name)

    @staticmethod
    def exception_to_str(exception: Exception | None) -> str:
        if exception is None:
            return ""

        try:
            return "".join(
                traceback.format_exception(
                    type(exception), exception, exception.__traceback__
                )
            )
        except Exception as formatting_error:
            return f"Error formatting exception: {formatting_error}"


def _log_exception_handler(
    exc_type: type[BaseException],
    exc_value: BaseException,
    exc_traceback,
) -> None:
    traceback_text = "".join(
        traceback.format_exception(exc_type, exc_value, exc_traceback)
    )
    _LOGGER.error("Uncaught exception: %s", traceback_text)


sys.excepthook = _log_exception_handler


class InfoFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return record.levelno < logging.ERROR


class SafeFileHandler(TimedRotatingFileHandler):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.namer = self._rotation_filename

    def _rotation_filename(self, default_name: str) -> str:
        directory, file_name = os.path.split(default_name)
        base_name = os.path.basename(self.baseFilename)
        prefix = base_name + "."
        if base_name.endswith(".log") and file_name.startswith(prefix):
            suffix = file_name[len(prefix) :]
            stem = base_name[:-4]
            return os.path.join(directory, f"{stem}.{suffix}.log")
        return default_name

    def emit(self, record: logging.LogRecord) -> None:
        if self.stream is None or self.stream.closed:
            return
        try:
            super().emit(record)
        except Exception:
            self.handleError(record)

    def getFilesToDelete(self) -> list[str]:  # noqa: N802
        if self.backupCount <= 0:
            return []

        directory, base_name = os.path.split(self.baseFilename)
        if not base_name.endswith(".log"):
            return super().getFilesToDelete()

        stem = base_name[:-4]
        date_pattern = self.extMatch.pattern
        new_pattern = re.compile(rf"^{re.escape(stem)}\.({date_pattern})\.log$")
        legacy_pattern = re.compile(
            rf"^{re.escape(base_name)}\.({date_pattern})$"
        )
        result: list[tuple[float, float, str]] = []

        for file_name in os.listdir(directory):
            if new_pattern.fullmatch(file_name) or legacy_pattern.fullmatch(file_name):
                file_path = os.path.join(directory, file_name)
                result.append(
                    (
                        os.path.getmtime(file_path),
                        os.path.getctime(file_path),
                        file_path,
                    )
                )

        if len(result) <= self.backupCount:
            return []

        result.sort()
        return [
            path for _, _, path in result[: len(result) - self.backupCount]
        ]


def config_logger(config: dict | None = None, name: str = "live-gpt") -> None:
    global _file_listener, _file_handler, _queue_handler

    shutdown_logger()

    parser = argparse.ArgumentParser(description="Live GPT logging options")
    parser.add_argument("--parent_pid", type=int, default=0)
    args, _ = parser.parse_known_args()

    if config is None:
        config = {"debug": True}

    log_level = logging.DEBUG if config.get("debug") else logging.INFO
    _LOGGER.setLevel(log_level)
    _LOGGER.propagate = False

    existing_stdout_handler = _get_stdout_handler()
    _LOGGER.handlers = (
        [existing_stdout_handler]
        if existing_stdout_handler is not None and args.parent_pid == 0
        else []
    )

    if args.parent_pid == 0:
        stdout_handler = existing_stdout_handler or logging.StreamHandler(sys.stdout)
        setattr(stdout_handler, _STDOUT_HANDLER_MARKER, True)
        stdout_handler.setFormatter(_LOG_FORMATTER)
        stdout_handler.filters = []
        stdout_handler.addFilter(InfoFilter())
        stdout_handler.setLevel(log_level)
        if stdout_handler not in _LOGGER.handlers:
            _LOGGER.addHandler(stdout_handler)

    stderr_handler = logging.StreamHandler(sys.stderr)
    stderr_handler.setFormatter(_LOG_FORMATTER)
    stderr_handler.setLevel(logging.ERROR)
    _LOGGER.addHandler(stderr_handler)

    if _should_skip_file_logging(config):
        return

    log_file = os.path.abspath(os.path.join("logs", f"{name}.log"))
    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    log_queue: queue.Queue = queue.Queue()
    _queue_handler = QueueHandler(log_queue)
    _LOGGER.addHandler(_queue_handler)

    _file_handler = SafeFileHandler(
        log_file,
        when="midnight",
        interval=1,
        backupCount=7,
        encoding="utf-8",
    )
    _file_handler.setFormatter(_LOG_FORMATTER)
    _file_handler.setLevel(logging.DEBUG)

    _file_listener = QueueListener(log_queue, _file_handler)
    _file_listener.start()


def shutdown_logger() -> None:
    global _file_listener, _file_handler, _queue_handler

    if _queue_handler is not None:
        _LOGGER.removeHandler(_queue_handler)
        _queue_handler.close()
        _queue_handler = None
    if _file_listener is not None:
        _file_listener.stop()
        _file_listener = None
    if _file_handler is not None:
        _file_handler.close()
        _file_handler = None


def _should_skip_file_logging(config: dict | None) -> bool:
    if config and config.get("disable_file_log"):
        return True
    return os.environ.get("LIVE_GPT_DISABLE_FILE_LOG") == "1" or "pytest" in sys.modules


def _get_stdout_handler() -> logging.Handler | None:
    for handler in _LOGGER.handlers:
        if getattr(handler, _STDOUT_HANDLER_MARKER, False):
            return handler
    return None
