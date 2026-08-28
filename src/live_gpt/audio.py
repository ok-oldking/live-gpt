from __future__ import annotations

import threading
import wave
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from time import monotonic

import sounddevice

from .logger import Logger


logger = Logger.get_logger(__name__)


@dataclass(frozen=True)
class RecordingResult:
    path: Path
    duration_seconds: float


@dataclass(frozen=True)
class AudioDevice:
    index: int
    name: str
    is_system_default: bool = False


def list_recording_devices() -> list[AudioDevice]:
    return _list_windows_audio_devices(
        channel_key="max_input_channels",
        default_key="default_input_device",
    )


def list_playback_devices() -> list[AudioDevice]:
    return _list_windows_audio_devices(
        channel_key="max_output_channels",
        default_key="default_output_device",
    )


def _list_windows_audio_devices(
    channel_key: str,
    default_key: str,
) -> list[AudioDevice]:
    devices = sounddevice.query_devices()
    host_apis = sounddevice.query_hostapis()
    wasapi = next(
        (
            host_api
            for host_api in host_apis
            if str(host_api["name"]).casefold() == "windows wasapi"
        ),
        None,
    )

    if wasapi is None:
        device_indices = range(len(devices))
        default_device = None
    else:
        device_indices = wasapi["devices"]
        configured_default = int(wasapi[default_key])
        default_device = configured_default if configured_default >= 0 else None

    result = [
        AudioDevice(
            index=int(index),
            name=str(devices[index]["name"]),
            is_system_default=int(index) == default_device,
        )
        for index in device_indices
        if int(devices[index][channel_key]) > 0
    ]
    result.sort(
        key=lambda device: (
            not device.is_system_default,
            device.name.casefold(),
        )
    )
    return result


class AudioRecorder:
    def __init__(
        self,
        output_directory: Path | str = "recordings",
        sample_rate: int = 16_000,
        input_device: int | None = None,
    ) -> None:
        self.output_directory = Path(output_directory)
        self.sample_rate = sample_rate
        self.channels = 1
        self.sample_width = 2
        self.input_device = input_device
        self._stream: sounddevice.RawInputStream | None = None
        self._chunks: list[bytes] = []
        self._chunks_lock = threading.Lock()
        self._started_at: float | None = None
        self._active_sample_rate: int | None = None
        self.audio_chunk_callback: Callable[[bytes, int], None] | None = None
        self.recording_started_callback: Callable[[int], None] | None = None
        self.recording_stopped_callback: Callable[[], None] | None = None

    @property
    def is_recording(self) -> bool:
        return self._stream is not None

    @property
    def active_sample_rate(self) -> int | None:
        return self._active_sample_rate

    def start(self) -> None:
        if self.is_recording:
            return

        with self._chunks_lock:
            self._chunks.clear()

        device_info = sounddevice.query_devices(self.input_device, "input")
        device_name = str(device_info["name"])
        native_sample_rate = int(round(float(device_info["default_samplerate"])))
        active_sample_rate = self.sample_rate

        try:
            sounddevice.check_input_settings(
                device=self.input_device,
                channels=self.channels,
                dtype="int16",
                samplerate=active_sample_rate,
            )
        except sounddevice.PortAudioError:
            if native_sample_rate == active_sample_rate:
                raise
            logger.warning(
                f"Recording device index={self.input_device} name={device_name!r} "
                f"does not support {active_sample_rate} Hz; "
                f"using native rate {native_sample_rate} Hz"
            )
            sounddevice.check_input_settings(
                device=self.input_device,
                channels=self.channels,
                dtype="int16",
                samplerate=native_sample_rate,
            )
            active_sample_rate = native_sample_rate

        logger.info(
            f"Opening recording device index={self.input_device} "
            f"name={device_name!r} sample_rate={active_sample_rate} Hz"
        )
        stream = sounddevice.RawInputStream(
            samplerate=active_sample_rate,
            channels=self.channels,
            dtype="int16",
            device=self.input_device,
            callback=self._audio_callback,
        )
        self._stream = stream
        self._started_at = monotonic()
        self._active_sample_rate = active_sample_rate

        try:
            if self.recording_started_callback is not None:
                self.recording_started_callback(active_sample_rate)
            stream.start()
        except Exception:
            self._stream = None
            self._started_at = None
            self._active_sample_rate = None
            stream.close()
            raise

        logger.info(
            f"Microphone recording started device={self.input_device} "
            f"name={device_name!r} sample_rate={active_sample_rate} Hz mono"
        )

    def stop(self) -> RecordingResult | None:
        stream = self._stream
        if stream is None:
            return None

        try:
            stream.stop()
        finally:
            stream.close()
            self._stream = None

        if self.recording_stopped_callback is not None:
            self.recording_stopped_callback()

        started_at = self._started_at
        self._started_at = None
        active_sample_rate = self._active_sample_rate or self.sample_rate
        self._active_sample_rate = None
        duration = monotonic() - started_at if started_at is not None else 0.0

        with self._chunks_lock:
            audio_data = b"".join(self._chunks)
            self._chunks.clear()

        if not audio_data:
            logger.warning("Microphone recording stopped without audio data")
            return None

        self.output_directory.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        output_path = self.output_directory / f"recording-{timestamp}.wav"

        with wave.open(str(output_path), "wb") as wav_file:
            wav_file.setnchannels(self.channels)
            wav_file.setsampwidth(self.sample_width)
            wav_file.setframerate(active_sample_rate)
            wav_file.writeframes(audio_data)

        result = RecordingResult(output_path.resolve(), duration)
        logger.info(
            f"Microphone recording saved to {result.path} "
            f"duration={result.duration_seconds:.2f}s"
        )
        return result

    def _audio_callback(self, input_data, frames, time_info, status) -> None:
        del frames, time_info
        if status:
            logger.warning(f"Microphone stream status: {status}")
        pcm_data = bytes(input_data)
        with self._chunks_lock:
            self._chunks.append(pcm_data)
        if self.audio_chunk_callback is not None:
            self.audio_chunk_callback(
                pcm_data,
                self._active_sample_rate or self.sample_rate,
            )
