from __future__ import annotations

import contextlib
import hashlib
import io
import json
import os
import sys
import tarfile
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, call, patch

from live_gpt.voice.dependencies import (
    OperationCancelled,
    PackageRequirement,
    PipProgressFormatter,
    PYPI_MIRRORS,
    _isolated_check,
    dependency_status,
    install_packages,
    run_logged_process,
)
from live_gpt.voice.sherpa_stt import (
    LocalDictationSession,
    STT_MODELS,
    SherpaSttProvider,
)


class VoiceModelManagerTests(unittest.TestCase):
    def test_isolated_dependency_check_rejects_cpu_only_torch(self) -> None:
        cpu_torch = SimpleNamespace(
            version=SimpleNamespace(cuda=None),
            cuda=SimpleNamespace(),
        )
        payload = json.dumps(
            {
                "requirements": [],
                "extra_imports": [],
                "require_nvidia_cuda": True,
            }
        )

        with patch.dict(sys.modules, {"torch": cpu_torch}):
            with self.assertRaisesRegex(RuntimeError, "CPU-only PyTorch"):
                _isolated_check(payload)

    def test_dependency_check_ignores_unrelated_process_output(self) -> None:
        completed = SimpleNamespace(
            stdout=(
                "library startup output\n"
                "__LIVE_GPT_DEPENDENCY_RESULT__:{\"ok\": false, "
                "\"message\": \"modelscope is missing\"}\n"
                "library shutdown output\n"
            ),
            stderr="",
        )
        with patch(
            "live_gpt.voice.dependencies.subprocess.run",
            return_value=completed,
        ):
            ok, message = dependency_status(
                (PackageRequirement("modelscope", "1.39.1"),)
            )

        self.assertFalse(ok)
        self.assertEqual(message, "modelscope is missing")

    def test_installer_process_streams_merged_output_lines(self) -> None:
        lines: list[str] = []

        return_code, tail = run_logged_process(
            [
                sys.executable,
                "-c",
                "import sys; print('pip stdout', flush=True); "
                "print('pip stderr', file=sys.stderr, flush=True)",
            ],
            lines.append,
            timeout=10,
        )

        self.assertEqual(return_code, 0)
        self.assertEqual(lines, ["pip stdout", "pip stderr"])
        self.assertIn("pip stdout", tail)
        self.assertIn("pip stderr", tail)

    def test_pip_raw_progress_is_formatted_for_the_second_status_line(self) -> None:
        formatter = PipProgressFormatter()
        with patch(
            "live_gpt.voice.dependencies.time.monotonic",
            side_effect=(10.0, 12.0),
        ):
            self.assertEqual(
                formatter.format("Progress 0 of 2600000000"),
                "   " + "─" * 40 + " 0.0/2.6 GB",
            )
            progress = formatter.format("Progress 1300000000 of 2600000000")

        self.assertIn("━" * 20 + "─" * 20, progress)
        self.assertIn("1.3/2.6 GB", progress)
        self.assertIn("650.0 MB/s", progress)
        self.assertIn("eta 0:00:02", progress)


    def test_shared_pip_installer_selects_requested_mirror(self) -> None:
        logs: list[str] = []
        with patch(
            "live_gpt.voice.dependencies.run_logged_process",
            return_value=(0, "installed"),
        ) as process:
            install_packages(
                ("example==1.0",),
                mirror="ali",
                log=logs.append,
                timeout=30,
                environment={"MAX_JOBS": "4"},
            )

        command = process.call_args.args[0]
        index = command.index("--index-url")
        self.assertEqual(command[index + 1], PYPI_MIRRORS["ali"].index_url)
        progress = command.index("--progress-bar")
        self.assertEqual(command[progress + 1], "raw")
        self.assertTrue(process.call_args.kwargs["pip_progress"])
        self.assertEqual(
            process.call_args.kwargs["environment"],
            {"MAX_JOBS": "4"},
        )
        self.assertTrue(any("MAX_JOBS=4" in line for line in logs))
        self.assertIn("Aliyun", logs[0])
        self.assertTrue(logs[1].startswith("Command: "))
        self.assertIn("example==1.0", logs[1])

    def test_custom_package_index_uses_selected_mirror_for_dependencies(self) -> None:
        logs: list[str] = []
        with patch(
            "live_gpt.voice.dependencies.run_logged_process",
            return_value=(0, "installed"),
        ) as process:
            install_packages(
                ("torch==2.11.0+cu126",),
                mirror="ali",
                log=logs.append,
                timeout=30,
                index_url="https://download.pytorch.org/whl/cu126",
                index_label="PyTorch NVIDIA CUDA wheels",
            )

        command = process.call_args.args[0]
        primary = command.index("--index-url")
        extra = command.index("--extra-index-url")
        self.assertEqual(
            command[primary + 1],
            "https://download.pytorch.org/whl/cu126",
        )
        self.assertEqual(
            command[extra + 1],
            PYPI_MIRRORS["ali"].index_url,
        )
        self.assertTrue(any("for dependencies" in line for line in logs))

    def test_logged_process_can_be_cancelled(self) -> None:
        cancellation = threading.Event()
        timer = threading.Timer(0.1, cancellation.set)
        timer.start()
        logs: list[str] = []
        try:
            with self.assertRaisesRegex(OperationCancelled, "cancelled"):
                run_logged_process(
                    [
                        sys.executable,
                        "-c",
                        "import time; print('started', flush=True); time.sleep(10)",
                    ],
                    logs.append,
                    timeout=15,
                    cancel_event=cancellation,
                )
        finally:
            timer.cancel()

        self.assertIn("started", logs)
        self.assertTrue(any("Cancellation requested" in line for line in logs))


    def test_bundle_status_detects_valid_and_modified_model_files(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manager = SherpaSttProvider(directory)
            model = STT_MODELS["en_nemo_conformer_ctc_small"]
            root = manager.model_directory("stt", model.key)
            hashes: dict[str, str] = {}
            for relative in model.required_files:
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(relative.encode("utf-8"))
                hashes[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
            (root / "manifest.json").write_text(
                json.dumps({"model_type": "stt", "model": model.key, "files": hashes}),
                encoding="utf-8",
            )

            self.assertTrue(manager.model_status("stt", model.key)[0])
            (root / model.required_files[0]).write_bytes(b"modified")
            ok, message = manager.model_status("stt", model.key)

            self.assertFalse(ok)
            self.assertIn("integrity", message)


    def test_safe_extract_rejects_parent_directory_entries(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = root / "unsafe.tar.bz2"
            with tarfile.open(archive, "w:bz2") as output:
                info = tarfile.TarInfo("../outside.txt")
                payload = b"unsafe"
                info.size = len(payload)
                output.addfile(info, io.BytesIO(payload))

            with self.assertRaisesRegex(RuntimeError, "unsafe path"):
                SherpaSttProvider._safe_extract(archive, root / "extract")

    def test_download_preserves_parent_permissions_and_verifies_model(self) -> None:
        model = STT_MODELS["en_nemo_conformer_ctc_small"]

        def download(asset, destination, progress, cancel_event):
            with tarfile.open(destination, "w:bz2") as output:
                for relative in model.required_files:
                    payload = relative.encode("utf-8")
                    info = tarfile.TarInfo(f"{asset.directory}/{relative}")
                    info.size = len(payload)
                    output.addfile(info, io.BytesIO(payload))

        with tempfile.TemporaryDirectory() as directory:
            manager = SherpaSttProvider(directory)
            with (
                patch.object(manager, "_download_asset", side_effect=download),
                patch("os.mkdir", wraps=os.mkdir) as mkdir,
            ):
                manager.download_model("stt", model.key)
            staging_calls = [
                invocation for invocation in mkdir.call_args_list
                if Path(invocation.args[0]).name.startswith(f".{model.key}-")
            ]
            self.assertEqual(len(staging_calls), 1)
            self.assertEqual(staging_calls[0].args[1], 0o777)
            self.assertTrue(manager.model_status("stt", model.key)[0])
            self.assertEqual(
                list((Path(directory) / "stt").iterdir()),
                [manager.model_directory("stt", model.key)],
            )

    def test_model_permission_error_explains_recovery(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manager = SherpaSttProvider(directory)
            with patch.object(Path, "read_text", side_effect=PermissionError("denied")):
                ok, message = manager.model_status("stt", "zh_sense_voice_small_int8")
            self.assertFalse(ok)
            self.assertIn("Restore inherited permissions", message)
            self.assertIn(str(manager.model_directory("stt", "zh_sense_voice_small_int8")), message)

    def test_sense_voice_language_reaches_factory_and_partitions_cache(self) -> None:
        factory = Mock(side_effect=lambda **kwargs: object())
        fake_sherpa = SimpleNamespace(OfflineRecognizer=SimpleNamespace(from_sense_voice=factory))
        with tempfile.TemporaryDirectory() as directory, patch.dict(sys.modules, {"sherpa_onnx": fake_sherpa}):
            manager = SherpaSttProvider(directory)
            for language in ("auto", "zh", "en"):
                first = manager._get_recognizer("zh_sense_voice_small_int8", language, verify=False)
                second = manager._get_recognizer("zh_sense_voice_small_int8", language, verify=False)
                self.assertIs(first, second)
            self.assertEqual([c.kwargs["language"] for c in factory.call_args_list], ["auto", "zh", "en"])
            manager._invalidate_model("zh_sense_voice_small_int8")
            self.assertEqual(manager._recognizers, {})
            for language in ("auto", "zh", "invalid"):
                with self.assertRaisesRegex(ValueError, "does not support language"):
                    manager._get_recognizer("en_moonshine_tiny_int8", language, verify=False)

    def test_offline_transcribe_uses_public_model_factories(self) -> None:
        cases = (
            ("zh_zipformer_ctc_int8_2025_07_03", "from_zipformer_ctc"),
            ("zh_paraformer_int8", "from_paraformer"),
            ("en_nemo_conformer_ctc_small", "from_nemo_ctc"),
            ("zh_sense_voice_small_int8", "from_sense_voice"),
            ("en_moonshine_tiny_int8", "from_moonshine"),
        )
        for model_key, factory_name in cases:
            with self.subTest(model=model_key):
                stream = Mock()
                stream.result = SimpleNamespace(text="recognized locally")
                recognizer = Mock()
                recognizer.create_stream.return_value = stream
                factory = Mock(return_value=recognizer)
                fake_sherpa = SimpleNamespace(
                    OfflineRecognizer=SimpleNamespace(
                        **{factory_name: factory}
                    )
                )

                with (
                    tempfile.TemporaryDirectory() as directory,
                    patch.dict(sys.modules, {"sherpa_onnx": fake_sherpa}),
                ):
                    manager = SherpaSttProvider(directory)
                    manager._resample = Mock(return_value=[0.0, 0.1, -0.1])
                    text = manager.transcribe(
                        model_key,
                        [0.0, 0.1, -0.1],
                        16_000,
                    )

                self.assertEqual(text, "recognized locally")
                factory.assert_called_once()
                stream.accept_waveform.assert_called_once()
                recognizer.decode_stream.assert_called_once_with(stream)

    def test_streaming_transcribe_uses_public_online_factory(self) -> None:
        stream = Mock()
        stream.result = SimpleNamespace(text="流式识别")
        recognizer = Mock()
        recognizer.create_stream.return_value = stream
        recognizer.is_ready.side_effect = [True, False]
        recognizer.get_result.return_value = "流式识别"
        factory = Mock(return_value=recognizer)
        fake_sherpa = SimpleNamespace(
            OnlineRecognizer=SimpleNamespace(from_zipformer2_ctc=factory)
        )

        with (
            tempfile.TemporaryDirectory() as directory,
            patch.dict(sys.modules, {"sherpa_onnx": fake_sherpa}),
        ):
            manager = SherpaSttProvider(directory)
            manager._resample = Mock(return_value=[0.0, 0.1, -0.1])
            text = manager.transcribe(
                "zh_streaming_zipformer_small_ctc_int8_2025_04_01",
                [0.0, 0.1, -0.1],
                16_000,
            )

        self.assertEqual(text, "流式识别")
        factory.assert_called_once()
        stream.input_finished.assert_called_once()
        recognizer.decode_stream.assert_called_once_with(stream)
        recognizer.get_result.assert_called_once_with(stream)

    def test_sherpa_prepare_verifies_and_loads_selected_model_once(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manager = SherpaSttProvider(directory)
            recognizer = object()
            manager.model_status = Mock(return_value=(True, "ready"))
            manager._create_recognizer = Mock(return_value=recognizer)

            with patch.dict(sys.modules, {"sounddevice": SimpleNamespace()}):
                first = manager.prepare("en_moonshine_tiny_int8")
                second = manager.prepare("en_moonshine_tiny_int8")

        self.assertIs(first, recognizer)
        self.assertIs(second, recognizer)
        manager.model_status.assert_called_once_with(
            "stt", "en_moonshine_tiny_int8"
        )
        manager._create_recognizer.assert_called_once_with(
            "en_moonshine_tiny_int8", "en"
        )

    def test_streaming_session_emits_partial_text_before_final_result(self) -> None:
        manager = Mock()
        stream = Mock()
        recognizer = Mock()
        recognizer.create_stream.return_value = stream
        manager.prepare.return_value = recognizer
        session = LocalDictationSession(
            manager,
            "zh_streaming_zipformer_small_ctc_int8_2025_04_01",
        )

        results = iter(("实时", "实时结尾", "实时完成"))

        def decode(_stream: object) -> None:
            if recognizer.decode_stream.call_count == 1:
                session.stop()

        recognizer.decode_stream.side_effect = decode
        recognizer.is_ready.side_effect = [
            True,
            False,
            True,
            False,
            True,
            False,
        ]
        recognizer.get_result.side_effect = lambda _stream: next(results)

        class FakeAudio:
            def __getitem__(self, _key: object) -> FakeAudio:
                return self

            def copy(self) -> list[float]:
                return [0.1, 0.2]

        input_options: dict[str, object] = {}

        timers: list[threading.Timer] = []

        class FakeInputStream:
            def __init__(self, **kwargs: object) -> None:
                input_options.update(kwargs)
                self.callback = kwargs["callback"]

            def __enter__(self) -> FakeInputStream:
                self.callback(FakeAudio(), 2, None, None)
                timer = threading.Timer(
                    0.05,
                    lambda: self.callback(FakeAudio(), 2, None, None),
                )
                timers.append(timer)
                timer.start()
                return self

            def __exit__(self, *_args: object) -> None:
                return None

        fake_numpy = SimpleNamespace(concatenate=lambda chunks: chunks[0])
        fake_sounddevice = SimpleNamespace(InputStream=FakeInputStream)
        partials: list[str] = []
        with patch.dict(
            sys.modules,
            {"numpy": fake_numpy, "sounddevice": fake_sounddevice},
        ):
            success, text, message = session.run(lambda: None, partials.append)
        for timer in timers:
            timer.join(timeout=1)

        self.assertTrue(success)
        self.assertEqual(text, "实时完成")
        self.assertEqual(partials, ["实时", "实时结尾", "实时完成"])
        self.assertTrue(message.startswith("Finalized in "))
        manager.transcribe.assert_not_called()
        self.assertEqual(stream.accept_waveform.call_count, 2)
        self.assertEqual(recognizer.get_result.call_count, 3)
        self.assertEqual(input_options["latency"], "low")


if __name__ == "__main__":
    unittest.main()
