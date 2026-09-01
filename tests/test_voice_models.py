from __future__ import annotations

import hashlib
import io
import json
import sys
import tarfile
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from live_gpt.voice.qwen_tts import QwenTtsProvider, TTS_MODELS
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

    def test_qwen_pip_lines_reach_ui_callback_and_app_logger(self) -> None:
        displayed: list[str] = []
        install_calls: list[tuple[tuple[str, ...], dict[str, object]]] = []

        def fake_install(
            packages: object,
            *,
            mirror: str,
            log: object,
            timeout: float,
            options: object = (),
            cancel_event: object = None,
            index_url: str = "",
            index_label: str = "",
        ) -> None:
            self.assertIsNotNone(cancel_event)
            install_calls.append(
                (
                    tuple(packages),
                    {
                        "mirror": mirror,
                        "timeout": timeout,
                        "options": options,
                        "index_url": index_url,
                        "index_label": index_label,
                    },
                )
            )
            log("Collecting qwen-tts")

        with (
            patch(
                "live_gpt.voice.qwen_tts.install_packages",
                side_effect=fake_install,
            ),
            patch.object(
                QwenTtsProvider,
                "dependency_status",
                return_value=(True, "runtime verified"),
            ),
            patch.object(
                QwenTtsProvider,
                "nvidia_gpu_status",
                return_value=(True, "NVIDIA Test GPU, driver 1"),
            ),
            patch("live_gpt.voice.qwen_tts.logger.info") as app_log,
        ):
            QwenTtsProvider.install_dependencies(
                log=displayed.append,
                mirror="ali",
                cancel_event=threading.Event(),
            )

        self.assertIn("Collecting qwen-tts", displayed)
        app_log.assert_any_call("pip:Collecting qwen-tts")
        self.assertEqual(len(install_calls), 2)
        self.assertIn("torch==2.11.0", install_calls[0][0])
        self.assertIn("torchaudio==2.11.0", install_calls[0][0])
        self.assertEqual(
            install_calls[0][1]["index_url"],
            "https://mirrors.aliyun.com/pytorch-wheels/cu126/",
        )
        self.assertEqual(install_calls[0][1]["mirror"], "ali")
        self.assertEqual(install_calls[1][1]["mirror"], "ali")
        self.assertEqual(install_calls[1][1]["timeout"], 3600)

    def test_qwen_official_pytorch_index_keeps_cuda_version_suffix(self) -> None:
        install_calls: list[tuple[str, ...]] = []

        def fake_install(packages, **_kwargs):
            install_calls.append(tuple(packages))

        with (
            patch(
                "live_gpt.voice.qwen_tts.install_packages",
                side_effect=fake_install,
            ),
            patch.object(
                QwenTtsProvider,
                "dependency_status",
                return_value=(True, "runtime verified"),
            ),
            patch.object(
                QwenTtsProvider,
                "nvidia_gpu_status",
                return_value=(True, "NVIDIA Test GPU, driver 1"),
            ),
        ):
            QwenTtsProvider.install_dependencies(mirror="default")

        self.assertIn("torch==2.11.0+cu126", install_calls[0])
        self.assertIn("torchaudio==2.11.0+cu126", install_calls[0])

    def test_qwen_sjtug_uses_sjtug_pip_and_cuda_indexes(self) -> None:
        install_calls: list[tuple[tuple[str, ...], dict[str, object]]] = []

        def fake_install(packages, **kwargs):
            install_calls.append((tuple(packages), kwargs))

        with (
            patch(
                "live_gpt.voice.qwen_tts.install_packages",
                side_effect=fake_install,
            ),
            patch.object(
                QwenTtsProvider,
                "dependency_status",
                return_value=(True, "runtime verified"),
            ),
            patch.object(
                QwenTtsProvider,
                "nvidia_gpu_status",
                return_value=(True, "NVIDIA Test GPU, driver 1"),
            ),
        ):
            QwenTtsProvider.install_dependencies(mirror="sjtug")

        self.assertIn("torch==2.11.0+cu126", install_calls[0][0])
        self.assertIn("torchaudio==2.11.0+cu126", install_calls[0][0])
        self.assertEqual(
            install_calls[0][1]["index_url"],
            "https://mirror.sjtu.edu.cn/pytorch-wheels/cu126",
        )
        self.assertEqual(install_calls[0][1]["mirror"], "sjtug")
        self.assertEqual(install_calls[1][1]["mirror"], "sjtug")

    def test_qwen_load_rejects_cpu_only_pytorch(self) -> None:
        manager = QwenTtsProvider()
        manager.model_status = Mock(return_value=(True, "verified"))
        cpu_torch = SimpleNamespace(
            version=SimpleNamespace(cuda=None),
            cuda=SimpleNamespace(
                is_available=Mock(return_value=False),
                device_count=Mock(return_value=0),
            ),
        )

        with patch.dict(sys.modules, {"torch": cpu_torch}):
            with self.assertRaisesRegex(RuntimeError, "NVIDIA GPU"):
                manager._load("qwen3_tts_0_6b_custom_voice")

    def test_qwen_download_installs_missing_modelscope_client(self) -> None:
        with (
            patch(
                "live_gpt.voice.qwen_tts.dependency_status",
                side_effect=[(False, "missing"), (True, "")],
            ),
            patch("live_gpt.voice.qwen_tts.install_packages") as install,
        ):
            QwenTtsProvider.ensure_download_client(
                mirror="ali",
                cancel_event=threading.Event(),
                source="modelscope",
            )

        self.assertEqual(
            install.call_args.args[0],
            ("modelscope==1.39.1",),
        )
        self.assertEqual(install.call_args.kwargs["mirror"], "ali")

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
            )

        command = process.call_args.args[0]
        index = command.index("--index-url")
        self.assertEqual(command[index + 1], PYPI_MIRRORS["ali"].index_url)
        progress = command.index("--progress-bar")
        self.assertEqual(command[progress + 1], "raw")
        self.assertTrue(process.call_args.kwargs["pip_progress"])
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

    def test_catalog_has_requested_independent_stt_and_tts_models(self) -> None:
        self.assertEqual(len(STT_MODELS), 10)
        self.assertEqual(len(TTS_MODELS), 2)
        self.assertEqual(
            len(TTS_MODELS["qwen3_tts_0_6b_custom_voice"].speakers), 9
        )
        self.assertFalse(hasattr(SherpaSttProvider, "synthesize"))
        for model in STT_MODELS.values():
            self.assertGreater(model.asset.size, 0)
            self.assertEqual(len(model.asset.sha256), 64)
            int(model.asset.sha256, 16)
            self.assertTrue(model.asset.url.startswith("https://github.com/"))
        for model in TTS_MODELS.values():
            self.assertGreater(model.download_size, 0)
            self.assertEqual(len(model.revision), 40)
            int(model.revision, 16)
            self.assertTrue(model.repository.startswith("Qwen/"))

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

    def test_qwen_bundle_status_checks_pinned_revision_and_file_hashes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manager = QwenTtsProvider(directory)
            model = TTS_MODELS["qwen3_tts_0_6b_custom_voice"]
            root = manager.model_directory("tts", model.key)
            files = (
                "config.json",
                "model.safetensors",
                "speech_tokenizer/model.safetensors",
            )
            metadata: dict[str, dict[str, object]] = {}
            for relative in files:
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(relative.encode("utf-8"))
                metadata[relative] = {
                    "size": path.stat().st_size,
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                }
            (root / "manifest.json").write_text(
                json.dumps(
                    {
                        "model": model.key,
                        "repository": model.repository,
                        "revision": model.revision,
                        "files": metadata,
                    }
                ),
                encoding="utf-8",
            )

            self.assertTrue(manager.model_status("tts", model.key)[0])
            (root / "model.safetensors").write_bytes(b"modified")
            ok, message = manager.model_status("tts", model.key)

            self.assertFalse(ok)
            self.assertIn("changed", message)

    def test_qwen_download_uses_selected_hub_and_working_models_folder(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            working_directory = Path(directory)
            commands: list[list[str]] = []
            download_logs: list[str] = []

            def fake_download(
                command: object,
                _log: object,
                *,
                timeout: float,
                cancel_event: object = None,
            ) -> tuple[int, str]:
                self.assertEqual(timeout, 21_600)
                self.assertIsNotNone(cancel_event)
                command_parts = list(command)
                commands.append(command_parts)
                staging = Path(command_parts[-1])
                for relative in (
                    "config.json",
                    "model.safetensors",
                    "speech_tokenizer/model.safetensors",
                ):
                    path = staging / relative
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(relative.encode("utf-8"))
                return 0, "downloaded"

            with (
                patch.object(Path, "cwd", return_value=working_directory),
                patch(
                    "live_gpt.voice.qwen_tts.run_logged_process",
                    side_effect=fake_download,
                ),
            ):
                manager = QwenTtsProvider()
                self.assertEqual(manager.model_root, working_directory / "models")
                manager.download_model(
                    "tts",
                    "qwen3_tts_0_6b_custom_voice",
                    cancel_event=threading.Event(),
                    log=download_logs.append,
                    source="huggingface",
                )
                manager.download_model(
                    "tts",
                    "qwen3_tts_0_6b_custom_voice",
                    cancel_event=threading.Event(),
                    log=download_logs.append,
                    source="modelscope",
                )

            self.assertIn("huggingface_hub", commands[0][2])
            self.assertIn("modelscope", commands[1][2])
            self.assertTrue(
                any(
                    line.startswith("Command: ") and "from modelscope" in line
                    for line in download_logs
                )
            )
            manifest = json.loads(
                (
                    working_directory
                    / "models"
                    / "qwen3_tts_0_6b_custom_voice"
                    / "manifest.json"
                ).read_text(encoding="utf-8")
            )
            self.assertEqual(manifest["source"], "modelscope")
            self.assertEqual(manifest["revision"], "master")

    def test_qwen_synthesis_uses_custom_voice_and_named_speaker(self) -> None:
        manager = QwenTtsProvider()
        runtime = Mock()
        runtime.generate_custom_voice.return_value = ([[0.1, -0.1]], 24_000)
        manager._load = Mock(return_value=runtime)

        samples, sample_rate = manager.synthesize(
            "qwen3_tts_0_6b_custom_voice",
            "你好",
            "Vivian",
        )

        self.assertEqual(samples, [0.1, -0.1])
        self.assertEqual(sample_rate, 24_000)
        runtime.generate_custom_voice.assert_called_once_with(
            text="你好",
            language="Auto",
            speaker="Vivian",
        )

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

    def test_streaming_session_emits_partial_text_before_final_result(self) -> None:
        manager = Mock()
        manager.model_status.return_value = (True, "ready")
        stream = Mock()
        recognizer = Mock()
        recognizer.create_stream.return_value = stream
        manager.create_streaming_recognizer.return_value = recognizer
        session = LocalDictationSession(
            manager,
            "zh_streaming_zipformer_small_ctc_int8_2025_04_01",
        )

        results = iter(("实时", "实时完成"))

        def decode(_stream: object) -> None:
            if recognizer.decode_stream.call_count == 1:
                session.stop()

        recognizer.decode_stream.side_effect = decode
        recognizer.is_ready.side_effect = [True, False, True, False]
        recognizer.get_result.side_effect = lambda _stream: next(results)

        class FakeAudio:
            def __getitem__(self, _key: object) -> FakeAudio:
                return self

            def copy(self) -> list[float]:
                return [0.1, 0.2]

        class FakeInputStream:
            def __init__(self, **kwargs: object) -> None:
                self.callback = kwargs["callback"]

            def __enter__(self) -> FakeInputStream:
                self.callback(FakeAudio(), 2, None, None)
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

        self.assertTrue(success)
        self.assertEqual(text, "实时完成")
        self.assertEqual(partials, ["实时", "实时完成"])
        self.assertTrue(message.startswith("Finalized in "))
        manager.transcribe.assert_not_called()
        self.assertEqual(recognizer.get_result.call_count, 2)


if __name__ == "__main__":
    unittest.main()
