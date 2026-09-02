from __future__ import annotations

import contextlib
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
from unittest.mock import Mock, call, patch

from live_gpt.voice.qwen_tts import QwenTtsProvider, TTS_MODELS
from live_gpt.voice.cosyvoice_tts import (
    COSYVOICE_RUNTIME_IMPORTS,
    COSYVOICE_TTS_MODELS,
    CosyVoiceTtsProvider,
    _block_optional_import,
    _filter_onnx_providers,
    _load_wav_with_soundfile,
    _remove_tree,
)
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
            environment: object = None,
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
                        "environment": environment,
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
            patch.object(QwenTtsProvider, "_cleanup_invalid_pip_remnants"),
            patch.object(
                QwenTtsProvider,
                "_short_build_temporary_directory",
                return_value=contextlib.nullcontext("X:\\"),
            ),
            patch.object(
                QwenTtsProvider,
                "_verified_nvidia_flash_source",
                return_value=contextlib.nullcontext(Path("X:/flash")),
            ),
            patch.object(
                QwenTtsProvider,
                "_nvidia_pip_cuda_environment",
                return_value={"CUDA_HOME": "X:\\cuda"},
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
        self.assertEqual(len(install_calls), 4)
        self.assertIn("torch==2.11.0", install_calls[0][0])
        self.assertIn("torchaudio==2.11.0", install_calls[0][0])
        self.assertEqual(
            install_calls[0][1]["index_url"],
            "https://mirrors.aliyun.com/pytorch-wheels/cu126/",
        )
        self.assertEqual(install_calls[0][1]["mirror"], "ali")
        self.assertEqual(install_calls[0][1]["options"], ())
        self.assertEqual(install_calls[1][1]["mirror"], "ali")
        self.assertEqual(install_calls[1][1]["timeout"], 3600)
        self.assertIn("nvidia-cuda-nvcc-cu12==12.6.85", install_calls[2][0])
        self.assertEqual(install_calls[3][0], (str(Path("X:/flash")),))
        self.assertEqual(
            install_calls[3][1]["options"],
            ("--no-build-isolation",),
        )
        flash_environment = install_calls[3][1]["environment"]
        self.assertEqual(flash_environment["MAX_JOBS"], "4")
        self.assertEqual(flash_environment["CUDA_HOME"], "X:\\cuda")
        self.assertEqual(flash_environment["TEMP"], flash_environment["TMP"])
        self.assertEqual(flash_environment["TEMP"], flash_environment["TMPDIR"])

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
            patch.object(QwenTtsProvider, "_cleanup_invalid_pip_remnants"),
            patch.object(
                QwenTtsProvider,
                "_short_build_temporary_directory",
                return_value=contextlib.nullcontext("X:\\"),
            ),
            patch.object(
                QwenTtsProvider,
                "_verified_nvidia_flash_source",
                return_value=contextlib.nullcontext(Path("X:/flash")),
            ),
            patch.object(
                QwenTtsProvider,
                "_nvidia_pip_cuda_environment",
                return_value={"CUDA_HOME": "X:\\cuda"},
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
            patch.object(QwenTtsProvider, "_cleanup_invalid_pip_remnants"),
            patch.object(
                QwenTtsProvider,
                "_short_build_temporary_directory",
                return_value=contextlib.nullcontext("X:\\"),
            ),
            patch.object(
                QwenTtsProvider,
                "_verified_nvidia_flash_source",
                return_value=contextlib.nullcontext(Path("X:/flash")),
            ),
            patch.object(
                QwenTtsProvider,
                "_nvidia_pip_cuda_environment",
                return_value={"CUDA_HOME": "X:\\cuda"},
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

    def test_qwen_flash_attention_retries_official_pypi_after_mirror_error(
        self,
    ) -> None:
        flash_mirrors: list[str] = []

        def fake_install(packages, **kwargs):
            if tuple(packages) == ("flash-attn==2.8.3",):
                flash_mirrors.append(kwargs["mirror"])
                if kwargs["mirror"] == "sjtug":
                    raise RuntimeError("mirror returned 403")

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
                "flash_attention_status",
                return_value=(True, "FlashAttention verified"),
            ),
            patch.object(
                QwenTtsProvider,
                "nvidia_gpu_status",
                return_value=(True, "NVIDIA Test GPU, driver 1"),
            ),
            patch.object(QwenTtsProvider, "_cleanup_invalid_pip_remnants"),
            patch.object(
                QwenTtsProvider,
                "_short_build_temporary_directory",
                return_value=contextlib.nullcontext("X:\\"),
            ),
            patch("live_gpt.voice.qwen_tts.sys.platform", "linux"),
        ):
            QwenTtsProvider.install_dependencies(mirror="sjtug")

        self.assertEqual(flash_mirrors, ["sjtug", "default"])

    def test_qwen_cleanup_removes_only_known_invalid_pip_backups(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "~orch").mkdir()
            (root / "~orch-2.11.dist-info").mkdir()
            (root / "~custom").mkdir()
            logs: list[str] = []

            with patch(
                "live_gpt.voice.qwen_tts.sysconfig.get_path",
                return_value=str(root),
            ):
                QwenTtsProvider._cleanup_invalid_pip_remnants(logs.append)

            self.assertFalse((root / "~orch").exists())
            self.assertFalse((root / "~orch-2.11.dist-info").exists())
            self.assertTrue((root / "~custom").exists())
            self.assertTrue(any("~orch" in line for line in logs))

    def test_qwen_windows_flash_source_omits_unused_amd_tree(self) -> None:
        archive_buffer = io.BytesIO()
        with tarfile.open(fileobj=archive_buffer, mode="w:gz") as archive:
            for name, content in (
                ("flash_attn-2.8.3/setup.py", b"from setuptools import setup"),
                (
                    "flash_attn-2.8.3/csrc/cutlass/include/cutlass/cutlass.h",
                    b"// NVIDIA dependency",
                ),
                (
                    "flash_attn-2.8.3/csrc/composable_kernel/library/src/"
                    + "very_long_amd_directory/" * 8
                    + "unused.cpp",
                    b"// AMD-only dependency",
                ),
            ):
                info = tarfile.TarInfo(name)
                info.size = len(content)
                archive.addfile(info, io.BytesIO(content))
        archive_bytes = archive_buffer.getvalue()
        metadata = json.dumps(
            {
                "urls": [
                    {
                        "packagetype": "sdist",
                        "filename": "flash_attn-2.8.3.tar.gz",
                        "url": "https://files.example/flash_attn.tar.gz",
                        "digests": {
                            "sha256": hashlib.sha256(archive_bytes).hexdigest()
                        },
                    }
                ]
            }
        ).encode()

        class FakeResponse(io.BytesIO):
            def __init__(self, content: bytes) -> None:
                super().__init__(content)
                self.headers = {"Content-Length": str(len(content))}

            def __enter__(self):
                return self

            def __exit__(self, *_args):
                self.close()

        def fake_open(url, **_kwargs):
            return FakeResponse(
                metadata if str(url).endswith("/json") else archive_bytes
            )

        with tempfile.TemporaryDirectory() as directory:
            logs: list[str] = []
            with patch(
                "live_gpt.voice.qwen_tts.urllib.request.urlopen",
                side_effect=fake_open,
            ):
                with QwenTtsProvider._verified_nvidia_flash_source(
                    directory,
                    logs.append,
                    threading.Event(),
                ) as source:
                    self.assertTrue((source / "setup.py").is_file())
                    self.assertTrue(
                        (
                            source
                            / "csrc"
                            / "cutlass"
                            / "include"
                            / "cutlass"
                            / "cutlass.h"
                        ).is_file()
                    )
                    self.assertFalse((source / "csrc" / "composable_kernel").exists())
                self.assertFalse(source.exists())

        self.assertTrue(any("omitted 1 AMD" in line for line in logs))

    def test_qwen_locates_pip_installed_nvidia_cuda_build_tools(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            relative_files = {
                "nvidia-cuda-nvcc-cu12": Path(
                    "nvidia/cuda_nvcc/bin/nvcc.exe"
                ),
                "nvidia-cuda-runtime-cu12": Path(
                    "nvidia/cuda_runtime/include/cuda_runtime_api.h"
                ),
                "runtime-library": Path(
                    "nvidia/cuda_runtime/lib/x64/cudart.lib"
                ),
                "nvidia-cuda-cccl-cu12": Path(
                    "nvidia/cuda_cccl/include/cub/cub.cuh"
                ),
            }
            for relative in relative_files.values():
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"test")

            distributions = {
                "nvidia-cuda-nvcc-cu12": SimpleNamespace(
                    files=[relative_files["nvidia-cuda-nvcc-cu12"]],
                    locate_file=lambda item: root / item,
                ),
                "nvidia-cuda-runtime-cu12": SimpleNamespace(
                    files=[
                        relative_files["nvidia-cuda-runtime-cu12"],
                        relative_files["runtime-library"],
                    ],
                    locate_file=lambda item: root / item,
                ),
                "nvidia-cuda-cccl-cu12": SimpleNamespace(
                    files=[relative_files["nvidia-cuda-cccl-cu12"]],
                    locate_file=lambda item: root / item,
                ),
            }
            with patch(
                "live_gpt.voice.qwen_tts.importlib.metadata.distribution",
                side_effect=lambda name: distributions[name],
            ):
                environment = QwenTtsProvider._nvidia_pip_cuda_environment()

            self.assertTrue(environment["CUDA_HOME"].endswith("cuda_nvcc"))
            self.assertIn("cuda_runtime", environment["INCLUDE"])
            self.assertIn("cuda_cccl", environment["INCLUDE"])
            self.assertIn("cuda_runtime", environment["LIB"])

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
            non_streaming_mode=False,
            do_sample=False,
            subtalker_dosample=False,
        )

    def test_qwen_streaming_chunks_use_streaming_text_mode(self) -> None:
        manager = QwenTtsProvider()
        manager.stream_text_chunks = Mock(return_value=("First.", "Second."))
        runtime = Mock()
        runtime.generate_custom_voice.side_effect = (
            ([[0.1]], 24_000),
            ([[0.2]], 24_000),
        )
        manager._load = Mock(return_value=runtime)

        chunks = list(
            manager.synthesize_stream(
                "qwen3_tts_0_6b_custom_voice",
                "First. Second.",
                "Ryan",
                "English",
            )
        )

        self.assertEqual(
            chunks,
            [([0.1], 24_000, "First."), ([0.2], 24_000, "Second.")],
        )
        self.assertEqual(
            runtime.generate_custom_voice.call_args_list,
            [
                call(
                    text=["First."],
                    language=["English"],
                    speaker=["Ryan"],
                    non_streaming_mode=False,
                    do_sample=False,
                    subtalker_dosample=False,
                ),
                call(
                    text=["Second."],
                    language=["English"],
                    speaker=["Ryan"],
                    non_streaming_mode=False,
                    do_sample=False,
                    subtalker_dosample=False,
                )
            ],
        )

    def test_qwen_streaming_chunks_long_english_reply_at_natural_pauses(self) -> None:
        text = (
            "Yes. For Cybernetic Creed, I’d research Integrated Cybernetics "
            "immediately if it appears. The reason is simple: your origin lets "
            "you open the Cybernetics tradition tree as soon as Integrated "
            "Cybernetics is researched, so getting that tech early accelerates "
            "the entire build. Current 4.3 Cybernetic Creed rush builds "
            "explicitly do this."
        )

        chunks = QwenTtsProvider.stream_text_chunks(text)

        self.assertGreater(len(chunks), 1)
        self.assertTrue(chunks[0].startswith("Yes. For Cybernetic Creed"))
        self.assertTrue(all(len(chunk) <= 220 for chunk in chunks))
        self.assertEqual(" ".join(chunks), " ".join(text.split()))

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
            "en_moonshine_tiny_int8"
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

    def test_cosyvoice_bundle_status_checks_every_downloaded_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manager = CosyVoiceTtsProvider(directory)
            model = COSYVOICE_TTS_MODELS["fun_cosyvoice3_0_5b_2512"]
            model_directory = manager.model_directory("tts", model.key)
            required = (
                "cosyvoice3.yaml",
                "llm.pt",
                "flow.pt",
                "hift.pt",
                "campplus.onnx",
                "speech_tokenizer_v3.onnx",
            )
            files: dict[str, dict[str, object]] = {}
            for index, relative in enumerate(required):
                path = model_directory / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(f"model-{index}".encode())
                files[relative] = {
                    "size": path.stat().st_size,
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                }
            (model_directory / "manifest.json").write_text(
                json.dumps(
                    {
                        "model": model.key,
                        "repository": model.repository,
                        "source": "huggingface",
                        "revision": "main",
                        "files": files,
                    }
                ),
                encoding="utf-8",
            )

            self.assertTrue(manager.model_status("tts", model.key)[0])
            (model_directory / "flow.pt").write_bytes(b"changed")
            valid, message = manager.model_status("tts", model.key)

            self.assertFalse(valid)
            self.assertIn("failed integrity check", message)

    def test_cosyvoice_uses_native_streaming_and_reference_voice(self) -> None:
        import numpy as np

        class Tensor:
            def __init__(self, values: list[float]) -> None:
                self.values = values

            def detach(self) -> Tensor:
                return self

            def float(self) -> Tensor:
                return self

            def cpu(self) -> Tensor:
                return self

            def numpy(self) -> object:
                return np.asarray([self.values], dtype=np.float32)

        with tempfile.TemporaryDirectory() as directory:
            prompt = Path(directory) / "reference.wav"
            prompt.write_bytes(b"RIFF-test")
            runtime = Mock()
            runtime.sample_rate = 24_000
            runtime.model = SimpleNamespace(
                llm=SimpleNamespace(inference_bistream=Mock()),
                token_hop_len=100,
            )
            runtime.inference_zero_shot.return_value = iter(
                (
                    {"tts_speech": Tensor([0.1, 0.2])},
                    {"tts_speech": Tensor([0.3, 0.4])},
                )
            )
            manager = CosyVoiceTtsProvider(directory)
            manager._load = Mock(return_value=runtime)
            manager._initial_token_hop_len = 25

            chunks = list(
                manager.synthesize_stream(
                    "fun_cosyvoice3_0_5b_2512",
                    "Hello world",
                    str(prompt),
                    "This is my voice",
                )
            )

            self.assertEqual(
                [chunk[1] for chunk in chunks],
                [24_000, 24_000, 24_000],
            )
            self.assertEqual(
                [chunk[2] for chunk in chunks],
                ["", "", "Hello world"],
            )
            self.assertEqual(runtime.model.token_hop_len, 25)
            args = runtime.inference_zero_shot.call_args.args
            self.assertEqual(list(args[0]), ["Hello world"])
            self.assertIn("<|endofprompt|>This is my voice", args[1])
            self.assertEqual(Path(args[2]), prompt.resolve())
            self.assertTrue(runtime.inference_zero_shot.call_args.kwargs["stream"])
            runtime.add_zero_shot_spk.assert_called_once()

    def test_cosyvoice_uses_multilingual_bistream_for_english_with_chinese_prompt(
        self,
    ) -> None:
        import numpy as np

        tensor = Mock()
        tensor.detach.return_value.float.return_value.cpu.return_value.numpy.return_value = (
            np.asarray([[0.1, 0.2]], dtype=np.float32)
        )
        with tempfile.TemporaryDirectory() as directory:
            prompt = Path(directory) / "reference.wav"
            prompt.write_bytes(b"RIFF-test")
            runtime = Mock()
            runtime.sample_rate = 24_000
            runtime.model.llm.inference_bistream = Mock()
            runtime.inference_zero_shot.return_value = iter(
                ({"tts_speech": tensor},)
            )
            manager = CosyVoiceTtsProvider(directory)
            manager._load = Mock(return_value=runtime)

            list(
                manager.synthesize_stream(
                    "fun_cosyvoice3_0_5b_2512",
                    "This should be spoken in English.",
                    str(prompt),
                    "希望你以后能够做的比我还好呦。",
                )
            )

            runtime.inference_zero_shot.assert_called_once()
            streamed_text = runtime.inference_zero_shot.call_args.args[0]
            self.assertEqual(
                list(streamed_text),
                ["This should be spoken in English."],
            )
            prompt_text = runtime.inference_zero_shot.call_args.args[1]
            self.assertIn("<|endofprompt|>", prompt_text)
            runtime.inference_cross_lingual.assert_not_called()

    def test_cosyvoice_complete_text_fallback_can_use_cross_lingual_mode(
        self,
    ) -> None:
        import numpy as np

        tensor = Mock()
        tensor.detach.return_value.float.return_value.cpu.return_value.numpy.return_value = (
            np.asarray([[0.1, 0.2]], dtype=np.float32)
        )
        with tempfile.TemporaryDirectory() as directory:
            prompt = Path(directory) / "reference.wav"
            prompt.write_bytes(b"RIFF-test")
            runtime = Mock()
            runtime.model = SimpleNamespace(llm=SimpleNamespace())
            runtime.sample_rate = 24_000
            runtime.inference_cross_lingual.return_value = iter(
                ({"tts_speech": tensor},)
            )
            manager = CosyVoiceTtsProvider(directory)
            manager._load = Mock(return_value=runtime)

            list(
                manager.synthesize_stream(
                    "fun_cosyvoice3_0_5b_2512",
                    "This should be spoken in English.",
                    str(prompt),
                    "希望你以后能够做的比我还好呦。",
                )
            )

            cross_text = runtime.inference_cross_lingual.call_args.args[0]
            self.assertEqual(
                cross_text,
                "You are a helpful assistant.<|endofprompt|>"
                "This should be spoken in English.",
            )

    def test_cosyvoice_bistream_chunks_preserve_text_at_natural_pauses(
        self,
    ) -> None:
        text = (
            "First sentence is long enough to start quickly. "
            "Second sentence continues naturally. "
            "最后一句使用中文，保持自然停顿。"
        )

        chunks = CosyVoiceTtsProvider.bistream_text_chunks(text, target_size=40)

        self.assertGreater(len(chunks), 1)
        self.assertEqual("".join(chunks), text)
        self.assertTrue(chunks[0].endswith("."))

    def test_cosyvoice_keeps_long_bistream_in_one_session_and_resets_hop_size(
        self,
    ) -> None:
        import numpy as np

        tensor = Mock()
        tensor.detach.return_value.float.return_value.cpu.return_value.numpy.return_value = (
            np.asarray([[0.1, 0.2]], dtype=np.float32)
        )
        with tempfile.TemporaryDirectory() as directory:
            prompt = Path(directory) / "reference.wav"
            prompt.write_bytes(b"RIFF-test")
            runtime = Mock()
            runtime.model = SimpleNamespace(
                llm=SimpleNamespace(inference_bistream=Mock()),
                token_hop_len=100,
            )
            runtime.sample_rate = 24_000
            hop_starts: list[int] = []

            def infer(*_args: object, **_kwargs: object) -> object:
                hop_starts.append(runtime.model.token_hop_len)
                runtime.model.token_hop_len = 100
                return iter(({"tts_speech": tensor},))

            runtime.inference_zero_shot.side_effect = infer
            manager = CosyVoiceTtsProvider(directory)
            manager._load = Mock(return_value=runtime)
            manager._initial_token_hop_len = 25
            text = "a" * 700

            chunks = list(
                manager.synthesize_stream(
                    "fun_cosyvoice3_0_5b_2512",
                    text,
                    str(prompt),
                    "English reference",
                )
            )

            self.assertEqual(hop_starts, [25])
            self.assertEqual(runtime.inference_zero_shot.call_count, 1)
            self.assertEqual(chunks[-1][2], text)

    def test_cosyvoice_falls_back_when_runtime_has_no_bistream_api(self) -> None:
        import numpy as np

        tensor = Mock()
        tensor.detach.return_value.float.return_value.cpu.return_value.numpy.return_value = (
            np.asarray([[0.1, 0.2]], dtype=np.float32)
        )
        with tempfile.TemporaryDirectory() as directory:
            prompt = Path(directory) / "reference.wav"
            prompt.write_bytes(b"RIFF-test")
            runtime = Mock()
            runtime.model = SimpleNamespace(llm=SimpleNamespace())
            runtime.sample_rate = 24_000
            runtime.inference_zero_shot.return_value = iter(
                ({"tts_speech": tensor},)
            )
            manager = CosyVoiceTtsProvider(directory)
            manager._load = Mock(return_value=runtime)

            list(
                manager.synthesize_stream(
                    "fun_cosyvoice3_0_5b_2512",
                    "Fallback text",
                    str(prompt),
                    "English reference",
                )
            )

            self.assertEqual(
                runtime.inference_zero_shot.call_args.args[0],
                "Fallback text",
            )

    def test_cosyvoice_treats_chinese_with_latin_product_names_as_chinese(
        self,
    ) -> None:
        text = "你好，Live GPT的CosyVoice 3流式语音已经可以使用。"

        self.assertEqual(CosyVoiceTtsProvider._text_script(text), "zh")

    def test_cosyvoice_filters_unavailable_onnx_provider(self) -> None:
        session = Mock(return_value="session")
        fake_onnxruntime = SimpleNamespace(
            InferenceSession=session,
            get_available_providers=Mock(
                return_value=["AzureExecutionProvider", "CPUExecutionProvider"]
            ),
        )

        with patch.dict(sys.modules, {"onnxruntime": fake_onnxruntime}):
            with _filter_onnx_providers() as providers:
                result = fake_onnxruntime.InferenceSession(
                    "model.onnx",
                    providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
                )

        self.assertEqual(result, "session")
        self.assertEqual(
            providers,
            ("AzureExecutionProvider", "CPUExecutionProvider"),
        )
        session.assert_called_once_with(
            "model.onnx", providers=["CPUExecutionProvider"]
        )
        self.assertIs(fake_onnxruntime.InferenceSession, session)

    def test_cosyvoice_reference_features_are_cached_once(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            prompt = Path(directory) / "reference.wav"
            prompt.write_bytes(b"RIFF-test")
            runtime = Mock()
            manager = CosyVoiceTtsProvider(directory)

            first = manager._ensure_reference_cached(runtime, prompt, "prompt")
            second = manager._ensure_reference_cached(runtime, prompt, "prompt")

            self.assertEqual(first, second)
            runtime.add_zero_shot_spk.assert_called_once()

    def test_cosyvoice_preload_is_idempotent(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manager = CosyVoiceTtsProvider(directory)
            runtime = Mock()

            def load(model_key: str) -> object:
                manager._loaded_key = model_key
                manager._loaded_model = runtime
                return runtime

            manager._load = Mock(side_effect=load)
            first = manager.preload("fun_cosyvoice3_0_5b_2512")
            second = manager.preload("fun_cosyvoice3_0_5b_2512")

            self.assertIn("preloaded in", first)
            self.assertIn("already preloaded", second)
            manager._load.assert_called_once_with(
                "fun_cosyvoice3_0_5b_2512"
            )

    def test_cosyvoice_cleanup_removes_read_only_git_objects(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            checkout = Path(directory) / "checkout"
            packed_object = checkout / ".git" / "objects" / "aa" / "object"
            packed_object.parent.mkdir(parents=True)
            packed_object.write_bytes(b"git object")
            packed_object.chmod(0o444)

            _remove_tree(checkout)

            self.assertFalse(checkout.exists())

    def test_cosyvoice_prompt_audio_uses_soundfile_without_torchcodec(self) -> None:
        import numpy as np

        samples = np.asarray([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32)
        tensor = Mock()
        averaged = Mock()
        tensor.mean.return_value = averaged
        fake_soundfile = SimpleNamespace(
            read=Mock(return_value=(samples, 24_000))
        )
        fake_torch = SimpleNamespace(from_numpy=Mock(return_value=tensor))

        with patch.dict(
            sys.modules,
            {"soundfile": fake_soundfile, "torch": fake_torch},
        ):
            result = _load_wav_with_soundfile("prompt.wav", 24_000)

        self.assertIs(result, averaged)
        fake_soundfile.read.assert_called_once_with(
            "prompt.wav", dtype="float32", always_2d=True
        )
        fake_torch.from_numpy.assert_called_once()
        tensor.mean.assert_called_once_with(dim=0, keepdim=True)

    def test_cosyvoice_blocks_wetext_only_during_model_startup(self) -> None:
        sentinel = object()
        with patch.dict(sys.modules, {"wetext": sentinel}):
            with _block_optional_import("wetext"):
                self.assertIsNone(sys.modules["wetext"])
            self.assertIs(sys.modules["wetext"], sentinel)

    def test_cosyvoice_runtime_import_probe_uses_installed_checkout(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manager = CosyVoiceTtsProvider(directory)
            matcha_root = manager.runtime_root / "third_party" / "Matcha-TTS"
            for module_name in COSYVOICE_RUNTIME_IMPORTS:
                base = (
                    matcha_root
                    if module_name.startswith("matcha.")
                    else manager.runtime_root
                )
                path = base / (module_name.replace(".", "/") + ".py")
                path.parent.mkdir(parents=True, exist_ok=True)
                package = path.parent
                while package != base:
                    (package / "__init__.py").touch()
                    package = package.parent
                path.write_text("", encoding="utf-8")

            valid, message = manager.runtime_import_status()

            self.assertTrue(valid, message)
            self.assertEqual(message, "CosyVoice runtime imports passed checks")

    def test_cosyvoice_installer_uses_python_312_compatible_audio_packages(
        self,
    ) -> None:
        with (
            tempfile.TemporaryDirectory() as directory,
            patch.object(
                CosyVoiceTtsProvider,
                "runtime_source_status",
                return_value=(True, "runtime verified"),
            ),
            patch.object(
                QwenTtsProvider,
                "nvidia_gpu_status",
                return_value=(True, "NVIDIA Test GPU"),
            ),
            patch(
                "live_gpt.voice.cosyvoice_tts.install_packages"
            ) as install,
            patch.object(
                CosyVoiceTtsProvider,
                "dependency_status",
                return_value=(True, "dependencies verified"),
            ),
        ):
            CosyVoiceTtsProvider(directory).install_dependencies()

        inference_packages = install.call_args_list[1].args[0]
        self.assertIn("openai-whisper==20250625", inference_packages)
        self.assertIn("gdown==5.1.0", inference_packages)
        self.assertIn("hydra-core==1.3.2", inference_packages)
        self.assertIn("lightning==2.2.4", inference_packages)
        self.assertIn("pytorch-lightning==2.2.4", inference_packages)
        self.assertIn("matplotlib==3.9.4", inference_packages)
        self.assertIn("pyarrow==18.1.0", inference_packages)
        self.assertIn("rich==15.0.0", inference_packages)
        self.assertIn("setuptools==80.9.0", inference_packages)
        self.assertIn("modelscope==1.39.1", inference_packages)
        self.assertIn("transformers==4.57.3", inference_packages)
        if sys.platform == "win32":
            self.assertIn(
                "pyworld-prebuilt==0.3.5.post2", inference_packages
            )
            self.assertNotIn("pyworld==0.3.4", inference_packages)


if __name__ == "__main__":
    unittest.main()
