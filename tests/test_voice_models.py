from __future__ import annotations

import hashlib
import io
import json
import sys
import tarfile
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from live_gpt.voice_models import (
    LocalDictationSession,
    STT_MODELS,
    SherpaVoiceManager,
    TTS_MODELS,
)


class VoiceModelManagerTests(unittest.TestCase):
    def test_catalog_has_requested_independent_stt_and_tts_models(self) -> None:
        self.assertEqual(len(STT_MODELS), 10)
        self.assertEqual(len(TTS_MODELS), 5)
        self.assertEqual(TTS_MODELS["kokoro_multilang_v1_0"].speakers, 53)
        self.assertEqual(TTS_MODELS["piper_libritts"].speakers, 904)
        for model in (*STT_MODELS.values(), *TTS_MODELS.values()):
            self.assertGreater(model.asset.size, 0)
            self.assertEqual(len(model.asset.sha256), 64)
            int(model.asset.sha256, 16)
            self.assertTrue(model.asset.url.startswith("https://github.com/"))

    def test_bundle_status_detects_valid_and_modified_model_files(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manager = SherpaVoiceManager(directory)
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
                SherpaVoiceManager._safe_extract(archive, root / "extract")

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
                    manager = SherpaVoiceManager(directory)
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
            manager = SherpaVoiceManager(directory)
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
