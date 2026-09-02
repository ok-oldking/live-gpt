"""Small standalone GPT-SoVITS streaming server copied into an installation.

This file intentionally only imports GPT-SoVITS/runtime dependencies so it can
be launched by the installation's embedded Python rather than Live GPT's Python.
"""
from __future__ import annotations

import argparse
import json
import os
import struct
import sys
import threading

import numpy as np
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
import uvicorn


ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(ROOT)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "GPT_SoVITS"))

from GPT_SoVITS.TTS_infer_pack.TTS import TTS, TTS_Config  # noqa: E402


parser = argparse.ArgumentParser()
parser.add_argument("--host", default="127.0.0.1")
parser.add_argument("--port", type=int, required=True)
parser.add_argument(
    "--config", default="GPT_SoVITS/configs/tts_infer.yaml"
)
args = parser.parse_args()

# Construction loads and retains all model weights. TTS itself retains the
# semantic prompt and reference spectrogram when the same reference is reused.
pipeline = TTS(TTS_Config(args.config))
pipeline_lock = threading.Lock()
app = FastAPI()


@app.get("/health")
def health():
    return {"ready": True, "version": pipeline.configs.version}


def _frames(payload: dict):
    # GPU inference is serialized because the shared pipeline and its prompt
    # cache are mutable. The lock remains held for the lifetime of the stream.
    with pipeline_lock:
        for sample_rate, samples in pipeline.run(payload):
            pcm = np.asarray(samples, dtype="<i2").reshape(-1).tobytes()
            yield struct.pack("!II", int(sample_rate), len(pcm)) + pcm


@app.post("/tts")
async def tts(request: Request):
    try:
        payload = await request.json()
        text = str(payload.get("text", "")).strip()
        text_lang = str(payload.get("text_lang", "")).lower()
        ref_audio = str(payload.get("ref_audio_path", "")).strip()
        prompt_text = str(payload.get("prompt_text", "")).strip()
        prompt_lang = str(payload.get("prompt_lang", "")).lower()
        if not text:
            raise ValueError("text is required")
        if text_lang not in pipeline.configs.languages:
            raise ValueError(f"unsupported text_lang: {text_lang}")
        if not ref_audio or not os.path.isfile(ref_audio):
            raise ValueError("ref_audio_path must be an existing audio file")
        if prompt_text and prompt_lang not in pipeline.configs.languages:
            raise ValueError(f"unsupported prompt_lang: {prompt_lang}")
        payload.update(
            text=text,
            text_lang=text_lang,
            ref_audio_path=os.path.abspath(ref_audio),
            prompt_text=prompt_text,
            prompt_lang=prompt_lang if prompt_text else text_lang,
            streaming_mode=True,
            return_fragment=False,
            parallel_infer=False,
        )
        return StreamingResponse(
            _frames(payload), media_type="application/octet-stream"
        )
    except Exception as error:
        return JSONResponse(status_code=400, content={"message": str(error)})


if __name__ == "__main__":
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
