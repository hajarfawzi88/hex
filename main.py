import os
import io
import wave
import json
import time
import base64
from typing import Literal

import httpx
import requests
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Response, WebSocket, WebSocketDisconnect, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

load_dotenv()

HAMSA_API_KEY = os.getenv("HAMSA_API_KEY")
if not HAMSA_API_KEY:
    raise RuntimeError("Missing HAMSA_API_KEY in environment.")

HAMSA_TTS_URL = "https://api.tryhamsa.com/v1/realtime/tts"
HAMSA_STT_URL = "https://api.tryhamsa.com/v1/realtime/stt"

app = FastAPI(title="Hamsa Voice API (STT + TTS)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten in prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

Dialect = Literal["pls", "egy", "syr", "irq", "jor", "leb", "ksa", "bah", "uae", "qat", "msa"]


# -------------------------
# Models
# -------------------------
class TTSRequest(BaseModel):
    text: str = Field(..., min_length=1)
    speaker: str = Field(..., description="Hamsa voice name, e.g., 'Majd'")
    dialect: Dialect = Field("msa")
    mulaw: bool = Field(False)
    return_base64: bool = Field(False)


# -------------------------
# Helpers
# -------------------------
def _hamsa_headers() -> dict:
    return {
        "Authorization": f"Token {HAMSA_API_KEY}",
        "Content-Type": "application/json",
    }


async def _call_hamsa_tts(payload: dict) -> tuple[bytes, str]:
    async with httpx.AsyncClient(timeout=60) as client:
        r = await client.post(HAMSA_TTS_URL, headers=_hamsa_headers(), json=payload)

    if r.status_code != 200:
        raise HTTPException(status_code=502, detail={"hamsa_status": r.status_code, "hamsa_body": r.text})

    ctype = (r.headers.get("Content-Type") or "").lower()

    if "application/json" in ctype:
        j = r.json()
        b64 = j.get("audioBase64")
        if not b64:
            raise HTTPException(status_code=502, detail={"error": "No audioBase64 in Hamsa response", "body": j})
        return base64.b64decode(b64), "audio/wav"

    # raw bytes
    return r.content, "audio/wav"


def pcm16_to_wav_bytes(pcm_bytes: bytes, sample_rate: int = 16000) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_bytes)
    buf.seek(0)
    return buf.read()


def approx_duration_sec(num_pcm_bytes: int, sample_rate: int = 16000) -> float:
    # mono int16: 2 bytes/sample
    return num_pcm_bytes / (sample_rate * 2)


# -------------------------
# Routes
# -------------------------
@app.get("/")
def health():
    return {"ok": True}


@app.post("/tts")
async def tts(req: TTSRequest):
    payload = {"text": req.text, "speaker": req.speaker, "dialect": req.dialect, "mulaw": req.mulaw}
    audio_bytes, media_type = await _call_hamsa_tts(payload)

    if req.return_base64:
        return {"audioBase64": base64.b64encode(audio_bytes).decode("utf-8")}

    return Response(content=audio_bytes, media_type=media_type)


@app.post("/stt/file")
async def stt_file(file: UploadFile = File(...)):
    content = await file.read()
    b64 = base64.b64encode(content).decode("utf-8")
    payload = {"audioBase64": b64, "language": "ar", "isEosEnabled": False, "eosThreshold": 0.3}

    async with httpx.AsyncClient(timeout=60) as client:
        r = await client.post(HAMSA_STT_URL, headers=_hamsa_headers(), json=payload)

    if r.status_code != 200:
        raise HTTPException(status_code=502, detail=r.text)

    data = r.json()
    text = ((data.get("data") or {}).get("text") or "").strip()
    return {"text": text, "raw": data}


# -------------------------
# WebSocket: NON-streaming STT (base64 WAV)
# -------------------------
@app.websocket("/ws/stt")
async def ws_stt(ws: WebSocket):
    await ws.accept()
    try:
        while True:
            msg = await ws.receive_text()
            data = json.loads(msg)

            if data.get("event") == "end":
                break

            r = requests.post(
                HAMSA_STT_URL,
                headers={"Authorization": f"Token {HAMSA_API_KEY}", "Content-Type": "application/json"},
                json={"audioBase64": data["audioBase64"], "language": data.get("language", "ar"), "isEosEnabled": False},
                timeout=30,
            )

            if r.status_code == 200:
                text = (r.json().get("data", {}) or {}).get("text", "") or ""
                await ws.send_json({"text": text})
            else:
                await ws.send_json({"error": r.text})
    finally:
        await ws.close()


# -------------------------
# WebSocket: NON-streaming TTS (JSON -> audioBase64)
# -------------------------
@app.websocket("/ws/tts")
async def ws_tts(ws: WebSocket):
    await ws.accept()
    try:
        while True:
            msg = await ws.receive_text()
            data = json.loads(msg)

            if data.get("event") == "end":
                break

            r = requests.post(
                HAMSA_TTS_URL,
                headers={"Authorization": f"Token {HAMSA_API_KEY}", "Content-Type": "application/json"},
                json={
                    "text": data["text"],
                    "speaker": data.get("speaker", "Ahmed"),
                    "dialect": data.get("dialect", "msa"),
                    "mulaw": False,
                },
                timeout=30,
            )

            # Hamsa might return raw bytes OR JSON. Handle both.
            ctype = (r.headers.get("Content-Type") or "").lower()

            if r.status_code != 200:
                await ws.send_json({"error": r.text})
                continue

            if "application/json" in ctype:
                j = r.json()
                audio_b64 = j.get("audioBase64")
                if audio_b64:
                    await ws.send_json({"audioBase64": audio_b64})
                else:
                    await ws.send_json({"error": "No audioBase64 in response", "raw": j})
            else:
                # raw audio -> base64 for WS
                await ws.send_json({"audioBase64": base64.b64encode(r.content).decode("utf-8")})
    finally:
        await ws.close()


# -------------------------
# WebSocket: STREAMING STT (binary PCM + VAD client) + partials best-effort
# -------------------------
PARTIAL_EVERY_SEC = 0.9
MIN_PARTIAL_SEC = 0.8

@app.websocket("/ws/stt_stream")
async def ws_stt_stream(ws: WebSocket):
    await ws.accept()

    audio_buffer = bytearray()
    language = "ar"
    sample_rate = 16000

    last_partial_at = 0.0
    last_partial_len = 0
    last_partial_text = ""

    partial_inflight = False

    async def send_partial_if_needed(client: httpx.AsyncClient):
        nonlocal last_partial_at, last_partial_len, last_partial_text, partial_inflight

        now = time.monotonic()
        if now - last_partial_at < PARTIAL_EVERY_SEC:
            return
        if approx_duration_sec(len(audio_buffer), sample_rate) < MIN_PARTIAL_SEC:
            return
        if len(audio_buffer) <= last_partial_len:
            return
        if partial_inflight:
            return

        partial_inflight = True
        try:
            last_partial_at = now
            last_partial_len = len(audio_buffer)

            wav_bytes = pcm16_to_wav_bytes(bytes(audio_buffer), sample_rate)
            wav_b64 = base64.b64encode(wav_bytes).decode("utf-8")

            r = await client.post(
                HAMSA_STT_URL,
                headers=_hamsa_headers(),
                json={"audioBase64": wav_b64, "language": language, "isEosEnabled": False},
            )
            if r.status_code != 200:
                return

            text = ((r.json().get("data") or {}).get("text") or "").strip()
            if text and text != last_partial_text:
                last_partial_text = text
                await ws.send_json({"event": "partial", "text": text})
        finally:
            partial_inflight = False

    async with httpx.AsyncClient(timeout=60) as client:
        try:
            while True:
                message = await ws.receive()

                # control JSON
                if "text" in message:
                    data = json.loads(message["text"])
                    ev = data.get("event")

                    if ev == "start":
                        language = data.get("language", "ar")
                        sample_rate = int(data.get("sample_rate", 16000))
                        audio_buffer.clear()
                        last_partial_at = 0.0
                        last_partial_len = 0
                        last_partial_text = ""
                        await ws.send_json({"event": "ready"})

                    elif ev == "end":
                        if not audio_buffer:
                            await ws.send_json({"event": "final", "text": ""})
                            continue

                        wav_bytes = pcm16_to_wav_bytes(bytes(audio_buffer), sample_rate)
                        wav_b64 = base64.b64encode(wav_bytes).decode("utf-8")

                        r = await client.post(
                            HAMSA_STT_URL,
                            headers=_hamsa_headers(),
                            json={"audioBase64": wav_b64, "language": language, "isEosEnabled": False},
                        )

                        text = ""
                        if r.status_code == 200:
                            text = ((r.json().get("data") or {}).get("text") or "").strip()

                        await ws.send_json({"event": "final", "text": text})

                        # reset buffer for next utterance
                        audio_buffer.clear()
                        last_partial_at = 0.0
                        last_partial_len = 0
                        last_partial_text = ""

                    elif ev == "close":
                        break

                # binary audio
                elif "bytes" in message:
                    audio_buffer.extend(message["bytes"])
                    await send_partial_if_needed(client)

        except WebSocketDisconnect:
            pass
        finally:
            await ws.close()

