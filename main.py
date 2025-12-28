import os
import base64
from typing import Literal, Optional

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Response, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

load_dotenv()

HAMSA_API_KEY = os.getenv("HAMSA_API_KEY")
if not HAMSA_API_KEY:
    raise RuntimeError("Missing HAMSA_API_KEY in environment.")

# Hamsa realtime endpoints
HAMSA_TTS_URL = "https://api.tryhamsa.com/v1/realtime/tts"
HAMSA_STT_URL = "https://api.tryhamsa.com/v1/realtime/stt"

app = FastAPI(title="Hamsa Voice API (STT + TTS)")

# If you call this API from a browser frontend, CORS helps.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten this in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

Dialect = Literal[
    "pls", "egy", "syr", "irq", "jor", "leb", "ksa", "bah", "uae", "qat", "msa"
]

# -------------------------
# Models
# -------------------------
class TTSRequest(BaseModel):
    text: str = Field(..., min_length=1)
    speaker: str = Field(..., description="Hamsa voice name, e.g., 'Majd'")
    dialect: Dialect = Field("msa")
    mulaw: bool = Field(False)
    return_base64: bool = Field(False, description="If True, return JSON audioBase64")

class STTRequest(BaseModel):
    audioBase64: str = Field(..., min_length=1, description="Base64 of a WAV file")
    language: str = Field("ar")
    isEosEnabled: bool = Field(False)
    eosThreshold: float = Field(0.3)

class STTResponse(BaseModel):
    text: str
    raw: dict

# -------------------------
# Helpers
# -------------------------
def _hamsa_headers() -> dict:
    return {
        "Authorization": f"Token {HAMSA_API_KEY}",
        "Content-Type": "application/json",
    }

async def _call_hamsa_tts(payload: dict) -> tuple[bytes, str]:
    """
    Returns (audio_bytes, inferred_media_type)
    Hamsa may return raw audio bytes OR JSON {"audioBase64": "..."}.
    """
    async with httpx.AsyncClient(timeout=60) as client:
        r = await client.post(HAMSA_TTS_URL, headers=_hamsa_headers(), json=payload)

    if r.status_code != 200:
        raise HTTPException(
            status_code=502,
            detail={"hamsa_status": r.status_code, "hamsa_body": r.text},
        )

    ctype = (r.headers.get("Content-Type") or "").lower()

    if "application/json" in ctype:
        j = r.json()
        b64 = j.get("audioBase64")
        if not b64:
            raise HTTPException(status_code=502, detail={"error": "No audioBase64 in Hamsa response", "body": j})
        return base64.b64decode(b64), "audio/wav"

    # raw bytes
    return r.content, "audio/wav"

# -------------------------
# Routes
# -------------------------
@app.get("/health")
def health():
    return {"ok": True}

@app.post("/tts")
async def tts(req: TTSRequest):
    """
    POST /tts
    - returns audio/wav by default
    - if return_base64=True -> {"audioBase64": "..."}
    """
    payload = {
        "text": req.text,
        "speaker": req.speaker,
        "dialect": req.dialect,
        "mulaw": req.mulaw,
    }

    audio_bytes, media_type = await _call_hamsa_tts(payload)

    if req.return_base64:
        return {"audioBase64": base64.b64encode(audio_bytes).decode("utf-8")}

    return Response(content=audio_bytes, media_type=media_type)

from fastapi import UploadFile, File

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
# WebSocket: TTS
# -------------------------
@app.websocket("/ws/tts")
async def ws_tts(ws: WebSocket):
    """
    WebSocket protocol:

    Client sends JSON:
      {
        "text": "...",
        "speaker": "Majd",
        "dialect": "egy",
        "mulaw": false,
        "mode": "bytes" | "base64"   (optional, default "bytes")
      }

    Server responds:
      - if mode="bytes": sends audio bytes as a binary frame
      - if mode="base64": sends {"audioBase64": "..."} as JSON
    """
    await ws.accept()
    try:
        while True:
            msg = await ws.receive_json()

            text = (msg.get("text") or "").strip()
            speaker = (msg.get("speaker") or "").strip()
            dialect = (msg.get("dialect") or "msa").strip()
            mulaw = bool(msg.get("mulaw") or False)
            mode = (msg.get("mode") or "bytes").strip().lower()

            if not text or not speaker:
                await ws.send_json({"error": "text and speaker are required"})
                continue

            payload = {"text": text, "speaker": speaker, "dialect": dialect, "mulaw": mulaw}
            audio_bytes, _ = await _call_hamsa_tts(payload)

            if mode == "base64":
                await ws.send_json({"audioBase64": base64.b64encode(audio_bytes).decode("utf-8")})
            else:
                await ws.send_bytes(audio_bytes)

    except WebSocketDisconnect:
        return
