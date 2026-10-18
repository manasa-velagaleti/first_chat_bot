"""Speech in and speech out, via OpenAI's audio endpoints.

Both directions use OpenAI regardless of which model is answering. They are
separate services from the chat model, so a conversation running on Gemini or
Groq still gets voice - it only needs OPENAI_API_KEY, which the app already
requires for embeddings.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "embeddings_app"))

import core                      # noqa: E402  (loads .env)

TRANSCRIBE_MODEL = "whisper-1"
SPEAK_MODEL = "gpt-4o-mini-tts"
VOICES = ["alloy", "echo", "fable", "onyx", "nova", "shimmer"]

# Spoken answers are cached by their text, so replaying one - or re-rendering
# the page, which Streamlit does constantly - does not pay to synthesise it
# again. Bounded, because the audio is held in memory.
_SPOKEN: dict[str, bytes] = {}
_MAX_CACHED = 24


def _client():
    from openai import OpenAI
    return OpenAI()


def transcribe(audio_bytes: bytes, filename: str = "speech.wav") -> dict:
    """Audio in, text out. Returns {"text": ...} or {"error": ...}."""
    if not audio_bytes:
        return {"error": "No audio was recorded."}
    try:
        import io
        buf = io.BytesIO(audio_bytes)
        buf.name = filename              # the SDK infers format from the name
        result = _client().audio.transcriptions.create(
            model=TRANSCRIBE_MODEL,
            file=buf,
        )
        text = (result.text or "").strip()
        if not text:
            return {"error": "Nothing was heard in that recording."}
        return {"text": text}
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}


def speak(text: str, voice: str = "alloy") -> dict:
    """Text in, mp3 bytes out. Returns {"audio": ...} or {"error": ...}."""
    text = (text or "").strip()
    if not text:
        return {"error": "Nothing to say."}

    # Long answers are slow and expensive to synthesise, and nobody listens to
    # ten minutes of a model reading a 10-K. Speak the opening and say so.
    truncated = False
    if len(text) > 4000:
        text, truncated = text[:4000].rsplit(" ", 1)[0], True

    key = hashlib.sha256(f"{voice}:{text}".encode()).hexdigest()
    if key in _SPOKEN:
        return {"audio": _SPOKEN[key], "truncated": truncated, "cached": True}

    try:
        resp = _client().audio.speech.create(
            model=SPEAK_MODEL, voice=voice, input=text,
        )
        audio = resp.read() if hasattr(resp, "read") else resp.content
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}

    if len(_SPOKEN) >= _MAX_CACHED:
        _SPOKEN.pop(next(iter(_SPOKEN)))
    _SPOKEN[key] = audio
    return {"audio": audio, "truncated": truncated, "cached": False}


def available() -> bool:
    """Voice needs an OpenAI key even when another provider is answering."""
    import os
    return bool(os.getenv("OPENAI_API_KEY"))
