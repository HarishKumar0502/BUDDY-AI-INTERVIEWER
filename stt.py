"""
stt.py
──────
Speech-to-Text using Groq's Whisper API (whisper-large-v3-turbo).
Requires only the GROQ_API_KEY already in .env — no OpenAI key needed.

Fine-tuning improvements:
  • Uses whisper-large-v3-turbo (faster, equally accurate for English)
  • Sends a context prompt with interview vocabulary to boost technical term accuracy
  • Validates minimum audio size (avoids sending empty/noise clips)
  • Strips filler-only transcriptions (e.g. pure "[BLANK_AUDIO]")
"""

import io
import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Map MIME types to file extensions Groq/Whisper accepts
_MIME_TO_EXT = {
    "audio/webm":              "webm",
    "audio/webm;codecs=opus":  "webm",
    "audio/ogg":               "ogg",
    "audio/ogg;codecs=opus":   "ogg",
    "audio/mp4":               "mp4",
    "audio/mpeg":              "mp3",
    "audio/wav":               "wav",
    "audio/x-wav":             "wav",
}

# Minimum bytes to even attempt transcription (avoids empty recording errors)
_MIN_AUDIO_BYTES = 1500   # ~0.1 seconds of Opus audio

# A short interview-domain prompt helps Whisper bias towards technical words
_WHISPER_PROMPT = (
    "Interview answer. Technical discussion about software engineering, "
    "Python, JavaScript, machine learning, APIs, databases, system design, "
    "React, Docker, cloud, algorithms, data structures."
)

# Artifacts that indicate no real speech was captured
_BLANK_PATTERNS = {"[blank_audio]", "(blank audio)", "[inaudible]", "(inaudible)"}


def transcribe_audio_bytes(audio_bytes: bytes, content_type: str = "audio/webm") -> str:
    """
    Transcribe raw audio bytes using Groq Whisper.

    Parameters
    ----------
    audio_bytes  : raw bytes from the browser MediaRecorder blob
    content_type : MIME type reported by the browser (e.g. 'audio/webm')

    Returns
    -------
    Transcribed text string, or empty string on failure / no speech detected.
    """
    if not audio_bytes or len(audio_bytes) < _MIN_AUDIO_BYTES:
        print(f"[stt] Audio too short ({len(audio_bytes)} bytes) — skipping.")
        return ""

    # Strip codec suffix for lookup, keep the full string as fallback
    mime_base = content_type.split(";")[0].strip().lower()
    ext = _MIME_TO_EXT.get(content_type.lower().strip(),
          _MIME_TO_EXT.get(mime_base, "webm"))

    filename = f"audio.{ext}"

    try:
        transcription = _client.audio.transcriptions.create(
            file=(filename, io.BytesIO(audio_bytes)),
            model="whisper-large-v3-turbo",   # faster and equally accurate for English
            response_format="text",
            language="en",
            temperature=0.0,                  # deterministic — best for STT accuracy
            prompt=_WHISPER_PROMPT,           # domain context boosts technical terms
        )
        result = str(transcription).strip() if transcription else ""

        # Filter out blank / noise-only transcriptions
        if result.lower() in _BLANK_PATTERNS:
            print("[stt] Blank audio detected — ignoring.")
            return ""

        return result

    except Exception as e:
        print(f"[stt] Transcription error: {e}")
        return ""
