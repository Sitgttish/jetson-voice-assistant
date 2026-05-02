import io
import logging
import wave
import subprocess
import tempfile
import os
from typing import Optional

logger = logging.getLogger(__name__)

PIPER_VOICE = "en_US-lessac-medium"
_CACHE_DIR  = os.path.expanduser("~/.cache/piper-voices")
_VOICE_PATH = os.path.join(_CACHE_DIR, "en", "en_US", "lessac", "medium", f"{PIPER_VOICE}.onnx")

_voice = None  # cached PiperVoice instance


def _load_voice():
    global _voice
    if _voice is not None:
        return _voice
    try:
        from piper import PiperVoice
        voice_path = _ensure_voice()
        if voice_path:
            _voice = PiperVoice.load(voice_path)
            logger.info(f"Piper TTS loaded: {voice_path}")
    except Exception as e:
        logger.warning(f"Could not load Piper voice: {e}")
    return _voice


def _ensure_voice() -> Optional[str]:
    if os.path.exists(_VOICE_PATH):
        return _VOICE_PATH
    try:
        from huggingface_hub import hf_hub_download
        os.makedirs(_CACHE_DIR, exist_ok=True)
        prefix = "en/en_US/lessac/medium"
        for ext in [".onnx", ".onnx.json"]:
            hf_hub_download(
                repo_id="rhasspy/piper-voices",
                filename=f"{prefix}/{PIPER_VOICE}{ext}",
                local_dir=_CACHE_DIR,
            )
        return _VOICE_PATH if os.path.exists(_VOICE_PATH) else None
    except Exception as e:
        logger.warning(f"Could not download Piper voice: {e}")
        return None


def synthesize(text: str) -> Optional[bytes]:
    if not text.strip():
        return None
    voice = _load_voice()
    if voice:
        try:
            buf = io.BytesIO()
            with wave.open(buf, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(voice.config.sample_rate)
                voice.synthesize(text, wf)
            return buf.getvalue()
        except Exception as e:
            logger.error(f"Piper TTS error: {e}")

    return _espeak_fallback(text)


def _espeak_fallback(text: str) -> Optional[bytes]:
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            tmp = f.name
        subprocess.run(
            ["espeak-ng", "-v", "en-us", "-s", "150", "-w", tmp, text],
            check=True, capture_output=True,
        )
        with open(tmp, "rb") as f:
            return f.read()
    except Exception as e:
        logger.error(f"espeak fallback failed: {e}")
        return None
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)
