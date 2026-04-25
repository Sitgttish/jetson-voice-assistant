import io
import logging
import wave
import subprocess
import tempfile
import os
from typing import Optional

logger = logging.getLogger(__name__)

PIPER_VOICE = "en_US-lessac-medium"


def _pcm_to_wav(pcm_bytes: bytes, sample_rate: int = 22050) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_bytes)
    return buf.getvalue()


def synthesize(text: str) -> Optional[bytes]:
    """Synthesize text to WAV bytes using piper-tts (native on x86_64)."""
    if not text.strip():
        return None
    try:
        from piper import PiperVoice
        import numpy as np

        voice_path = _ensure_voice()
        if voice_path is None:
            return _espeak_fallback(text)

        voice = PiperVoice.load(voice_path)
        pcm_chunks = []
        for audio_bytes in voice.synthesize_stream_raw(text):
            pcm_chunks.append(audio_bytes)
        pcm = b"".join(pcm_chunks)
        return _pcm_to_wav(pcm, sample_rate=voice.config.sample_rate)

    except ImportError:
        logger.warning("piper-tts not installed, falling back to espeak.")
        return _espeak_fallback(text)
    except Exception as e:
        logger.error(f"TTS error: {e}")
        return _espeak_fallback(text)


def _ensure_voice() -> Optional[str]:
    """Download the Piper voice model if not already cached."""
    cache_dir = os.path.expanduser("~/.cache/piper-voices")
    onnx_path = os.path.join(cache_dir, f"{PIPER_VOICE}.onnx")
    if os.path.exists(onnx_path):
        return onnx_path
    try:
        from huggingface_hub import hf_hub_download
        os.makedirs(cache_dir, exist_ok=True)
        repo = "rhasspy/piper-voices"
        lang, name, quality = PIPER_VOICE.split("-", 2) if "-" in PIPER_VOICE else (None, None, None)
        prefix = f"en/en_US/lessac/medium"
        hf_hub_download(repo_id=repo, filename=f"{prefix}/{PIPER_VOICE}.onnx",
                        local_dir=cache_dir, local_dir_use_symlinks=False)
        hf_hub_download(repo_id=repo, filename=f"{prefix}/{PIPER_VOICE}.onnx.json",
                        local_dir=cache_dir, local_dir_use_symlinks=False)
        downloaded = os.path.join(cache_dir, prefix, f"{PIPER_VOICE}.onnx")
        return downloaded if os.path.exists(downloaded) else None
    except Exception as e:
        logger.warning(f"Could not download Piper voice: {e}")
        return None


def _espeak_fallback(text: str) -> Optional[bytes]:
    """Fallback TTS using espeak-ng."""
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
