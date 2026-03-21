"""
asr.py — Automatic Speech Recognition for the Jetson assistant.

Design
------
ASRBase defines the interface: a single `transcribe(pcm_bytes) -> str` method.
Concrete implementations live below.  To swap backends later, subclass ASRBase
and update config.py — nothing else in the codebase needs to change.

Implemented backends
--------------------
WhisperASR   — faster-whisper (recommended; GPU-accelerated on Jetson Nano)
"""
import abc
import logging
import io
import wave
from typing import Optional

import numpy as np

import config

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Abstract interface
# ---------------------------------------------------------------------------

class ASRBase(abc.ABC):
    """All ASR backends must implement this interface."""

    @abc.abstractmethod
    def transcribe(self, pcm_bytes: bytes) -> str:
        """
        Convert raw PCM audio to text.

        Parameters
        ----------
        pcm_bytes : bytes
            Raw int16 mono PCM at config.SAMPLE_RATE Hz.
            Empty bytes → return empty string.

        Returns
        -------
        str
            Transcribed text, stripped of leading/trailing whitespace.
            Returns empty string on failure or if nothing was heard.
        """

    def warm_up(self) -> None:
        """
        Optional: run a silent inference pass so the first real call isn't slow.
        Subclasses can override; default is a no-op.
        """


# ---------------------------------------------------------------------------
# faster-whisper backend
# ---------------------------------------------------------------------------

class WhisperASR(ASRBase):
    """
    ASR backend using faster-whisper (CTranslate2-based Whisper).

    Model is downloaded on first use to config.MODEL_DIR and cached there.
    Subsequent starts load from cache — no internet required on the Nano.

    Memory footprint (approximate):
        tiny   ~200 MB RAM / ~39 MB VRAM
        base   ~300 MB RAM / ~74 MB VRAM
        small  ~600 MB RAM / ~244 MB VRAM
    """

    def __init__(
        self,
        model_size:   str = config.WHISPER_MODEL_SIZE,
        device:       str = config.WHISPER_DEVICE,
        compute_type: str = config.WHISPER_COMPUTE_TYPE,
        language:     Optional[str] = config.WHISPER_LANGUAGE,
        download_root: str = config.MODEL_DIR,
    ):
        self.language = language

        logger.info(
            "Loading faster-whisper model '%s' on %s (%s) …",
            model_size, device, compute_type,
        )

        # Lazy import so the rest of the codebase doesn't break if
        # faster-whisper is not installed on a development machine.
        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:
            raise ImportError(
                "faster-whisper is not installed. "
                "Run: pip install faster-whisper"
            ) from exc

        self._model = WhisperModel(
            model_size,
            device=device,
            compute_type=compute_type,
            download_root=download_root,
        )
        logger.info("faster-whisper model ready.")

    # ------------------------------------------------------------------

    def transcribe(self, pcm_bytes: bytes) -> str:
        if not pcm_bytes:
            return ""

        # Convert int16 PCM → float32 in [-1, 1] (what Whisper expects)
        audio_np = _pcm_bytes_to_float32(pcm_bytes)

        if audio_np is None or len(audio_np) == 0:
            return ""

        segments, info = self._model.transcribe(
            audio_np,
            language=self.language,
            beam_size=5,
            vad_filter=True,          # built-in silero VAD — filters silence
            vad_parameters=dict(
                min_silence_duration_ms=500,
            ),
        )

        # `segments` is a generator; consume it
        parts = [seg.text for seg in segments]
        text  = " ".join(parts).strip()

        logger.info(
            "ASR [lang=%s, prob=%.2f]: %r",
            info.language,
            info.language_probability,
            text,
        )
        return text

    def warm_up(self) -> None:
        """Transcribe a silent buffer to JIT-compile CUDA kernels."""
        logger.info("Warming up ASR model …")
        silence = np.zeros(config.SAMPLE_RATE, dtype=np.float32)  # 1 s
        self._model.transcribe(silence, language=self.language)
        logger.info("ASR warm-up complete.")


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _pcm_bytes_to_float32(pcm_bytes: bytes) -> Optional[np.ndarray]:
    """Convert raw int16 PCM bytes to a float32 numpy array in [-1, 1]."""
    try:
        samples = np.frombuffer(pcm_bytes, dtype=np.int16)
        return samples.astype(np.float32) / 32768.0
    except Exception as exc:
        logger.error("PCM conversion failed: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Factory — instantiate whichever backend is configured
# ---------------------------------------------------------------------------

def create_asr() -> ASRBase:
    """Return the configured ASR backend (currently always WhisperASR)."""
    return WhisperASR()
