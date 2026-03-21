"""
tts.py — Text-to-Speech for the Jetson assistant.

Design
------
TTSBase defines the interface: synthesize(text) -> bytes (WAV).
AudioPlayer in audio_io.py consumes those bytes via play_wav().

Implemented backends
--------------------
PiperTTS    — piper-tts (neural, ONNX, good quality, ~60 MB model)
EspeakTTS   — espeak-ng subprocess (robotic but zero-download fallback)
"""
import abc
import io
import logging
import os
import subprocess
import wave
from typing import Optional

import config

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Abstract interface
# ---------------------------------------------------------------------------

class TTSBase(abc.ABC):
    """All TTS backends must implement this interface."""

    @abc.abstractmethod
    def synthesize(self, text: str) -> bytes:
        """
        Convert text to audio.

        Parameters
        ----------
        text : str
            The text to speak. Empty string returns empty bytes.

        Returns
        -------
        bytes
            WAV-format audio bytes ready for AudioPlayer.play_wav().
            Returns empty bytes on failure or empty input.
        """

    def warm_up(self) -> None:
        """Optional: run a silent pass to JIT-compile on first use."""


# ---------------------------------------------------------------------------
# Piper TTS backend
# ---------------------------------------------------------------------------

class PiperTTS(TTSBase):
    """
    Neural TTS using piper-tts (ONNX, CPU, ~60 MB for medium voices).

    The voice model (.onnx + .onnx.json) is downloaded automatically from
    Hugging Face on first use and cached in config.MODEL_DIR.

    Quality levels: x_low < low < medium < high
    Memory: medium ≈ 60 MB RAM
    Speed:  medium ≈ 0.3–0.5 s latency on Jetson Nano CPU
    """

    _HF_REPO   = "rhasspy/piper-voices"
    _HF_BRANCH = "v1.0.0"

    def __init__(self, voice: Optional[str] = None):
        voice = voice or config.PIPER_VOICE

        try:
            from piper.voice import PiperVoice
        except ImportError as exc:
            raise ImportError(
                "piper-tts is not installed. Run: pip install piper-tts"
            ) from exc

        model_path, config_path = self._ensure_model(voice)

        logger.info("Loading Piper voice '%s' …", voice)
        self._voice = PiperVoice.load(model_path, config_path=config_path)
        logger.info("Piper TTS ready.")

    # ------------------------------------------------------------------

    def synthesize(self, text: str) -> bytes:
        if not text.strip():
            return b""

        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            self._voice.synthesize(text, wf)

        wav_bytes = buf.getvalue()
        logger.debug("Synthesized %d bytes for %d chars", len(wav_bytes), len(text))
        return wav_bytes

    def warm_up(self) -> None:
        logger.info("Warming up Piper TTS …")
        self.synthesize("Hello.")
        logger.info("Piper TTS warm-up complete.")

    # ------------------------------------------------------------------
    # Model download helpers
    # ------------------------------------------------------------------

    def _ensure_model(self, voice: str):
        """
        Return (model_path, config_path) for the given voice name,
        downloading from Hugging Face if not already cached.
        """
        model_path  = os.path.join(config.MODEL_DIR, f"{voice}.onnx")
        config_path = os.path.join(config.MODEL_DIR, f"{voice}.onnx.json")

        if os.path.exists(model_path) and os.path.exists(config_path):
            logger.debug("Piper model already cached at %s", model_path)
            return model_path, config_path

        logger.info(
            "Piper model '%s' not found — downloading from Hugging Face …",
            voice,
        )
        self._download_model(voice, model_path, config_path)
        return model_path, config_path

    def _download_model(self, voice: str, model_path: str, config_path: str) -> None:
        """Download .onnx and .onnx.json from the rhasspy/piper-voices HF repo."""
        try:
            from huggingface_hub import hf_hub_download
        except ImportError as exc:
            raise ImportError(
                "huggingface_hub is required for model download. "
                "Run: pip install 'huggingface_hub<0.26'"
            ) from exc

        # Voices are stored under <lang>/<lang_region>/<voice_name>/
        # e.g. en/en_US/lessac/medium/en_US-lessac-medium.onnx
        parts   = voice.split("-")          # ["en_US", "lessac", "medium"]
        lang    = parts[0].split("_")[0]    # "en"
        subpath = "/".join([lang, parts[0], parts[1], parts[2]])

        for filename, dest in [
            (f"{subpath}/{voice}.onnx",      model_path),
            (f"{subpath}/{voice}.onnx.json", config_path),
        ]:
            logger.info("  Downloading %s …", filename)
            downloaded = hf_hub_download(
                repo_id=self._HF_REPO,
                filename=filename,
                revision=self._HF_BRANCH,
                local_dir=config.MODEL_DIR,
                local_dir_use_symlinks=False,
            )
            # hf_hub_download saves to a nested path — copy to flat MODEL_DIR
            if os.path.abspath(downloaded) != os.path.abspath(dest):
                import shutil
                shutil.copy2(downloaded, dest)

        logger.info("Piper model download complete.")


# ---------------------------------------------------------------------------
# Espeak-ng TTS backend (fallback)
# ---------------------------------------------------------------------------

class EspeakTTS(TTSBase):
    """
    Lightweight TTS using espeak-ng via subprocess.

    No model download required — uses the system espeak-ng package.
    Voice is robotic but works instantly with zero extra dependencies.
    Install: sudo apt-get install espeak-ng

    Use this if Piper is unavailable or you want near-zero latency.
    """

    def __init__(self, voice: str = "en-us", speed: int = 150):
        self._voice = voice
        self._speed = speed
        # Verify espeak-ng is available
        try:
            subprocess.run(
                ["espeak-ng", "--version"],
                capture_output=True, check=True,
            )
        except (FileNotFoundError, subprocess.CalledProcessError) as exc:
            raise RuntimeError(
                "espeak-ng not found. Run: sudo apt-get install espeak-ng"
            ) from exc
        logger.info("EspeakTTS ready (voice=%s, speed=%d).", voice, speed)

    def synthesize(self, text: str) -> bytes:
        if not text.strip():
            return b""

        result = subprocess.run(
            [
                "espeak-ng",
                "-v", self._voice,
                "-s", str(self._speed),
                "--stdout",
                text,
            ],
            capture_output=True,
            check=True,
        )
        logger.debug("espeak-ng synthesized %d bytes", len(result.stdout))
        return result.stdout  # WAV bytes


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def create_tts(backend: str = "piper") -> TTSBase:
    """
    Return the configured TTS backend.

    Parameters
    ----------
    backend : str
        "piper"   — PiperTTS (neural, recommended)
        "espeak"  — EspeakTTS (robotic, instant fallback)
    """
    if backend == "espeak":
        return EspeakTTS()
    return PiperTTS()
