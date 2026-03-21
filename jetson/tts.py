"""
tts.py — Text-to-Speech for the Jetson assistant.

Design
------
TTSBase defines the interface: synthesize(text) -> bytes (WAV).
AudioPlayer in audio_io.py consumes those bytes via play_wav().

Implemented backends
--------------------
PiperBinaryTTS — pre-compiled piper binary (neural, best quality, run download_piper.sh)
PiperTTS       — piper-tts Python package (NOT usable on aarch64 — no wheel)
EspeakTTS      — espeak-ng subprocess (robotic but zero-download fallback)
"""
import abc
import io
import json
import logging
import os
import struct
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
# Piper via Docker (works on JetPack 4.x / Ubuntu 18.04 / glibc 2.27)
# ---------------------------------------------------------------------------

class PiperDockerTTS(TTSBase):
    """
    Neural TTS using Piper inside a Docker container.

    Solves the glibc 2.29 incompatibility of the pre-compiled Piper binary
    on JetPack 4.x (Ubuntu 18.04, glibc 2.27).  The rhasspy/piper image
    runs on linux/arm64 and bundles everything needed.

    One-time setup:
        docker pull rhasspy/piper:latest

    Usage in config.py:
        TTS_BACKEND = "piper-docker"

    Latency: ~1.5–2.5 s per sentence (synthesis + docker run overhead).
    """

    IMAGE = "piper-local:latest"

    def __init__(self, voice: Optional[str] = None):
        voice = voice or config.PIPER_VOICE
        self._voice      = voice
        self._model_path = os.path.join(config.MODEL_DIR, f"{voice}.onnx")

        if not os.path.isfile(self._model_path):
            raise FileNotFoundError(
                f"Piper voice model not found at {self._model_path}. "
                "Run: bash download_piper.sh"
            )

        # Read sample rate from JSON config
        json_path = self._model_path + ".json"
        try:
            with open(json_path) as f:
                self._sample_rate = json.load(f)["audio"]["sample_rate"]
        except Exception:
            self._sample_rate = config.TTS_SAMPLE_RATE

        # Verify Docker is available
        try:
            subprocess.run(
                ["docker", "info"],
                capture_output=True, check=True,
            )
        except (FileNotFoundError, subprocess.CalledProcessError) as exc:
            raise RuntimeError(
                "Docker is not running or not installed. "
                "Install Docker and run: docker pull rhasspy/piper:latest"
            ) from exc

        logger.info(
            "PiperDockerTTS ready (voice=%s, sample_rate=%d).",
            voice, self._sample_rate,
        )

    def synthesize(self, text: str) -> bytes:
        if not text.strip():
            return b""

        result = subprocess.run(
            [
                "docker", "run", "--rm", "-i",
                "-v", f"{config.MODEL_DIR}:/models",
                self.IMAGE,
                "--model", f"/models/{os.path.basename(self._model_path)}",
                "--output_raw",
                "--quiet",
            ],
            input=text.encode("utf-8"),
            capture_output=True,
            check=True,
        )

        wav_bytes = _pcm_to_wav(result.stdout, self._sample_rate)
        logger.debug(
            "PiperDockerTTS synthesized %d bytes for %d chars",
            len(wav_bytes), len(text),
        )
        return wav_bytes

    def warm_up(self) -> None:
        logger.info("Warming up PiperDockerTTS (first run pulls image layers into cache) …")
        self.synthesize("Hello.")
        logger.info("PiperDockerTTS warm-up complete.")


# ---------------------------------------------------------------------------
# Piper binary TTS backend (requires glibc 2.29 — not usable on JetPack 4.x)
# ---------------------------------------------------------------------------

class PiperBinaryTTS(TTSBase):
    """
    Neural TTS using the pre-compiled piper binary (aarch64).

    The binary bundles its own ONNX runtime and phonemizer, so no Python
    packages are needed.  Run download_piper.sh once to fetch the binary
    and voice model.

    Usage in config.py:
        TTS_BACKEND  = "piper-binary"
        PIPER_BINARY = "/absolute/path/to/piper-bin/piper"
        PIPER_VOICE  = "en_US-lessac-medium"

    Latency: ~0.5–1 s on Jetson Nano CPU for a short sentence.
    """

    def __init__(
        self,
        binary_path: Optional[str] = None,
        voice: Optional[str] = None,
    ):
        self._binary = binary_path or config.PIPER_BINARY
        voice        = voice       or config.PIPER_VOICE

        if not os.path.isfile(self._binary):
            raise FileNotFoundError(
                f"Piper binary not found at {self._binary}. "
                "Run: bash download_piper.sh"
            )

        model_path  = os.path.join(config.MODEL_DIR, f"{voice}.onnx")
        config_path = os.path.join(config.MODEL_DIR, f"{voice}.onnx.json")

        if not os.path.isfile(model_path):
            raise FileNotFoundError(
                f"Piper voice model not found at {model_path}. "
                "Run: bash download_piper.sh"
            )

        self._model_path = model_path

        # Read sample rate from the voice config JSON
        try:
            with open(config_path) as f:
                info = json.load(f)
            self._sample_rate = info["audio"]["sample_rate"]
        except Exception:
            self._sample_rate = config.TTS_SAMPLE_RATE

        logger.info(
            "PiperBinaryTTS ready (voice=%s, sample_rate=%d).",
            voice, self._sample_rate,
        )

    def synthesize(self, text: str) -> bytes:
        if not text.strip():
            return b""

        result = subprocess.run(
            [
                self._binary,
                "--model",      self._model_path,
                "--output_raw",
                "--quiet",
            ],
            input=text.encode("utf-8"),
            capture_output=True,
            check=True,
        )

        raw_pcm = result.stdout
        wav_bytes = _pcm_to_wav(raw_pcm, self._sample_rate)
        logger.debug(
            "PiperBinaryTTS synthesized %d bytes for %d chars",
            len(wav_bytes), len(text),
        )
        return wav_bytes

    def warm_up(self) -> None:
        logger.info("Warming up Piper binary TTS …")
        self.synthesize("Hello.")
        logger.info("Piper binary TTS warm-up complete.")


def _pcm_to_wav(pcm_bytes: bytes, sample_rate: int,
                channels: int = 1, sampwidth: int = 2) -> bytes:
    """Wrap raw int16 PCM bytes in a WAV container."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(sampwidth)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_bytes)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Piper TTS backend (Python package — NOT usable on aarch64)
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

    def __init__(self, voice: Optional[str] = None, speed: Optional[int] = None):
        voice = voice or config.ESPEAK_VOICE
        speed = speed or config.ESPEAK_SPEED
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

def create_tts(backend: Optional[str] = None) -> TTSBase:
    """
    Return the configured TTS backend.

    Parameters
    ----------
    backend : str
        "espeak"  — EspeakTTS (default; system espeak-ng, works on aarch64)
        "piper"   — PiperTTS (neural quality; requires piper-phonemize
                    which has no aarch64 wheel yet — not usable on Jetson Nano)
    """
    backend = backend or config.TTS_BACKEND
    if backend == "piper-docker":
        return PiperDockerTTS()
    if backend == "piper-binary":
        return PiperBinaryTTS()
    if backend == "piper":
        return PiperTTS()
    return EspeakTTS()
