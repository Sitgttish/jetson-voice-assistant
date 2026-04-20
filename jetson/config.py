"""
Central configuration for the Jetson-side assistant.
Edit this file to tune hardware settings and model choices.
"""
import os

# ---------------------------------------------------------------------------
# Paths (defined first so other sections can reference them)
# ---------------------------------------------------------------------------
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
LOG_DIR     = os.path.join(BASE_DIR, "logs")
MODEL_DIR   = os.path.join(BASE_DIR, "models")

os.makedirs(LOG_DIR,   exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Audio hardware
# ---------------------------------------------------------------------------
ALSA_DEVICE = "plughw:2,0"   # HyperX USB headset on Jetson Nano
SAMPLE_RATE  = 16000          # Hz — Whisper expects 16 kHz
CHANNELS     = 1              # mono
DTYPE        = "int16"        # PCM format for ALSA

# ---------------------------------------------------------------------------
# Voice-activity detection (energy-based, no extra deps)
# ---------------------------------------------------------------------------
VAD_SILENCE_CHUNKS    = 20    # consecutive silent chunks before end-of-utterance
VAD_SILENCE_THRESHOLD = 500   # RMS below this = silence (raise if room is noisy)
VAD_MIN_SPEECH_CHUNKS = 5     # minimum speech chunks before VAD can stop
MAX_RECORD_SECONDS    = 15    # hard ceiling on recording length

# ---------------------------------------------------------------------------
# ASR — faster-whisper
# ---------------------------------------------------------------------------
WHISPER_MODEL_SIZE   = "tiny"
# "cpu" works out of the box; "cuda" requires CUDA-enabled ctranslate2 (not on PyPI aarch64)
WHISPER_DEVICE       = "cpu"
WHISPER_COMPUTE_TYPE = "int8"
WHISPER_LANGUAGE     = "en"   # set None for auto-detect (slower)

# ---------------------------------------------------------------------------
# TTS
# ---------------------------------------------------------------------------
# "espeak"        — espeak-ng subprocess (system package, always works)
# "piper-docker"  — Piper in Docker (neural quality, works on JetPack 4.x)
#                   requires: docker pull rhasspy/piper:latest
# "piper-binary"  — Piper pre-compiled binary (requires glibc 2.29, NOT on JetPack 4.x)
# "piper"         — piper-tts Python package (NOT usable on aarch64 — no wheel)
TTS_BACKEND      = "piper-docker"

# espeak-ng settings
ESPEAK_VOICE     = "en-us"
ESPEAK_SPEED     = 150        # words per minute

# Piper binary settings (only used when TTS_BACKEND = "piper-binary")
PIPER_BINARY     = os.path.join(os.path.dirname(BASE_DIR), "piper-bin", "piper")
PIPER_VOICE      = "en_US-lessac-medium"

# TTS output sample rate (Piper medium = 22050 Hz)
TTS_SAMPLE_RATE  = 22050

# ---------------------------------------------------------------------------
# LLM backend
# ---------------------------------------------------------------------------
# URL of the GCP server. Replace with your VM's external IP after provisioning.
GCP_SERVER_URL      = os.environ.get("GCP_SERVER_URL", "http://YOUR_GCP_IP:8000")
LLM_REQUEST_TIMEOUT = 30   # seconds to wait for a response

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
LOG_LEVEL = "INFO"
