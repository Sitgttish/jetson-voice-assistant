"""
Central configuration for the Jetson-side assistant.
Edit this file to tune hardware settings and model choices.
"""
import os

# ---------------------------------------------------------------------------
# Audio hardware
# ---------------------------------------------------------------------------
ALSA_DEVICE = "plughw:2,0"   # HyperX USB headset on Jetson Nano
SAMPLE_RATE  = 16000          # Hz — Whisper expects 16 kHz
CHANNELS     = 1              # mono
DTYPE        = "int16"        # PCM format for ALSA / pyaudio

# ---------------------------------------------------------------------------
# Voice-activity detection (energy-based, no extra deps)
# ---------------------------------------------------------------------------
# Recording stops after this many consecutive silent chunks
VAD_SILENCE_CHUNKS   = 20     # ~0.64 s at chunk=512, sr=16000
# RMS below this threshold is considered silence
VAD_SILENCE_THRESHOLD = 500   # tune up if mic is noisy, down if too sensitive
# Minimum speech chunks before VAD will trigger end-of-utterance
VAD_MIN_SPEECH_CHUNKS = 5
# Hard ceiling on recording length (seconds)
MAX_RECORD_SECONDS    = 15

# ---------------------------------------------------------------------------
# ASR — faster-whisper
# ---------------------------------------------------------------------------
# "tiny" ~39 MB VRAM; "base" ~74 MB; "small" ~244 MB
WHISPER_MODEL_SIZE   = "tiny"
# "cuda" to use Jetson GPU; "cpu" if you hit OOM issues
WHISPER_DEVICE       = "cuda"
# "float16" for GPU; "int8" is quantised and saves ~50% memory
WHISPER_COMPUTE_TYPE = "float16"
WHISPER_LANGUAGE     = "en"   # set None for auto-detect (slower)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
LOG_DIR     = os.path.join(BASE_DIR, "logs")
MODEL_DIR   = os.path.join(BASE_DIR, "models")  # faster-whisper caches here

os.makedirs(LOG_DIR,   exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
LOG_LEVEL = "INFO"
