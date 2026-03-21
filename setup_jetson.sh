#!/usr/bin/env bash
# setup_jetson.sh — Install Jetson-side dependencies for the voice assistant.
# Run once after cloning the repo:
#
#   bash setup_jetson.sh
#
# Tested on JetPack 4.6.x (Ubuntu 18.04, Python 3.6 stock).
# Installs Python 3.8 via deadsnakes PPA and creates a venv at
# ~/voice-assistant-env.

set -euo pipefail

VENV="$HOME/voice-assistant-env"

echo "============================================================"
echo "  Jetson Voice Assistant — dependency setup"
echo "============================================================"

# ---------------------------------------------------------------------------
# 1. System packages
# ---------------------------------------------------------------------------
echo "[1/5] Installing system packages …"
sudo apt-get update -qq
sudo apt-get install -y \
    portaudio19-dev \
    libsndfile1 \
    ffmpeg \
    alsa-utils \
    espeak-ng \
    python3-pip \
    python3-dev \
    libasound2-dev

# ---------------------------------------------------------------------------
# 2. Python 3.8 (JetPack 4.x ships 3.6 which is too old for faster-whisper)
# ---------------------------------------------------------------------------
echo "[2/5] Installing Python 3.8 via deadsnakes PPA …"
sudo add-apt-repository ppa:deadsnakes/ppa -y
sudo apt-get update -qq
sudo apt-get install -y python3.8 python3.8-venv python3.8-dev

# ---------------------------------------------------------------------------
# 3. Virtual environment
# ---------------------------------------------------------------------------
echo "[3/5] Creating venv at $VENV …"
if [ -d "$VENV" ]; then
    echo "  venv already exists — skipping creation"
else
    python3.8 -m venv "$VENV"
fi

# shellcheck disable=SC1090
source "$VENV/bin/activate"
pip install --upgrade pip --quiet

# ---------------------------------------------------------------------------
# 4. Python packages
#
# Pin notes (discovered on JetPack 4.6 aarch64):
#   tokenizers 0.21+ requires Rust to build — pin to 0.13.3 (has aarch64 wheel)
#   huggingface_hub 0.26+ pulls in hf-xet which also requires Rust — pin <0.26
#   ctranslate2 from PyPI has no CUDA support on aarch64 — CPU/int8 only
# ---------------------------------------------------------------------------
echo "[4/5] Installing Python packages …"

# Pin Rust-free versions first so pip doesn't try to upgrade them
pip install --quiet \
    "tokenizers==0.13.3" \
    "huggingface_hub<0.26"

# Core packages
pip install --quiet \
    numpy \
    faster-whisper \
    piper-tts \
    requests

# ---------------------------------------------------------------------------
# 5. Verify
# ---------------------------------------------------------------------------
echo "[5/5] Verifying imports …"
python3 - <<'EOF'
import importlib, sys

checks = [
    ("numpy",          "numpy"),
    ("faster_whisper", "faster-whisper"),
    ("piper",          "piper-tts"),
    ("requests",       "requests"),
]
ok = True
for mod, pkg in checks:
    try:
        importlib.import_module(mod)
        print(f"  {pkg:<20} OK")
    except ImportError as e:
        print(f"  {pkg:<20} MISSING — {e}")
        ok = False

# espeak-ng via subprocess
import subprocess
try:
    subprocess.run(["espeak-ng", "--version"],
                   capture_output=True, check=True)
    print(f"  {'espeak-ng':<20} OK")
except Exception as e:
    print(f"  {'espeak-ng':<20} MISSING — {e}")
    ok = False

sys.exit(0 if ok else 1)
EOF

echo ""
echo "Setup complete."
echo ""
echo "Activate the venv:"
echo "  source ~/voice-assistant-env/bin/activate"
echo ""
echo "Download the Piper TTS voice model (run once):"
echo "  cd jetson/ && python3 -c \"from tts import PiperTTS; PiperTTS()\""
echo ""
echo "Test ASR:"
echo "  cd jetson/ && python3 test_asr.py --save-wav"
echo ""
echo "Test TTS:"
echo "  cd jetson/ && python3 test_tts.py"
