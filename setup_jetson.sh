#!/usr/bin/env bash
# setup_jetson.sh — Install Jetson-side dependencies for the voice assistant.
# Run once after cloning/copying the project to the Nano:
#
#   bash setup_jetson.sh
#
# Assumes:
#   - JetPack 4.6.x (Ubuntu 18.04) or 5.x (Ubuntu 20.04)
#   - Python 3.8+
#   - pip3 available
#   - CUDA toolkit already installed (comes with JetPack)

set -euo pipefail

echo "============================================================"
echo "  Jetson Voice Assistant — dependency setup"
echo "============================================================"

# ---------------------------------------------------------------------------
# System packages
# ---------------------------------------------------------------------------
echo "[1/4] Installing system packages …"
sudo apt-get update -qq
sudo apt-get install -y \
    portaudio19-dev \
    libsndfile1 \
    ffmpeg \
    python3-pip \
    python3-dev \
    libasound2-dev

# ---------------------------------------------------------------------------
# Python packages — core
# ---------------------------------------------------------------------------
echo "[2/4] Installing Python packages …"
pip3 install --upgrade pip

# PyAudio — binds to PortAudio / ALSA
pip3 install pyaudio

# numpy — audio array processing
pip3 install "numpy>=1.21,<2.0"

# ---------------------------------------------------------------------------
# faster-whisper + CTranslate2
# ---------------------------------------------------------------------------
# CTranslate2 wheels for Jetson (ARM64 + CUDA) are published by the
# ctranslate2 project. If a pre-built wheel isn't available for your
# JetPack version you may need to build from source — see:
#   https://github.com/OpenNMT/CTranslate2
#
# For JetPack 5.x (Python 3.8, CUDA 11.4):
echo "[3/4] Installing faster-whisper …"
pip3 install faster-whisper

# If the above fails with no matching distribution, try pinning an older
# version that ships an ARM64 wheel:
#   pip3 install faster-whisper==0.10.1

# ---------------------------------------------------------------------------
# Optional: huggingface_hub for model download progress bars
# ---------------------------------------------------------------------------
pip3 install huggingface_hub

# ---------------------------------------------------------------------------
# Verify
# ---------------------------------------------------------------------------
echo "[4/4] Verifying imports …"
python3 - <<'EOF'
import pyaudio; print("  pyaudio        OK")
import numpy;   print("  numpy          OK")
try:
    import faster_whisper; print("  faster_whisper OK")
except ImportError as e:
    print(f"  faster_whisper MISSING — {e}")
    print("  You may need to build CTranslate2 from source for your JetPack.")
EOF

echo ""
echo "Setup complete."
echo ""
echo "Quick test:"
echo "  cd jetson/"
echo "  python3 test_asr.py --save-wav"
echo ""
echo "If the GPU runs out of memory, add --cpu to fall back to int8 CPU:"
echo "  python3 test_asr.py --cpu"
