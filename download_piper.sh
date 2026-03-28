#!/usr/bin/env bash
# download_piper.sh — Download the Piper TTS binary (aarch64) and voice model.
#
# Run once from the repo root:
#   bash download_piper.sh
#
# What this does:
#   1. Downloads the pre-compiled piper binary for Linux aarch64
#   2. Extracts it to ./piper-bin/
#   3. Downloads the en_US-lessac-medium voice model to jetson/models/
#
# The piper binary bundles its own ONNX runtime and phonemizer, so no
# Python packages are needed beyond what's already installed.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
BIN_DIR="$REPO_ROOT/piper-bin"
MODEL_DIR="$REPO_ROOT/jetson/models"

PIPER_VERSION="2023.11.14-2"
PIPER_URL="https://github.com/rhasspy/piper/releases/download/${PIPER_VERSION}/piper_linux_aarch64.tar.gz"

VOICE="en_US-lessac-medium"
HF_BASE="https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/lessac/medium"

echo "============================================================"
echo "  Piper TTS binary + voice download"
echo "============================================================"
echo "  Binary dir : $BIN_DIR"
echo "  Model dir  : $MODEL_DIR"
echo ""

mkdir -p "$BIN_DIR" "$MODEL_DIR"

# ---------------------------------------------------------------------------
# 1. Piper binary
# ---------------------------------------------------------------------------
PIPER_BIN="$BIN_DIR/piper"

if [ -f "$PIPER_BIN" ]; then
    echo "[1/2] Piper binary already present — skipping download."
else
    echo "[1/2] Downloading Piper binary (${PIPER_VERSION}) …"
    TMP_TAR="$(mktemp /tmp/piper_XXXXXX.tar.gz)"
    wget -q --show-progress -O "$TMP_TAR" "$PIPER_URL"
    echo "  Extracting …"
    # The tarball extracts to a 'piper/' subdirectory
    tar -xf "$TMP_TAR" -C "$BIN_DIR" --strip-components=1
    rm "$TMP_TAR"
    chmod +x "$PIPER_BIN"
    echo "  Binary ready at $PIPER_BIN"
fi

# Quick sanity check
if ! "$PIPER_BIN" --help &>/dev/null; then
    echo "ERROR: piper binary failed to run. Check architecture or missing libs."
    exit 1
fi

# ---------------------------------------------------------------------------
# 2. Voice model
# ---------------------------------------------------------------------------
MODEL_ONNX="$MODEL_DIR/${VOICE}.onnx"
MODEL_JSON="$MODEL_DIR/${VOICE}.onnx.json"

if [ -f "$MODEL_ONNX" ] && [ -f "$MODEL_JSON" ]; then
    echo "[2/2] Voice model already present — skipping download."
else
    echo "[2/2] Downloading voice model '${VOICE}' (~63 MB) …"
    wget -q --show-progress -O "$MODEL_ONNX" "${HF_BASE}/${VOICE}.onnx"
    wget -q --show-progress -O "$MODEL_JSON" "${HF_BASE}/${VOICE}.onnx.json"
    echo "  Model ready at $MODEL_ONNX"
fi

# ---------------------------------------------------------------------------
# 3. Smoke test
# ---------------------------------------------------------------------------
echo ""
echo "Running smoke test …"
echo "This is a test of the Piper text to speech system." \
    | "$PIPER_BIN" \
        --model "$MODEL_ONNX" \
        --output_raw \
        --quiet \
    | aplay -D plughw:2,0 -r 22050 -f S16_LE -c 1 -t raw --quiet

echo ""
echo "Done. If you heard speech, Piper is working."
echo ""
echo "Update config.py if needed:"
echo "  TTS_BACKEND  = \"piper-binary\""
echo "  PIPER_BINARY = \"$(realpath "$PIPER_BIN")\""
echo "  PIPER_VOICE  = \"${VOICE}\""
