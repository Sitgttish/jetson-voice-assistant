"""
test_asr.py — Interactive test for the audio → ASR pipeline.

Run on the Jetson Nano:
    python3 test_asr.py

The script will:
  1. Load the Whisper model (downloads on first run, ~39 MB for 'tiny').
  2. Warm up the model.
  3. Loop: record one utterance → print transcription → repeat.

Press Ctrl+C to exit.
Optional flags:
  --save-wav    also save each recording as /tmp/last_recording.wav
  --cpu         force CPU inference (slower but lower VRAM)
  --model tiny|base|small
"""
import argparse
import logging
import os
import sys
import time

# Allow running from the jetson/ directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
import audio_io
import asr as asr_module

# ---------------------------------------------------------------------------
# CLI args
# ---------------------------------------------------------------------------

parser = argparse.ArgumentParser(description="Test the ASR pipeline.")
parser.add_argument("--save-wav", action="store_true",
                    help="Save each recording to /tmp/last_recording.wav")
parser.add_argument("--cpu", action="store_true",
                    help="Use CPU instead of CUDA (override config)")
parser.add_argument("--model", default=None,
                    choices=["tiny", "base", "small"],
                    help="Whisper model size (overrides config)")
parser.add_argument("--no-warmup", action="store_true",
                    help="Skip model warm-up (first transcription will be slower)")
args = parser.parse_args()

# Apply CLI overrides to config before importing model
if args.cpu:
    config.WHISPER_DEVICE       = "cpu"
    config.WHISPER_COMPUTE_TYPE = "int8"
if args.model:
    config.WHISPER_MODEL_SIZE = args.model

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("test_asr")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 60)
    print("  Jetson Voice Assistant — ASR test")
    print(f"  ALSA device : {config.ALSA_DEVICE}")
    print(f"  Whisper     : {config.WHISPER_MODEL_SIZE} / "
          f"{config.WHISPER_DEVICE} / {config.WHISPER_COMPUTE_TYPE}")
    print("=" * 60)

    # Build components
    recorder = audio_io.AudioRecorder()
    player   = audio_io.AudioPlayer()
    asr      = asr_module.create_asr()

    if not args.no_warmup:
        asr.warm_up()

    print("\nReady. Speak into the headset. Press Ctrl+C to quit.\n")

    loop_count = 0
    while True:
        loop_count += 1
        print(f"[{loop_count}] Waiting for speech …")

        t0 = time.perf_counter()
        pcm = recorder.record()
        record_time = time.perf_counter() - t0

        if not pcm:
            print("  (nothing captured — try speaking louder or adjusting "
                  "VAD_SILENCE_THRESHOLD in config.py)\n")
            continue

        print(f"  Captured {len(pcm)//(config.SAMPLE_RATE*2):.1f}s of audio "
              f"({len(pcm):,} bytes) in {record_time:.2f}s")

        if args.save_wav:
            wav_path = "/tmp/last_recording.wav"
            audio_io.save_pcm_as_wav(pcm, wav_path)
            print(f"  Saved to {wav_path}")

        t1 = time.perf_counter()
        text = asr.transcribe(pcm)
        asr_time = time.perf_counter() - t1

        print(f"  Transcribed in {asr_time:.2f}s")
        if text:
            print(f"  TEXT: \"{text}\"\n")
        else:
            print("  (empty transcription — spoke too quietly?)\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nBye.")
