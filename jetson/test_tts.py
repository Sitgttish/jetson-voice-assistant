"""
test_tts.py — Interactive test for the TTS → speaker pipeline.

Run on the Jetson Nano:
    python3 test_tts.py

The script will:
  1. Load the Piper TTS model (downloads ~60 MB on first run).
  2. Speak a test sentence through the headset.
  3. Loop: type a line → hear it spoken → repeat.

Press Ctrl+C or type 'quit' to exit.

Optional flags:
  --espeak        use espeak-ng instead of Piper (no download, robotic voice)
  --save-wav      also save each utterance to /tmp/last_tts.wav
  --text "hello"  speak a single sentence and exit (non-interactive)
"""
import argparse
import logging
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
import audio_io
import tts as tts_module

# ---------------------------------------------------------------------------
# CLI args
# ---------------------------------------------------------------------------

parser = argparse.ArgumentParser(description="Test the TTS pipeline.")
parser.add_argument("--espeak",   action="store_true",
                    help="Use espeak-ng instead of Piper")
parser.add_argument("--save-wav", action="store_true",
                    help="Save each utterance to /tmp/last_tts.wav")
parser.add_argument("--text",     default=None,
                    help="Speak a single sentence and exit")
args = parser.parse_args()

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("test_tts")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

DEMO_TEXT = (
    "Hello, I am your Jetson voice assistant. "
    "The text to speech system is working correctly."
)


def speak(tts: tts_module.TTSBase, player: audio_io.AudioPlayer,
          text: str, save: bool) -> None:
    t0 = time.perf_counter()
    wav = tts.synthesize(text)
    synth_time = time.perf_counter() - t0

    if not wav:
        print("  (empty audio — nothing synthesized)")
        return

    print(f"  Synthesized {len(wav):,} bytes in {synth_time:.2f}s")

    if save:
        with open("/tmp/last_tts.wav", "wb") as f:
            f.write(wav)
        print("  Saved to /tmp/last_tts.wav")

    player.play_wav(wav)


def main() -> None:
    backend = "espeak" if args.espeak else config.TTS_BACKEND

    print("=" * 60)
    print("  Jetson Voice Assistant — TTS test")
    print(f"  ALSA speaker: {config.ALSA_SPEAKER_DEVICE}")
    print(f"  TTS backend : {backend}")
    print("=" * 60)

    player = audio_io.AudioPlayer()
    tts    = tts_module.create_tts(backend)
    tts.warm_up()

    # Non-interactive single-shot mode
    if args.text:
        speak(tts, player, args.text, args.save_wav)
        return

    # Demo sentence first
    print(f"\nDemo: \"{DEMO_TEXT}\"\n")
    speak(tts, player, DEMO_TEXT, args.save_wav)

    # Interactive loop
    print("\nType something to hear it spoken. Type 'quit' to exit.\n")
    while True:
        try:
            text = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            break

        if not text or text.lower() == "quit":
            break

        speak(tts, player, text, args.save_wav)


if __name__ == "__main__":
    main()
