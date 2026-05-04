"""
audio_io.py — microphone recording and speaker playback via ALSA.

Uses arecord/aplay subprocesses with the ALSA device string from config.
This avoids PyAudio's device-index scanning, which is unreliable on Jetson
when the card index doesn't match the PortAudio enumeration order.

arecord/aplay are part of alsa-utils and always respect plughw: strings
including the plug layer that handles sample-rate conversion.
"""
import audioop
import io
import logging
import subprocess
import wave
from typing import List, Optional

import config

logger = logging.getLogger(__name__)

# Bytes per chunk (int16 mono): CHUNK_FRAMES * 2 bytes/sample
CHUNK_FRAMES = 512
CHUNK_BYTES  = CHUNK_FRAMES * 2  # int16 = 2 bytes per sample


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _arecord_cmd(duration_s: Optional[float] = None) -> List[str]:
    """Build the arecord command for the configured device."""
    cmd = [
        "arecord",
        "-D", config.ALSA_MIC_DEVICE,
        "-f", "S16_LE",
        "-r", str(config.SAMPLE_RATE),
        "-c", str(config.CHANNELS),
        "-t", "raw",          # raw PCM on stdout, no WAV header
        "--quiet",
    ]
    if duration_s is not None:
        cmd += ["--duration", str(int(duration_s))]
    return cmd


def _aplay_cmd() -> List[str]:
    """Build the aplay command for the configured device."""
    return [
        "aplay",
        "-D", config.ALSA_SPEAKER_DEVICE,
        "-f", "S16_LE",
        "-r", str(config.SAMPLE_RATE),
        "-c", str(config.CHANNELS),
        "-t", "raw",
        "--quiet",
    ]


# ---------------------------------------------------------------------------
# Recorder
# ---------------------------------------------------------------------------

class AudioRecorder:
    """
    Records a single utterance from the microphone using arecord.

    Usage::

        recorder = AudioRecorder()
        pcm_bytes = recorder.record()   # blocks until end-of-speech

    Returns raw PCM bytes (int16, mono, config.SAMPLE_RATE Hz).
    """

    def record(self) -> bytes:
        """
        Block until an utterance is complete and return raw PCM bytes.

        Algorithm:
        1. Spawn arecord, read raw PCM chunks from its stdout.
        2. Wait for RMS > VAD_SILENCE_THRESHOLD (speech onset).
        3. Collect until VAD_SILENCE_CHUNKS consecutive silent chunks
           follow at least VAD_MIN_SPEECH_CHUNKS of speech.
        4. Hard-stop after MAX_RECORD_SECONDS.
        """
        cmd = _arecord_cmd()
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )

        frames: list[bytes] = []
        silent_chunks  = 0
        speech_chunks  = 0
        recording      = False
        max_chunks     = int(config.MAX_RECORD_SECONDS * config.SAMPLE_RATE
                             / CHUNK_FRAMES)

        logger.info("Listening… (waiting for speech)")

        try:
            for _ in range(max_chunks):
                chunk = proc.stdout.read(CHUNK_BYTES)
                if not chunk:
                    break

                rms = audioop.rms(chunk, 2)

                if rms > config.VAD_SILENCE_THRESHOLD:
                    if not recording:
                        logger.info("Speech detected — recording.")
                    recording     = True
                    speech_chunks += 1
                    silent_chunks  = 0
                else:
                    silent_chunks += 1

                if recording:
                    frames.append(chunk)
                    if (
                        speech_chunks >= config.VAD_MIN_SPEECH_CHUNKS
                        and silent_chunks >= config.VAD_SILENCE_CHUNKS
                    ):
                        logger.info("End of utterance detected.")
                        break
        finally:
            proc.terminate()
            proc.wait()

        if not frames:
            logger.warning("No speech captured.")
            return b""

        raw = b"".join(frames)
        logger.debug(
            "Recorded %d bytes (%.2f s)",
            len(raw),
            len(raw) / (config.SAMPLE_RATE * 2),
        )
        return raw


# ---------------------------------------------------------------------------
# Player
# ---------------------------------------------------------------------------

class AudioPlayer:
    """
    Plays raw PCM or WAV bytes through the speaker using aplay.

    Usage::

        player = AudioPlayer()
        player.play_pcm(raw_bytes)   # int16 mono at config.SAMPLE_RATE
        player.play_wav(wav_bytes)   # WAV file bytes
    """

    def play_pcm(self, pcm_bytes: bytes) -> None:
        """Play raw int16 mono PCM at config.SAMPLE_RATE."""
        if not pcm_bytes:
            return
        proc = subprocess.Popen(
            _aplay_cmd(),
            stdin=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
        proc.communicate(input=pcm_bytes)

    def play_wav(self, wav_bytes: bytes) -> None:
        """
        Play WAV-format audio. Decodes the WAV header and plays raw PCM
        so we stay within the aplay raw pipeline.
        """
        if not wav_bytes:
            return
        buf = io.BytesIO(wav_bytes)
        with wave.open(buf, "rb") as wf:
            pcm = wf.readframes(wf.getnframes())
            sr  = wf.getframerate()
            ch  = wf.getnchannels()
            sw  = wf.getsampwidth()

        cmd = [
            "aplay",
            "-D", config.ALSA_SPEAKER_DEVICE,
            "-f", f"S{sw * 8}_LE",
            "-r", str(sr),
            "-c", str(ch),
            "-t", "raw",
            "--quiet",
        ]
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE,
                                stderr=subprocess.DEVNULL)
        proc.communicate(input=pcm)


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def save_pcm_as_wav(pcm_bytes: bytes, path: str,
                    sample_rate: int = config.SAMPLE_RATE) -> None:
    """Save raw int16 mono PCM to a WAV file (useful for debugging)."""
    with wave.open(path, "wb") as wf:
        wf.setnchannels(config.CHANNELS)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_bytes)
    logger.debug("Saved WAV to %s", path)
