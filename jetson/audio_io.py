"""
audio_io.py — microphone recording and speaker playback via ALSA.

Uses PyAudio so we can pass the raw ALSA device string from config.
The AudioRecorder uses a simple energy-based VAD: it starts collecting
audio once speech is detected, and stops after a configurable run of
silent chunks — no extra dependencies needed.
"""
import audioop
import logging
import wave
import io
import time
from typing import Optional

import numpy as np
import pyaudio

import config

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _open_pyaudio_stream(pa: pyaudio.PyAudio, input: bool, output: bool):
    """Open a PyAudio stream bound to the ALSA device in config."""
    # PyAudio's 'input_device_index' expects an integer index, not a hw string.
    # We locate the device by scanning available devices for the one whose
    # hostApi name contains 'ALSA' and whose name contains the card/device
    # numbers from config.ALSA_DEVICE ("plughw:2,0" → card 2 device 0).
    target = config.ALSA_DEVICE  # e.g. "plughw:2,0"
    # Extract card index from string like "plughw:2,0"
    try:
        card_str = target.split(":")[1].split(",")[0]
        card_index = int(card_str)
    except (IndexError, ValueError):
        card_index = None

    device_index: Optional[int] = None
    for i in range(pa.get_device_count()):
        info = pa.get_device_info_by_index(i)
        # Match by card number embedded in the device name (ALSA names look
        # like "HyperX 7.1 Audio: USB Audio (hw:2,0)" on Jetson).
        name: str = info["name"]
        if card_index is not None and f"hw:{card_index}," in name:
            if input and info["maxInputChannels"] > 0:
                device_index = i
                break
            if output and info["maxOutputChannels"] > 0:
                device_index = i
                break

    if device_index is None:
        logger.warning(
            "Could not find PyAudio device matching %s; using default.", target
        )

    kwargs = dict(
        format=pyaudio.paInt16,
        channels=config.CHANNELS,
        rate=config.SAMPLE_RATE,
        frames_per_buffer=512,
    )
    if input:
        kwargs["input"] = True
        if device_index is not None:
            kwargs["input_device_index"] = device_index
    if output:
        kwargs["output"] = True
        if device_index is not None:
            kwargs["output_device_index"] = device_index

    return pa.open(**kwargs)


# ---------------------------------------------------------------------------
# Recorder
# ---------------------------------------------------------------------------

class AudioRecorder:
    """
    Records a single utterance from the microphone.

    Usage::

        recorder = AudioRecorder()
        audio_bytes = recorder.record()   # blocks until end-of-speech

    Returns raw PCM bytes (int16, mono, 16 kHz) suitable for passing
    directly to the ASR module.
    """

    CHUNK = 512  # frames per read; ~32 ms at 16 kHz

    def record(self) -> bytes:
        """
        Block until an utterance is complete and return raw PCM bytes.

        The algorithm:
        1. Wait for the RMS to exceed VAD_SILENCE_THRESHOLD (speech onset).
        2. Collect audio until VAD_SILENCE_CHUNKS consecutive silent chunks
           follow the speech (end of utterance).
        3. Hard-stop after MAX_RECORD_SECONDS regardless.
        """
        pa = pyaudio.PyAudio()
        stream = _open_pyaudio_stream(pa, input=True, output=False)

        frames: list[bytes] = []
        silent_chunks   = 0
        speech_chunks   = 0
        recording       = False
        max_chunks      = int(
            config.MAX_RECORD_SECONDS * config.SAMPLE_RATE / self.CHUNK
        )

        logger.info("Listening… (waiting for speech)")

        try:
            for _ in range(max_chunks):
                chunk = stream.read(self.CHUNK, exception_on_overflow=False)
                rms   = audioop.rms(chunk, 2)  # 2 bytes per int16 sample

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
            stream.stop_stream()
            stream.close()
            pa.terminate()

        if not frames:
            logger.warning("No speech captured.")
            return b""

        raw = b"".join(frames)
        logger.debug("Recorded %d bytes (%.2f s)", len(raw),
                     len(raw) / (config.SAMPLE_RATE * 2))
        return raw


# ---------------------------------------------------------------------------
# Player
# ---------------------------------------------------------------------------

class AudioPlayer:
    """
    Plays raw PCM or WAV bytes through the speaker.

    Usage::

        player = AudioPlayer()
        player.play_pcm(raw_bytes)   # int16 mono 16 kHz
        player.play_wav(wav_bytes)   # WAV file bytes
    """

    def play_pcm(self, pcm_bytes: bytes,
                 sample_rate: int = config.SAMPLE_RATE) -> None:
        """Play raw int16 mono PCM."""
        if not pcm_bytes:
            return
        pa = pyaudio.PyAudio()
        stream = _open_pyaudio_stream(pa, input=False, output=True)
        try:
            stream.write(pcm_bytes)
        finally:
            stream.stop_stream()
            stream.close()
            pa.terminate()

    def play_wav(self, wav_bytes: bytes) -> None:
        """Play WAV-format audio (as returned by most TTS engines)."""
        if not wav_bytes:
            return
        buf = io.BytesIO(wav_bytes)
        with wave.open(buf, "rb") as wf:
            pa = pyaudio.PyAudio()
            stream = pa.open(
                format=pa.get_format_from_width(wf.getsampwidth()),
                channels=wf.getnchannels(),
                rate=wf.getframerate(),
                output=True,
            )
            try:
                data = wf.readframes(1024)
                while data:
                    stream.write(data)
                    data = wf.readframes(1024)
            finally:
                stream.stop_stream()
                stream.close()
                pa.terminate()


# ---------------------------------------------------------------------------
# Utility: save PCM to WAV file (useful for debugging)
# ---------------------------------------------------------------------------

def save_pcm_as_wav(pcm_bytes: bytes, path: str,
                    sample_rate: int = config.SAMPLE_RATE) -> None:
    with wave.open(path, "wb") as wf:
        wf.setnchannels(config.CHANNELS)
        wf.setsampwidth(2)  # int16 = 2 bytes
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_bytes)
    logger.debug("Saved WAV to %s", path)
