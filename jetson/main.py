import logging
import sys
import time
import config

logging.basicConfig(
    level=config.LOG_LEVEL,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(f"{config.LOG_DIR}/assistant.log"),
    ],
)
logger = logging.getLogger(__name__)

from audio_io import AudioRecorder, AudioPlayer
from asr import create_asr
from tts import create_tts
from llm_client import create_llm_client


def log_latency(record_ms, asr_ms, network_ms, server_latency, local_tts_ms, playback_ms):
    # Response latency = time from user stops talking to audio starts playing
    # TTS is now on the server side (inside network_ms); local_tts_ms is only used as fallback
    response_latency_ms = asr_ms + network_ms + local_tts_ms
    lines = [
        "--- Latency Breakdown ---",
        f"  Recording      : {record_ms:>7.0f} ms  (user speaking, not part of response latency)",
        f"  ASR            : {asr_ms:>7.0f} ms",
        f"  Network RT     : {network_ms:>7.0f} ms",
    ]
    if server_latency:
        if server_latency.get("search_ms"):
            lines.append(f"    └─ Search     : {server_latency['search_ms']:>7.0f} ms")
        lines.append(f"    └─ LLM        : {server_latency['llm_ms']:>7.0f} ms")
        lines.append(f"    └─ TTS (cloud): {server_latency['tts_ms']:>7.0f} ms")
    if local_tts_ms > 0:
        lines.append(f"  TTS (local fallback): {local_tts_ms:>7.0f} ms")
    lines += [
        f"  Playback       : {playback_ms:>7.0f} ms  (response length, not part of response latency)",
        f"  *** Response Latency : {response_latency_ms:>7.0f} ms ***",
        "-------------------------",
    ]
    logger.info("\n".join(lines))


def run():
    logger.info("Initializing assistant...")

    recorder = AudioRecorder()
    player = AudioPlayer()
    asr = create_asr()
    tts = create_tts()
    llm = create_llm_client()

    asr.warm_up()
    tts.warm_up()

    logger.info("Assistant ready. Listening...")
    player.play_wav(tts.synthesize("Hello! I'm ready. How can I help you?"))

    while True:
        try:
            # 1. Record
            logger.info("Waiting for speech...")
            t0 = time.perf_counter()
            pcm = recorder.record()
            record_ms = (time.perf_counter() - t0) * 1000
            if not pcm:
                continue

            # Mark the moment user finishes talking — response latency starts here
            t_user_done = time.perf_counter()

            # 2. ASR
            transcript = asr.transcribe(pcm)
            asr_ms = (time.perf_counter() - t_user_done) * 1000
            if not transcript:
                logger.info("No speech detected, skipping.")
                continue
            logger.info(f"Heard: {transcript!r} (ASR: {asr_ms:.0f}ms)")

            # 3. LLM + cloud TTS (network round trip)
            t0 = time.perf_counter()
            response, wav, server_latency = llm.chat(transcript)
            network_ms = (time.perf_counter() - t0) * 1000
            local_tts_ms = 0.0

            if not response:
                response = "Sorry, I couldn't reach the server. Please try again."
                server_latency = None

            logger.info(f"Response: {response[:80]!r} (network RT: {network_ms:.0f}ms)")

            # 4. Local TTS fallback — only if cloud didn't return audio
            if not wav:
                t0 = time.perf_counter()
                wav = tts.synthesize(response)
                local_tts_ms = (time.perf_counter() - t0) * 1000

            # Audio starts playing here — response latency ends
            t0 = time.perf_counter()
            if wav:
                player.play_wav(wav)
            playback_ms = (time.perf_counter() - t0) * 1000

            log_latency(record_ms, asr_ms, network_ms, server_latency, local_tts_ms, playback_ms)

        except KeyboardInterrupt:
            logger.info("Shutting down.")
            break
        except Exception as e:
            logger.error(f"Pipeline error: {e}", exc_info=True)


if __name__ == "__main__":
    run()
