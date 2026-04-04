import logging
import sys
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
            # 1. Record until silence
            logger.info("Waiting for speech...")
            pcm = recorder.record_until_silence()
            if not pcm:
                continue

            # 2. ASR: audio -> text
            transcript = asr.transcribe(pcm)
            if not transcript:
                logger.info("No speech detected, skipping.")
                continue
            logger.info(f"Heard: {transcript!r}")

            # 3. LLM: text -> text
            response = llm.chat(transcript)
            if not response:
                response = "Sorry, I couldn't reach the server. Please try again."
            logger.info(f"Response: {response!r}")

            # 4. TTS: text -> audio
            wav = tts.synthesize(response)
            if wav:
                player.play_wav(wav)

        except KeyboardInterrupt:
            logger.info("Shutting down.")
            break
        except Exception as e:
            logger.error(f"Pipeline error: {e}", exc_info=True)


if __name__ == "__main__":
    run()
