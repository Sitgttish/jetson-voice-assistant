from abc import ABC, abstractmethod
from typing import Optional, Tuple
import logging
import base64
import requests

logger = logging.getLogger(__name__)


class LLMClientBase(ABC):
    @abstractmethod
    def chat(self, message: str) -> Tuple[Optional[str], Optional[bytes], Optional[dict]]:
        """Return (response_text, wav_bytes, latency_ms_dict). wav_bytes is None if not available."""
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Return True if this backend is reachable."""
        pass


class CloudLLMClient(LLMClientBase):
    def __init__(self, base_url: str, timeout: int = 30):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def is_available(self) -> bool:
        try:
            r = requests.get(f"{self.base_url}/health", timeout=3)
            return r.status_code == 200
        except Exception:
            return False

    def chat(self, message: str) -> Tuple[Optional[str], Optional[bytes], Optional[dict]]:
        try:
            r = requests.post(
                f"{self.base_url}/chat",
                json={"message": message},
                timeout=self.timeout,
            )
            r.raise_for_status()
            data = r.json()
            wav = base64.b64decode(data["audio_b64"]) if data.get("audio_b64") else None
            return data["response"], wav, data.get("latency_ms")
        except Exception as e:
            logger.error(f"CloudLLMClient error: {e}")
            return None, None, None


# Placeholder — will be implemented when local LLM is deployed on Jetson
class LocalLLMClient(LLMClientBase):
    def is_available(self) -> bool:
        return False

    def chat(self, message: str) -> Tuple[Optional[str], Optional[bytes], Optional[dict]]:
        raise NotImplementedError("Local LLM not yet deployed")


def create_llm_client() -> LLMClientBase:
    import config
    cloud = CloudLLMClient(base_url=config.GCP_SERVER_URL, timeout=config.LLM_REQUEST_TIMEOUT)
    if cloud.is_available():
        logger.info("Using cloud LLM.")
        return cloud

    logger.warning("Cloud LLM unavailable, trying local fallback.")
    local = LocalLLMClient()
    if local.is_available():
        logger.info("Using local LLM.")
        return local

    # Return cloud client anyway — it will log errors per request
    logger.warning("No LLM backend available. Will retry on each request.")
    return cloud
