import logging
import sys
import os
import time
import base64

# Allow imports from gcp/ directory
sys.path.insert(0, os.path.dirname(__file__))

import config
from llm import create_llm
from search import build_prompt_with_search, needs_search
import tts as tts_module

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(title="Voice Assistant Backend")
llm = None  # Loaded at startup


class ChatRequest(BaseModel):
    message: str


class ChatResponse(BaseModel):
    response: str
    audio_b64: str        # base64-encoded WAV bytes
    latency_ms: dict      # {"search_ms": float|None, "llm_ms": float, "tts_ms": float, "total_ms": float}


@app.on_event("startup")
async def startup():
    global llm
    llm = create_llm()


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    if not req.message.strip():
        raise HTTPException(status_code=400, detail="Empty message")

    t_total = time.perf_counter()

    # Search (optional)
    search_ms = None
    if needs_search(req.message):
        t_search = time.perf_counter()
        prompt = build_prompt_with_search(req.message, max_results=config.SEARCH_MAX_RESULTS)
        search_ms = (time.perf_counter() - t_search) * 1000
    else:
        prompt = req.message

    # LLM generation
    t_llm = time.perf_counter()
    response = llm.generate(prompt, system_prompt=config.SYSTEM_PROMPT)
    llm_ms = (time.perf_counter() - t_llm) * 1000

    # TTS synthesis
    t_tts = time.perf_counter()
    wav = tts_module.synthesize(response)
    tts_ms = (time.perf_counter() - t_tts) * 1000
    audio_b64 = base64.b64encode(wav).decode() if wav else ""

    total_ms = (time.perf_counter() - t_total) * 1000

    logger.info(
        f"Latency — search: {f'{search_ms:.0f}ms' if search_ms else 'N/A'}, "
        f"llm: {llm_ms:.0f}ms, tts: {tts_ms:.0f}ms, total: {total_ms:.0f}ms"
    )
    logger.info(f"User: {req.message!r} -> Response: {response[:80]!r}...")

    return ChatResponse(
        response=response,
        audio_b64=audio_b64,
        latency_ms={
            "search_ms": round(search_ms, 1) if search_ms is not None else None,
            "llm_ms": round(llm_ms, 1),
            "tts_ms": round(tts_ms, 1),
            "total_ms": round(total_ms, 1),
        },
    )


@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": llm is not None}


if __name__ == "__main__":
    uvicorn.run(app, host=config.HOST, port=config.PORT)
