import logging
import sys
import os

# Allow imports from gcp/ directory
sys.path.insert(0, os.path.dirname(__file__))

import config
from llm import create_llm
from search import build_prompt_with_search

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


@app.on_event("startup")
async def startup():
    global llm
    llm = create_llm()


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    if not req.message.strip():
        raise HTTPException(status_code=400, detail="Empty message")

    prompt = build_prompt_with_search(req.message, max_results=config.SEARCH_MAX_RESULTS)
    response = llm.generate(prompt, system_prompt=config.SYSTEM_PROMPT)
    logger.info(f"User: {req.message!r} -> Response: {response[:80]!r}...")
    return ChatResponse(response=response)


@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": llm is not None}


if __name__ == "__main__":
    uvicorn.run(app, host=config.HOST, port=config.PORT)
