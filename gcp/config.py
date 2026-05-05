import os

# Server
HOST = "0.0.0.0"
PORT = 8000

# LLM
# LLM_BACKEND: "huggingface" (Linux/CUDA, GCP VM) or "mlx" (Apple Silicon, local dev)
LLM_BACKEND = os.environ.get("LLM_BACKEND", "mlx")

# LLM_MODEL: "qwen" or "llama" (llama requires HuggingFace access approval)
_MODEL_CHOICE = os.environ.get("LLM_MODEL", "qwen")

_MODELS = {
    "qwen":  {
        "hf":  "Qwen/Qwen2.5-7B-Instruct",
        "mlx": "mlx-community/Qwen2.5-7B-Instruct-4bit",
    },
    "llama": {
        "hf":  "meta-llama/Llama-3.1-8B-Instruct",
        "mlx": "mlx-community/Meta-Llama-3.1-8B-Instruct-4bit",
    },
}

if _MODEL_CHOICE not in _MODELS:
    raise ValueError(f"Unknown LLM_MODEL: {_MODEL_CHOICE!r}. Use 'qwen' or 'llama'.")

MODEL_ID     = _MODELS[_MODEL_CHOICE]["hf"]
MLX_MODEL_ID = _MODELS[_MODEL_CHOICE]["mlx"]

LOAD_IN_4BIT   = True
MAX_NEW_TOKENS = 120   # ~2-3 spoken sentences; keeps LLM + TTS + playback fast
TEMPERATURE    = 0.7

# Web search
SEARCH_MAX_RESULTS = 3

# Base system prompt — user memory and schedule context are appended at request time
SYSTEM_PROMPT_BASE = (
    "You are a helpful personal voice assistant. "
    "Reply in 2-3 sentences maximum — short enough to speak aloud in under 10 seconds. "
    "Never use markdown, bullet points, numbers, or lists. Speak naturally and directly. "
    "Use any provided user facts or schedule context to give personalized answers. "
    "If you use a web search result, incorporate it naturally into your answer."
)
