import os

# Server
HOST = "0.0.0.0"
PORT = 8000

# LLM
# "huggingface" — transformers + bitsandbytes 4-bit (use on Linux/CUDA, i.e. GCP VM)
# "mlx"         — Apple MLX (use on Apple Silicon Mac for local development)
LLM_BACKEND = os.environ.get("LLM_BACKEND", "mlx")
MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"
# MLX model: a pre-quantized MLX version from HuggingFace (faster download, no conversion needed)
MLX_MODEL_ID = os.environ.get("MLX_MODEL_ID", "mlx-community/Meta-Llama-3.1-8B-Instruct-4bit")
LOAD_IN_4BIT = True
MAX_NEW_TOKENS = 512
TEMPERATURE = 0.7

# Web search
SEARCH_MAX_RESULTS = 3

# System prompt
SYSTEM_PROMPT = (
    "You are a helpful voice assistant. "
    "Give concise, natural responses suitable for spoken audio — avoid markdown, bullet points, or lists. "
    "If you use a web search result, incorporate it naturally into your answer."
)
