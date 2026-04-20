import os

# Server
HOST = "0.0.0.0"
PORT = 8000

# LLM
LLM_BACKEND = "cloud"  # "cloud" = HuggingFace transformers on this VM
MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"
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
