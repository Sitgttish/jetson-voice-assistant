import logging
from typing import Optional

logger = logging.getLogger(__name__)

# Keywords that suggest the query needs live data
_SEARCH_TRIGGERS = [
    "weather", "today", "tomorrow", "current", "latest", "news",
    "score", "price", "stock", "what time", "when is", "search",
    "look up", "find", "who is", "what is happening",
]


def needs_search(text: str) -> bool:
    lower = text.lower()
    return any(trigger in lower for trigger in _SEARCH_TRIGGERS)


def web_search(query: str, max_results: int = 3) -> Optional[str]:
    try:
        from duckduckgo_search import DDGS
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
        if not results:
            return None
        snippets = [f"{r['title']}: {r['body']}" for r in results]
        return "\n".join(snippets)
    except Exception as e:
        logger.warning(f"Search failed: {e}")
        return None


def build_prompt_with_search(user_message: str, max_results: int = 3) -> str:
    if not needs_search(user_message):
        return user_message

    logger.info(f"Running web search for: {user_message!r}")
    results = web_search(user_message, max_results=max_results)
    if not results:
        return user_message

    return (
        f"Web search results:\n{results}\n\n"
        f"Using the above information if relevant, answer: {user_message}"
    )
