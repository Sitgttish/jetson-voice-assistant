"""
Tool execution — parses and applies [ACTION: {...}] commands from LLM responses.
Modifies schedule.SCHEDULE and user_memory.USER_FACTS in-memory.
"""
import json
import logging
import re
from typing import Tuple

import schedule as schedule_module
import user_memory

logger = logging.getLogger(__name__)

ACTION_PATTERN = re.compile(r"\[ACTION:\s*(\{.*?\})\s*\]", re.DOTALL)

# Instructions appended to the system prompt so the LLM knows how to emit actions
TOOL_INSTRUCTIONS = """
If the user asks to add, update, or remove a schedule event, or mentions a change to their personal information (name, location, etc.), append ONE action command on its own line at the very end of your response, in this exact format:
[ACTION: {"type": "<action_type>", ...fields}]

Action types and required fields:
- add_event:        {"type": "add_event", "date": "YYYY-MM-DD", "time": "HH:MM", "title": "...", "location": "..."}
- remove_event:     {"type": "remove_event", "title": "...", "date": "YYYY-MM-DD"}
- update_user_fact: {"type": "update_user_fact", "key": "location|name|timezone", "value": "..."}

Only emit an action when the user explicitly requests a change. Do not emit one for read-only queries.
""".strip()


def extract_and_execute(response: str) -> Tuple[str, bool]:
    """
    Find [ACTION: ...] in the response, execute it, and return
    (cleaned_response, action_was_executed).
    """
    match = ACTION_PATTERN.search(response)
    if not match:
        return response, False

    clean_response = ACTION_PATTERN.sub("", response).strip()

    try:
        action = json.loads(match.group(1))
        _execute(action)
        return clean_response, True
    except Exception as e:
        logger.error(f"Failed to parse/execute action: {e} | raw: {match.group(0)}")
        return clean_response, False


def _execute(action: dict):
    t = action.get("type")

    if t == "add_event":
        entry = {
            "date":     action["date"],
            "time":     action["time"],
            "title":    action["title"],
            "location": action.get("location", ""),
        }
        schedule_module.SCHEDULE.append(entry)
        logger.info(f"Added event: {entry}")

    elif t == "remove_event":
        before = len(schedule_module.SCHEDULE)
        schedule_module.SCHEDULE[:] = [
            e for e in schedule_module.SCHEDULE
            if not (e["title"].lower() == action["title"].lower()
                    and e.get("date", "") == action.get("date", e.get("date", "")))
        ]
        removed = before - len(schedule_module.SCHEDULE)
        logger.info(f"Removed {removed} event(s) matching title={action['title']!r}")

    elif t == "update_user_fact":
        key, val = action["key"], action["value"]
        user_memory.USER_FACTS[key] = val
        logger.info(f"Updated user fact: {key!r} = {val!r}")

    else:
        logger.warning(f"Unknown action type: {t!r}")
