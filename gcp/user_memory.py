"""
User memory — facts about the user injected into every system prompt.
Edit USER_FACTS to add/update what the assistant knows about the user.
"""

USER_FACTS = {
    "name": "Jianyu",
    "location": "Manhattan, New York City",
    "timezone": "Eastern Time (ET)",
}


def get_user_context() -> str:
    lines = ["Facts about the user:"]
    for key, val in USER_FACTS.items():
        lines.append(f"- {key.replace('_', ' ').capitalize()}: {val}")
    return "\n".join(lines)
