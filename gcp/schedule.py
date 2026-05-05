"""
Mock schedule database for the user.
Add/edit events here. Dates are in YYYY-MM-DD format.
"""
from datetime import date, timedelta
from typing import Optional

today = date.today()

SCHEDULE = [
    {"date": str(today),               "time": "09:00", "title": "Team standup",              "location": "Zoom"},
    {"date": str(today),               "time": "12:30", "title": "Lunch with Sarah",           "location": "Sweetgreen, SoHo"},
    {"date": str(today),               "time": "15:00", "title": "DL project demo prep",       "location": "Columbia campus"},
    {"date": str(today + timedelta(1)),"time": "10:00", "title": "Doctor appointment",         "location": "Columbia Medical Center"},
    {"date": str(today + timedelta(1)),"time": "14:00", "title": "Coffee with Alex",           "location": "Blue Bottle, West Village"},
    {"date": str(today + timedelta(2)),"time": "09:30", "title": "DL final project demo",      "location": "Columbia CEPSR room 750"},
    {"date": str(today + timedelta(3)),"time": "18:00", "title": "Dinner with family",         "location": "Home"},
    {"date": str(today + timedelta(7)),"time": "11:00", "title": "Summer internship interview","location": "Remote (Google Meet)"},
]

_SCHEDULE_TRIGGERS = [
    "schedule", "calendar", "appointment", "meeting", "event",
    "what do i have", "what's on", "what is on", "am i free",
    "am i busy", "today", "tomorrow", "this week",
]


def needs_schedule(text: str) -> bool:
    lower = text.lower()
    return any(t in lower for t in _SCHEDULE_TRIGGERS)


def get_schedule_context(text: str) -> Optional[str]:
    lower = text.lower()
    if "tomorrow" in lower:
        target = str(today + timedelta(1))
        label = "tomorrow"
    elif "this week" in lower:
        target = None
        label = "this week"
    else:
        target = str(today)
        label = "today"

    if target:
        events = [e for e in SCHEDULE if e["date"] == target]
    else:
        week_end = today + timedelta(7)
        events = [e for e in SCHEDULE if str(today) <= e["date"] <= str(week_end)]

    if not events:
        return f"The user has no scheduled events {label}."

    lines = [f"User's schedule for {label}:"]
    for e in sorted(events, key=lambda x: x["time"]):
        lines.append(f"- {e['time']}: {e['title']} @ {e['location']}")
    return "\n".join(lines)
