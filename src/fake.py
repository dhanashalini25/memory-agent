"""Canned replies for MODEL=fake: durability verdicts, summaries and chat."""
from __future__ import annotations


def respond(messages: list[dict]) -> str:
    last = messages[-1]["content"]
    if "YES or NO" in last:
        ephemeral = ("hello", "hi", "thanks", "ok", "what time")
        body = last.split("\n\n")[-1].lower()
        return "NO" if any(w in body for w in ephemeral) else "YES"
    if last.startswith("Summarise this conversation"):
        return "The user introduced themselves and stated their preferences. No open questions remain."
    system = messages[0]["content"] if messages and messages[0]["role"] == "system" else ""
    if "Known about this user" in system:
        return "Noted - and going on what you've told me before, that fits."
    return "Got it, I'll remember that."
