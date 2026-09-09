"""Memory-Enabled Conversational Agent - two tiers that survive a restart.

Short term is a token-bounded buffer that summarises its oldest half when it
overflows. Long term is a per-user vector store on disk: a classifier decides
what is worth keeping, and recall is scored and thresholded before anything
enters the prompt.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from .llm import complete
from .logging_setup import log
from .store import Store

BUFFER_TOKENS = 400
RECALL_K = 4
RECALL_FLOOR = 0.12
STORE_PATH = "memory_store.json"
DEMO = "I'm Dhanashalini, I'm in Bengaluru, and I prefer Python over Java."


def count_tokens(text: str) -> int:
    """Rough but dependency-free. Swap in tiktoken if you need precision."""
    return max(1, len(text) // 4)


@dataclass
class Memory:
    user_id: str
    store_path: str = STORE_PATH
    buffer: list[dict] = field(default_factory=list)
    summary: str = ""
    _store: Store | None = None

    @property
    def store(self) -> Store:
        if self._store is None:
            self._store = Store(path=self.store_path)
        return self._store

    # ---------------------------------------------------------- short term
    def tokens(self) -> int:
        return sum(count_tokens(m["content"]) for m in self.buffer)

    def append(self, role: str, content: str) -> None:
        self.buffer.append({"role": role, "content": content})
        if self.tokens() > BUFFER_TOKENS:
            self.compress()

    def compress(self) -> None:
        """Summarise the oldest half; keep the rest verbatim."""
        half = max(1, len(self.buffer) // 2)
        old, self.buffer = self.buffer[:half], self.buffer[half:]
        transcript = "\n".join(f"{m['role']}: {m['content']}" for m in old)
        self.summary = complete([{
            "role": "user",
            "content": f"Summarise this conversation in two sentences, keeping facts:\n{transcript}",
        }]).strip()
        log.info("buffer_compressed", extra={"user": self.user_id, "dropped": half})

    # ----------------------------------------------------------- long term
    def is_durable(self, text: str) -> bool:
        """Ask the model whether this is worth remembering next week."""
        verdict = complete([{
            "role": "user",
            "content": (
                "Is this worth remembering about the user long-term "
                "(a stable fact, preference or goal)? Answer YES or NO only.\n\n" + text
            ),
        }])
        return verdict.strip().upper().startswith("YES")

    def maybe_remember(self, text: str) -> bool:
        if not self.is_durable(text):
            log.info("skipped_ephemeral", extra={"user": self.user_id})
            return False
        key = f"{self.user_id}:{len(self.store.records)}:{abs(hash(text)) % 10**6}"
        self.store.add(key, text, user=self.user_id)
        log.info("remembered", extra={"user": self.user_id, "key": key})
        return True

    def recall(self, query: str, k: int = RECALL_K) -> list[str]:
        hits = self.store.search(query, k=k, user=self.user_id)
        return [rec.text for rec, score in hits if score >= RECALL_FLOOR]


SYSTEM = "You are a helpful assistant with memory.{facts}{summary}"


def chat(mem: Memory, message: str) -> str:
    facts = mem.recall(message)
    system = SYSTEM.format(
        facts=("\n\nKnown about this user:\n" + "\n".join(f"- {f}" for f in facts)) if facts else "",
        summary=(f"\n\nEarlier in this conversation: {mem.summary}") if mem.summary else "",
    )
    mem.append("user", message)
    reply = complete([{"role": "system", "content": system}, *mem.buffer])
    mem.append("assistant", reply)
    mem.maybe_remember(message)
    return reply


def run(prompt: str) -> str:
    mem = Memory(user_id="demo")
    reply = chat(mem, prompt)
    return f"{reply}\n\n[remembered {len(mem.store)} facts across all sessions]"
