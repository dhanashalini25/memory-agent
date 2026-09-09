import pytest

from src import agent
from src.agent import BUFFER_TOKENS, RECALL_FLOOR, Memory, chat, count_tokens


@pytest.fixture()
def sp(tmp_path):
    return str(tmp_path / "mem.json")


def test_thresholds_are_sane():
    assert BUFFER_TOKENS > 0 and 0 < RECALL_FLOOR < 1


def test_append_grows_buffer(sp):
    m = Memory(user_id="u1", store_path=sp)
    m.append("user", "hello")
    assert len(m.buffer) == 1


def test_compression_on_overflow(monkeypatch, sp):
    monkeypatch.setattr(agent, "complete", lambda msgs, **k: "A short summary.")
    m = Memory(user_id="u1", store_path=sp)
    for i in range(40):
        m.append("user", "word " * 40)
    assert m.summary == "A short summary."
    assert m.tokens() <= BUFFER_TOKENS * 2


def test_durable_facts_are_stored(monkeypatch, sp):
    monkeypatch.setattr(agent, "complete", lambda msgs, **k: "YES")
    m = Memory(user_id="u1", store_path=sp)
    assert m.maybe_remember("I prefer Python over Java")
    assert len(m.store) == 1


def test_ephemeral_chatter_is_not_stored(monkeypatch, sp):
    monkeypatch.setattr(agent, "complete", lambda msgs, **k: "NO")
    m = Memory(user_id="u1", store_path=sp)
    assert not m.maybe_remember("hi there")
    assert len(m.store) == 0


def test_recall_is_scoped_to_user(monkeypatch, sp):
    monkeypatch.setattr(agent, "complete", lambda msgs, **k: "YES")
    a = Memory(user_id="alice", store_path=sp)
    a.maybe_remember("Alice prefers Python over Java")
    b = Memory(user_id="bob", store_path=sp)
    assert b.recall("what does Alice think about Python and Java") == []
    assert a.recall("what does Alice think about Python and Java")


def test_recall_drops_low_relevance(monkeypatch, sp):
    monkeypatch.setattr(agent, "complete", lambda msgs, **k: "YES")
    m = Memory(user_id="u1", store_path=sp)
    m.maybe_remember("Deploys happen on Friday afternoons")
    assert m.recall("recipe for dosa") == []


def test_memory_survives_restart(monkeypatch, sp):
    monkeypatch.setattr(agent, "complete", lambda msgs, **k: "YES")
    first = Memory(user_id="u1", store_path=sp)
    first.maybe_remember("Dhanashalini is based in Bengaluru")

    reborn = Memory(user_id="u1", store_path=sp)  # brand new object, same file
    assert reborn.recall("Dhanashalini Bengaluru")


def test_chat_injects_recalled_facts(monkeypatch, sp):
    monkeypatch.setattr(agent, "complete", lambda msgs, **k: "YES")
    m = Memory(user_id="u1", store_path=sp)
    m.maybe_remember("The user prefers Python over Java")

    seen = {}

    def capture(msgs, **kwargs):
        if msgs and msgs[0]["role"] == "system":
            seen["system"] = msgs[0]["content"]
        return "YES"

    monkeypatch.setattr(agent, "complete", capture)
    chat(m, "should I use Python or Java for this")
    assert "Python" in seen.get("system", "")


def test_token_counter_monotonic():
    assert count_tokens("a" * 400) > count_tokens("a" * 40)
