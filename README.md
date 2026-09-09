# 05 - Memory-Enabled Conversational Agent

> Two tiers of memory that survive a restart.

**What it demonstrates:** Persisting user context across sessions without unbounded prompts

**Status:** working implementation with passing tests. Built as a learning project to understand the pattern, not as a production service.

---

## Run it right now

No API key needed - every project ships with `MODEL=fake`, a deterministic
offline responder, so you can see the whole flow work before spending anything.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
cp .env.example .env               # Windows: copy .env.example .env
python -m src.main
pytest -q
```

To use a real model, edit `.env`:

```
MODEL=gpt-4o-mini            # + OPENAI_API_KEY
MODEL=claude-3-5-haiku-latest  # + ANTHROPIC_API_KEY
MODEL=ollama/llama3.1        # free, runs locally
```

## How it works

Short-term memory is a token-bounded buffer. When it overflows, the oldest half is summarised into a single sentence and the rest is kept verbatim - so the conversation compresses rather than truncates.

Long-term memory is a vector store on disk, namespaced by user id. Not everything is stored: `is_durable()` asks the model whether a message is a stable fact worth keeping, so greetings and small talk never enter the store. Recall is scored and anything below `RECALL_FLOOR` is dropped before it reaches the prompt.

Because the store is a file, a brand-new `Memory` object with the same user id recovers everything - which is exactly what the restart test checks.

## What "done" means here

- A token-bounded buffer plus a persistent per-user vector store
- Buffer overflow is summarised, not silently dropped
- A classifier decides what is durable; chatter is never stored
- Recall is relevance-scored and thresholded before entering the prompt
- One user's memories are invisible to another user
- A fresh process with the same user id recovers prior facts

Every one of those lines has a test behind it in `tests/` - `pytest -q` is the
proof, not the README.

## Layout

```
src/llm.py             provider-agnostic completion, plus offline fake mode
src/fake.py            the canned responses that make MODEL=fake work
src/logging_setup.py   structured JSON logging
src/agent.py           the pattern itself
src/main.py            CLI entrypoint
tests/                 10 tests, all passing
```

## Next steps

- Replace `count_tokens` with tiktoken for exact budgeting
- Handle contradictions - a newer fact should supersede an older one
- Add memory decay so facts nobody recalls eventually age out

## Reference

https://github.com/FareedKhan-dev/langgraph-long-memory

---

Part of a 12-project agentic AI series - [github.com/dhanashalini25](https://github.com/dhanashalini25?tab=repositories)
