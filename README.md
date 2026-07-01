# SHL Assessment Recommender

A stateless conversational agent that recommends SHL Individual Test
Solutions through dialogue. Built for the SHL Labs AI Intern take-home
assignment.

## What's here

```
app/
  main.py      FastAPI app: GET /health, POST /chat
  models.py    Pydantic request/response schemas
  catalog.py   Catalogue loader + TF-IDF/faceted search + name grounding
  agent.py     Conversation logic: guardrails, LLM call, validation
data/
  catalogue.json           377 scraped SHL Individual Test Solutions
GenAI_SampleConversations/
  C1.md ... C10.md         10 reference conversation traces
eval/
  run_eval.py              Local regression harness against the traces
Dockerfile, render.yaml    Deployment
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Add at least one LLM key to `.env` (tried in this order, first one present that succeeds wins):
- `GEMINI_API_KEY` — free tier at https://aistudio.google.com/apikey
- `GROQ_API_KEY` — free tier at https://console.groq.com
- `OPENROUTER_API_KEY` — free tier at https://openrouter.ai/keys (defaults to the auto-routing `openrouter/free` model)

Having more than one set is a good idea — each provider has its own free-tier
token/request budget, so if one gets rate-limited mid-conversation the
service automatically falls through to the next.

If neither key is set, the service still runs using a deterministic
rule-based fallback (TF-IDF search, no LLM) so `/health` and `/chat` stay
reachable — but conversation quality (clarify/refine/compare nuance) drops
substantially. Set a real key before submitting.

## Run locally

```bash
uvicorn app.main:app --reload
```

- `GET http://localhost:8000/health` → `{"status": "ok"}`
- `POST http://localhost:8000/chat` with body `{"messages": [{"role": "user", "content": "..."}]}`

## Evaluate against the reference traces

```bash
pip install requests
uvicorn app.main:app --reload &
python eval/run_eval.py
```

This replays each of the 10 traces' user turns against your running
service and reports schema compliance, hallucination checks (every
returned name/URL must exist in `data/catalogue.json`), and recall against
the reference shortlist shown in each trace. See the docstring in
`eval/run_eval.py` for why this is a useful dev-time regression check but
not a stand-in for the real (reactive, LLM-simulated-user) evaluator.

## Deploy (Render, free tier)

1. Push this folder to a GitHub repo.
2. On Render: New → Web Service → connect the repo. `render.yaml` already
   defines the build/start commands.
3. Add the `GEMINI_API_KEY` env var in the Render dashboard (marked
   `sync: false` in `render.yaml` so it isn't committed to source).
4. Deploy. First `/health` call after a cold start can take up to ~2
   minutes to wake the free-tier instance — expected per the assignment.

Any other free host (Fly.io, Railway, Modal, Hugging Face Spaces) works
too; the `Dockerfile` is host-agnostic.

## Design notes

See `approach.docx` for the full write-up (design choices, retrieval
setup, prompt design, evaluation approach, what didn't work). In short:

- **Retrieval**: TF-IDF cosine similarity over name/description/category/
  job-level text, merged with an exact substring/token match pass so
  COMPARE questions stay grounded to the literal assessments named in the
  conversation rather than just "similar" ones.
- **Guardrails**: prompt-injection, legal-advice, and general-hiring-advice
  patterns are checked deterministically *before* the LLM is called, so
  refusal behavior can't be talked out of by the model.
- **Turn budget**: the agent is told which turn (of the evaluator's 8-turn
  cap) it's on, so it commits to a shortlist instead of over-clarifying.
- **Validation**: every LLM-proposed recommendation is re-checked against
  the catalogue after generation; anything not found is dropped rather
  than passed through, with a catalogue-search fallback if that leaves the
  list empty despite the model clearly intending to recommend.
