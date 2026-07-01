# Approach Document — SHL Assessment Recommender

**Author:** Drishti Sharma · **Role applied for:** AI Intern, SHL Labs

## 1. Problem framing

Hiring managers describe roles in natural language, not catalogue vocabulary.
The task is to turn that vague intent into a grounded, cited shortlist through
dialogue — asking when information is missing, retrieving from the real SHL
catalogue rather than the model's prior knowledge, and refusing to drift into
legal or general hiring advice. The hard constraint underneath all of this:
the service is stateless, so every turn has to re-derive "where are we in the
conversation" purely from the message history the evaluator resends.

## 2. Catalogue & retrieval

The 377 Individual Test Solutions were scraped into `data/catalogue.json`
(name, URL, description, job levels, languages, duration, remote/adaptive
flags, and category keys). Each assessment's category keys are mapped to
SHL's single-letter test_type scheme (A/B/C/D/E/K/P/S); assessments spanning
multiple categories get a joined code like `K,S`, matching how the reference
conversations render bundled solutions.

Retrieval is a hybrid, not a single vector index:

- **TF-IDF cosine similarity** over name + description + categories + job
  levels handles the general "what fits this need" case (e.g. "Java
  developer, 4 years, works with stakeholders").
- **Exact substring/token grounding** (`catalog.find_by_substring`) scans the
  full conversation for literal assessment names or their meaningful
  sub-tokens (e.g. "OPQ", "DSI", "GSA") and pulls those catalogue entries in
  regardless of TF-IDF score. This exists specifically for COMPARE turns and
  follow-ups about a previously-named assessment ("what's the difference
  between OPQ and GSA?") — similarity search alone tends to under-rank the
  exact items being asked about once the conversation has moved past the
  words that made them score highly.

Both result sets are merged (named matches first) and handed to the LLM as
grounding context; nothing enters the final response unless it also survives
a second, independent validation pass (§4).

I chose TF-IDF over an embedding/vector store deliberately: 377 documents is
small enough that a sparse lexical method retrieves well, needs no external
embedding API call (extra latency + a second point of failure inside the
30-second timeout), and is trivial to unit-test deterministically.

## 3. Agent design

**Four behaviors, one state machine inferred fresh each call.** Because the
API is stateless, "clarify vs. recommend vs. refine vs. compare" is decided
from the message history alone, on every request. The LLM (Gemini 2.0 Flash,
with Groq/Llama as a secondary provider) makes that call, guided by a system
prompt that spells out each behavior and a **turn-budget hint** ("this is
message N of 8") so the agent stops asking questions and commits to a
shortlist as the evaluator's cap approaches, instead of the conversation
timing out with nothing.

**Deterministic guardrails run before the LLM, not instead of it.** Prompt
injection ("ignore previous instructions", "reveal your system prompt"),
legal/compliance questions, and general hiring-advice questions are matched
against a small regex set and short-circuited with a fixed refusal —
`recommendations: []`, `end_of_conversation: false` — without ever reaching
the model. This was a direct response to the assignment's warning that
"insufficient evaluation rigor" and "hallucination" are the most common
failure modes: a system prompt is a strong suggestion, not a guarantee, and
behavior probes should not depend on the model reliably choosing to comply
with it turn after turn.

**`end_of_conversation` is conservative by design.** It's true only when a
shortlist has already been delivered *and* the latest user message reads as
a clear acceptance with no new constraint (checked both by the LLM's own
judgement and a `_is_confirmation` regex used in the non-LLM fallback path).
Early drafts set it true on the same turn a shortlist first appeared, which
is wrong per the traces — several (e.g. C1, C4, C6) show a shortlist
persisting across 1–3 more refinement turns before the user actually
confirms.

**Validation, not trust.** Every recommendation the LLM proposes is checked
against the catalogue by exact name, then a length-guarded fuzzy match;
anything unmatched is dropped silently. If the model clearly intended to
recommend but every proposed name failed validation (garbled name, off-catalogue
invention), a catalogue search over the conversation fills the gap rather
than shipping an empty array that would contradict the reply text.

## 4. API

`GET /health` → `{"status": "ok"}`. `POST /chat` takes the full message
history and returns `{reply, recommendations[0..10], end_of_conversation}`,
matching the spec exactly — this was the first bug I fixed in the initial
scaffold: `end_of_conversation` was missing from the response model
entirely, which would have failed every schema check in the automated
harness regardless of conversational quality.

## 5. Evaluation approach

`eval/run_eval.py` parses the 10 provided traces and replays each trace's
scripted user turns against a running instance of the service, checking (a)
schema compliance on every response, (b) that every returned name/URL exists
in the catalogue (zero tolerance — any miss is a hallucination), and (c)
recall against the reference shortlist shown at each turn that has one. This
is a *static* replay: it always sends the trace's fixed next line rather than
reacting to what our agent just said, so it's a regression check during
development, not a stand-in for the real evaluator's LLM-simulated user (who
answers *our* clarifying questions, not the reference script's). I relied on
it to catch schema and hallucination regressions cheaply, and read the
traces manually to calibrate tone, refusal phrasing, and the
confirmation/refinement boundary that governs `end_of_conversation`.

## 6. What didn't work / changed

- **Single test_type letter per assessment** — the initial mapping picked
  the *first* matching category, silently dropping information for bundled
  solutions. Traces showed multi-code strings (`K,S`, `P,C`, `A,S`); fixed
  by joining all matched codes.
- **`Simulations → "SIM"`** — wrong; the traces consistently use the
  single letter `S`.
- **Setting `end_of_conversation` on first recommend** — corrected after
  re-reading the traces (§3).
- **Pure substring "contains" matching for recommendation validation** —
  matched too eagerly on short names; now requires a minimum length and
  prefers the longest overlapping catalogue name.
- **Retrieval context was too large for free-tier LLM budgets** — widening
  candidate retrieval from 12 to 30 catalogue entries (to fix low recall on
  less-obvious items) pushed each `/chat` call to 5,000-8,500 tokens, which
  burned through Groq's 100k-token/day free tier in well under 40 turns.
  Fixed by compacting the per-item context string (~150 tokens → ~50-60
  tokens) and capping merged retrieval at 18 items — the real lesson being
  that token budget is a hard constraint worth designing for from the
  start, not something to patch after building a bigger prompt.
- **Provider calls could hang close to the evaluator's 30s/call limit** —
  free-tier LLM routing (especially OpenRouter's shared free-model pool) is
  sometimes slow, and the provider SDKs are synchronous, so a slow call
  blocked the async event loop with no time budget left to fall through.
  Fixed with a short per-provider timeout (8s) enforced at both the
  SDK/client level and an `asyncio.wait_for` backstop, and by running each
  provider call in a thread so it doesn't block the server while waiting.
- **Reply text and the `recommendations` array could disagree** — observed
  live: the model's `reply` said "I've included a personality measure
  (OPQ32r)" while the `recommendations` array it returned in the same
  response didn't actually contain OPQ32r. This is exactly the
  "conversational incoherence" failure mode the assignment calls out.
  Added an explicit consistency rule instructing the model to treat the
  array as the source of truth and write the reply to match it, not the
  reverse. Residual risk: this is a prompted behavior, not a hard
  guarantee, so it's worth spot-checking in the eval harness output rather
  than assumed fixed.
- **`end_of_conversation` was never true on a plain confirmation turn** —
  the system prompt originally told the model to return an empty
  `recommendations` array whenever the shortlist hadn't changed, which
  meant a user saying "that works, thanks" got zero recommendations back
  instead of the standing shortlist re-confirmed. Traces show the shortlist
  should stay in every response from the first time it's presented onward;
  fixed the prompt accordingly.

## 7. AI tool usage disclosure

I used Claude (agentic coding assistant) to review an initial scaffold I had
built, cross-check it line-by-line against the assignment PDF and the 10
reference traces, fix the schema/logic bugs described above, and write the
guardrail and evaluation code. I read and understand every change; the
design decisions and trade-offs above are mine to defend.
