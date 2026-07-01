"""
Local evaluation harness for the SHL Assessment Recommender.

Replays the 10 provided GenAI_SampleConversations traces against a running
instance of the service (default: http://localhost:8000) using the exact
user messages from each trace, and checks:

  - Schema compliance of every /chat response (reply: str, recommendations:
    list of 0-10 {name,url,test_type}, end_of_conversation: bool).
  - Every recommended name/url actually exists in data/catalogue.json.
  - Recall against the reference shortlist shown in the trace, for turns
    where the trace shows one.

Caveat: this is a STATIC replay (it always sends the trace's scripted user
message, not a live reactive one), so it is a useful regression check
during development but is not a substitute for the real evaluator, which
uses an LLM to play the user and will react to whatever our agent actually
says. Use it to catch schema/hallucination bugs before submitting, not to
claim a final Recall@10 score.

Usage:
    pip install requests
    uvicorn app.main:app --reload &
    python eval/run_eval.py
"""

import glob
import os
import re
import sys

import requests

BASE_URL = os.environ.get("EVAL_BASE_URL", "http://localhost:8000")
TRACES_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "GenAI_SampleConversations")
CATALOGUE_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "catalogue.json")


def load_catalogue_names():
    import json
    with open(CATALOGUE_PATH, "r", encoding="utf-8-sig") as f:
        data = json.load(f)
    return {item["name"].lower() for item in data}, {item["link"] for item in data}


def parse_trace(path):
    """Parse a trace .md file into a list of turns: dict(user, ref_names, ref_eoc, has_table)."""
    text = open(path, "r", encoding="utf-8").read()
    blocks = re.split(r"^### Turn \d+\s*$", text, flags=re.MULTILINE)[1:]

    turns = []
    for block in blocks:
        user_match = re.search(r"\*\*User\*\*\s*\n\s*((?:>.*\n?)+)", block)
        if not user_match:
            continue
        raw_lines = user_match.group(1).splitlines()
        cleaned = [re.sub(r"^>\s?", "", ln) for ln in raw_lines]
        user_msg = "\n".join(cleaned).strip()

        eoc_match = re.search(r"end_of_conversation`:\s*\*\*(true|false)\*\*", block)
        ref_eoc = eoc_match.group(1) == "true" if eoc_match else None

        row_matches = re.findall(r"^\|\s*\d+\s*\|\s*(.*?)\s*\|", block, flags=re.MULTILINE)
        ref_names = [name.strip() for name in row_matches]

        turns.append({
            "user": user_msg,
            "ref_names": ref_names,
            "ref_eoc": ref_eoc,
            "has_table": bool(ref_names),
        })
    return turns


def check_schema(resp_json):
    errors = []
    if not isinstance(resp_json.get("reply"), str):
        errors.append("reply is not a string")
    recs = resp_json.get("recommendations")
    if not isinstance(recs, list):
        errors.append("recommendations is not a list")
    elif len(recs) > 10:
        errors.append(f"recommendations has {len(recs)} items (> 10)")
    else:
        for r in recs:
            if not all(k in r for k in ("name", "url", "test_type")):
                errors.append(f"recommendation missing keys: {r}")
    if not isinstance(resp_json.get("end_of_conversation"), bool):
        errors.append("end_of_conversation is not a boolean")
    return errors


def recall_at_10(returned_names, ref_names):
    if not ref_names:
        return None
    returned_lower = {n.lower() for n in returned_names}
    ref_lower = {n.lower() for n in ref_names}
    hit = len(returned_lower & ref_lower)
    return hit / len(ref_lower)


def run():
    catalogue_names, catalogue_urls = load_catalogue_names()
    trace_files = sorted(glob.glob(os.path.join(TRACES_DIR, "*.md")))

    if not trace_files:
        print(f"No trace files found in {TRACES_DIR}")
        sys.exit(1)

    all_recalls = []
    schema_failures = 0
    hallucination_failures = 0
    total_turns = 0

    for path in trace_files:
        name = os.path.basename(path)
        turns = parse_trace(path)
        messages = []
        print(f"\n=== {name} ({len(turns)} turns) ===")

        for i, turn in enumerate(turns, 1):
            messages.append({"role": "user", "content": turn["user"]})
            total_turns += 1

            try:
                resp = requests.post(f"{BASE_URL}/chat", json={"messages": messages}, timeout=30)
                resp.raise_for_status()
                data = resp.json()
            except Exception as e:
                print(f"  Turn {i}: REQUEST FAILED: {e}")
                schema_failures += 1
                messages.append({"role": "assistant", "content": ""})
                continue

            errors = check_schema(data)
            if errors:
                schema_failures += 1
                print(f"  Turn {i}: SCHEMA ERRORS: {errors}")

            recs = data.get("recommendations", []) or []
            for r in recs:
                if r.get("name", "").lower() not in catalogue_names:
                    hallucination_failures += 1
                    print(f"  Turn {i}: HALLUCINATED NAME not in catalogue: {r.get('name')}")
                elif r.get("url") not in catalogue_urls:
                    hallucination_failures += 1
                    print(f"  Turn {i}: URL doesn't match catalogue entry: {r.get('url')}")

            if turn["has_table"]:
                recall = recall_at_10([r.get("name", "") for r in recs], turn["ref_names"])
                if recall is not None:
                    all_recalls.append(recall)
                    print(f"  Turn {i}: recall={recall:.2f}  (got {len(recs)} recs, ref had {len(turn['ref_names'])})")

            messages.append({"role": "assistant", "content": data.get("reply", "")})

    print("\n=== SUMMARY ===")
    print(f"Total turns replayed: {total_turns}")
    print(f"Schema failures: {schema_failures}")
    print(f"Hallucination failures (name/url not in catalogue): {hallucination_failures}")
    if all_recalls:
        print(f"Mean recall vs reference shortlists: {sum(all_recalls)/len(all_recalls):.3f} over {len(all_recalls)} recommend-turns")
    else:
        print("No reference shortlists found to score.")


if __name__ == "__main__":
    run()
