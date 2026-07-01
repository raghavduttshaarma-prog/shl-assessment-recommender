"""One-off diagnostic: call OpenRouter directly and print the raw response."""
import os
import json
from dotenv import load_dotenv

load_dotenv()

import requests

api_key = os.environ.get("OPENROUTER_API_KEY")
print(f"Key present: {bool(api_key)}, starts with: {api_key[:10] if api_key else None}")

resp = requests.post(
    "https://openrouter.ai/api/v1/chat/completions",
    headers={
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    },
    json={
        "model": os.environ.get("OPENROUTER_MODEL", "openrouter/free"),
        "messages": [{"role": "user", "content": "Say hi in one short sentence."}],
        "temperature": 0.3,
        "max_tokens": 100,
    },
    timeout=25,
)

print(f"HTTP status: {resp.status_code}")
print("Raw body:")
print(json.dumps(resp.json(), indent=2))
