"""
FastAPI application for SHL Assessment Recommender.

Endpoints:
- GET /health → {"status": "ok"}
- POST /chat  → {"reply": "...", "recommendations": [...], "end_of_conversation": false}
"""

import os
from contextlib import asynccontextmanager

from dotenv import load_dotenv

load_dotenv()  # populate os.environ from .env before anything reads GEMINI_API_KEY etc.

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.models import ChatRequest, ChatResponse, HealthResponse, Recommendation
from app.catalog import catalog
from app.agent import process_chat


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load catalogue on startup."""
    print("Loading SHL catalogue...")
    catalog.load()
    print(f"Loaded {len(catalog.assessments)} assessments.")
    yield
    print("Shutting down.")


app = FastAPI(
    title="SHL Assessment Recommender",
    description="Conversational agent for recommending SHL assessments",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS for any frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check endpoint."""
    return HealthResponse(status="ok")


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Stateless chat endpoint.

    Takes full conversation history and returns the next agent reply
    plus optional assessment recommendations.
    """
    if not request.messages:
        raise HTTPException(status_code=400, detail="messages array cannot be empty")

    try:
        reply, recommendations, end_of_conversation = await process_chat(request.messages)

        return ChatResponse(
            reply=reply,
            recommendations=recommendations,
            end_of_conversation=end_of_conversation,
        )
    except Exception as e:
        print(f"Error processing chat: {e}")
        # Return a graceful error response that still matches schema
        return ChatResponse(
            reply="I apologize, but I encountered an issue processing your request. Could you please rephrase your question about SHL assessments?",
            recommendations=[],
            end_of_conversation=False,
        )


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("app.main:app", host="0.0.0.0", port=port, reload=True)
