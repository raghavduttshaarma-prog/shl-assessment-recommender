"""Pydantic models for request/response schemas."""

from pydantic import BaseModel, Field
from typing import List, Optional


class ChatMessage(BaseModel):
    role: str = Field(..., description="Either 'user' or 'assistant'")
    content: str = Field(..., description="Message content")


class ChatRequest(BaseModel):
    messages: List[ChatMessage] = Field(..., description="Full conversation history")


class Recommendation(BaseModel):
    name: str = Field(..., description="Assessment name from catalogue")
    url: str = Field(..., description="Catalogue URL")
    test_type: str = Field(
        ...,
        description="Catalogue test_type code(s): A, B, C, D, E, K, P, S, or a comma-joined combination (e.g. 'K,S')",
    )


class ChatResponse(BaseModel):
    reply: str = Field(..., description="Agent's text reply")
    recommendations: List[Recommendation] = Field(
        default_factory=list,
        description="1-10 assessment recommendations, or empty if still gathering context"
    )
    end_of_conversation: bool = Field(
        default=False,
        description="True only when the agent considers the task complete"
    )


class HealthResponse(BaseModel):
    status: str = "ok"
