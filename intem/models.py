from typing import Any, TypedDict
from pydantic import BaseModel, Field


class Question(BaseModel):
    question: str = Field(min_length=3, max_length=6000)


class Plan(BaseModel):
    queries: list[str] = Field(min_length=1, max_length=5)


class Claim(BaseModel):
    text: str = Field(min_length=1)
    source_id: str
    quote: str = ""


class Draft(BaseModel):
    answer: str = Field(default="", max_length=12000)
    claims: list[Claim] = Field(default_factory=list, max_length=15)
    limitations: list[str] = Field(default_factory=list)
    missing: list[str] = Field(default_factory=list)


class State(TypedDict, total=False):
    question: str
    queries: list[str]
    evidence: list[dict]
    draft: dict
    verification: dict
    answer: str
    refused: bool
    answer_status: str
    trace: list[dict[str, Any]]
    timings: list[dict]
