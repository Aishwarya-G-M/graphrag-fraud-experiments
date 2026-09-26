from typing import Any, Literal
from pydantic import BaseModel, Field

import json
import re

class AbstentionRequest(BaseModel):
    query: str
    parameters: dict[str, Any] = Field(default_factory=dict)


class AbstentionResponse(BaseModel):
    query: str
    backend: str = "graphrag"
    abstention_status: Literal["answer", "abstain"]
    is_spam: bool | None = None
    answer: str | None = None
    abstention_reason: str | None = None
    retrieval_metadata: dict[str, Any] = Field(
        default_factory=dict
    )

def parse_abstention_output(raw_output: str) -> dict:
    cleaned = raw_output.strip()

    if cleaned.startswith("```"):
        cleaned = re.sub(
            r"^```(?:json)?\s*|\s*```$",
            "",
            cleaned,
            flags=re.IGNORECASE,
        ).strip()

    return json.loads(cleaned)