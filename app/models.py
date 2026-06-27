from pydantic import BaseModel, Field

DISCLAIMER = (
    "This is an engineering demonstration, not a medical device. "
    "It does not provide medical advice."
)


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=500)
    top_k: int = Field(default=8, ge=1, le=12)


class RetrievedContext(BaseModel):
    text: str
    score: float
    source_title: str
    source_url: str


class QueryResponse(BaseModel):
    question: str
    answer: str
    sources: list[str]
    contexts: list[RetrievedContext]
    refused: bool
    latency_ms: float | None = None
    cost_usd: float = 0.0
    model: str | None = None
    disclaimer: str = DISCLAIMER
