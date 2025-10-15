from pydantic import BaseModel
from typing import List, Optional


class IngestResponse(BaseModel):
    inserted: int
    message: str


class QueryRequest(BaseModel):
    q: str
    top_k: Optional[int] = 5
    use_llm: Optional[bool] = None


class Passage(BaseModel):
    id: str
    text: str
    score: float
    source: Optional[str]


class QueryResponse(BaseModel):
    answer: str
    passages: List[Passage]