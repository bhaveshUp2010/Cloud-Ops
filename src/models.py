from pydantic import BaseModel, Field, field_validator
from typing import List, Optional, Literal

class ChatRequest(BaseModel):
    question: str = Field(..., min_length=2, max_length=252)
    thread_id: str = Field(..., min_length=2, max_length=252)

    @field_validator("question", "thread_id", mode="before")
    @classmethod
    def strip_whitespace(cls, value: object) -> object:
        return value.strip() if isinstance(value, str) else value

class SourceItem(BaseModel):
    type: Literal["internal", "web"]
    title: str = ""
    source: str = ""
    url: Optional[str] = None
    page: Optional[int] = None


class ChatResponse(BaseModel):
    answer: str
    route: str
    used_web_search: bool
    support_status: str = ""
    usefulness: str = ""
    sources: List[SourceItem] = Field(default_factory=list)
    trace: List[str] = Field(default_factory=list)
    thread_id: str
    memory_turns: int = 0


class UploadResponse(BaseModel):
    filename: str
    chunks_indexed: int
    namespace: str