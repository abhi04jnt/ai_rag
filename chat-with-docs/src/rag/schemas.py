from pydantic import BaseModel, Field

class ChatMessage(BaseModel):
    role: str = Field(..., examples=["user", "assistant"])
    content: str

class ChatRequest(BaseModel):
    question: str
    history: list[ChatMessage] = []

class ChatResponse(BaseModel):
    answer: str
    retrieved: list[dict]
