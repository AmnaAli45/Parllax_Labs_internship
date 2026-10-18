"""
FastAPI app jo RAG pipeline ko web endpoints ke zariye expose karti hai.

Endpoints:
  GET  /health         -> app chal rahi hai ya nahi, check karne ke liye
  POST /ask            -> ek sawal bhejo, jawab + status + timing wapas milega
"""
from fastapi import FastAPI
from pydantic import BaseModel


app = FastAPI(title="RAG API", version="1.0")


class AskRequest(BaseModel):
    """User jo bhejega: sirf ek query, aur optional top_k."""
    query: str
    top_k: int = 3


class AskResponse(BaseModel):
    """User ko jo wapas milega."""
    answer: str
    status: str
    retrieval_ms: float
    generation_ms: float
    total_ms: float


@app.get("/health")
def health_check():
    """Simple check: app zinda hai ya nahi."""
    return {"status": "ok"}


@app.post("/ask", response_model=AskResponse)
def ask_question(request: AskRequest):
    """
    Ek query leta hai, RAG pipeline chalata hai, jawab deta hai.
    """
    # TODO: yahan asal answer_question() call hogi (Week 3/4 se)
    return AskResponse(
        answer=f"This is a placeholder answer for: {request.query}",
        status="answered",
        retrieval_ms=12.5,
        generation_ms=450.0,
        total_ms=462.5,
    )