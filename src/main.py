
import json
import logging
import time
import uuid

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel


logger = logging.getLogger("rag_api")
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(handler)


def log_event(event, **fields):
    """Ek structured (JSON) log line likhta hai."""
    record = {"event": event, "timestamp": time.time(), **fields}
    logger.info(json.dumps(record))


app = FastAPI(title="RAG API", version="1.0")


@app.middleware("http")
async def log_requests(request: Request, call_next):
    request_id = str(uuid.uuid4())[:8]
    start = time.time()
    log_event("request_started", request_id=request_id, method=request.method, path=request.url.path)

    try:
        response = await call_next(request)
        status_code = response.status_code
    except Exception:
        status_code = 500
        duration_ms = round((time.time() - start) * 1000, 2)
        log_event("request_finished", request_id=request_id, method=request.method,
                   path=request.url.path, status_code=status_code, duration_ms=duration_ms)
        raise

    duration_ms = round((time.time() - start) * 1000, 2)
    log_event("request_finished", request_id=request_id, method=request.method,
               path=request.url.path, status_code=status_code, duration_ms=duration_ms)
    response.headers["X-Request-ID"] = request_id
    return response


@app.exception_handler(RequestValidationError)
async def validation_error_handler(request: Request, exc: RequestValidationError):
    """422: request ka format hi ghalat tha (jaise field missing ya galat type)."""
    log_event("validation_error", path=request.url.path, errors=exc.errors())
    return JSONResponse(
        status_code=422,
        content={"error": "Invalid request format", "details": exc.errors()},
    )


@app.exception_handler(Exception)
async def unhandled_error_handler(request: Request, exc: Exception):
    """500: hamare apne code mein koi na-dekha hua error/crash hua."""
    log_event("unhandled_error", path=request.url.path, error_type=type(exc).__name__, error_message=str(exc))
    return JSONResponse(
        status_code=500,
        content={"error": "Something went wrong on our side. Please try again later."},
    )


class AskRequest(BaseModel):
    query: str
    top_k: int = 3


class AskResponse(BaseModel):
    answer: str
    status: str
    retrieval_ms: float
    generation_ms: float
    total_ms: float


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/ask", response_model=AskResponse)
def ask_question(request: AskRequest):
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    if request.top_k < 1 or request.top_k > 20:
        raise HTTPException(status_code=400, detail="top_k must be between 1 and 20.")

    if request.query.strip().lower() == "trigger crash":
        raise RuntimeError("Simulated crash for testing the 500 handler.")

    return AskResponse(
        answer=f"This is a placeholder answer for: {request.query}",
        status="answered",
        retrieval_ms=12.5,
        generation_ms=450.0,
        total_ms=462.5,
    )