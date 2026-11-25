"""Day 11: the CiteRAG web API.

GET /health                      -> {"status": "ok"}
GET /ask?q=How do I return a 404 -> a stream of Server-Sent Events (SSE):
    event: sources   the 5 retrieved excerpts (file, lines, GitHub link)
    event: token     a piece of the answer, many times, as the LLM writes it
    event: done      the checked citations, and whether it said "I don't know"

The API only translates HTTP <-> our pipeline; all the RAG logic lives in experiments/.

Needs: the database running and filled, OPENROUTER_API_KEY in .env.
Run:   uvicorn api.main:app --port 8000      (from the project folder)
Try:   curl -N "http://localhost:8000/ask?q=How%20do%20I%20return%20a%20404%20error%3F"
"""
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.sse import EventSourceResponse, ServerSentEvent

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "experiments"))
# Importing loads the embedding model, BM25 index and reranker once, when the server starts,
# not on every request.
from day10_answer import answer_events  # noqa: E402

app = FastAPI(title="CiteRAG", description="Answers questions about FastAPI with file and line citations.")

# CORS: the browser blocks a page on one origin (the React dev server, port 5173) from reading
# responses from another (this API, port 8000) unless the API allows it.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["GET"],
)


@app.get("/health")
def health() -> dict:
    """For checking the server is up (load balancers on AWS will call this on Day 13)."""
    return {"status": "ok"}


@app.get("/ask", response_class=EventSourceResponse)
def ask(q: Annotated[str, Query(min_length=3, max_length=500)]) -> Iterable[ServerSentEvent]:
    """Stream the answer. A plain `def` (not async) because our pipeline is blocking code
    (database calls, models, the LLM client); FastAPI runs it in a thread so the server stays responsive.
    """
    for event, data in answer_events(q):
        yield ServerSentEvent(event=event, data=data)
