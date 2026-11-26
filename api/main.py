"""Day 11: the CiteRAG web API. Day 13: CORS from settings and a rate limit, for deployment.

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
import hashlib
import os
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import Annotated

import psycopg
from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.sse import EventSourceResponse, ServerSentEvent

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "experiments"))
# Importing loads the embedding model and BM25 index once, when the server starts,
# not on every request.
from day3_ask import DATABASE_URL  # noqa: E402
from day10_answer import answer_events  # noqa: E402

# Settings come from environment variables, so the same code runs locally and on Vercel.
# ALLOWED_ORIGINS: the web pages allowed to call this API (comma-separated).
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "http://localhost:5173").split(",")
# Every question costs two LLM calls, paid from our OpenRouter key, so a public site needs limits.
QUESTIONS_PER_HOUR = int(os.getenv("QUESTIONS_PER_HOUR", "20"))     # per visitor
QUESTIONS_PER_DAY = int(os.getenv("QUESTIONS_PER_DAY", "300"))      # for everyone together

app = FastAPI(title="CiteRAG", description="Answers questions about FastAPI with file and line citations.")

# CORS: the browser blocks a page on one origin (the React site) from reading responses from
# another (this API) unless the API allows it.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[origin.strip() for origin in ALLOWED_ORIGINS],
    allow_methods=["GET"],
)


# --- Rate limit, stored in Postgres -----------------------------------------------------------
# On a serverless host, many copies of this app can run at once and each has its own memory,
# so a counter in a Python variable wouldn't be shared. The database is shared by all copies.

with psycopg.connect(DATABASE_URL) as conn:
    conn.execute("""
        CREATE TABLE IF NOT EXISTS request_log (
            id          bigserial   PRIMARY KEY,
            visitor     text        NOT NULL,   -- a hash of the IP address, not the IP itself
            created_at  timestamptz NOT NULL DEFAULT now()
        )
    """)


def visitor_id(request: Request) -> str:
    """Who is asking. Behind Vercel's proxy the real IP is the first entry in X-Forwarded-For.
    We store a hash so the database never holds visitors' IP addresses."""
    forwarded = request.headers.get("x-forwarded-for", "")
    ip = forwarded.split(",")[0].strip() or (request.client.host if request.client else "unknown")
    return hashlib.sha256(ip.encode()).hexdigest()[:16]


def rate_limit(request: Request) -> None:
    """A FastAPI dependency: runs before /ask. Raises 429 Too Many Requests when over a limit."""
    visitor = visitor_id(request)
    with psycopg.connect(DATABASE_URL) as conn:
        this_hour, today = conn.execute(
            """
            SELECT count(*) FILTER (WHERE visitor = %s AND created_at > now() - interval '1 hour'),
                   count(*)
            FROM request_log
            WHERE created_at > now() - interval '1 day'
            """,
            (visitor,),
        ).fetchone()
        if this_hour >= QUESTIONS_PER_HOUR:
            raise HTTPException(429, f"Limit of {QUESTIONS_PER_HOUR} questions per hour reached. Try again later.")
        if today >= QUESTIONS_PER_DAY:
            raise HTTPException(429, "The demo has reached its daily question limit. Try again tomorrow.")
        conn.execute("INSERT INTO request_log (visitor) VALUES (%s)", (visitor,))


# --- Endpoints -------------------------------------------------------------------------------

@app.get("/health")
def health() -> dict:
    """For checking the server is up."""
    return {"status": "ok"}


@app.get("/ask", response_class=EventSourceResponse, dependencies=[Depends(rate_limit)])
def ask(q: Annotated[str, Query(min_length=3, max_length=500)]) -> Iterable[ServerSentEvent]:
    """Stream the answer. A plain `def` (not async) because our pipeline is blocking code
    (database calls, the model, the LLM client); FastAPI runs it in a thread so the server stays responsive.
    """
    for event, data in answer_events(q):
        yield ServerSentEvent(event=event, data=data)
