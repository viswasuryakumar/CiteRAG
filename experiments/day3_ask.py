"""Day 3, step 3: ask a question, retrieve the closest chunks, and have an LLM answer from them.

Needs: the database filled by day3_ingest.py, and OPENROUTER_API_KEY in .env.
Run:   python experiments/day3_ask.py "How do I return a 404 error?"
"""
import os
import sys
from pathlib import Path

import psycopg
from dotenv import load_dotenv
from openai import OpenAI
from pgvector.psycopg import register_vector
from sentence_transformers import SentenceTransformer

PROJECT_DIR = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_DIR / ".env")   # loads OPENROUTER_API_KEY and DATABASE_URL from .env
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://rag:rag@localhost:5432/rag")

TOP_K = 5
# Any model id from openrouter.ai/models works here. Claude Haiku 5.5: $0.10 / $0.50 per 1M tokens.
MODEL = "anthropic/claude-haiku-5.5"

SYSTEM_PROMPT = """You answer questions about FastAPI using only the excerpts provided
(from its documentation and its source code).
Cite the excerpts you used by their number, like [2].
If the excerpts don't contain the answer, say you don't know instead of guessing."""

embedder = SentenceTransformer("all-MiniLM-L6-v2")
# OpenRouter speaks the same API format as OpenAI, so the OpenAI client works with a different URL.
client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=os.getenv("OPENROUTER_API_KEY"))


def retrieve(question: str, k: int = TOP_K) -> list[dict]:
    """R in RAG: embed the question and let pgvector find the k closest chunks."""
    question_vector = embedder.encode(question)   # held in memory only, never stored

    with psycopg.connect(DATABASE_URL) as conn:
        register_vector(conn)
        rows = conn.execute(
            """
            SELECT file, start_line, end_line, text,
                   1 - (embedding <=> %(q)s) AS similarity
            FROM chunks
            ORDER BY embedding <=> %(q)s
            LIMIT %(k)s
            """,
            {"q": question_vector, "k": k},
        ).fetchall()

    return [
        {"file": f, "start_line": s, "end_line": e, "text": t, "similarity": sim}
        for f, s, e, t, sim in rows
    ]


def build_prompt(question: str, chunks: list[dict]) -> str:
    """A in RAG: paste the chunks' text (not their vectors) into the prompt, numbered."""
    excerpts = "\n\n".join(
        f"[{i}] {c['file']}, lines {c['start_line']}-{c['end_line']}\n{c['text']}"
        for i, c in enumerate(chunks, start=1)
    )
    return f"Excerpts:\n\n{excerpts}\n\nQuestion: {question}"


def generate(question: str, chunks: list[dict]) -> str:
    """G in RAG: the LLM writes the answer from the excerpts."""
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_prompt(question, chunks)},
        ],
    )
    return response.choices[0].message.content


def main() -> None:
    question = " ".join(sys.argv[1:]) or "How do I return a 404 error?"
    chunks = retrieve(question)

    print(f"Q: {question}\n\nRetrieved chunks:")
    for i, c in enumerate(chunks, start=1):
        print(f"  [{i}] {c['similarity']:.3f}  {c['file']}:{c['start_line']}-{c['end_line']}")

    print("\nAnswer:\n" + generate(question, chunks))


if __name__ == "__main__":
    main()
