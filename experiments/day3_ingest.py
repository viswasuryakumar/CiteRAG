"""Day 3, step 2: embed every chunk and store it in Postgres + pgvector.

Needs: the database running (docker compose up -d) and data/chunks.jsonl (run day3_chunk.py first).
Run:   python experiments/day3_ingest.py
"""
import json
import os
from pathlib import Path

import psycopg
from dotenv import load_dotenv
from pgvector.psycopg import register_vector
from sentence_transformers import SentenceTransformer

PROJECT_DIR = Path(__file__).resolve().parent.parent
CHUNKS_FILE = PROJECT_DIR / "data" / "chunks.jsonl"

load_dotenv(PROJECT_DIR / ".env")
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://rag:rag@localhost:5432/rag")


def embedding_text(chunk: dict) -> str:
    """What we feed the model: the heading path, then the chunk text.

    A chunk like "The same error would appear if you provided a float..." doesn't say
    what it's about. Putting "Path Parameters > Data validation" in front tells the model.
    """
    return f"{chunk['heading']}\n\n{chunk['text']}"


def main() -> None:
    chunks = [json.loads(line) for line in CHUNKS_FILE.open(encoding="utf-8")]

    # 1. Embed all chunks in one call. Batching (64 at a time) is much faster than one by one.
    model = SentenceTransformer("all-MiniLM-L6-v2")
    print(f"embedding {len(chunks)} chunks...")
    embeddings = model.encode(
        [embedding_text(c) for c in chunks], batch_size=64, show_progress_bar=True
    )

    with psycopg.connect(DATABASE_URL) as conn:
        # 2. Switch on pgvector, then teach the Python driver to send numpy arrays as vectors.
        conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        conn.commit()
        register_vector(conn)

        # 3. Start fresh each run, so re-running never creates duplicate rows.
        conn.execute("DROP TABLE IF EXISTS chunks")
        conn.execute("""
            CREATE TABLE chunks (
                id          serial PRIMARY KEY,
                file        text NOT NULL,
                heading     text NOT NULL,
                start_line  int  NOT NULL,
                end_line    int  NOT NULL,
                text        text NOT NULL,
                embedding   vector(384) NOT NULL
            )
        """)

        # 4. One row per chunk: the text (for Claude), file + lines (for citations),
        #    and the vector (for search).
        with conn.cursor() as cur:
            cur.executemany(
                "INSERT INTO chunks (file, heading, start_line, end_line, text, embedding)"
                " VALUES (%s, %s, %s, %s, %s, %s)",
                [
                    (c["file"], c["heading"], c["start_line"], c["end_line"], c["text"], vector)
                    for c, vector in zip(chunks, embeddings)
                ],
            )

        # 5. Build the HNSW index after inserting: building once is faster than
        #    updating the graph 1,649 times. vector_cosine_ops matches the <=> operator.
        conn.execute("CREATE INDEX ON chunks USING hnsw (embedding vector_cosine_ops)")

        count = conn.execute("SELECT count(*) FROM chunks").fetchone()[0]
    print(f"stored {count} chunks")


if __name__ == "__main__":
    main()
