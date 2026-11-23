"""Day 6: hybrid search = vector search + BM25 keyword search, merged with Reciprocal Rank Fusion.

Both searches run separately over ALL chunks. Each gives its own ranked list of chunk ids.
RRF merges the two lists: a chunk scores 1 / (60 + rank) for every list it appears in.

Needs: the database filled by day3_ingest.py (and OPENROUTER_API_KEY in .env to get an answer).
Run:   python experiments/day6_hybrid.py "How do I return a 404 error?"
"""
import math
import re
import sys
from collections import Counter, defaultdict

import psycopg
from pgvector.psycopg import register_vector

from day3_ask import DATABASE_URL, TOP_K, embedder, generate

CANDIDATES = 50   # how many results each search hands to RRF
RRF_K = 60        # the standard RRF constant; stops rank 1 from dominating


# Stopwords: words that say nothing about the topic. Without this, "how do I" in a question
# gets a high IDF (the docs rarely say "how do I") and BM25 matches on filler words.
# The usual English list (as in Lucene) plus words that start questions.
STOPWORDS = set("""
a an and are as at be but by for if in into is it no not of on or such that the their then there
these they this to was will with
how do does did i my me we our you your can could should would what which when where why who
""".split())


def tokenize(text: str) -> list[str]:
    """Lowercase words and numbers, minus stopwords. 'status_code=404' -> ['status', 'code', '404']."""
    return [word for word in re.findall(r"[a-z0-9]+", text.lower()) if word not in STOPWORDS]


class BM25:
    """Keyword search. Scores a chunk by the question words it contains.

    k1 limits how much repeating a word helps (term frequency saturates).
    b  controls how much long chunks are penalised (length normalisation).
    1.2 and 0.75 are the usual defaults (Elasticsearch / Lucene use them).
    """

    def __init__(self, documents: list[str], k1: float = 1.2, b: float = 0.75):
        self.k1 = k1
        self.b = b
        # For each chunk: how many times each word appears in it.
        self.word_counts = [Counter(tokenize(doc)) for doc in documents]
        self.lengths = [sum(counts.values()) for counts in self.word_counts]
        self.average_length = sum(self.lengths) / len(self.lengths)

        # Document frequency: in how many chunks does each word appear?
        doc_freq = Counter()
        for counts in self.word_counts:
            doc_freq.update(counts.keys())
        # IDF: rare words get a big weight, words found in almost every chunk get nearly 0.
        n = len(documents)
        self.idf = {word: math.log(1 + (n - df + 0.5) / (df + 0.5)) for word, df in doc_freq.items()}

    def scores(self, query: str) -> list[float]:
        """One BM25 score per chunk, in the same order as the documents."""
        query_words = set(tokenize(query))
        results = []
        for counts, length in zip(self.word_counts, self.lengths):
            score = 0.0
            for word in query_words:
                tf = counts[word]   # term frequency: how often the word is in this chunk
                if tf == 0:
                    continue
                length_factor = 1 - self.b + self.b * length / self.average_length
                score += self.idf[word] * tf * (self.k1 + 1) / (tf + self.k1 * length_factor)
            results.append(score)
        return results


def load_chunks() -> list[dict]:
    """Read every chunk from Postgres once. BM25 needs all the text in memory."""
    with psycopg.connect(DATABASE_URL) as conn:
        rows = conn.execute(
            "SELECT id, file, heading, start_line, end_line, text FROM chunks ORDER BY id"
        ).fetchall()
    return [
        {"id": i, "file": f, "heading": h, "start_line": s, "end_line": e, "text": t}
        for i, f, h, s, e, t in rows
    ]


# Built once when this file is imported (a second or two for 2,355 chunks).
CHUNKS = load_chunks()
CHUNKS_BY_ID = {c["id"]: c for c in CHUNKS}
# Same text the embeddings saw: heading + chunk text.
bm25 = BM25([f"{c['heading']}\n\n{c['text']}" for c in CHUNKS])


def vector_search(question: str, k: int = CANDIDATES) -> list[int]:
    """Search 1: ids of the k chunks closest in meaning (same query as Day 3, but returns ids)."""
    with psycopg.connect(DATABASE_URL) as conn:
        register_vector(conn)
        rows = conn.execute(
            "SELECT id FROM chunks ORDER BY embedding <=> %s LIMIT %s",
            (embedder.encode(question), k),
        ).fetchall()
    return [row[0] for row in rows]


def bm25_search(question: str, k: int = CANDIDATES) -> list[int]:
    """Search 2: ids of the k chunks with the highest keyword score (chunks scoring 0 are dropped)."""
    scores = bm25.scores(question)
    best = sorted(range(len(CHUNKS)), key=lambda i: scores[i], reverse=True)[:k]
    return [CHUNKS[i]["id"] for i in best if scores[i] > 0]


def reciprocal_rank_fusion(ranked_lists: list[list[int]], k: int = RRF_K) -> dict[int, float]:
    """Merge ranked lists. Each chunk gets 1 / (k + rank) from every list it is in.

    Rank 1 is best, so a small rank gives a BIG score. Appearing in both lists adds two scores.
    Returns {chunk_id: rrf_score}, sorted from highest score to lowest.
    """
    scores = defaultdict(float)
    for ranked in ranked_lists:
        for rank, chunk_id in enumerate(ranked, start=1):
            scores[chunk_id] += 1 / (k + rank)
    return dict(sorted(scores.items(), key=lambda item: item[1], reverse=True))


def retrieve(question: str, k: int = TOP_K) -> list[dict]:
    """Hybrid retrieval: run both searches, fuse them, return the top k chunks."""
    fused = reciprocal_rank_fusion([vector_search(question), bm25_search(question)])
    top_ids = list(fused)[:k]
    return [{**CHUNKS_BY_ID[i], "score": fused[i]} for i in top_ids]


def main() -> None:
    question = " ".join(sys.argv[1:]) or "How do I return a 404 error?"
    vector_ids = vector_search(question)
    bm25_ids = bm25_search(question)

    def rank_in(ids: list[int], chunk_id: int) -> str:
        return str(ids.index(chunk_id) + 1) if chunk_id in ids else "-"

    chunks = retrieve(question)
    print(f"Q: {question}\n")
    print("      RRF     vector  BM25   chunk")
    for i, c in enumerate(chunks, start=1):
        print(f"  [{i}] {c['score']:.4f}  {rank_in(vector_ids, c['id']):<7} "
              f"{rank_in(bm25_ids, c['id']):<6} {c['file']}:{c['start_line']}-{c['end_line']}")

    print("\nAnswer:\n" + generate(question, chunks))


if __name__ == "__main__":
    main()
