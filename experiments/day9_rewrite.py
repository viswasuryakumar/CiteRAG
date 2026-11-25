"""Day 9: query rewriting. An LLM rewrites the user's question into the words the docs/code use.

Users say "a value from the URL"; the docs say "path parameter". Searching with a rewritten
query can find chunks the user's own wording misses.

The risk: a rewrite can drop exact words (like `OAuth2PasswordBearer`) that BM25 needs.
So we search with BOTH the original and the rewritten question (vector + BM25 for each),
and merge all 4 ranked lists with RRF.

Needs: the database filled by day3_ingest.py, and OPENROUTER_API_KEY in .env.
Run:   python experiments/day9_rewrite.py "How do I read a value from the URL, like the 5 in /items/5?"
"""
import sys
from functools import lru_cache

from langsmith import traceable

from day3_ask import MODEL, TOP_K, client, generate
from day6_hybrid import CHUNKS_BY_ID, bm25_search, reciprocal_rank_fusion, vector_search
from day7_rerank import RERANK_CANDIDATES, rerank

REWRITE_PROMPT = """You rewrite questions about FastAPI into a search query for its documentation and source code.
Use the technical terms the FastAPI docs and code would use.
Keep every exact name from the question unchanged: classes, functions, parameters, headers, numbers.
Reply with only the rewritten query, one line, no explanation."""


@traceable(name="rewrite query")
@lru_cache(maxsize=None)   # same question -> reuse the rewrite instead of paying for another LLM call
def rewrite(question: str) -> str:
    response = client.chat.completions.create(
        model=MODEL,
        temperature=0,   # as repeatable as possible, so eval runs are comparable
        messages=[
            {"role": "system", "content": REWRITE_PROMPT},
            {"role": "user", "content": question},
        ],
    )
    return response.choices[0].message.content.strip()


@traceable(name="hybrid search", run_type="retriever")
def search_both(question: str, k: int) -> list[dict]:
    """Vector + BM25 for the original AND the rewritten question, merged with RRF."""
    ranked_lists = []
    for query in (question, rewrite(question)):
        ranked_lists.append(vector_search(query))
        ranked_lists.append(bm25_search(query))
    fused = reciprocal_rank_fusion(ranked_lists)
    return [{**CHUNKS_BY_ID[i], "score": fused[i]} for i in list(fused)[:k]]


def retrieve(question: str, k: int = TOP_K) -> list[dict]:
    """Hybrid search with query rewriting, no reranker."""
    return search_both(question, k)


def retrieve_rerank(question: str, k: int = TOP_K) -> list[dict]:
    """Hybrid search with query rewriting, then the reranker.

    The reranker scores against the ORIGINAL question: relevance is judged by what the user asked.
    """
    candidates = search_both(question, max(k, RERANK_CANDIDATES))
    return rerank(question, candidates, k)


def main() -> None:
    question = " ".join(sys.argv[1:]) or "How do I read a value from the URL, like the 5 in /items/5?"
    print(f"Q:         {question}\nRewritten: {rewrite(question)}\n")
    chunks = retrieve(question)
    for i, c in enumerate(chunks, start=1):
        print(f"  [{i}] {c['file']}:{c['start_line']}-{c['end_line']}  {c['heading'][:60]}")
    print("\nAnswer:\n" + generate(question, chunks))


if __name__ == "__main__":
    main()
