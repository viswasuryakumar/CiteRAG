"""Day 7: two-stage retrieval = hybrid search for candidates, then a cross-encoder reranks them.

Stage 1 (fast):  hybrid search (Day 6) picks the top 20 candidate chunks.
Stage 2 (smart): a cross-encoder reads question + chunk TOGETHER and gives each pair one score.
                 We sort the 20 by that score and keep the best 5.

Needs: the database filled by day3_ingest.py (and OPENROUTER_API_KEY in .env to get an answer).
Run:   python experiments/day7_rerank.py "How do I return a 404 error?"
"""
import sys

from sentence_transformers import CrossEncoder

from day3_ask import TOP_K, generate
from day6_hybrid import retrieve as hybrid_retrieve

RERANK_CANDIDATES = 20   # how many hybrid results the cross-encoder rescores

# A different model from our embedder: it outputs one relevance score per (question, chunk) pair,
# not a vector. Picked after comparing three rerankers on the golden set (hybrid top 20 -> top 5):
#   cross-encoder/ms-marco-MiniLM-L6-v2   recall@5 0.76   0.45 s/question  (trained on web search only)
#   mixedbread-ai/mxbai-rerank-xsmall-v1  recall@5 0.92   1.4 s/question   <- chosen
#   BAAI/bge-reranker-base                recall@5 0.78   37 s/question on this CPU
reranker = CrossEncoder("mixedbread-ai/mxbai-rerank-xsmall-v1")


def retrieve(question: str, k: int = TOP_K) -> list[dict]:
    """Hybrid top 20 -> rerank with the cross-encoder -> return the top k."""
    candidates = hybrid_retrieve(question, k=max(k, RERANK_CANDIDATES))
    return rerank(question, candidates, k)


def rerank(question: str, candidates: list[dict], k: int) -> list[dict]:
    """Rescore the first RERANK_CANDIDATES candidates with the cross-encoder, return the top k."""
    for hybrid_rank, chunk in enumerate(candidates, start=1):
        chunk["hybrid_rank"] = hybrid_rank

    to_rerank = candidates[:RERANK_CANDIDATES]
    # One (question, chunk text) pair per candidate. The model joins each pair into one input:
    #   [CLS] question [SEP] heading + chunk text [SEP]  ->  one score (higher = more relevant)
    pairs = [(question, f"{c['heading']}\n\n{c['text']}") for c in to_rerank]
    scores = reranker.predict(pairs)

    for chunk, score in zip(to_rerank, scores):
        chunk["rerank_score"] = float(score)
    reranked = sorted(to_rerank, key=lambda c: c["rerank_score"], reverse=True)

    # Only the top 20 are reranked. If more than 20 are asked for (the eval looks up to 50 deep
    # to report where a miss was), the rest keep their hybrid order after the reranked ones.
    return (reranked + candidates[RERANK_CANDIDATES:])[:k]


def main() -> None:
    question = " ".join(sys.argv[1:]) or "How do I return a 404 error?"
    chunks = retrieve(question)

    print(f"Q: {question}\n")
    print("      rerank   hybrid  chunk")
    for i, c in enumerate(chunks, start=1):
        print(f"  [{i}] {c['rerank_score']:>6.2f}   {c['hybrid_rank']:<7} "
              f"{c['file']}:{c['start_line']}-{c['end_line']}")

    print("\nAnswer:\n" + generate(question, chunks))


if __name__ == "__main__":
    main()
