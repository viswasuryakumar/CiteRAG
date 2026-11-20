"""Day 4: measure retrieval quality on the golden set with recall@k and MRR.

Needs: the database filled by day3_ingest.py.
Run:   python eval/run_retrieval_eval.py            (saves results as "naive")
       python eval/run_retrieval_eval.py hybrid     (saves results under another name)
"""
import json
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR / "experiments"))
from day3_ask import retrieve  # noqa: E402  (the retrieval we are measuring)

GOLDEN_SET = PROJECT_DIR / "eval" / "golden_set.jsonl"
RESULTS_DIR = PROJECT_DIR / "eval" / "results"

K = 5           # we send the top 5 chunks to the LLM, so that's what we score
LOOK_DEEPER = 50  # also search further, only to report where a missed chunk was


def is_relevant(chunk: dict, relevant: list[dict]) -> bool:
    """A chunk counts as correct if it's from a labelled file and its lines overlap the labelled range.

    We label by file + line range, not chunk id, so the labels stay valid after re-chunking.
    """
    for target in relevant:
        start, end = target["lines"]
        same_file = chunk["file"] == target["file"]
        lines_overlap = chunk["start_line"] <= end and chunk["end_line"] >= start
        if same_file and lines_overlap:
            return True
    return False


def first_correct_rank(question: str, relevant: list[dict]) -> int | None:
    """Rank (1 = top) of the first correct chunk, or None if not in the top LOOK_DEEPER."""
    chunks = retrieve(question, k=LOOK_DEEPER)
    for rank, chunk in enumerate(chunks, start=1):
        if is_relevant(chunk, relevant):
            return rank
    return None


def main() -> None:
    run_name = sys.argv[1] if len(sys.argv) > 1 else "naive"
    items = [json.loads(line) for line in GOLDEN_SET.open(encoding="utf-8")]

    rows = []
    for item in items:
        rank = first_correct_rank(item["question"], item["relevant"])
        hit = rank is not None and rank <= K
        rows.append({
            "id": item["id"],
            "question": item["question"],
            "rank": rank,
            "hit": hit,                              # for recall@K
            "reciprocal_rank": 1 / rank if hit else 0.0,  # for MRR@K
        })

    recall = sum(r["hit"] for r in rows) / len(rows)
    mrr = sum(r["reciprocal_rank"] for r in rows) / len(rows)

    print(f"\nMisses (correct chunk not in top {K}):")
    for r in rows:
        if not r["hit"]:
            where = f"rank {r['rank']}" if r["rank"] else f"not in top {LOOK_DEEPER}"
            print(f"  #{r['id']:<3} {where:<16} {r['question']}")

    print(f"\n{run_name}: {len(rows)} questions")
    print(f"  recall@{K} = {recall:.2f}")
    print(f"  MRR@{K}    = {mrr:.2f}")

    RESULTS_DIR.mkdir(exist_ok=True)
    out = RESULTS_DIR / f"{run_name}.json"
    out.write_text(json.dumps({"recall_at_k": recall, "mrr_at_k": mrr, "k": K, "rows": rows}, indent=2))
    print(f"  saved to {out.relative_to(PROJECT_DIR)}")


if __name__ == "__main__":
    main()
