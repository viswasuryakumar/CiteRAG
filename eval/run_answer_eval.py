"""Day 10: score the final answers on citations and on saying "I don't know".

For each golden question, run the full pipeline (experiments/day10_answer.py) and check:
  Answerable questions (60):
    - false refusal: it said "I don't know" although the answer was findable      (want low)
    - correct citation: at least one cited source overlaps the labelled lines      (want high)
    - uncited: it answered but cited nothing                                       (want 0)
    - invalid citation: it cited [n] for an excerpt that doesn't exist             (want 0)
  Unanswerable questions (10, empty "relevant" in the golden set):
    - abstained: it said "I don't know" instead of making something up             (want high)

Needs: the database filled by day3_ingest.py, and OPENROUTER_API_KEY in .env.
Run:   python eval/run_answer_eval.py
Cost:  two LLM calls per question (rewrite + answer); a few cents for all 70.
"""
import json
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR / "experiments"))
from day10_answer import answer  # noqa: E402
from run_retrieval_eval import is_relevant  # noqa: E402

GOLDEN_SET = PROJECT_DIR / "eval" / "golden_set.jsonl"
RESULTS_FILE = PROJECT_DIR / "eval" / "results" / "answers.json"


def main() -> None:
    items = [json.loads(line) for line in GOLDEN_SET.open(encoding="utf-8")]

    rows = []
    for item in items:
        result = answer(item["question"])
        answerable = bool(item["relevant"])
        row = {
            "id": item["id"],
            "question": item["question"],
            "answerable": answerable,
            "answer": result["answer"],
            "abstained": result["abstained"],
            "sources": result["sources"],
            "invalid_citations": result["invalid_citations"],
            "uncited": result["uncited"],
            # Did it cite at least one chunk that our labels say is correct?
            "correct_citation": answerable and any(is_relevant(s, item["relevant"]) for s in result["sources"]),
        }
        rows.append(row)
        status = "I don't know" if row["abstained"] else f"cited {[s['n'] for s in row['sources']]}"
        print(f"#{item['id']:<3} {'answerable' if answerable else 'UNANSWERABLE':<13} {status:<16} {item['question'][:60]}")

    answerable_rows = [r for r in rows if r["answerable"]]
    unanswerable_rows = [r for r in rows if not r["answerable"]]
    answered = [r for r in answerable_rows if not r["abstained"]]

    def share(count: int, total: int) -> str:
        return f"{count}/{total} = {count / total:.2f}" if total else "n/a"

    print(f"\nAnswerable questions ({len(answerable_rows)}):")
    print(f"  false refusals (said I don't know): {share(sum(r['abstained'] for r in answerable_rows), len(answerable_rows))}")
    print(f"  cited a correct source:             {share(sum(r['correct_citation'] for r in answered), len(answered))}  (of the ones it answered)")
    print(f"  answered without citing:            {sum(r['uncited'] for r in answered)}")
    print(f"  invalid citation numbers:           {sum(bool(r['invalid_citations']) for r in rows)}")
    print(f"Unanswerable questions ({len(unanswerable_rows)}):")
    print(f"  correctly said I don't know:        {share(sum(r['abstained'] for r in unanswerable_rows), len(unanswerable_rows))}")

    print("\nAnswered an unanswerable question (possible made-up answers):")
    for r in unanswerable_rows:
        if not r["abstained"]:
            print(f"  #{r['id']} {r['question']}\n      -> {r['answer'][:150]}")
    print("Refused an answerable question:")
    for r in answerable_rows:
        if r["abstained"]:
            print(f"  #{r['id']} {r['question']}")

    RESULTS_FILE.parent.mkdir(exist_ok=True)
    RESULTS_FILE.write_text(json.dumps(rows, indent=2))
    print(f"\nsaved to {RESULTS_FILE.relative_to(PROJECT_DIR)}")


if __name__ == "__main__":
    main()
