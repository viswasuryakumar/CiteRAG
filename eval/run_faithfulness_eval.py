"""Day 5: measure faithfulness - does the answer only say things the retrieved chunks support?

For each golden question: retrieve -> generate an answer -> an LLM judge splits the answer into
claims and checks each claim against the retrieved chunks.
    faithfulness = supported claims / total claims

Needs: the database filled by day3_ingest.py, and OPENROUTER_API_KEY in .env.
Run:   python eval/run_faithfulness_eval.py              (all 50 questions, saved as "naive")
       python eval/run_faithfulness_eval.py naive 10     (only the first 10 questions)
       python eval/run_faithfulness_eval.py hybrid       (answers built from hybrid retrieval)
Cost:  two LLM calls per question; about $0.05 for all 50 with Claude Haiku 5.5.
"""
import importlib
import json
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR / "experiments"))
from day3_ask import build_prompt, client, generate  # noqa: E402

GOLDEN_SET = PROJECT_DIR / "eval" / "golden_set.jsonl"
RESULTS_DIR = PROJECT_DIR / "eval" / "results"

# Run name -> the file whose retrieve() feeds the answers.
RETRIEVERS = {"naive": "day3_ask", "hybrid": "day6_hybrid", "hybrid_rerank": "day7_rerank"}

# The judge. Using the same cheap model that wrote the answers keeps costs low, but a model
# grading its own family can be lenient; swap in a stronger model here to compare.
JUDGE_MODEL = "anthropic/claude-haiku-5.5"

JUDGE_INSTRUCTIONS = """You check whether an answer is supported by documentation excerpts.

1. Split the ANSWER into short, standalone factual claims.
   Skip sentences that only say the excerpts don't contain something, and skip citation markers like [2].
2. For each claim, decide if the EXCERPTS directly support it. Use only the excerpts, not your own knowledge.
   If a claim is true in general but not stated in the excerpts, it is NOT supported.

Reply with only JSON, no other text, in this shape:
{"claims": [{"claim": "...", "supported": true}, {"claim": "...", "supported": false}]}"""


def judge(question: str, chunks: list[dict], answer: str) -> list[dict]:
    """The LLM-as-judge step: returns [{"claim": ..., "supported": bool}, ...]."""
    excerpts_and_question = build_prompt(question, chunks)   # the same excerpts the answer saw
    response = client.chat.completions.create(
        model=JUDGE_MODEL,
        messages=[
            {"role": "system", "content": JUDGE_INSTRUCTIONS},
            {"role": "user", "content": f"EXCERPTS:\n{excerpts_and_question}\n\nANSWER:\n{answer}"},
        ],
    )
    text = response.choices[0].message.content
    # Models sometimes wrap JSON in ```json fences or add a sentence; take the outermost {...}.
    json_text = text[text.find("{"): text.rfind("}") + 1]
    return json.loads(json_text)["claims"]


def main() -> None:
    run_name = sys.argv[1] if len(sys.argv) > 1 else "naive"
    limit = int(sys.argv[2]) if len(sys.argv) > 2 else None
    retrieve = importlib.import_module(RETRIEVERS[run_name]).retrieve
    items = [json.loads(line) for line in GOLDEN_SET.open(encoding="utf-8")][:limit]

    rows = []
    for item in items:
        question = item["question"]
        chunks = retrieve(question)
        answer = generate(question, chunks)
        claims = judge(question, chunks, answer)

        supported = sum(c["supported"] for c in claims)
        # An answer that only says "I don't know" has no claims: nothing to be unfaithful about.
        score = supported / len(claims) if claims else None
        rows.append({"id": item["id"], "question": question, "answer": answer,
                     "claims": claims, "faithfulness": score})

        shown = f"{score:.2f}" if score is not None else "n/a (no claims)"
        print(f"#{item['id']:<3} {shown:<16} {supported}/{len(claims)} claims supported  {question}")

    scored = [r["faithfulness"] for r in rows if r["faithfulness"] is not None]
    average = sum(scored) / len(scored)

    print("\nUnsupported claims (possible hallucinations):")
    for r in rows:
        for c in r["claims"]:
            if not c["supported"]:
                print(f"  #{r['id']:<3} {c['claim']}")

    print(f"\n{run_name}: faithfulness = {average:.2f}  "
          f"(over {len(scored)} answers; {len(rows) - len(scored)} had no claims)")

    RESULTS_DIR.mkdir(exist_ok=True)
    out = RESULTS_DIR / f"faithfulness_{run_name}.json"
    out.write_text(json.dumps({"faithfulness": average, "rows": rows}, indent=2))
    print(f"saved to {out.relative_to(PROJECT_DIR)}")


if __name__ == "__main__":
    main()
