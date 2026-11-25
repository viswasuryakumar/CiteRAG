"""Day 10: the final answer step. Every fact cites its source, and the model says "I don't know"
when the retrieved chunks don't contain the answer.

Pipeline: query rewriting + hybrid search (Day 9, our best retriever) -> LLM answer with [n] markers
-> we turn each [n] into a file + line range + GitHub link, and check the markers are valid.

Needs: the database filled by day3_ingest.py, and OPENROUTER_API_KEY in .env.
Run:   python experiments/day10_answer.py "How do I return a 404 error?"
"""
import re
import sys

from day3_ask import MODEL, build_prompt, client
from day9_rewrite import retrieve

# The FastAPI commit we indexed. Pinning it keeps links pointing at the exact lines we read,
# even after the repo changes.
FASTAPI_COMMIT = "b2831196bfee3eb9ce303a3c6440f2a104832954"

I_DONT_KNOW = "I don't know based on the FastAPI docs and code."

SYSTEM_PROMPT = f"""You answer questions about FastAPI using only the numbered excerpts provided
(from its documentation and its source code).

Rules:
1. End every sentence that states a fact with the number of the excerpt that supports it, like [2].
   If several excerpts support it, write [1][3].
2. Use only the excerpts. Do not add facts from your own knowledge, even if you are sure they are true.
3. If the excerpts do not contain the answer, reply with exactly this sentence and nothing else:
   {I_DONT_KNOW}"""


def github_link(chunk: dict) -> str:
    """A link that opens the exact lines on GitHub, e.g. .../handling-errors.md#L22-L40."""
    return (f"https://github.com/fastapi/fastapi/blob/{FASTAPI_COMMIT}/{chunk['file']}"
            f"#L{chunk['start_line']}-L{chunk['end_line']}")


def answer(question: str) -> dict:
    """Retrieve, generate, then turn the [n] markers into real sources."""
    chunks = retrieve(question)
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_prompt(question, chunks)},
        ],
    )
    text = response.choices[0].message.content.strip()

    # Every [n] in the answer. A number outside 1..len(chunks) points at nothing: an invalid citation.
    cited = sorted({int(n) for n in re.findall(r"\[(\d+)\]", text)})
    valid = [n for n in cited if 1 <= n <= len(chunks)]
    invalid = [n for n in cited if n not in valid]

    sources = [
        {"n": n, "file": chunks[n - 1]["file"], "start_line": chunks[n - 1]["start_line"],
         "end_line": chunks[n - 1]["end_line"], "url": github_link(chunks[n - 1])}
        for n in valid
    ]
    return {
        "answer": text,
        "abstained": text.startswith("I don't know"),
        "sources": sources,               # only the chunks the answer actually used
        "invalid_citations": invalid,
        "uncited": not cited and not text.startswith("I don't know"),   # answered, but cited nothing
        "chunks": chunks,
    }


def main() -> None:
    question = " ".join(sys.argv[1:]) or "How do I return a 404 error?"
    result = answer(question)

    print(f"Q: {question}\n\n{result['answer']}\n")
    if result["sources"]:
        print("Sources:")
        for s in result["sources"]:
            print(f"  [{s['n']}] {s['file']}:{s['start_line']}-{s['end_line']}\n      {s['url']}")
    if result["invalid_citations"]:
        print(f"Warning: cites excerpts that don't exist: {result['invalid_citations']}")
    if result["uncited"]:
        print("Warning: the answer cites no excerpts.")


if __name__ == "__main__":
    main()
