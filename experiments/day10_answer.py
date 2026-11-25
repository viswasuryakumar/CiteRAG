"""Day 10: the final answer step. Every fact cites its source, and the model says "I don't know"
when the retrieved chunks don't contain the answer.
Day 11: the answer can also be streamed token by token (answer_events), for the web API.

Pipeline: query rewriting + hybrid search (Day 9, our best retriever) -> LLM answer with [n] markers
-> we turn each [n] into a file + line range + GitHub link, and check the markers are valid.

Needs: the database filled by day3_ingest.py, and OPENROUTER_API_KEY in .env.
Run:   python experiments/day10_answer.py "How do I return a 404 error?"
"""
import re
import sys
from collections.abc import Iterator

from langsmith import traceable

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


def source_info(n: int, chunk: dict) -> dict:
    """What the user sees for excerpt [n]: where it is, not its full text."""
    return {"n": n, "file": chunk["file"], "start_line": chunk["start_line"],
            "end_line": chunk["end_line"], "url": github_link(chunk)}


def parse_citations(text: str, chunks: list[dict]) -> dict:
    """Turn the [n] markers in a finished answer into sources, and flag problems."""
    # Every [n] in the answer. A number outside 1..len(chunks) points at nothing: an invalid citation.
    cited = sorted({int(n) for n in re.findall(r"\[(\d+)\]", text)})
    valid = [n for n in cited if 1 <= n <= len(chunks)]
    abstained = text.strip().startswith("I don't know")
    return {
        "answer": text.strip(),
        "abstained": abstained,
        "sources": [source_info(n, chunks[n - 1]) for n in valid],   # only the ones actually used
        "invalid_citations": [n for n in cited if n not in valid],
        "uncited": not cited and not abstained,   # answered, but cited nothing
    }


def stream_generate(question: str, chunks: list[dict]) -> Iterator[str]:
    """The LLM call with stream=True: yields the answer a few characters at a time."""
    stream = client.chat.completions.create(
        model=MODEL,
        stream=True,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_prompt(question, chunks)},
        ],
    )
    for event in stream:
        if event.choices and event.choices[0].delta.content:
            yield event.choices[0].delta.content


@traceable(name="CiteRAG answer")   # in LangSmith, the whole request shows as one trace
def answer_events(question: str) -> Iterator[tuple[str, object]]:
    """The full pipeline as a stream of (event name, data) pairs:
         ("sources", [...])  first: the 5 retrieved excerpts, so the page can show them early
         ("token", "...")    many times: pieces of the answer as the LLM writes them
         ("done", {...})     last: citations checked, abstained or not
    """
    chunks = retrieve(question)
    yield "sources", [source_info(n, c) for n, c in enumerate(chunks, start=1)]

    text = ""
    for token in stream_generate(question, chunks):
        text += token
        yield "token", token

    yield "done", parse_citations(text, chunks)


def answer(question: str) -> dict:
    """Not streamed: run the whole pipeline and return the final result (used by the eval)."""
    for event, data in answer_events(question):
        if event == "done":
            return data


def main() -> None:
    question = " ".join(sys.argv[1:]) or "How do I return a 404 error?"
    print(f"Q: {question}\n")

    for event, data in answer_events(question):
        if event == "token":
            print(data, end="", flush=True)   # appears word by word, like in a chat app
        elif event == "done":
            result = data

    print()
    if result["sources"]:
        print("\nSources:")
        for s in result["sources"]:
            print(f"  [{s['n']}] {s['file']}:{s['start_line']}-{s['end_line']}\n      {s['url']}")
    if result["invalid_citations"]:
        print(f"Warning: cites excerpts that don't exist: {result['invalid_citations']}")
    if result["uncited"]:
        print("Warning: the answer cites no excerpts.")


if __name__ == "__main__":
    main()
