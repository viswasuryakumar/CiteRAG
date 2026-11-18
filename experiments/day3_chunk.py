"""Day 3, step 1: split the FastAPI docs into chunks.

Reads:  data/fastapi/docs/en/docs/**/*.md
Writes: data/chunks.jsonl  (one JSON object per line, one chunk per object)
Run:    python experiments/day3_chunk.py
"""
import json
import re
from pathlib import Path

from transformers import AutoTokenizer

PROJECT_DIR = Path(__file__).resolve().parent.parent
REPO_DIR = PROJECT_DIR / "data" / "fastapi"
DOCS_DIR = REPO_DIR / "docs" / "en" / "docs"
OUTPUT_FILE = PROJECT_DIR / "data" / "chunks.jsonl"

# all-MiniLM-L6-v2 reads at most 256 tokens and silently ignores the rest.
# 200 leaves room for the heading we put in front of each chunk when embedding.
MAX_TOKENS = 200
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

# The changelog is huge and full of version numbers, not explanations.
SKIP_FILES = {"release-notes.md"}

# Matches "# Title", "## Title", "### Title" and drops the "{ #anchor }" suffix.
HEADING = re.compile(r"^(#{1,3}) (.+?)(?: \{ #[\w-]+ \})?\s*$")



def count_tokens(text: str) -> int:
    return len(tokenizer.encode(text, add_special_tokens=False))


def make_paragraph(headings: list[str], numbered_lines: list[tuple[int, str]]) -> dict:
    """Bundle some consecutive lines into one paragraph record."""
    return {
        "heading": " > ".join(headings),             # e.g. "Path Parameters > Data validation"
        "start_line": numbered_lines[0][0],
        "end_line": numbered_lines[-1][0],
        "text": "\n".join(text for _, text in numbered_lines),
    }


def parse_paragraphs(lines: list[str]) -> list[dict]:
    """Split a markdown file into paragraphs, each tagged with its heading path and line numbers."""
    paragraphs = []
    headings = []        # the headings we are currently under
    current = []         # (line_number, text) pairs of the paragraph being built
    current_tokens = 0
    in_code = False      # are we inside a ``` code block?

    for line_number, line in enumerate(lines, start=1):
        stripped = line.strip()
        if stripped.startswith("```"):
            in_code = not in_code

        # Inside a code block, "# ..." is a Python comment and blank lines don't end anything.
        heading = None if in_code else HEADING.match(line)
        ends_paragraph = heading or (stripped == "" and not in_code)

        if ends_paragraph:
            if current:
                paragraphs.append(make_paragraph(headings, current))
                current, current_tokens = [], 0
            if heading:
                level = len(heading.group(1))                  # "##" -> 2
                headings = headings[: level - 1] + [heading.group(2)]
        elif not stripped.startswith("{*"):
            # "{* ../../docs_src/... *}" lines pull in code examples at build time; skip them.
            line_tokens = count_tokens(line)
            # One paragraph bigger than the limit (usually a long code block): cut it here.
            if current and current_tokens + line_tokens > MAX_TOKENS:
                paragraphs.append(make_paragraph(headings, current))
                current, current_tokens = [], 0
            current.append((line_number, line))
            current_tokens += line_tokens

    if current:
        paragraphs.append(make_paragraph(headings, current))
    return paragraphs


def merge(paragraphs: list[dict]) -> dict:
    """Join consecutive paragraphs from the same section into one chunk."""
    return {
        "heading": paragraphs[0]["heading"],
        "start_line": paragraphs[0]["start_line"],
        "end_line": paragraphs[-1]["end_line"],
        "text": "\n\n".join(p["text"] for p in paragraphs),
    }


def pack_into_chunks(paragraphs: list[dict]) -> list[dict]:
    """Greedily add paragraphs to a chunk until it would pass MAX_TOKENS or the section changes."""
    chunks = []
    current = []
    current_tokens = 0
    for paragraph in paragraphs:
        paragraph_tokens = count_tokens(paragraph["text"])
        same_section = current and current[-1]["heading"] == paragraph["heading"]
        if same_section and current_tokens + paragraph_tokens <= MAX_TOKENS:
            current.append(paragraph)
            current_tokens += paragraph_tokens
        else:
            if current:
                chunks.append(merge(current))
            current, current_tokens = [paragraph], paragraph_tokens
    if current:
        chunks.append(merge(current))
    return chunks


def main() -> None:
    all_chunks = []
    for path in sorted(DOCS_DIR.rglob("*.md")):
        if path.name in SKIP_FILES:
            continue
        lines = path.read_text(encoding="utf-8").splitlines()
        file_name = path.relative_to(REPO_DIR).as_posix()   # e.g. docs/en/docs/tutorial/path-params.md
        for chunk in pack_into_chunks(parse_paragraphs(lines)):
            all_chunks.append({"file": file_name, **chunk})

    with OUTPUT_FILE.open("w", encoding="utf-8") as f:
        for chunk in all_chunks:
            f.write(json.dumps(chunk) + "\n")

    sizes = [count_tokens(c["text"]) for c in all_chunks]
    print(f"chunks: {len(all_chunks)}")
    print(f"average size: {sum(sizes) // len(sizes)} tokens, largest: {max(sizes)} tokens")
    print(f"over {MAX_TOKENS} tokens: {sum(s > MAX_TOKENS for s in sizes)}")
    example = next(c for c in all_chunks if c["file"].endswith("tutorial/path-params.md"))
    print("\nexample chunk:")
    print(json.dumps(example, indent=2))


if __name__ == "__main__":
    main()
