"""Day 8: chunk FastAPI's Python source code by its structure, using the AST.

Docs are split at headings and blank lines, but in code a blank line can sit inside a function,
so that would cut functions in half. Instead Python's own parser (the `ast` module) tells us
exactly where every function and class starts and ends.

Rules:
  - every top-level function or class becomes one chunk, if it fits in MAX_TOKENS
  - a class that is too big is split: its header (class line, docstring, attributes),
    then each method on its own
  - a function that is too big is cut into pieces of consecutive lines
  - heading = file and name path, e.g. "fastapi/routing.py > APIRouter > include_router"

Needs: the FastAPI source in data/fastapi/fastapi (git sparse-checkout add fastapi).
Run:   python experiments/day8_code_chunk.py   -> writes data/code_chunks.jsonl
"""
import ast
import json

from day3_chunk import MAX_TOKENS, PROJECT_DIR, REPO_DIR, count_tokens

SOURCE_DIR = REPO_DIR / "fastapi"
OUTPUT_FILE = PROJECT_DIR / "data" / "code_chunks.jsonl"

DEFINITIONS = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)


def start_line(node: ast.AST) -> int:
    """First line of a function/class, including decorators like @app.get(...) above it."""
    return min([node.lineno] + [d.lineno for d in node.decorator_list])


def make_chunk(file: str, heading: str, start: int, end: int, lines: list[str]) -> dict:
    """Same shape as the doc chunks, so ingest, search and citations work unchanged."""
    return {"file": file, "heading": heading, "start_line": start, "end_line": end,
            "text": "\n".join(lines[start - 1:end])}


def split_by_lines(file: str, heading: str, start: int, end: int, lines: list[str]) -> list[dict]:
    """Cut lines start..end into pieces of at most MAX_TOKENS, never splitting a line."""
    chunks = []
    piece_start, piece_tokens = start, 0
    for line_no in range(start, end + 1):
        line_tokens = count_tokens(lines[line_no - 1])
        if piece_tokens and piece_tokens + line_tokens > MAX_TOKENS:
            chunks.append(make_chunk(file, heading, piece_start, line_no - 1, lines))
            piece_start, piece_tokens = line_no, 0
        piece_tokens += line_tokens
    chunks.append(make_chunk(file, heading, piece_start, end, lines))
    # Drop pieces that are only blank lines.
    return [c for c in chunks if c["text"].strip()]


def chunk_definition(node: ast.AST, parents: list[str], file: str, lines: list[str]) -> list[dict]:
    """Turn one function or class into chunks (calls itself for the methods of a big class)."""
    heading = " > ".join([file, *parents, node.name])
    start, end = start_line(node), node.end_lineno   # line numbers straight from the AST

    # Fits? Then the whole function/class is one chunk.
    if count_tokens("\n".join(lines[start - 1:end])) <= MAX_TOKENS:
        return [make_chunk(file, heading, start, end, lines)]

    if isinstance(node, ast.ClassDef):
        inner = [child for child in node.body if isinstance(child, DEFINITIONS)]
        if inner:
            # Header = from the class line up to the first method.
            header_end = start_line(inner[0]) - 1
            chunks = split_by_lines(file, heading, start, header_end, lines)
            for child in inner:
                chunks += chunk_definition(child, [*parents, node.name], file, lines)
            return chunks

    # A big function (or a big class with no methods): cut it into line pieces.
    return split_by_lines(file, heading, start, end, lines)


def main() -> None:
    chunks = []
    for path in sorted(SOURCE_DIR.rglob("*.py")):
        source = path.read_text(encoding="utf-8")
        lines = source.splitlines()
        file = path.relative_to(REPO_DIR).as_posix()   # e.g. "fastapi/routing.py"

        tree = ast.parse(source)   # the whole file as a tree of nodes
        # Only top-level functions and classes. Imports and module constants are skipped.
        for node in tree.body:
            if isinstance(node, DEFINITIONS):
                chunks += chunk_definition(node, [], file, lines)

    with OUTPUT_FILE.open("w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(chunk) + "\n")

    sizes = [count_tokens(c["text"]) for c in chunks]
    print(f"{len(chunks)} code chunks from {len(list(SOURCE_DIR.rglob('*.py')))} files")
    print(f"average {sum(sizes) / len(sizes):.0f} tokens, largest {max(sizes)}, "
          f"over {MAX_TOKENS}: {sum(s > MAX_TOKENS for s in sizes)}")
    print(f"wrote {OUTPUT_FILE.relative_to(PROJECT_DIR)}")


if __name__ == "__main__":
    main()
