"""Day 1 build task: a tiny semantic search engine.

Run:  python experiments/day01_exercise.py
"""
import numpy as np
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

DOCS = [
    "Path parameters are declared with curly braces in the route path, like /items/{item_id}.",
    "Query parameters are function arguments that are not part of the path.",
    "Use a Pydantic model as a function argument to declare a JSON request body.",
    "Depends() lets you declare dependencies that FastAPI resolves for each request.",
    "Use BackgroundTasks to run work after returning the response.",
    "CORSMiddleware lets browsers on other origins call your API.",
    "Raise HTTPException to return an error response with a status code.",
    "Use response_model to filter and validate the data your endpoint returns.",
    "OAuth2PasswordBearer extracts a bearer token from the Authorization header.",
    "TestClient lets you call your app in tests without starting a server.",
]


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine of the angle between two vectors: 1 = same meaning, 0 = unrelated."""
    # dot(a, b) grows with both the angle AND the vector lengths.
    # Dividing by the two lengths removes length, leaving only the angle.
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def search(query: str, docs: list[str], k: int = 3) -> list[tuple[float, str]]:
    """Return the k docs most similar to the query, best first, as (score, doc)."""
    # 1. Embed every doc: one row of 384 numbers per doc -> shape (len(docs), 384).
    doc_embeddings = model.encode(docs)

    # 2. Embed the query with the SAME model, so it lands in the same vector space.
    query_embedding = model.encode(query)  # shape (384,)

    # 3. Score every doc against the query.
    scored = [
        (cosine_similarity(query_embedding, doc_embedding), doc)
        for doc_embedding, doc in zip(doc_embeddings, docs)
    ]

    # 4. Highest score first, then keep only the top k.
    scored.sort(key=lambda pair: pair[0], reverse=True)
    return scored[:k]


if __name__ == "__main__":
    for query in [
        "how do I return a 404?",
        "run something after the request finishes",
        "how can my React frontend call the API?",
    ]:
        print(f"\nQ: {query}")
        for score, doc in search(query, DOCS):
            print(f"  {score:.3f}  {doc}")
