"""Day 1-2: semantic search in plain Python.

Run:  python experiments/day1_search.py
"""
import numpy as np
from sentence_transformers import SentenceTransformer

# 1. Load the embedding model (downloads once, then runs locally).
model = SentenceTransformer("all-MiniLM-L6-v2")

# 2. Twelve sentences on three topics.
SENTENCES = [
    # Web APIs / FastAPI
    "Path parameters are declared with curly braces in the FastAPI route.",
    "A POST request sends JSON data to the server in the request body.",
    "The server returns status code 404 when a resource is not found.",
    "Use an API key in the header to authenticate each request.",
    # Cooking
    "Salt the boiling water before adding the spaghetti.",
    "Knead the bread dough for ten minutes until it is smooth.",
    "Roast the chicken in the oven until the skin is golden.",
    "Chop the onions finely and fry them in butter.",
    # Sports
    "The striker scored a goal in the final minute of the match.",
    "She won the marathon after running for just over two hours.",
    "The tennis player served an ace to win the set.",
    "The basketball team practiced free throws before the playoffs.",
]

# 3. Embed all sentences once, up front. One row per sentence.
sentence_embeddings = model.encode(SENTENCES)
print("shape:", sentence_embeddings.shape)  # expect (12, 384)


# 4. Cosine similarity by hand, so the math is visible.
def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    dot_product = np.dot(a, b)       # multiply matching positions, add them up
    length_a = np.linalg.norm(a)     # length of a: sqrt(sum of squares)
    length_b = np.linalg.norm(b)
    return float(dot_product / (length_a * length_b))


# 5. Search: embed the query, score every sentence, return the top k.
def search(query: str, k: int = 3) -> list[tuple[float, str]]:
    query_embedding = model.encode(query)  # shape (384,)

    scored = []
    for sentence, sentence_embedding in zip(SENTENCES, sentence_embeddings):
        score = cosine_similarity(query_embedding, sentence_embedding)
        scored.append((score, sentence))

    scored.sort(key=lambda pair: pair[0], reverse=True)  # highest score first
    return scored[:k]


# 6. Three test queries.
if __name__ == "__main__":
    queries = [
        ("shares words", "How do I declare path parameters in FastAPI?"),
        ("same meaning, no shared words", "seasoning noodles while they cook"),
        ("off-topic", "What is the capital of France?"),
    ]
    for label, query in queries:
        print(f"\n[{label}] Q: {query}")
        for score, sentence in search(query):
            print(f"  {score:.3f}  {sentence}")
