"""Day 1 demo: turn sentences into vectors and compare them.

Run:  python experiments/day01_embeddings.py
"""
import numpy as np
from sentence_transformers import SentenceTransformer

# A small, fast embedding model. Each sentence becomes a list of 384 numbers.
model = SentenceTransformer("all-MiniLM-L6-v2")

sentences = [
    "How do I declare a path parameter in FastAPI?",   # 0
    "Path parameters are defined in the route URL.",   # 1  (same topic as 0)
    "Use Depends() to inject a database session.",     # 2  (different topic)
    "My cat sleeps on the keyboard all day.",          # 3  (unrelated)
]

embeddings = model.encode(sentences)  # numpy array, one row per sentence
print("shape:", embeddings.shape)     # (4, 384) -> 4 sentences, 384 dimensions
print("first 5 numbers of sentence 0:", embeddings[0][:5].round(3))


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine of the angle between two vectors: 1 = same direction, 0 = unrelated."""
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


print("\nSimilarity of sentence 0 to each sentence:")
for i, sentence in enumerate(sentences):
    score = cosine_similarity(embeddings[0], embeddings[i])
    print(f"  {score:.3f}  {sentence}")
