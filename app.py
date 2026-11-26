"""Entry point for Vercel: it looks for a FastAPI instance named `app` in a root file like app.py.

Locally you can still run: uvicorn api.main:app --port 8000
"""
import os

# Vercel's disk is read-only except /tmp, so the embedding model is downloaded there.
# Must be set before huggingface_hub is imported (it reads HF_HOME once, at import).
if os.getenv("VERCEL"):
    os.environ.setdefault("HF_HOME", "/tmp/huggingface")

from api.main import app  # noqa: E402,F401
