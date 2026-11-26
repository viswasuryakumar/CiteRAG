"""Day 13: the question embedder without PyTorch, so the API fits on a free host.

Same model (all-MiniLM-L6-v2), exported to ONNX: a portable format that ONNX Runtime can run
without the framework the model was trained in. PyTorch + sentence-transformers need ~1.5 GB
of RAM; ONNX Runtime + this model need a fraction of that.

We copy what sentence-transformers does for this model (see its modules.json):
  1. tokenize (WordPiece, cut at 256 tokens)
  2. run the transformer -> one 384-number vector per token
  3. mean pooling: average the token vectors, ignoring padding
  4. normalize to length 1
"""
import os

import numpy as np
import onnxruntime as ort
from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer

MODEL_REPO = "sentence-transformers/all-MiniLM-L6-v2"
MODEL_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"   # pinned: the exact files we tested
MAX_TOKENS = 256   # the model's max sequence length


class OnnxEmbedder:
    def __init__(self):
        # Downloaded once from Hugging Face, then read from the local cache.
        # On a server with a read-only disk, set HF_HOME to a writable folder like /tmp.
        model_path = hf_hub_download(MODEL_REPO, "onnx/model.onnx", revision=MODEL_REVISION)
        tokenizer_path = hf_hub_download(MODEL_REPO, "tokenizer.json", revision=MODEL_REVISION)

        self.tokenizer = Tokenizer.from_file(tokenizer_path)
        self.tokenizer.enable_truncation(max_length=MAX_TOKENS)
        options = ort.SessionOptions()
        options.intra_op_num_threads = int(os.getenv("ONNX_THREADS", "0"))   # 0 = let ONNX decide
        self.session = ort.InferenceSession(model_path, options, providers=["CPUExecutionProvider"])

    def encode(self, text: str) -> np.ndarray:
        """One text -> one normalized 384-number vector (same as SentenceTransformer.encode)."""
        tokens = self.tokenizer.encode(text)                     # adds [CLS] ... [SEP]
        input_ids = np.array([tokens.ids], dtype=np.int64)
        attention_mask = np.array([tokens.attention_mask], dtype=np.int64)
        token_type_ids = np.zeros_like(input_ids)

        token_vectors = self.session.run(None, {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
        })[0]                                                     # shape (1, tokens, 384)

        # Mean pooling: average over real tokens only (mask is 1 for real, 0 for padding).
        mask = attention_mask[..., None].astype(np.float32)
        vector = (token_vectors * mask).sum(axis=1) / mask.sum(axis=1)
        vector = vector[0] / np.linalg.norm(vector[0])            # length 1, like the Normalize module
        return vector.astype(np.float32)
