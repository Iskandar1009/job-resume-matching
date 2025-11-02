import os
import hashlib
import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer

CACHE_DIR = "cache/embeddings"
os.makedirs(CACHE_DIR, exist_ok=True)

class EmbeddingModel:
    def __init__(self, model_name="Qwen/Qwen3-Embedding-0.6B", **kwargs):
        """
        model_name: the HF model identifier for the Qwen3 embedding model
        kwargs: extra keyword args to pass to SentenceTransformer (e.g. revision, model_kwargs, tokenizer_kwargs)
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading model: {model_name} on {self.device}")
        self.model = SentenceTransformer(model_name, device=self.device, **kwargs)
        print("Model loaded successfully.")

    def _hash_text(self, text: str) -> str:
        return hashlib.md5(text.encode("utf-8")).hexdigest()

    def _cache_path(self, text: str) -> str:
        return os.path.join(CACHE_DIR, f"{self._hash_text(text)}.pt")

    def _load_cache(self, text: str):
        path = self._cache_path(text)
        if os.path.exists(path):
            return torch.load(path)
        return None

    def _save_cache(self, text: str, emb: torch.Tensor):
        torch.save(emb, self._cache_path(text))

    def encode(self, texts):
        embs = []
        for text in texts:
            text = text.strip()
            if not text:
                embs.append(torch.zeros(1, 1024))
                continue
            cached = self._load_cache(text)
            if cached is not None:
                embs.append(cached)
                continue
            with torch.no_grad():
                emb = self.model.encode(text, convert_to_tensor=True, device=self.device)
                emb = F.normalize(emb, p=2, dim=-1).cpu()
                self._save_cache(text, emb)
                embs.append(emb.unsqueeze(0))
        return torch.cat(embs, dim=0)
