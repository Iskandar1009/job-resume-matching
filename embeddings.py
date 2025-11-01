# embeddings.py
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
import hashlib
import json
import os

CACHE_DIR = "cache_embeddings"
os.makedirs(CACHE_DIR, exist_ok=True)

class EmbeddingModel:
    def __init__(self, model_name="Qwen/Qwen3-Embedding-0.6B"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

    def _hash_text(self, text: str) -> str:
        return hashlib.md5(text.encode("utf-8")).hexdigest()

    def _load_cache(self, text: str):
        h = self._hash_text(text)
        path = os.path.join(CACHE_DIR, f"{h}.pt")
        if os.path.exists(path):
            return torch.load(path)
        return None

    def _save_cache(self, text: str, emb: torch.Tensor):
        h = self._hash_text(text)
        path = os.path.join(CACHE_DIR, f"{h}.pt")
        torch.save(emb, path)

    def encode(self, texts):
        embs = []
        for text in texts:
            cached = self._load_cache(text)
            if cached is not None:
                embs.append(cached)
                continue
            with torch.no_grad():
                inputs = self.tokenizer(
                    text, padding=True, truncation=True,
                    max_length=512, return_tensors="pt"
                ).to(self.device)
                outputs = self.model(**inputs)
                emb = outputs.last_hidden_state.mean(dim=1)
                emb = F.normalize(emb, p=2, dim=1).cpu()
                self._save_cache(text, emb)
                embs.append(emb)
        return torch.cat(embs, dim=0)
