import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

class EmbeddingModel:
    def __init__(self, model_name="intfloat/e5-base-v2"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

    def _avg_pool(self, hidden, mask):
        hidden = hidden.masked_fill(~mask[..., None].bool(), 0.0)
        summed = hidden.sum(dim=1)
        counts = mask.sum(dim=1)[..., None]
        return summed / counts

    def encode(self, texts):
        with torch.no_grad():
            batch = self.tokenizer(
                texts, padding=True, truncation=True,
                max_length=512, return_tensors="pt"
            ).to(self.device)
            out = self.model(**batch)
            pooled = self._avg_pool(out.last_hidden_state, batch.attention_mask)
            return F.normalize(pooled, p=2, dim=1).cpu()
