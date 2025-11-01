# matching.py
import torch
from embeddings import EmbeddingModel
from pdf_utils import extract_text_from_pdf
import os, re, json, hashlib

TEXT_CACHE_DIR = "cache_texts"
os.makedirs(TEXT_CACHE_DIR, exist_ok=True)


def _hash_file(path: str) -> str:
    return hashlib.md5(open(path, "rb").read()).hexdigest()


def get_cached_text(path: str) -> str:
    h = _hash_file(path)
    cache_path = os.path.join(TEXT_CACHE_DIR, f"{h}.txt")
    if os.path.exists(cache_path):
        return open(cache_path, encoding="utf-8").read()
    text = extract_text_from_pdf(path)
    with open(cache_path, "w", encoding="utf-8") as f:
        f.write(text)
    return text


def cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> float:
    return (a @ b.T).item()


def truncate_text(text: str, max_chars=2000) -> str:
    return text[:max_chars]


def extract_field(text: str, field_name: str):
    """Extract section text by simple regex patterns."""
    pattern = rf"{field_name}[:\-–]\s*(.+)"
    match = re.search(pattern, text, re.IGNORECASE)
    return match.group(1).strip() if match else ""


def section_similarity(resume_text: str, job_text: str, model: EmbeddingModel):
    # Extract sections
    resume_title = extract_field(resume_text, "должность") or extract_field(resume_text, "позиция")
    job_title = extract_field(job_text, "должность")

    resume_skills = extract_field(resume_text, "навыки")
    job_reqs = extract_field(job_text, "требования")

    resume_exp = extract_field(resume_text, "опыт работы")
    job_resp = extract_field(job_text, "обязанности")

    sections = {
        "title": (resume_title, job_title, 0.4),
        "skills": (resume_skills, job_reqs, 0.4),
        "experience": (resume_exp, job_resp, 0.2)
    }

    scores = {}
    for name, (r_text, j_text, weight) in sections.items():
        if not r_text or not j_text:
            scores[name] = 0.0
            continue
        r_emb = model.encode([r_text])
        j_emb = model.encode([j_text])
        sim = cosine_similarity(r_emb, j_emb)
        scores[name] = round(sim * 100, 2)

    total = sum(scores[k] * sections[k][2] for k in scores)
    return round(total, 2), scores


def generate_explanation(scores: dict):
    if not scores:
        return "Not enough data to explain match."
    best = max(scores, key=scores.get)
    worst = min(scores, key=scores.get)
    return (
        f"Strong alignment in {best} ({scores[best]}%). "
        f"Weak alignment in {worst} ({scores[worst]}%)."
    )


def score_pair(resume_path: str, job_path: str, model: EmbeddingModel):
    resume_text = truncate_text(get_cached_text(resume_path))
    job_text = truncate_text(get_cached_text(job_path))
    if not resume_text or not job_text:
        return {"score": 0.0, "sections": {}, "explanation": "Empty text"}

    total, section_scores = section_similarity(resume_text, job_text, model)
    explanation = generate_explanation(section_scores)
    return {"score": total, "section_scores": section_scores, "explanation": explanation}


def normalize_scores(scores):
    if not scores:
        return []
    min_s, max_s = min(scores), max(scores)
    if max_s == min_s:
        return [50 for _ in scores]
    return [(s - min_s) / (max_s - min_s) * 100 for s in scores]
