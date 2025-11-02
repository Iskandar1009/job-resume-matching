import torch
from app.models.embeddings import EmbeddingModel

def cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> float:
    a_norm = a / a.norm(dim=-1, keepdim=True)
    b_norm = b / b.norm(dim=-1, keepdim=True)
    return float((a_norm * b_norm).sum(dim=-1).mean())


def section_similarity(resume_sections: dict, job_sections: dict, model: EmbeddingModel):
    resume_all_skills = "\n".join([
        resume_sections.get("skills", ""),
        resume_sections.get("about", ""),
    ]).strip()
    resume_full_exp = "\n".join([
        resume_sections.get("experience", ""),
        resume_sections.get("about", ""),
    ]).strip()

    pairs = {
        "title": (resume_sections.get("position", ""), job_sections.get("title", ""), 0.5),
        "skills": (resume_all_skills, job_sections.get("requirements", ""), 0.2),
        "experience": (resume_full_exp, job_sections.get("responsibilities", ""), 0.2),
        "education": (resume_sections.get("education", ""), job_sections.get("education", ""), 0.1),
    }

    scores = {}
    for key, (r_text, j_text, weight) in pairs.items():
        if not r_text.strip() or not j_text.strip():
            scores[key] = 0.0
            continue
        r_emb = model.encode([r_text])
        j_emb = model.encode([j_text])
        sim = cosine_similarity(r_emb, j_emb)
        scores[key] = round(sim * 100, 2)

    total = sum(scores[k] * pairs[k][2] for k in scores)
    return round(total, 2), scores
