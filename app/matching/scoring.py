import os
import logging
from app.utils.cache_utils import get_cached_text, truncate_text
from .extractors import extract_resume_sections, extract_job_sections
from .similarity import section_similarity

logger = logging.getLogger(__name__)

def generate_explanation(scores: dict, total: float) -> str:
    if not scores:
        return "Недостаточно данных для объяснения совпадения."
    non_zero = {k: v for k, v in scores.items() if v > 0}
    if not non_zero:
        return "Совпадений не обнаружено. Проверьте формат документов."
    labels = {
        "title": "название должности",
        "skills": "навыки и требования",
        "experience": "опыт и обязанности",
        "education": "образование"
    }
    parts = []
    if total >= 60:
        parts.append(f"Отличное совпадение ({total}%).")
    elif total >= 40:
        parts.append(f"Хорошее совпадение ({total}%).")
    elif total >= 20:
        parts.append(f"Среднее совпадение ({total}%).")
    else:
        parts.append(f"Низкое совпадение ({total}%).")
    best = max(non_zero, key=non_zero.get)
    parts.append(f"Сильнее всего: «{labels.get(best, best)}» ({scores[best]}%).")
    if len(non_zero) > 1:
        worst = min(non_zero, key=non_zero.get)
        parts.append(f"Слабее всего: «{labels.get(worst, worst)}» ({scores[worst]}%).")
    return " ".join(parts)


def score_pair(resume_path: str, job_path: str, model) -> dict:
    try:
        resume_text = truncate_text(get_cached_text(resume_path))
        job_text = truncate_text(get_cached_text(job_path))
    except Exception as e:
        return {"score": 0.0, "section_scores": {}, "explanation": f"Ошибка извлечения текста: {e}"}

    if not resume_text or not job_text:
        return {"score": 0.0, "section_scores": {}, "explanation": "Не удалось извлечь текст из документов."}

    resume_sections = extract_resume_sections(resume_text)
    job_sections = extract_job_sections(job_text)
    total, section_scores = section_similarity(resume_sections, job_sections, model)
    explanation = generate_explanation(section_scores, total)
    return {"score": total, "section_scores": section_scores, "explanation": explanation}


def normalize_scores(scores: list) -> list:
    if not scores:
        return []
    min_s, max_s = min(scores), max(scores)
    if max_s == min_s:
        return [50.0 for _ in scores]
    return [(s - min_s) / (max_s - min_s) * 100 for s in scores]
