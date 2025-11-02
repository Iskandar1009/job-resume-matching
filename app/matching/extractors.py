import re
import logging

logger = logging.getLogger(__name__)

HEADER_ALIASES = {
    "position": [
        r"Желаемая\s+(?:позиция|должность).*?:?\s*(?:\n)?([^\n]+)",
        r"Целевая\s+(?:позиция|роль).*?:?\s*(?:\n)?([^\n]+)",
        r"Desired\s+(?:position|role|job).*?:?\s*(?:\n)?([^\n]+)",
        r"Objective\s*:?\s*(?:\n)?([^\n]+)",
        r"(?m)^(?:[A-ZА-ЯЁ][\w\-\s\/&]{2,60})$",
    ],
    "experience_block": [
        r"Опыт\s+работы.*?:(.*?)(?=Образование|Сертификаты|Навыки|$)",
        r"Work\s+Experience.*?:(.*?)(?=Education|Skills|Certificates|$)",
    ],
    "skills": [
        r"Навыки.*?:(.*?)(?=Опыт работы|Образование|$)",
        r"Skills.*?:(.*?)(?=Experience|Education|$)",
    ],
    "education": [
        r"Образование.*?:(.*?)(?=Сертификаты|Навыки|$)",
        r"Education.*?:(.*?)(?=Certificates|Skills|Experience|$)",
    ],
    "about": [
        r"(Обо мне|Профиль).*?:(.*?)(?=Навыки|Опыт работы|$)",
        r"(Profile|Summary|About\s+me).*?:(.*?)(?=Skills|Experience|Education|$)",
    ],
}


def _first_match(text: str, patterns, dotall=True) -> str:
    flags = re.IGNORECASE | (re.DOTALL if dotall else 0)
    for pat in patterns:
        m = re.search(pat, text, flags)
        if m:
            groups = [g for g in m.groups() if g]
            if groups:
                val = re.sub(r"\s+", " ", groups[-1].strip())
                if val:
                    return val
    return ""


def extract_resume_sections(text: str) -> dict:
    sections = {}
    text = text.strip()
    sections["position"] = _first_match(text, HEADER_ALIASES["position"], dotall=False)
    sections["about"] = _first_match(text, HEADER_ALIASES["about"])
    sections["skills"] = _first_match(text, HEADER_ALIASES["skills"])
    sections["experience"] = _first_match(text, HEADER_ALIASES["experience_block"])
    sections["education"] = _first_match(text, HEADER_ALIASES["education"])

    if not sections["position"]:
        for line in text.splitlines()[:15]:
            line = line.strip()
            if re.search(r"(тел|email|citizenship|гражданство|возраст)", line, re.IGNORECASE):
                continue
            if 2 <= len(line.split()) <= 6 and line[0].isupper():
                sections["position"] = line
                break

    if not sections["skills"]:
        m = re.search(r"Дополнительная информация\s*:?\s*(.*)", text, re.IGNORECASE | re.DOTALL)
        if m:
            sections["skills"] = re.sub(r"\s+", " ", m.group(1).strip())

    if not sections["skills"] and sections.get("about"):
        sections["skills"] = sections["about"]

    for k in sections:
        if sections[k]:
            sections[k] = re.sub(r"\s+", " ", sections[k]).strip()

    return sections


def extract_job_sections(text: str) -> dict:
    sections = {}
    c = re.search(r'Название компании.*?:\s*(.*?)(?=Название вакансии|Локация|Требования|$)', text, re.I | re.S)
    if c:
        sections['company'] = re.sub(r'\s+', ' ', c.group(1).strip())
    t = re.search(r'Название вакансии:\s*(.*?)(?=Локация|Требования|$)', text, re.I | re.S)
    if t:
        sections['title'] = re.sub(r'\s+', ' ', t.group(1).strip())
    l = re.search(r'Локация:\s*(.*?)(?=Требования|Обязанности|$)', text, re.I | re.S)
    if l:
        sections['location'] = re.sub(r'\s+', ' ', l.group(1).strip())
    r = re.search(r'Требования.*?:\s*(.*?)(?=Обязанности|$)', text, re.I | re.S)
    if r:
        sections['requirements'] = re.sub(r'\s+', ' ', r.group(1).strip())
    rs = re.search(r'Обязанности.*?:\s*(.*?)$', text, re.I | re.S)
    if rs:
        sections['responsibilities'] = re.sub(r'\s+', ' ', rs.group(1).strip())

    if 'requirements' in sections:
        e = re.search(r'(Высшее образование.*?)', sections['requirements'], re.I)
        if e:
            sections['education'] = e.group(1).strip()

    return sections
