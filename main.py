from fastapi import FastAPI, UploadFile
from typing import List
import tempfile, json, torch
from pdf_utils import extract_text_from_pdf
from embeddings import EmbeddingModel

app = FastAPI(title="AI Resume Matcher")

model = None 


def get_model():
    global model
    if model is None:
        print("Loading model (one-time)...")
        model = EmbeddingModel()
        print("Model loaded!")
    return model


def cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> float:
    return (a @ b.T).item()


def score_pair(resume_path: str, job_path: str) -> float:
    resume_text = extract_text_from_pdf(resume_path)
    job_text = extract_text_from_pdf(job_path)
    if not resume_text or not job_text:
        return 0.0
    mdl = get_model()
    emb_resume = mdl.encode([resume_text])
    emb_job = mdl.encode([job_text])
    sim = cosine_similarity(emb_resume, emb_job)
    return round(sim * 100, 2)


@app.post("/match/")
async def match_resumes(jobs: List[UploadFile], resumes: List[UploadFile]):
    results = {}
    for job in jobs:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as jf:
            jf.write(await job.read())
            job_path = jf.name
        job_results = []
        for resume in resumes:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as rf:
                rf.write(await resume.read())
                resume_path = rf.name
            percent = score_pair(resume_path, job_path)
            job_results.append({
                "resume": resume.filename,
                "match_percent": percent
            })
        results[job.filename] = job_results
    return json.dumps(results, ensure_ascii=False, indent=2)
