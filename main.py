# app.py
from fastapi import FastAPI, UploadFile
from typing import List
import tempfile, json
from embeddings import EmbeddingModel
from matching import score_pair
from matching import normalize_scores
from fastapi.responses import JSONResponse

app = FastAPI(title="AI Resume Matcher")

model = None

def get_model():
    global model
    if model is None:
        print("Loading Qwen3-Embedding-0.6B...")
        model = EmbeddingModel()
        print("Model loaded.")
    return model

@app.post("/match/")
async def match_resumes(jobs: List[UploadFile], resumes: List[UploadFile]):
    mdl = get_model()
    results = {}

    for job in jobs:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as jf:
            jf.write(await job.read())
            job_path = jf.name

        job_scores = []
        for resume in resumes:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as rf:
                rf.write(await resume.read())
                resume_path = rf.name

            res = score_pair(resume_path, job_path, mdl)
            job_scores.append({
                "resume": resume.filename,
                "match_percent": res["score"],
                "section_scores": res["section_scores"],
                "explanation": res["explanation"]
            })

        # normalize for better ranking evaluation
        normalized = normalize_scores([r["match_percent"] for r in job_scores])
        for i, val in enumerate(normalized):
            job_scores[i]["normalized_match_percent"] = round(val, 2)

        results[job.filename] = job_scores

    return JSONResponse(content=results)





