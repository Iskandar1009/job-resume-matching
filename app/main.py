from fastapi import FastAPI, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import List
import tempfile
import os
import logging

from starlette.responses import FileResponse

from app.models.embeddings import EmbeddingModel
from app.matching.scoring import score_pair, normalize_scores
from app.utils.pdf_utils import is_valid_pdf

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="AI Resume Matcher")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

INDEX_PATH = os.path.join(os.path.dirname(__file__), "..", "index.html")

model = None
def get_model():
    global model
    if model is None:
        logger.info("Loading embedding model...")
        model = EmbeddingModel()
        logger.info("Model loaded successfully.")
    return model


@app.post("/match/")
async def match_resumes(resumes: List[UploadFile], jobs: List[UploadFile]):
    """
    Match resumes against job descriptions.
    Returns JSON with match scores for each job-resume pair sorted by match_percent.
    """
    if not resumes or len(resumes) == 0 or not jobs or len(jobs) == 0:
        raise HTTPException(status_code=400, detail="Must provide at least one resume and one job description")

    mdl = get_model()
    results = {}
    temp_files = []

    try:
        for job in jobs:
            if not job.filename.lower().endswith(".pdf"):
                logger.warning(f"Skipping non-PDF job file: {job.filename}")
                continue

            job_content = await job.read()
            await job.seek(0)

            if len(job_content) < 50:
                raise HTTPException(status_code=400, detail=f"Job file appears empty or corrupted: {job.filename}")

            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as jf:
                jf.write(job_content)
                jf.flush()
                job_path = jf.name
                temp_files.append(job_path)

            if not is_valid_pdf(job_path):
                raise HTTPException(status_code=400, detail=f"Invalid PDF file: {job.filename}")

            job_scores = []

            for resume in resumes:
                if not resume.filename.lower().endswith(".pdf"):
                    logger.warning(f"Skipping non-PDF resume file: {resume.filename}")
                    continue

                resume_content = await resume.read()
                await resume.seek(0)

                if len(resume_content) < 50:
                    raise HTTPException(status_code=400, detail=f"Resume file appears empty or corrupted: {resume.filename}")

                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as rf:
                    rf.write(resume_content)
                    rf.flush()
                    resume_path = rf.name
                    temp_files.append(resume_path)

                if not is_valid_pdf(resume_path):
                    raise HTTPException(status_code=400, detail=f"Invalid PDF file: {resume.filename}")

                try:
                    res = score_pair(resume_path, job_path, mdl)
                    job_scores.append({
                        "resume": resume.filename,
                        "match_percent": res["score"],
                        "section_scores": res["section_scores"],
                        "explanation": res["explanation"]
                    })
                except Exception as e:
                    logger.error(f"Error scoring {resume.filename} vs {job.filename}: {e}")
                    job_scores.append({
                        "resume": resume.filename,
                        "match_percent": 0.0,
                        "section_scores": {},
                        "explanation": f"Error processing files: {str(e)}"
                    })

            job_scores = sorted(job_scores, key=lambda x: x["match_percent"], reverse=True)
            results[job.filename] = job_scores

    finally:
        for temp_file in temp_files:
            try:
                if os.path.exists(temp_file):
                    os.unlink(temp_file)
            except Exception as e:
                logger.warning(f"Failed to delete temp file {temp_file}: {e}")

    return JSONResponse(content=results)

@app.get("/")
async def serve_index():
    return FileResponse(INDEX_PATH)