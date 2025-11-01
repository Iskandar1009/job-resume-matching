# Automated Resume Matcher (Qwen3-Embedding-0.6B)

## Run
pip install -r requirements.txt
uvicorn app:app --reload

## Endpoint
POST /match/
Form-data:
  jobs: list of job description PDFs
  resumes: list of candidate resume PDFs

## Output Example
{
  "job_description.pdf": [
    {
      "resume": "john_doe.pdf",
      "match_percent": 78.5,
      "section_scores": {
        "skills": 82.1,
        "experience": 77.3,
        "education": 65.4
      },
      "explanation": "Strong match in skills (82%) and experience (77%). Weaker alignment in education (65%).",
      "normalized_match_percent": 91.2
    }
  ]
}
