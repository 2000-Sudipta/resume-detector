# Talent Prism ATS

Talent Prism ATS is a Python + scikit-learn hiring dashboard that ranks resumes against a job description and explains why a candidate should be shortlisted, rejected, or reviewed manually.

## What the project does

- Reads one or many resumes from pasted text or uploaded `.txt`, `.md`, `.pdf`, and `.docx` files
- Requires a job description so scoring is tied to a real role
- Applies ATS-style logic for skills match, experience match, education fit, culture fit, salary alignment, and risk flags
- Uses a trained scikit-learn model to estimate shortlist probability
- Ranks candidates best to worst and shows quick compare mode
- Explains decisions in HR-friendly language instead of opaque AI output

## Tech stack

- Python
- Flask
- scikit-learn
- pandas
- numpy
- pypdf
- python-docx
- Vanilla HTML, CSS, and JavaScript

## How the ML logic works

The ATS model is trained on a synthetic dataset because real hiring-decision datasets are usually private.

### Features used by the model

- Skills match
- Must-have skill coverage
- Job-description keyword match
- Experience match
- Education match
- Culture match
- Location match
- Salary alignment
- Project strength
- Achievement score
- Authenticity score
- Resume completeness
- Stability score
- Red flag penalty

### Training flow

1. Synthetic candidate-job examples are generated with realistic hiring-style signals.
2. A Random Forest classifier learns which combinations usually lead to shortlist decisions.
3. The trained model is saved to `artifacts/hr_ats_ranker.joblib`.
4. During analysis, the app combines rule-based ATS scoring with ML shortlist probability.

## Website flow

1. HR uploads resumes or pastes multiple resume blocks.
2. HR adds the job description and optional filters.
3. The backend extracts role, skills, education, years of experience, and risk signals.
4. The ATS model predicts shortlist probability for each candidate.
5. The frontend renders:
   - ATS match score
   - shortlist / reject / review verdict
   - reasoning
   - key highlights
   - red flags
   - ranking
   - quick compare mode

## Run locally

```powershell
python -m venv .venv
python -m pip install --target .venv\Lib\site-packages -r requirements.txt
.\.venv\Scripts\python.exe train_model.py
.\.venv\Scripts\python.exe app.py
```

Open `http://127.0.0.1:5000`.

## Important note

This is a strong portfolio-grade ATS prototype. It supports practical HR screening logic, but it is still a decision-support system, not a replacement for human hiring judgment.
