from __future__ import annotations

from flask import Flask, jsonify, render_template, request

from resume_truth_detector.ats_engine import ATSResumeEngine, PRESET_SKILLS, build_candidate_inputs
from resume_truth_detector.ats_modeling import load_ats_model_bundle
from resume_truth_detector.file_parsers import SUPPORTED_EXTENSIONS, extract_text_from_upload


app = Flask(__name__)
model_bundle = load_ats_model_bundle()
engine = ATSResumeEngine(model_bundle)


def _parse_optional_float(value: str | None) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None


@app.get("/")
def index():
    return render_template(
        "index.html",
        model_metrics=model_bundle["metrics"],
        supported_extensions=sorted(SUPPORTED_EXTENSIONS),
        preset_skills=PRESET_SKILLS,
    )


@app.get("/health")
def health():
    return jsonify({"status": "ok", "model_metrics": model_bundle["metrics"]})


@app.post("/analyze")
def analyze():
    pasted_resumes = request.form.get("resume_texts", "").strip()
    job_description = request.form.get("job_description", "").strip()
    must_have_skills = request.form.get("must_have_skills", "").strip()
    selected_skills = request.form.getlist("must_have_skill_chip")
    min_experience = _parse_optional_float(request.form.get("min_experience")) or 0.0
    location_preference = request.form.get("location_preference", "").strip()
    salary_min = _parse_optional_float(request.form.get("salary_min"))
    salary_max = _parse_optional_float(request.form.get("salary_max"))
    culture_keywords = request.form.get("culture_keywords", "").strip()
    priority_mode = request.form.get("priority_mode", "balanced").strip()

    uploaded_resumes = []
    for index, uploaded in enumerate(request.files.getlist("resume_files"), start=1):
        if not uploaded or not uploaded.filename:
            continue
        uploaded_text = extract_text_from_upload(uploaded)
        uploaded_resumes.append({"label": uploaded.filename or f"Uploaded Candidate {index}", "text": uploaded_text})

    if not job_description:
        return jsonify({"error": "Job description is required so the ATS can score against the target role."}), 400

    candidates = build_candidate_inputs(pasted_resumes, uploaded_resumes)
    if not candidates:
        return jsonify({"error": "Upload at least one resume or paste one or more resume blocks first."}), 400

    job_profile = engine.build_job_profile(
        job_description=job_description,
        must_have_skills=must_have_skills,
        checkbox_skills=selected_skills,
        min_experience_filter=min_experience,
        location_preference=location_preference,
        salary_min=salary_min,
        salary_max=salary_max,
        culture_keywords=culture_keywords,
        priority_mode=priority_mode,
    )
    result = engine.analyze(candidates, job_profile)
    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True)
