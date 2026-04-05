from __future__ import annotations

import re
from dataclasses import dataclass

from .ats_modeling import predict_shortlist_probability


PRESET_SKILLS = [
    "Python",
    "SQL",
    "Machine Learning",
    "Deep Learning",
    "NLP",
    "Data Analysis",
    "scikit-learn",
    "TensorFlow",
    "PyTorch",
    "AWS",
    "Azure",
    "Docker",
    "Kubernetes",
    "Flask",
    "FastAPI",
    "React",
]

SKILL_PATTERNS: dict[str, list[str]] = {
    "Python": ["python"],
    "SQL": ["sql", "postgresql", "mysql", "sqlite", "sql server"],
    "Machine Learning": ["machine learning", "ml", "predictive modeling"],
    "Deep Learning": ["deep learning", "neural network", "cnn", "rnn"],
    "NLP": ["nlp", "natural language processing", "sentiment analysis", "text classification", "transformer"],
    "Data Analysis": ["data analysis", "analytics", "eda", "data cleaning", "visualization"],
    "scikit-learn": ["scikit-learn", "sklearn"],
    "TensorFlow": ["tensorflow"],
    "PyTorch": ["pytorch"],
    "AWS": ["aws", "amazon web services"],
    "Azure": ["azure"],
    "GCP": ["gcp", "google cloud"],
    "Docker": ["docker", "containerization"],
    "Kubernetes": ["kubernetes", "k8s"],
    "Flask": ["flask"],
    "FastAPI": ["fastapi"],
    "Django": ["django"],
    "React": ["react", "react.js"],
    "JavaScript": ["javascript", "js"],
    "TypeScript": ["typescript", "ts"],
    "Power BI": ["power bi"],
    "Tableau": ["tableau"],
    "Spark": ["spark", "pyspark"],
    "Airflow": ["airflow"],
    "MLOps": ["mlops", "mlflow", "model deployment", "model monitoring"],
    "APIs": ["api", "rest api", "microservice"],
}

ROLE_PATTERNS = {
    "Data Scientist": ["data scientist", "data science"],
    "ML Engineer": ["ml engineer", "machine learning engineer", "ai engineer"],
    "Backend Engineer": ["backend engineer", "backend developer", "api developer"],
    "Frontend Engineer": ["frontend engineer", "frontend developer", "ui developer"],
    "Cloud Engineer": ["cloud engineer", "devops engineer", "platform engineer"],
    "Data Analyst": ["data analyst", "business analyst"],
}

ROLE_FAMILY_MAP = {
    "Data Scientist": "Data",
    "ML Engineer": "ML",
    "Backend Engineer": "Backend",
    "Frontend Engineer": "Frontend",
    "Cloud Engineer": "Cloud",
    "Data Analyst": "Data",
}

EDUCATION_LEVELS = {
    "Doctorate": ["phd", "doctorate"],
    "Masters": ["master", "m.tech", "ms", "msc", "mba"],
    "Bachelors": ["bachelor", "b.tech", "be ", "b.e", "bsc", "bca"],
    "Diploma": ["diploma"],
}

EDUCATION_RANK = {"Unknown": 0, "Diploma": 1, "Bachelors": 2, "Masters": 3, "Doctorate": 4}

ACTION_VERBS = [
    "built",
    "implemented",
    "designed",
    "deployed",
    "improved",
    "optimized",
    "reduced",
    "increased",
    "led",
    "created",
    "developed",
]
PROJECT_TERMS = ["project", "projects", "portfolio", "hackathon", "research", "internship", "product"]
MEASUREMENT_TERMS = ["%", "percent", "$", "kpi", "revenue", "latency", "accuracy", "f1", "auc", "users"]
CULTURE_HINTS = {
    "startup": ["startup", "0 to 1", "ownership", "fast-paced"],
    "corporate": ["corporate", "enterprise", "process", "stakeholder"],
    "collaboration": ["collaborative", "cross-functional", "teamwork", "stakeholder"],
    "leadership": ["lead", "mentored", "managed", "ownership"],
}

EMAIL_PATTERN = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
LOCATION_PATTERN = re.compile(r"\b(remote|hybrid|onsite|on-site|bangalore|bengaluru|mumbai|delhi|pune|hyderabad|chennai|kolkata)\b", re.IGNORECASE)
SALARY_PATTERN = re.compile(r"(\d+(?:\.\d+)?)\s*(?:lpa|lakhs?|k|usd|inr|per annum)", re.IGNORECASE)
YEARS_PATTERN = re.compile(r"(\d+(?:\.\d+)?)\+?\s*(?:years?|yrs?)", re.IGNORECASE)
YEAR_RANGE_PATTERN = re.compile(r"((?:19|20)\d{2})\s*(?:-|to)\s*((?:19|20)\d{2}|present|current|now)", re.IGNORECASE)
SECTION_PATTERN = re.compile(r"^\s*(summary|skills|experience|projects|education|certifications?)\s*:?\s*$", re.IGNORECASE | re.MULTILINE)
TEXT_SPLIT_PATTERN = re.compile(r"^\s*(?:---+|===+)\s*(?:candidate(?:\s*\d+)?)?\s*(?:---+|===+)?\s*$", re.IGNORECASE)


@dataclass
class JobProfile:
    role: str
    role_family: str
    description: str
    required_skills: list[str]
    preferred_skills: list[str]
    required_experience: float
    required_education: str
    min_experience_filter: float
    location_preference: str
    salary_min: float | None
    salary_max: float | None
    culture_keywords: list[str]
    priority_mode: str


def _term_pattern(term: str) -> re.Pattern[str]:
    return re.compile(rf"(?<!\w){re.escape(term.lower())}(?!\w)")


def _contains_term(text: str, term: str) -> bool:
    return bool(_term_pattern(term).search(text.lower()))


def _count_term(text: str, term: str) -> int:
    return len(_term_pattern(term).findall(text.lower()))


def _split_csv_terms(text: str) -> list[str]:
    return [item.strip() for item in re.split(r"[\n,;/]+", text) if item.strip()]


def extract_skills(text: str) -> list[str]:
    lowered = text.lower()
    matches: list[str] = []
    for skill, aliases in SKILL_PATTERNS.items():
        if any(_contains_term(lowered, alias) for alias in aliases):
            matches.append(skill)
    return sorted(matches)


def extract_role(text: str) -> str:
    lowered = text.lower()
    for role, aliases in ROLE_PATTERNS.items():
        if any(_contains_term(lowered, alias) for alias in aliases):
            return role
    return "General Hiring"


def extract_education_level(text: str) -> str:
    lowered = text.lower()
    best = "Unknown"
    best_rank = 0
    for level, aliases in EDUCATION_LEVELS.items():
        if any(_contains_term(lowered, alias) for alias in aliases):
            rank = EDUCATION_RANK[level]
            if rank > best_rank:
                best = level
                best_rank = rank
    return best


def split_resume_blocks(text: str) -> list[str]:
    blocks: list[str] = []
    current: list[str] = []
    for line in (text or "").splitlines():
        if TEXT_SPLIT_PATTERN.match(line):
            block = "\n".join(current).strip()
            if block:
                blocks.append(block)
            current = []
            continue
        current.append(line)

    final_block = "\n".join(current).strip()
    if final_block:
        blocks.append(final_block)
    return blocks


def normalize_must_have_skills(manual_skills: str, checkbox_skills: list[str], job_description: str) -> list[str]:
    manual_matches = []
    for item in _split_csv_terms(manual_skills):
        for skill, aliases in SKILL_PATTERNS.items():
            if item.lower() == skill.lower() or any(_contains_term(item, alias) for alias in aliases):
                manual_matches.append(skill)
                break
    extracted = extract_skills(job_description)
    merged = []
    for skill in [*checkbox_skills, *manual_matches, *extracted]:
        if skill and skill not in merged:
            merged.append(skill)
    return merged[:8]


def estimate_experience_years(text: str) -> float:
    explicit_years = max([float(item) for item in YEARS_PATTERN.findall(text)] or [0.0])
    covered_years: set[int] = set()
    for start, end in YEAR_RANGE_PATTERN.findall(text):
        start_year = int(start)
        end_year = 2026 if end.lower() in {"present", "current", "now"} else int(end)
        for year in range(start_year, min(end_year, 2026) + 1):
            covered_years.add(year)
    range_years = float(max(len(covered_years) - 1, 0))
    return round(max(explicit_years, range_years), 1)


def extract_candidate_name(text: str, fallback: str) -> str:
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if len(stripped.split()) <= 5 and not any(char.isdigit() for char in stripped):
            if stripped.lower() not in {"summary", "skills", "experience", "projects", "education"}:
                return stripped
    return fallback


def extract_salary_expectation(text: str) -> float | None:
    match = SALARY_PATTERN.search(text)
    if not match:
        return None
    value = float(match.group(1))
    token = match.group(0).lower()
    if "k" in token and "lpa" not in token:
        return value * 1000
    return value


def extract_location(text: str) -> str:
    match = LOCATION_PATTERN.search(text)
    if not match:
        return "Not specified"
    return match.group(1).title()


def extract_email(text: str) -> str:
    match = EMAIL_PATTERN.search(text)
    return match.group(0) if match else "Not found"


def score_location(preference: str, candidate_location: str, resume_text: str) -> float:
    if not preference:
        return 100.0
    lowered_pref = preference.lower()
    lowered_resume = resume_text.lower()
    if "remote" in lowered_pref and "remote" in lowered_resume:
        return 100.0
    if lowered_pref in lowered_resume or lowered_pref in candidate_location.lower():
        return 100.0
    if candidate_location == "Not specified":
        return 60.0
    return 35.0


def score_salary(min_salary: float | None, max_salary: float | None, expected_salary: float | None) -> float:
    if min_salary is None and max_salary is None:
        return 100.0
    if expected_salary is None:
        return 65.0
    if min_salary is not None and expected_salary < min_salary:
        return 70.0
    if max_salary is not None and expected_salary <= max_salary:
        return 100.0
    if max_salary is not None and expected_salary <= max_salary * 1.1:
        return 72.0
    return 35.0


def score_education(required_level: str, candidate_level: str) -> float:
    if required_level == "Unknown":
        return 100.0
    candidate_rank = EDUCATION_RANK[candidate_level]
    required_rank = EDUCATION_RANK[required_level]
    if candidate_rank >= required_rank:
        return 100.0
    gap = required_rank - candidate_rank
    return max(25.0, 100.0 - gap * 28.0)


def score_experience(required_years: float, candidate_years: float) -> float:
    if required_years <= 0:
        return min(100.0, 65.0 + candidate_years * 8.0)
    ratio = candidate_years / required_years
    if ratio >= 1.1:
        return 100.0
    if ratio >= 1:
        return 92.0
    if ratio >= 0.8:
        return 75.0 + ratio * 10.0
    if ratio >= 0.5:
        return 48.0 + ratio * 20.0
    return max(15.0, ratio * 60.0)


def score_culture(culture_keywords: list[str], resume_text: str) -> float:
    if not culture_keywords:
        return 100.0
    lowered = resume_text.lower()
    matched = 0
    for keyword in culture_keywords:
        if _contains_term(lowered, keyword):
            matched += 1
            continue
        for alias in CULTURE_HINTS.get(keyword.lower(), []):
            if _contains_term(lowered, alias):
                matched += 1
                break
    return round((matched / max(len(culture_keywords), 1)) * 100, 1)


def score_resume_completeness(text: str) -> float:
    section_hits = len(SECTION_PATTERN.findall(text))
    skill_hits = len(extract_skills(text))
    score = 35 + min(section_hits, 5) * 10 + min(skill_hits, 8) * 3
    return float(min(score, 100))


def score_achievements(text: str) -> float:
    lowered = text.lower()
    quantified_lines = sum(
        1
        for line in text.splitlines()
        if any(token in line.lower() for token in MEASUREMENT_TERMS) or re.search(r"\d", line)
    )
    action_hits = sum(lowered.count(term) for term in ACTION_VERBS)
    score = quantified_lines * 16 + action_hits * 6
    return float(min(score, 100))


def score_project_strength(text: str, target_skills: list[str]) -> float:
    lowered = text.lower()
    project_hits = sum(lowered.count(term) for term in PROJECT_TERMS)
    related_skill_hits = sum(_count_term(text, alias) for skill in target_skills for alias in SKILL_PATTERNS.get(skill, [skill]))
    score = project_hits * 16 + related_skill_hits * 6
    return float(min(score, 100))


def score_authenticity(text: str, matched_skills: list[str], years: float) -> float:
    if not matched_skills:
        return 25.0
    lowered = text.lower()
    support = 0.0
    max_support = len(matched_skills) * 3
    for skill in matched_skills:
        aliases = SKILL_PATTERNS.get(skill, [skill])
        mentions = sum(_count_term(text, alias) for alias in aliases)
        if mentions >= 2:
            support += 1
        if any(_contains_term(lowered, verb) for verb in ACTION_VERBS) and mentions >= 1:
            support += 1
        if any(_contains_term(lowered, term) for term in PROJECT_TERMS) and mentions >= 1:
            support += 1
    if years >= 2:
        support += 1
        max_support += 1
    return round((support / max(max_support, 1)) * 100, 1)


def score_stability(text: str) -> tuple[float, list[str]]:
    ranges = []
    for start, end in YEAR_RANGE_PATTERN.findall(text):
        start_year = int(start)
        end_year = 2026 if end.lower() in {"present", "current", "now"} else int(end)
        ranges.append((start_year, end_year))
    if not ranges:
        return 68.0, []

    ranges.sort()
    flags: list[str] = []
    gaps = 0
    for index in range(1, len(ranges)):
        prev_end = ranges[index - 1][1]
        start_year = ranges[index][0]
        if start_year - prev_end > 1:
            gaps += 1
    if gaps:
        flags.append("Employment gaps detected in the timeline.")

    total_span = max(ranges[-1][1] - ranges[0][0], 1)
    switches = max(len(ranges) - 1, 0)
    if switches >= 4 and total_span <= 6:
        flags.append("Frequent job switches in a short time span.")

    penalty = gaps * 18 + max(switches - 2, 0) * 10
    return max(18.0, 100.0 - penalty), flags


class ATSResumeEngine:
    def __init__(self, model_bundle: dict) -> None:
        self.model_bundle = model_bundle

    def build_job_profile(
        self,
        job_description: str,
        must_have_skills: str,
        checkbox_skills: list[str],
        min_experience_filter: float,
        location_preference: str,
        salary_min: float | None,
        salary_max: float | None,
        culture_keywords: str,
        priority_mode: str,
    ) -> JobProfile:
        role = extract_role(job_description)
        role_family = ROLE_FAMILY_MAP.get(role, "Data")
        required_skills = normalize_must_have_skills(must_have_skills, checkbox_skills, job_description)
        preferred_skills = [skill for skill in extract_skills(job_description) if skill not in required_skills][:6]
        required_experience = max(max([float(item) for item in YEARS_PATTERN.findall(job_description)] or [0.0]), min_experience_filter)
        required_education = extract_education_level(job_description)
        culture_list = [item.lower() for item in _split_csv_terms(culture_keywords)]
        if priority_mode not in {"balanced", "skills", "experience", "education"}:
            priority_mode = "balanced"

        return JobProfile(
            role=role,
            role_family=role_family,
            description=job_description.strip(),
            required_skills=required_skills,
            preferred_skills=preferred_skills,
            required_experience=required_experience,
            required_education=required_education,
            min_experience_filter=min_experience_filter,
            location_preference=location_preference.strip(),
            salary_min=salary_min,
            salary_max=salary_max,
            culture_keywords=culture_list,
            priority_mode=priority_mode,
        )

    def analyze(self, candidate_inputs: list[dict], job_profile: JobProfile) -> dict:
        candidates = [self._analyze_candidate(item, index, job_profile) for index, item in enumerate(candidate_inputs, start=1)]
        candidates.sort(key=lambda item: item["overall_score"], reverse=True)
        for rank, candidate in enumerate(candidates, start=1):
            candidate["rank"] = rank

        shortlist_count = sum(1 for item in candidates if item["verdict"] == "Shortlist")
        review_count = sum(1 for item in candidates if item["verdict"] == "Review Manually")
        reject_count = sum(1 for item in candidates if item["verdict"] == "Reject")

        ranking = [
            {
                "rank": item["rank"],
                "id": item["id"],
                "name": item["name"],
                "score": item["overall_score"],
                "verdict": item["verdict"],
                "skills_match": item["breakdown"]["skills_match"],
                "experience_match": item["breakdown"]["experience_match"],
            }
            for item in candidates[:10]
        ]

        compare = self._build_compare_payload(candidates)
        return {
            "job": {
                "role": job_profile.role,
                "required_skills": job_profile.required_skills,
                "preferred_skills": job_profile.preferred_skills,
                "required_experience": job_profile.required_experience,
                "required_education": job_profile.required_education,
                "priority_mode": job_profile.priority_mode,
            },
            "summary": {
                "candidate_count": len(candidates),
                "shortlist_count": shortlist_count,
                "review_count": review_count,
                "reject_count": reject_count,
                "top_candidate": candidates[0]["name"] if candidates else "None",
            },
            "candidates": candidates,
            "ranking": ranking,
            "compare": compare,
        }

    def _analyze_candidate(self, candidate_input: dict, index: int, job_profile: JobProfile) -> dict:
        text = candidate_input["text"]
        name = extract_candidate_name(text, candidate_input["label"] or f"Candidate {index}")
        resume_skills = extract_skills(text)
        matched_skills = [skill for skill in job_profile.required_skills if skill in resume_skills]
        missing_skills = [skill for skill in job_profile.required_skills if skill not in resume_skills]
        nice_matches = [skill for skill in job_profile.preferred_skills if skill in resume_skills]

        experience_years = estimate_experience_years(text)
        education_level = extract_education_level(text)
        candidate_location = extract_location(text)
        expected_salary = extract_salary_expectation(text)

        jd_keywords = [word for word in re.findall(r"[a-zA-Z][a-zA-Z.+-]+", job_profile.description.lower()) if len(word) > 3]
        top_jd_keywords = []
        for keyword in jd_keywords:
            if keyword not in top_jd_keywords:
                top_jd_keywords.append(keyword)
            if len(top_jd_keywords) >= 18:
                break
        jd_keyword_hits = sum(1 for keyword in top_jd_keywords if _contains_term(text, keyword))
        jd_keyword_match = round((jd_keyword_hits / max(len(top_jd_keywords), 1)) * 100, 1)

        if job_profile.required_skills:
            skill_ratio = len(matched_skills) / max(len(job_profile.required_skills), 1)
            preferred_ratio = len(nice_matches) / max(len(job_profile.preferred_skills), 1) if job_profile.preferred_skills else 1.0
            skills_match = round(skill_ratio * 75 + preferred_ratio * 25, 1)
            must_have_match = round(skill_ratio * 100, 1)
        else:
            skills_match = round(max(55.0, jd_keyword_match), 1)
            must_have_match = 100.0

        experience_match = round(score_experience(max(job_profile.required_experience, job_profile.min_experience_filter), experience_years), 1)
        education_match = round(score_education(job_profile.required_education, education_level), 1)
        culture_match = round(score_culture(job_profile.culture_keywords, text), 1)
        location_match = round(score_location(job_profile.location_preference, candidate_location, text), 1)
        salary_match = round(score_salary(job_profile.salary_min, job_profile.salary_max, expected_salary), 1)
        project_strength = round(score_project_strength(text, job_profile.required_skills or resume_skills[:4]), 1)
        achievement_score = round(score_achievements(text), 1)
        authenticity_score = round(score_authenticity(text, matched_skills or resume_skills[:4], experience_years), 1)
        resume_completeness = round(score_resume_completeness(text), 1)
        stability_score, stability_flags = score_stability(text)
        stability_score = round(stability_score, 1)

        red_flags = self._collect_red_flags(
            missing_skills=missing_skills,
            experience_match=experience_match,
            education_match=education_match,
            achievement_score=achievement_score,
            authenticity_score=authenticity_score,
            stability_flags=stability_flags,
        )
        red_flag_penalty = round(min(len(red_flags) * 18 + max(0.0, 55 - authenticity_score) * 0.25, 100), 1)

        feature_row = {
            "role_family": job_profile.role_family,
            "priority_mode": job_profile.priority_mode,
            "skills_match": skills_match,
            "must_have_match": must_have_match,
            "jd_keyword_match": jd_keyword_match,
            "experience_match": experience_match,
            "education_match": education_match,
            "culture_match": culture_match,
            "location_match": location_match,
            "salary_match": salary_match,
            "project_strength": project_strength,
            "achievement_score": achievement_score,
            "authenticity_score": authenticity_score,
            "resume_completeness": resume_completeness,
            "stability_score": stability_score,
            "red_flag_penalty": red_flag_penalty,
        }
        ml_probability = predict_shortlist_probability(self.model_bundle, feature_row)
        rule_score = self._rule_score(job_profile.priority_mode, feature_row)
        overall_score = round(max(0.0, min(100.0, rule_score * 0.62 + ml_probability * 100 * 0.38)), 1)
        verdict, verdict_reason = self._verdict(overall_score, must_have_match, experience_match, red_flags)

        reasoning = self._build_reasoning(
            job_profile=job_profile,
            matched_skills=matched_skills,
            missing_skills=missing_skills,
            experience_years=experience_years,
            experience_match=experience_match,
            education_level=education_level,
            project_strength=project_strength,
            authenticity_score=authenticity_score,
        )
        highlights = self._build_highlights(
            name=name,
            matched_skills=matched_skills,
            experience_years=experience_years,
            project_strength=project_strength,
            culture_match=culture_match,
        )

        return {
            "id": f"candidate-{index}",
            "name": name,
            "source": candidate_input["label"],
            "verdict": verdict,
            "verdict_reason": verdict_reason,
            "overall_score": overall_score,
            "ml_confidence": round(ml_probability * 100, 1),
            "contact": {
                "email": extract_email(text),
                "location": candidate_location,
            },
            "experience_years": experience_years,
            "education_level": education_level,
            "matched_skills": matched_skills,
            "missing_skills": missing_skills,
            "reasoning": reasoning,
            "highlights": highlights,
            "red_flags": red_flags,
            "breakdown": {
                "skills_match": skills_match,
                "experience_match": experience_match,
                "education_match": education_match,
                "culture_match": culture_match,
                "authenticity_score": authenticity_score,
            },
            "meta": {
                "must_have_match": must_have_match,
                "project_strength": project_strength,
                "achievement_score": achievement_score,
                "location_match": location_match,
                "salary_match": salary_match,
                "resume_completeness": resume_completeness,
                "stability_score": stability_score,
            },
        }

    @staticmethod
    def _rule_score(priority_mode: str, feature_row: dict) -> float:
        if priority_mode == "skills":
            weights = {"skills": 0.3, "must": 0.24, "experience": 0.14, "education": 0.06}
        elif priority_mode == "experience":
            weights = {"skills": 0.22, "must": 0.18, "experience": 0.28, "education": 0.08}
        elif priority_mode == "education":
            weights = {"skills": 0.2, "must": 0.18, "experience": 0.16, "education": 0.18}
        else:
            weights = {"skills": 0.26, "must": 0.2, "experience": 0.2, "education": 0.08}

        score = (
            weights["skills"] * feature_row["skills_match"]
            + weights["must"] * feature_row["must_have_match"]
            + weights["experience"] * feature_row["experience_match"]
            + weights["education"] * feature_row["education_match"]
            + 0.06 * feature_row["culture_match"]
            + 0.05 * feature_row["project_strength"]
            + 0.05 * feature_row["achievement_score"]
            + 0.06 * feature_row["authenticity_score"]
            + 0.03 * feature_row["location_match"]
            + 0.02 * feature_row["salary_match"]
            + 0.03 * feature_row["resume_completeness"]
            + 0.03 * feature_row["stability_score"]
            - 0.22 * feature_row["red_flag_penalty"]
        )
        return round(max(0.0, min(score, 100.0)), 1)

    @staticmethod
    def _verdict(score: float, must_have_match: float, experience_match: float, red_flags: list[str]) -> tuple[str, str]:
        if must_have_match <= 40:
            return "Reject", "Too many must-have skill gaps for this role."
        if score >= 78 and experience_match >= 65 and len(red_flags) <= 3:
            return "Shortlist", "Strong alignment across required skills, experience, and evidence."
        if score >= 56:
            return "Review Manually", "Promising profile, but a few gaps need human review."
        return "Reject", "Overall match is below the shortlist threshold."

    @staticmethod
    def _collect_red_flags(
        missing_skills: list[str],
        experience_match: float,
        education_match: float,
        achievement_score: float,
        authenticity_score: float,
        stability_flags: list[str],
    ) -> list[str]:
        flags = []
        if missing_skills:
            flags.append(f"Missing must-have skills: {', '.join(missing_skills[:4])}.")
        if experience_match < 55:
            flags.append("Relevant experience is below the target for this role.")
        if education_match < 55:
            flags.append("Education level is below the stated requirement.")
        if achievement_score < 35:
            flags.append("Resume has few measurable achievements or quantified outcomes.")
        if authenticity_score < 45:
            flags.append("Several claimed skills have weak supporting evidence in the resume.")
        flags.extend(stability_flags)
        return flags[:5]

    @staticmethod
    def _build_reasoning(
        job_profile: JobProfile,
        matched_skills: list[str],
        missing_skills: list[str],
        experience_years: float,
        experience_match: float,
        education_level: str,
        project_strength: float,
        authenticity_score: float,
    ) -> list[str]:
        reasons = []
        if matched_skills:
            reasons.append(f"Strong in {', '.join(matched_skills[:4])}.")
        if experience_match >= 70:
            reasons.append(f"{experience_years:.1f} years of relevant experience is close to or above the target.")
        elif experience_years > 0:
            reasons.append(f"Has {experience_years:.1f} years of relevant experience, but it may be slightly below target.")
        if education_level != "Unknown":
            reasons.append(f"Education detected: {education_level}.")
        if missing_skills:
            reasons.append(f"Missing key requirement(s): {', '.join(missing_skills[:3])}.")
        if project_strength >= 55:
            reasons.append("Project work appears relevant to the job description.")
        if authenticity_score < 45:
            reasons.append("Some skills are mentioned without enough project or achievement support.")
        return reasons[:5]

    @staticmethod
    def _build_highlights(
        name: str,
        matched_skills: list[str],
        experience_years: float,
        project_strength: float,
        culture_match: float,
    ) -> list[str]:
        items = []
        if matched_skills:
            items.append(f"{name} matches {len(matched_skills)} required skill(s).")
        if experience_years > 0:
            items.append(f"Estimated relevant experience: {experience_years:.1f} years.")
        if project_strength >= 55:
            items.append("Project portfolio looks aligned with the target role.")
        if culture_match >= 65:
            items.append("Resume language reflects the requested team or work style.")
        return items[:4]

    @staticmethod
    def _build_compare_payload(candidates: list[dict]) -> dict | None:
        if len(candidates) < 2:
            return None
        left, right = candidates[0], candidates[1]
        return {
            "left": {"name": left["name"], "verdict": left["verdict"]},
            "right": {"name": right["name"], "verdict": right["verdict"]},
            "rows": [
                {"feature": "ATS Match", "left": f"{left['overall_score']}%", "right": f"{right['overall_score']}%"},
                {"feature": "Skills", "left": f"{left['breakdown']['skills_match']}%", "right": f"{right['breakdown']['skills_match']}%"},
                {"feature": "Experience", "left": f"{left['experience_years']:.1f} yrs", "right": f"{right['experience_years']:.1f} yrs"},
                {"feature": "Education", "left": left["education_level"], "right": right["education_level"]},
                {"feature": "Verdict", "left": left["verdict"], "right": right["verdict"]},
            ],
        }


def build_candidate_inputs(pasted_resumes: str, uploaded_resumes: list[dict]) -> list[dict]:
    candidates: list[dict] = []
    for index, block in enumerate(split_resume_blocks(pasted_resumes), start=1):
        candidates.append({"label": f"Pasted Candidate {index}", "text": block})
    candidates.extend(uploaded_resumes)
    return candidates
