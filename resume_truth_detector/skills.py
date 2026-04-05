from __future__ import annotations

import re
from typing import Iterable


SKILL_LIBRARY: dict[str, dict[str, list[str]]] = {
    "Machine Learning": {
        "aliases": ["machine learning", "ml", "predictive modeling", "model training"],
        "evidence_keywords": [
            "regression",
            "classification",
            "feature engineering",
            "cross-validation",
            "random forest",
            "xgboost",
            "scikit-learn",
            "model evaluation",
            "pipeline",
        ],
        "repo_keywords": ["ml", "model", "classification", "regression", "predict", "sklearn"],
        "languages": ["python", "jupyter"],
    },
    "Data Science": {
        "aliases": ["data science", "data analysis", "analytics", "data scientist"],
        "evidence_keywords": [
            "pandas",
            "numpy",
            "statistics",
            "visualization",
            "data cleaning",
            "eda",
            "tableau",
            "matplotlib",
        ],
        "repo_keywords": ["analysis", "dashboard", "eda", "dataset", "insights", "visualization"],
        "languages": ["python", "r", "sql"],
    },
    "NLP": {
        "aliases": ["nlp", "natural language processing", "language model", "text mining"],
        "evidence_keywords": [
            "tokenization",
            "transformer",
            "bert",
            "llm",
            "named entity recognition",
            "sentiment analysis",
            "text classification",
        ],
        "repo_keywords": ["nlp", "text", "transformer", "bert", "llm", "chatbot"],
        "languages": ["python"],
    },
    "Computer Vision": {
        "aliases": ["computer vision", "cv", "image processing", "object detection"],
        "evidence_keywords": [
            "opencv",
            "cnn",
            "image classification",
            "object detection",
            "segmentation",
            "yolo",
            "vision transformer",
        ],
        "repo_keywords": ["vision", "opencv", "image", "yolo", "detection", "segmentation"],
        "languages": ["python"],
    },
    "MLOps": {
        "aliases": ["mlops", "model deployment", "ml pipeline", "model serving"],
        "evidence_keywords": [
            "docker",
            "fastapi",
            "flask",
            "ci/cd",
            "monitoring",
            "deployment",
            "airflow",
            "mlflow",
        ],
        "repo_keywords": ["deploy", "pipeline", "serving", "mlops", "airflow", "mlflow"],
        "languages": ["python", "yaml"],
    },
    "Backend Engineering": {
        "aliases": ["backend", "backend development", "api development", "server-side"],
        "evidence_keywords": [
            "api",
            "microservice",
            "database",
            "authentication",
            "flask",
            "fastapi",
            "django",
        ],
        "repo_keywords": ["api", "backend", "service", "server", "django", "flask", "fastapi"],
        "languages": ["python", "java", "go", "node"],
    },
    "Frontend Engineering": {
        "aliases": ["frontend", "frontend development", "ui", "web interface"],
        "evidence_keywords": [
            "react",
            "javascript",
            "typescript",
            "css",
            "animation",
            "responsive design",
            "accessibility",
        ],
        "repo_keywords": ["frontend", "ui", "react", "web", "dashboard", "landing"],
        "languages": ["javascript", "typescript", "css", "html"],
    },
    "Cloud Engineering": {
        "aliases": ["cloud", "aws", "azure", "gcp", "cloud architecture"],
        "evidence_keywords": [
            "aws",
            "azure",
            "gcp",
            "terraform",
            "kubernetes",
            "iam",
            "serverless",
        ],
        "repo_keywords": ["cloud", "aws", "azure", "gcp", "infra", "terraform", "k8s"],
        "languages": ["yaml", "terraform", "python"],
    },
}


CLAIM_STRENGTH_PATTERNS = (
    (3, re.compile(r"\b(expert|advanced|specialist|deep expertise|lead|architect)\b", re.IGNORECASE)),
    (2, re.compile(r"\b(proficient|strong|solid|experienced|hands-on|professional)\b", re.IGNORECASE)),
    (1, re.compile(r"\b(familiar|basic|beginner|exposure|learning)\b", re.IGNORECASE)),
)


def _lower_terms(terms: Iterable[str]) -> list[str]:
    return [term.lower() for term in terms]


def _term_pattern(term: str) -> re.Pattern[str]:
    return re.compile(rf"(?<!\w){re.escape(term.lower())}(?!\w)")


def _contains_term(text: str, term: str) -> bool:
    return bool(_term_pattern(term).search(text.lower()))


def _count_term(text: str, term: str) -> int:
    return len(_term_pattern(term).findall(text.lower()))


def normalize_skill_name(raw_text: str) -> str | None:
    normalized = raw_text.strip().lower()
    if not normalized:
        return None

    for skill_name, profile in SKILL_LIBRARY.items():
        aliases = _lower_terms([skill_name, *profile["aliases"]])
        if any(_contains_term(normalized, alias) for alias in aliases):
            return skill_name
    return None


def detect_claim_strength(text: str) -> int:
    for score, pattern in CLAIM_STRENGTH_PATTERNS:
        if pattern.search(text):
            return score
    return 1


def find_skills_in_text(text: str) -> list[str]:
    lowered = text.lower()
    matches: list[str] = []
    for skill_name, profile in SKILL_LIBRARY.items():
        aliases = _lower_terms([skill_name, *profile["aliases"]])
        if any(_contains_term(lowered, alias) for alias in aliases):
            matches.append(skill_name)
    return matches


def keyword_hits(text: str, skill_name: str, field: str = "evidence_keywords") -> int:
    keywords = _lower_terms(SKILL_LIBRARY[skill_name][field])
    return sum(_count_term(text, keyword) for keyword in keywords)
