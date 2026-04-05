from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


ROOT_DIR = Path(__file__).resolve().parent.parent
ARTIFACT_PATH = ROOT_DIR / "artifacts" / "hr_ats_ranker.joblib"

CATEGORICAL_FEATURES = ["role_family", "priority_mode"]
NUMERIC_FEATURES = [
    "skills_match",
    "must_have_match",
    "jd_keyword_match",
    "experience_match",
    "education_match",
    "culture_match",
    "location_match",
    "salary_match",
    "project_strength",
    "achievement_score",
    "authenticity_score",
    "resume_completeness",
    "stability_score",
    "red_flag_penalty",
]
FEATURE_COLUMNS = [*CATEGORICAL_FEATURES, *NUMERIC_FEATURES]


def _priority_weights(mode: str) -> dict[str, float]:
    if mode == "skills":
        return {"skills": 0.32, "must_have": 0.22, "experience": 0.12, "education": 0.06}
    if mode == "experience":
        return {"skills": 0.22, "must_have": 0.18, "experience": 0.26, "education": 0.08}
    if mode == "education":
        return {"skills": 0.20, "must_have": 0.18, "experience": 0.16, "education": 0.18}
    return {"skills": 0.26, "must_have": 0.20, "experience": 0.18, "education": 0.08}


def _build_dataset(samples: int = 4200, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    roles = ["Data", "Backend", "Frontend", "Cloud", "ML"]
    priorities = ["balanced", "skills", "experience", "education"]
    rows: list[dict[str, float | str | int]] = []

    for _ in range(samples):
        role_family = str(rng.choice(roles))
        priority_mode = str(rng.choice(priorities, p=[0.48, 0.24, 0.2, 0.08]))
        latent_fit = float(np.clip(rng.beta(2.3, 2.0) + rng.normal(0, 0.05), 0, 1))

        skills_match = float(np.clip(rng.normal(28 + latent_fit * 60, 12), 0, 100))
        must_have_match = float(np.clip(skills_match + rng.normal(4, 10), 0, 100))
        jd_keyword_match = float(np.clip(rng.normal(26 + latent_fit * 58, 13), 0, 100))
        experience_match = float(np.clip(rng.normal(24 + latent_fit * 65, 14), 0, 100))
        education_match = float(np.clip(rng.normal(35 + latent_fit * 45, 16), 0, 100))
        culture_match = float(np.clip(rng.normal(38 + latent_fit * 42, 15), 0, 100))
        location_match = float(np.clip(rng.normal(60 + latent_fit * 28, 16), 0, 100))
        salary_match = float(np.clip(rng.normal(55 + latent_fit * 32, 18), 0, 100))
        project_strength = float(np.clip(rng.normal(18 + latent_fit * 70, 15), 0, 100))
        achievement_score = float(np.clip(rng.normal(16 + latent_fit * 66, 14), 0, 100))
        authenticity_score = float(np.clip(rng.normal(24 + latent_fit * 62, 14), 0, 100))
        resume_completeness = float(np.clip(rng.normal(44 + latent_fit * 42, 12), 0, 100))
        stability_score = float(np.clip(rng.normal(46 + latent_fit * 38, 18), 0, 100))

        red_flag_penalty = float(
            np.clip(
                rng.normal(42 - latent_fit * 36, 14)
                + (18 if must_have_match < 45 else 0)
                + (14 if experience_match < 40 else 0),
                0,
                100,
            )
        )

        weights = _priority_weights(priority_mode)
        rule_score = (
            weights["skills"] * skills_match
            + weights["must_have"] * must_have_match
            + weights["experience"] * experience_match
            + weights["education"] * education_match
            + 0.08 * jd_keyword_match
            + 0.06 * culture_match
            + 0.04 * location_match
            + 0.03 * salary_match
            + 0.06 * project_strength
            + 0.05 * achievement_score
            + 0.06 * authenticity_score
            + 0.03 * resume_completeness
            + 0.03 * stability_score
            - 0.28 * red_flag_penalty
        )

        hard_gate = must_have_match < 32 or (priority_mode == "experience" and experience_match < 28)
        shortlist_probability = 1 / (1 + np.exp(-((rule_score - 41) / 8)))
        if hard_gate:
            shortlist_probability *= 0.25
        shortlist_probability = float(np.clip(shortlist_probability + rng.normal(0, 0.04), 0, 1))
        label = int(rng.binomial(1, shortlist_probability))

        rows.append(
            {
                "role_family": role_family,
                "priority_mode": priority_mode,
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
                "label": label,
            }
        )

    return pd.DataFrame(rows)


def train_and_save_ats_model(force: bool = False) -> dict:
    if ARTIFACT_PATH.exists() and not force:
        return joblib.load(ARTIFACT_PATH)

    dataset = _build_dataset()
    train_frame, test_frame = train_test_split(
        dataset,
        test_size=0.2,
        random_state=42,
        stratify=dataset["label"],
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("categorical", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_FEATURES),
            ("numeric", StandardScaler(), NUMERIC_FEATURES),
        ]
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=320,
                    max_depth=10,
                    min_samples_leaf=2,
                    class_weight="balanced_subsample",
                    random_state=42,
                ),
            ),
        ]
    )

    x_train = train_frame[FEATURE_COLUMNS]
    y_train = train_frame["label"]
    x_test = test_frame[FEATURE_COLUMNS]
    y_test = test_frame["label"]

    pipeline.fit(x_train, y_train)
    probabilities = pipeline.predict_proba(x_test)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)

    artifact = {
        "pipeline": pipeline,
        "feature_columns": FEATURE_COLUMNS,
        "metrics": {
            "accuracy": round(float(accuracy_score(y_test, predictions)), 4),
            "roc_auc": round(float(roc_auc_score(y_test, probabilities)), 4),
            "samples": int(len(dataset)),
        },
    }

    ARTIFACT_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, ARTIFACT_PATH)
    return artifact


def load_ats_model_bundle() -> dict:
    return train_and_save_ats_model(force=False)


def predict_shortlist_probability(model_bundle: dict, feature_row: dict) -> float:
    frame = pd.DataFrame([feature_row], columns=model_bundle["feature_columns"])
    probability = model_bundle["pipeline"].predict_proba(frame)[0, 1]
    return float(probability)
