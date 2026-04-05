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

from .skills import SKILL_LIBRARY


ROOT_DIR = Path(__file__).resolve().parent.parent
ARTIFACT_PATH = ROOT_DIR / "artifacts" / "resume_truth_detector.joblib"
CATEGORICAL_FEATURES = ["skill_area"]
NUMERIC_FEATURES = [
    "claim_level",
    "resume_keyword_hits",
    "action_hits",
    "project_mentions",
    "years_signal",
    "certification_hits",
    "skill_sentence_hits",
    "github_relevant_repos",
    "github_recent_repos",
    "github_relevant_stars",
    "github_profile_score",
    "external_link_hits",
]
FEATURE_COLUMNS = [*CATEGORICAL_FEATURES, *NUMERIC_FEATURES]


def _build_synthetic_dataset(samples: int = 3200, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    skill_areas = list(SKILL_LIBRARY.keys())
    rows: list[dict[str, float | int | str]] = []

    for _ in range(samples):
        skill_area = str(rng.choice(skill_areas))
        claim_level = int(rng.choice([1, 2, 3], p=[0.28, 0.46, 0.26]))
        latent_evidence = float(np.clip(rng.beta(2.4, 2.2) + rng.normal(0, 0.05), 0, 1))

        resume_keyword_hits = int(max(0, rng.poisson(1 + latent_evidence * 9)))
        action_hits = int(max(0, rng.poisson(0.5 + latent_evidence * 4)))
        project_mentions = int(max(0, rng.poisson(0.4 + latent_evidence * 4.5)))
        years_signal = float(np.clip(rng.normal(0.8 + latent_evidence * 4.2, 1.1), 0, 10))
        certification_hits = int(max(0, rng.poisson(0.15 + latent_evidence * 1.4)))
        skill_sentence_hits = int(max(0, rng.poisson(0.8 + latent_evidence * 4.2)))
        github_relevant_repos = int(max(0, rng.poisson(latent_evidence * 5.5)))
        github_recent_repos = int(max(0, rng.poisson(latent_evidence * 4.2)))
        github_relevant_stars = int(max(0, rng.normal(latent_evidence * 40, 12)))
        github_profile_score = float(np.clip(rng.normal(15 + latent_evidence * 70, 9), 0, 100))
        external_link_hits = int(max(0, rng.poisson(0.2 + latent_evidence * 3.4)))

        evidence_score = (
            0.21 * min(resume_keyword_hits, 6) / 6
            + 0.14 * min(action_hits, 4) / 4
            + 0.14 * min(project_mentions, 4) / 4
            + 0.10 * min(years_signal, 5) / 5
            + 0.05 * min(certification_hits, 2) / 2
            + 0.12 * min(skill_sentence_hits, 3) / 3
            + 0.10 * min(github_relevant_repos, 4) / 4
            + 0.05 * min(github_recent_repos, 3) / 3
            + 0.04 * min(github_relevant_stars, 30) / 30
            + 0.03 * github_profile_score / 100
            + 0.02 * min(external_link_hits, 3) / 3
        )
        claim_pressure = 0.10 + claim_level * 0.11
        realism_margin = evidence_score - claim_pressure + rng.normal(0, 0.07)
        realistic_probability = 1 / (1 + np.exp(-10 * realism_margin))
        label = int(rng.binomial(1, realistic_probability))

        rows.append(
            {
                "skill_area": skill_area,
                "claim_level": claim_level,
                "resume_keyword_hits": resume_keyword_hits,
                "action_hits": action_hits,
                "project_mentions": project_mentions,
                "years_signal": years_signal,
                "certification_hits": certification_hits,
                "skill_sentence_hits": skill_sentence_hits,
                "github_relevant_repos": github_relevant_repos,
                "github_recent_repos": github_recent_repos,
                "github_relevant_stars": github_relevant_stars,
                "github_profile_score": github_profile_score,
                "external_link_hits": external_link_hits,
                "label": label,
            }
        )

    return pd.DataFrame(rows)


def train_and_save_model(force: bool = False) -> dict:
    if ARTIFACT_PATH.exists() and not force:
        return joblib.load(ARTIFACT_PATH)

    dataset = _build_synthetic_dataset()
    train_frame, test_frame = train_test_split(
        dataset,
        test_size=0.22,
        random_state=42,
        stratify=dataset["label"],
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("categorical", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_FEATURES),
            ("numeric", StandardScaler(), NUMERIC_FEATURES),
        ]
    )

    classifier = RandomForestClassifier(
        n_estimators=320,
        max_depth=10,
        min_samples_leaf=2,
        class_weight="balanced_subsample",
        random_state=42,
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", classifier),
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
        "metrics": {
            "accuracy": round(float(accuracy_score(y_test, predictions)), 4),
            "roc_auc": round(float(roc_auc_score(y_test, probabilities)), 4),
            "samples": int(len(dataset)),
        },
        "feature_columns": FEATURE_COLUMNS,
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
    }

    ARTIFACT_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, ARTIFACT_PATH)
    return artifact


def load_model_bundle() -> dict:
    return train_and_save_model(force=False)


def predict_realism_probability(model_bundle: dict, feature_row: dict) -> float:
    frame = pd.DataFrame([feature_row], columns=model_bundle["feature_columns"])
    probability = model_bundle["pipeline"].predict_proba(frame)[0, 1]
    return float(probability)
