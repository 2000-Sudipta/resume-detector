from __future__ import annotations

from pprint import pprint

from resume_truth_detector.ats_modeling import train_and_save_ats_model


if __name__ == "__main__":
    artifact = train_and_save_ats_model(force=True)
    pprint(artifact["metrics"])
