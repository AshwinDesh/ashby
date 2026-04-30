from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path
from typing import Any

import joblib
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.pipeline import Pipeline

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.features import extract_model_features

DEFAULT_TRAINING_DATA_PATH = PROJECT_ROOT / "relevant_priors_public.json"
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "model.joblib"
RANDOM_STATE = 20260426
DEFAULT_DECISION_THRESHOLD = 0.65
ESTIMATOR_COUNT = 300
MIN_SAMPLES_PER_LEAF = 8
MODEL_WORKER_COUNT = 1


def _load_training_payload(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _build_training_rows(payload: dict[str, Any]) -> tuple[list[dict[str, Any]], list[bool]]:
    labels = {
        (row["case_id"], row["study_id"]): row["is_relevant_to_current"]
        for row in payload["truth"]
    }
    features: list[dict[str, Any]] = []
    targets: list[bool] = []

    for case in payload["cases"]:
        current_study = case["current_study"]
        current_study_date = date.fromisoformat(current_study["study_date"])
        for prior_study in case["prior_studies"]:
            key = (case["case_id"], prior_study["study_id"])
            if key not in labels:
                raise ValueError(f"Missing training label for case/study {key}")

            features.append(
                extract_model_features(
                    current_study_description=current_study["study_description"],
                    current_study_date=current_study_date,
                    prior_study_description=prior_study["study_description"],
                    prior_study_date=date.fromisoformat(prior_study["study_date"]),
                )
            )
            targets.append(labels[key])

    return features, targets


def train_and_save(
    *,
    training_data_path: Path = DEFAULT_TRAINING_DATA_PATH,
    output_path: Path = DEFAULT_MODEL_PATH,
    threshold: float = DEFAULT_DECISION_THRESHOLD,
) -> None:
    features, targets = _build_training_rows(_load_training_payload(training_data_path))
    estimator = Pipeline(
        steps=[
            ("vectorizer", DictVectorizer(sparse=False)),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=ESTIMATOR_COUNT,
                    min_samples_leaf=MIN_SAMPLES_PER_LEAF,
                    max_features="sqrt",
                    class_weight="balanced_subsample",
                    random_state=RANDOM_STATE,
                    n_jobs=MODEL_WORKER_COUNT,
                ),
            ),
        ]
    )
    estimator.fit(features, targets)

    serialized_model = {
        "name": "random_forest_relevant_prior_v1",
        "estimator": estimator,
        "threshold": threshold,
        "training_data": str(training_data_path),
        "training_rows": len(targets),
        "positive_rows": sum(targets),
        "random_state": RANDOM_STATE,
        "model_worker_count": MODEL_WORKER_COUNT,
        "sklearn_version": sklearn.__version__,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(serialized_model, output_path)
    print(f"Saved random forest model artifact to {output_path}")
    print(f"Training rows: {len(targets)}")
    print(f"Positive rows: {sum(targets)}")
    print(f"Decision threshold: {threshold}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and save the relevant-priors model artifact.")
    parser.add_argument("--training-data", type=Path, default=DEFAULT_TRAINING_DATA_PATH)
    parser.add_argument("--output-model", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--threshold", type=float, default=DEFAULT_DECISION_THRESHOLD)
    args = parser.parse_args()
    train_and_save(
        training_data_path=args.training_data,
        output_path=args.output_model,
        threshold=args.threshold,
    )
