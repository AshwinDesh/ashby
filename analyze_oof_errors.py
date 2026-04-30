from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline

from app.features import extract_features, extract_model_features
from app.train import ESTIMATOR_COUNT, MIN_SAMPLES_PER_LEAF, MODEL_WORKER_COUNT, RANDOM_STATE


DEFAULT_PAYLOAD_PATH = Path("relevant_priors_public.json")
DEFAULT_THRESHOLD = 0.64
DEFAULT_FOLDS = 5
PATTERN_LIMIT = 20
EXAMPLE_LIMIT = 12


@dataclass(frozen=True)
class StudyRow:
    case_id: str
    study_id: str
    current_description: str
    prior_description: str
    current_date: date
    prior_date: date
    expected: bool


def _load_payload(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _age_bucket(days: int) -> str:
    if days == 0:
        return "same_day"
    if days <= 30:
        return "within_30_days"
    if days <= 365:
        return "within_1_year"
    if days <= 365 * 5:
        return "within_5_years"
    return "older_than_5_years"


def _pattern_key(row: StudyRow) -> tuple[str, ...]:
    features = extract_features(
        current_study_description=row.current_description,
        current_study_date=row.current_date,
        prior_study_description=row.prior_description,
        prior_study_date=row.prior_date,
    )
    return (
        f"modality={features.current_modality}->{features.prior_modality}",
        f"region={features.current_body_region}->{features.prior_body_region}",
        f"age={_age_bucket(features.days_between_studies)}",
        f"desc_match={features.descriptions_match}",
        f"body_match={features.body_region_matches}",
        f"laterality_match={features.laterality_matches}",
        f"contrast_match={features.contrast_matches}",
        f"jaccard={features.token_jaccard:.1f}",
    )


def _build_rows(payload: dict[str, Any]) -> tuple[list[StudyRow], list[dict[str, Any]], np.ndarray, np.ndarray]:
    truth = {
        (row["case_id"], row["study_id"]): row["is_relevant_to_current"]
        for row in payload["truth"]
    }
    rows: list[StudyRow] = []
    feature_rows: list[dict[str, Any]] = []
    targets: list[int] = []
    groups: list[str] = []

    for case in payload["cases"]:
        current_study = case["current_study"]
        current_date = date.fromisoformat(current_study["study_date"])
        for prior_study in case["prior_studies"]:
            key = (case["case_id"], prior_study["study_id"])
            expected = bool(truth[key])
            prior_date = date.fromisoformat(prior_study["study_date"])
            rows.append(
                StudyRow(
                    case_id=case["case_id"],
                    study_id=prior_study["study_id"],
                    current_description=current_study["study_description"],
                    prior_description=prior_study["study_description"],
                    current_date=current_date,
                    prior_date=prior_date,
                    expected=expected,
                )
            )
            feature_rows.append(
                extract_model_features(
                    current_study_description=current_study["study_description"],
                    current_study_date=current_date,
                    prior_study_description=prior_study["study_description"],
                    prior_study_date=prior_date,
                )
            )
            targets.append(int(expected))
            groups.append(case["case_id"])

    return rows, feature_rows, np.array(targets), np.array(groups)


def _make_model() -> Pipeline:
    return Pipeline(
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


def _print_patterns(title: str, rows: list[StudyRow]) -> None:
    print()
    print(title)
    pattern_counts = Counter(_pattern_key(row) for row in rows)
    for pattern, count in pattern_counts.most_common(PATTERN_LIMIT):
        print(f"  {count}: {' | '.join(pattern)}")


def _print_examples(title: str, rows: list[StudyRow]) -> None:
    print()
    print(title)
    for row in rows[:EXAMPLE_LIMIT]:
        print(
            "  "
            f"{row.case_id}/{row.study_id}: "
            f"{row.current_description} [{row.current_date.isoformat()}] <> "
            f"{row.prior_description} [{row.prior_date.isoformat()}]"
        )


def analyze(payload_path: Path, folds: int, threshold: float) -> None:
    rows, feature_rows, targets, groups = _build_rows(_load_payload(payload_path))
    probabilities = np.zeros(len(rows), dtype=float)

    for fold_index, (train_idx, test_idx) in enumerate(
        GroupKFold(n_splits=folds).split(feature_rows, targets, groups),
        start=1,
    ):
        model = _make_model()
        model.fit([feature_rows[i] for i in train_idx], targets[train_idx])
        probabilities[test_idx] = model.predict_proba([feature_rows[i] for i in test_idx])[:, 1]
        print(f"fold {fold_index}: train={len(train_idx)} test={len(test_idx)}")

    predictions = probabilities >= threshold
    precision, recall, f1, _ = precision_recall_fscore_support(
        targets,
        predictions,
        average="binary",
        zero_division=0,
    )
    false_positives = [
        row for row, predicted, expected in zip(rows, predictions, targets, strict=True)
        if predicted and not expected
    ]
    false_negatives = [
        row for row, predicted, expected in zip(rows, predictions, targets, strict=True)
        if not predicted and expected
    ]

    print()
    print(f"threshold: {threshold:.2f}")
    print(f"accuracy: {accuracy_score(targets, predictions):.6f}")
    print(f"precision: {precision:.6f}")
    print(f"recall: {recall:.6f}")
    print(f"f1: {f1:.6f}")
    print(f"auc: {roc_auc_score(targets, probabilities):.6f}")
    print(f"false positives: {len(false_positives)}")
    print(f"false negatives: {len(false_negatives)}")

    _print_patterns("Top out-of-fold false positive patterns:", false_positives)
    _print_patterns("Top out-of-fold false negative patterns:", false_negatives)
    _print_examples("Out-of-fold false positive examples:", false_positives)
    _print_examples("Out-of-fold false negative examples:", false_negatives)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze case-level out-of-fold prediction errors.")
    parser.add_argument("--payload", type=Path, default=DEFAULT_PAYLOAD_PATH)
    parser.add_argument("--folds", type=int, default=DEFAULT_FOLDS)
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    args = parser.parse_args()
    analyze(args.payload, args.folds, args.threshold)


if __name__ == "__main__":
    main()
