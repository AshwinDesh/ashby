from __future__ import annotations

import json
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path
from typing import Any

from app.features import extract_features
from app.model import RelevantPriorModel, StudyComparison


PAYLOAD_PATH = Path("relevant_priors_public.json")
EXAMPLE_LIMIT = 8
PATTERN_LIMIT = 12


def _load_payload(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _pattern_key(
    *,
    current_description: str,
    prior_description: str,
    current_study_date: date,
    prior_study_date: date,
) -> tuple[str, str, str, str, str]:
    features = extract_features(
        current_study_description=current_description,
        current_study_date=current_study_date,
        prior_study_description=prior_description,
        prior_study_date=prior_study_date,
    )
    if features.days_between_studies == 0:
        age_bucket = "same_day"
    elif features.days_between_studies <= 30:
        age_bucket = "within_30_days"
    elif features.days_between_studies <= 365:
        age_bucket = "within_1_year"
    elif features.days_between_studies <= 365 * 5:
        age_bucket = "within_5_years"
    else:
        age_bucket = "older_than_5_years"

    return (
        f"{features.current_modality}->{features.prior_modality}",
        f"{features.current_body_region}->{features.prior_body_region}",
        age_bucket,
        f"desc_match={features.descriptions_match}",
        f"token_jaccard={features.token_jaccard:.1f}",
    )


def main() -> None:
    payload = _load_payload(PAYLOAD_PATH)
    truth = {
        (row["case_id"], row["study_id"]): row["is_relevant_to_current"]
        for row in payload["truth"]
    }
    model = RelevantPriorModel()

    correct = 0
    false_positives: list[dict[str, Any]] = []
    false_negatives: list[dict[str, Any]] = []
    pattern_counts: dict[str, Counter[tuple[str, str, str, str, str]]] = defaultdict(Counter)
    rows: list[dict[str, Any]] = []
    comparisons: list[StudyComparison] = []

    for case in payload["cases"]:
        current_study = case["current_study"]
        current_study_date = date.fromisoformat(current_study["study_date"])
        for prior_study in case["prior_studies"]:
            prior_study_date = date.fromisoformat(prior_study["study_date"])
            rows.append(
                {
                    "case_id": case["case_id"],
                    "study_id": prior_study["study_id"],
                    "current": current_study["study_description"],
                    "prior": prior_study["study_description"],
                    "current_date": current_study["study_date"],
                    "prior_date": prior_study["study_date"],
                    "expected": truth[(case["case_id"], prior_study["study_id"])],
                }
            )
            comparisons.append(
                StudyComparison(
                    current_study_description=current_study["study_description"],
                    current_study_date=current_study_date,
                    prior_study_description=prior_study["study_description"],
                    prior_study_date=prior_study_date,
                )
            )

    for row, predicted in zip(rows, model.predict_is_relevant_many(comparisons), strict=True):
        if predicted == row["expected"]:
            correct += 1
            continue

        error_type = "false_positive" if predicted else "false_negative"
        pattern_counts[error_type][
            _pattern_key(
                current_description=row["current"],
                prior_description=row["prior"],
                current_study_date=date.fromisoformat(row["current_date"]),
                prior_study_date=date.fromisoformat(row["prior_date"]),
            )
        ] += 1

        if predicted:
            false_positives.append(row)
        else:
            false_negatives.append(row)

    total = len(truth)
    print(f"accuracy: {correct / total:.6f}")
    print(f"correct: {correct}")
    print(f"incorrect: {total - correct}")
    print(f"false positives: {len(false_positives)}")
    print(f"false negatives: {len(false_negatives)}")

    for error_type in ("false_positive", "false_negative"):
        print()
        print(f"Top {error_type} patterns:")
        for pattern, count in pattern_counts[error_type].most_common(PATTERN_LIMIT):
            print(f"  {count}: {' | '.join(pattern)}")

    for title, examples in (
        ("False positive examples", false_positives),
        ("False negative examples", false_negatives),
    ):
        print()
        print(title)
        for row in examples[:EXAMPLE_LIMIT]:
            print(
                "  "
                f"{row['case_id']}/{row['study_id']}: "
                f"{row['current']} [{row['current_date']}] <> "
                f"{row['prior']} [{row['prior_date']}]"
            )


if __name__ == "__main__":
    main()
