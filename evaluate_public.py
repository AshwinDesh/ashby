from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Any

from app.model import RelevantPriorModel, StudyComparison


def load_eval_payload(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def generate_predictions(payload: dict[str, Any]) -> dict[tuple[str, str], bool]:
    model = RelevantPriorModel()
    predictions: dict[tuple[str, str], bool] = {}
    keys: list[tuple[str, str]] = []
    comparisons: list[StudyComparison] = []

    for case in payload["cases"]:
        case_id = case["case_id"]
        current_study = case["current_study"]
        current_study_date = date.fromisoformat(current_study["study_date"])
        for prior in case["prior_studies"]:
            keys.append((case_id, prior["study_id"]))
            comparisons.append(
                StudyComparison(
                    current_study_description=current_study["study_description"],
                    current_study_date=current_study_date,
                    prior_study_description=prior["study_description"],
                    prior_study_date=date.fromisoformat(prior["study_date"]),
                )
            )

    for key, prediction in zip(keys, model.predict_is_relevant_many(comparisons), strict=True):
        predictions[key] = prediction

    return predictions


def evaluate(payload: dict[str, Any]) -> dict[str, float | int]:
    predictions = generate_predictions(payload)
    truth = payload["truth"]

    total_cases = len(payload["cases"])
    total_priors = sum(len(case["prior_studies"]) for case in payload["cases"])

    correct = 0
    missing = 0

    for row in truth:
        key = (row["case_id"], row["study_id"])
        expected = row["is_relevant_to_current"]
        predicted = predictions.get(key)

        if predicted is None:
            missing += 1
            continue

        if predicted == expected:
            correct += 1

    incorrect = len(truth) - correct
    accuracy = correct / len(truth) if truth else 0.0

    return {
        "total_cases": total_cases,
        "total_priors": total_priors,
        "correct": correct,
        "incorrect": incorrect,
        "missing": missing,
        "accuracy": accuracy,
    }


def main() -> None:
    payload_path = Path("relevant_priors_public.json")
    payload = load_eval_payload(payload_path)
    metrics = evaluate(payload)

    print(f"total cases: {metrics['total_cases']}")
    print(f"total priors: {metrics['total_priors']}")
    print(f"correct: {metrics['correct']}")
    print(f"incorrect: {metrics['incorrect']}")
    print(f"missing: {metrics['missing']}")
    print(f"accuracy: {metrics['accuracy']:.6f}")


if __name__ == "__main__":
    main()
