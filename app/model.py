from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Sequence

import joblib

from app.features import extract_model_features


DEFAULT_MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "model.joblib"


@dataclass(frozen=True)
class StudyComparison:
    current_study_description: str
    current_study_date: date
    prior_study_description: str
    prior_study_date: date


class RelevantPriorModel:
    def __init__(self, model_path: Path = DEFAULT_MODEL_PATH) -> None:
        if not model_path.exists():
            raise FileNotFoundError(
                f"Model artifact not found at {model_path}. Run `python3 app/train.py` first."
            )

        artifact: dict[str, Any] = joblib.load(model_path)
        self._estimator = artifact["estimator"]
        self._threshold = artifact["threshold"]

    @staticmethod
    def _feature_row(comparison: StudyComparison) -> dict[str, Any]:
        return extract_model_features(
            current_study_description=comparison.current_study_description,
            current_study_date=comparison.current_study_date,
            prior_study_description=comparison.prior_study_description,
            prior_study_date=comparison.prior_study_date,
        )

    def predict_is_relevant_many(self, comparisons: Sequence[StudyComparison]) -> list[bool]:
        if not comparisons:
            return []

        feature_rows = [self._feature_row(comparison) for comparison in comparisons]
        probabilities = self._estimator.predict_proba(feature_rows)[:, 1]
        return [bool(probability >= self._threshold) for probability in probabilities]

    def predict_is_relevant(
        self,
        *,
        current_study_description: str,
        current_study_date: date,
        prior_study_description: str,
        prior_study_date: date,
    ) -> bool:
        return self.predict_is_relevant_many(
            [
                StudyComparison(
                    current_study_description=current_study_description,
                    current_study_date=current_study_date,
                    prior_study_description=prior_study_description,
                    prior_study_date=prior_study_date,
                )
            ]
        )[0]
