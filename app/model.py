from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Sequence

import joblib
import sklearn

from app.features import extract_model_features


DEFAULT_MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "model.joblib"
MODEL_WORKER_COUNT = 1


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
        artifact_sklearn_version = artifact.get("sklearn_version")
        if artifact_sklearn_version is not None and artifact_sklearn_version != sklearn.__version__:
            raise RuntimeError(
                "Model artifact was trained with scikit-learn "
                f"{artifact_sklearn_version}, but runtime has {sklearn.__version__}. "
                "Run `python3 -B app/train.py` from the active environment to regenerate it."
            )

        self._estimator = artifact["estimator"]
        self._threshold = artifact["threshold"]
        self.metadata = {
            "name": artifact.get("name"),
            "threshold": self._threshold,
            "training_rows": artifact.get("training_rows"),
            "positive_rows": artifact.get("positive_rows"),
            "sklearn_version": artifact.get("sklearn_version"),
            "model_worker_count": MODEL_WORKER_COUNT,
        }
        self._force_single_worker_estimator()

    def _force_single_worker_estimator(self) -> None:
        classifier = getattr(self._estimator, "named_steps", {}).get("classifier")
        if classifier is not None and hasattr(classifier, "n_jobs"):
            classifier.n_jobs = MODEL_WORKER_COUNT

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
