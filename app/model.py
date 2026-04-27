from __future__ import annotations

from datetime import date

from app.features import extract_features


class RelevantPriorModel:
    def predict_is_relevant(
        self,
        *,
        current_study_description: str,
        current_study_date: date,
        prior_study_description: str,
        prior_study_date: date,
    ) -> bool:
        features = extract_features(
            current_study_description=current_study_description,
            current_study_date=current_study_date,
            prior_study_description=prior_study_description,
            prior_study_date=prior_study_date,
        )

        if features.descriptions_match:
            return True

        if features.modality_matches and features.days_between_studies <= 365 * 5:
            return True

        return False
