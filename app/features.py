from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date


@dataclass(frozen=True)
class PriorFeatures:
    days_between_studies: int
    descriptions_match: bool
    modality_matches: bool


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def _modality(description: str) -> str:
    normalized = _normalize(description)
    if normalized.startswith("mri"):
        return "mri"
    if normalized.startswith("ct"):
        return "ct"
    return "other"


def extract_features(
    *,
    current_study_description: str,
    current_study_date: date,
    prior_study_description: str,
    prior_study_date: date,
) -> PriorFeatures:
    current_desc = _normalize(current_study_description)
    prior_desc = _normalize(prior_study_description)
    return PriorFeatures(
        days_between_studies=abs((current_study_date - prior_study_date).days),
        descriptions_match=current_desc == prior_desc,
        modality_matches=_modality(current_desc) == _modality(prior_desc),
    )
