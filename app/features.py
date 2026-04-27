from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date
from math import log1p
from typing import Any


TOKEN_PATTERN = re.compile(r"[a-z0-9]+")

MODALITY_PREFIXES = {
    "mri": ("mri", "mr "),
    "ct": ("ct", "cta"),
    "xr": ("xr", "xray", "x-ray", "radiograph"),
    "mam": ("mam", "mammo", "mammography"),
    "us": ("us", "ultrasound"),
    "nm": ("nm", "nuclear"),
    "echo": ("echo", "tte"),
    "dxa": ("dxa",),
    "vas": ("vas", "vascular", "venous"),
    "pet": ("pet",),
}

BODY_REGION_TERMS = {
    "head_neuro": ("head", "brain", "skull", "neuro", "cerebral"),
    "neck_spine": ("neck", "cervical", "cervicl", "spine", "lumbar", "thoracic"),
    "chest_cardiac": ("chest", "lung", "cardiac", "heart", "coronary", "echo"),
    "breast": ("breast", "mam", "mammo", "mammography"),
    "abdomen_pelvis": ("abdomen", "abdominal", "pelvis", "pelvic", "renal", "kidney", "liver"),
    "upper_extremity": ("shoulder", "humerus", "elbow", "wrist", "hand", "finger"),
    "lower_extremity": ("hip", "knee", "ankle", "foot", "femur", "tibia"),
}

LATERALITY_TERMS = {
    "left": ("left", "lt"),
    "right": ("right", "rt"),
    "bilateral": ("bilat", "bilateral", "bi"),
}

CONTRAST_TERMS = {
    "with_contrast": ("with contrast", "with cntrst", "w con", "wo/w con", "w/contrast"),
    "without_contrast": ("without contrast", "without cntrst", "wo con", "w/o contrast", "w/o cntrst"),
}


@dataclass(frozen=True)
class PriorFeatures:
    days_between_studies: int
    descriptions_match: bool
    modality_matches: bool
    current_modality: str
    prior_modality: str
    current_body_region: str
    prior_body_region: str
    body_region_matches: bool
    token_overlap_count: int
    token_jaccard: float
    laterality_matches: bool
    contrast_matches: bool


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def _tokens(description: str) -> set[str]:
    return set(TOKEN_PATTERN.findall(_normalize(description)))


def _modality(description: str) -> str:
    normalized = _normalize(description)
    for modality, prefixes in MODALITY_PREFIXES.items():
        if normalized.startswith(prefixes):
            return modality
    return "other"


def _first_matching_group(description: str, groups: dict[str, tuple[str, ...]], default: str) -> str:
    normalized = _normalize(description)
    tokens = _tokens(normalized)
    for group_name, terms in groups.items():
        if any(term in tokens or term in normalized for term in terms):
            return group_name
    return default


def _body_region(description: str) -> str:
    return _first_matching_group(description, BODY_REGION_TERMS, "other")


def _laterality(description: str) -> str:
    return _first_matching_group(description, LATERALITY_TERMS, "unspecified")


def _contrast(description: str) -> str:
    return _first_matching_group(description, CONTRAST_TERMS, "unspecified")


def extract_features(
    *,
    current_study_description: str,
    current_study_date: date,
    prior_study_description: str,
    prior_study_date: date,
) -> PriorFeatures:
    current_desc = _normalize(current_study_description)
    prior_desc = _normalize(prior_study_description)
    current_tokens = _tokens(current_desc)
    prior_tokens = _tokens(prior_desc)
    token_overlap = current_tokens & prior_tokens
    token_union = current_tokens | prior_tokens
    current_modality = _modality(current_desc)
    prior_modality = _modality(prior_desc)
    current_body_region = _body_region(current_desc)
    prior_body_region = _body_region(prior_desc)
    return PriorFeatures(
        days_between_studies=abs((current_study_date - prior_study_date).days),
        descriptions_match=current_desc == prior_desc,
        modality_matches=current_modality == prior_modality,
        current_modality=current_modality,
        prior_modality=prior_modality,
        current_body_region=current_body_region,
        prior_body_region=prior_body_region,
        body_region_matches=current_body_region == prior_body_region,
        token_overlap_count=len(token_overlap),
        token_jaccard=len(token_overlap) / len(token_union) if token_union else 0.0,
        laterality_matches=_laterality(current_desc) == _laterality(prior_desc),
        contrast_matches=_contrast(current_desc) == _contrast(prior_desc),
    )


def extract_model_features(
    *,
    current_study_description: str,
    current_study_date: date,
    prior_study_description: str,
    prior_study_date: date,
) -> dict[str, Any]:
    features = extract_features(
        current_study_description=current_study_description,
        current_study_date=current_study_date,
        prior_study_description=prior_study_description,
        prior_study_date=prior_study_date,
    )
    return {
        "days_between_studies": features.days_between_studies,
        "log_days_between_studies": log1p(features.days_between_studies),
        "same_day": features.days_between_studies == 0,
        "within_30_days": features.days_between_studies <= 30,
        "within_1_year": features.days_between_studies <= 365,
        "within_5_years": features.days_between_studies <= 365 * 5,
        "descriptions_match": features.descriptions_match,
        "modality_matches": features.modality_matches,
        "body_region_matches": features.body_region_matches,
        "token_overlap_count": features.token_overlap_count,
        "token_jaccard": features.token_jaccard,
        "laterality_matches": features.laterality_matches,
        "contrast_matches": features.contrast_matches,
        "current_modality": features.current_modality,
        "prior_modality": features.prior_modality,
        "current_body_region": features.current_body_region,
        "prior_body_region": features.prior_body_region,
    }
