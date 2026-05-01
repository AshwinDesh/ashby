from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date
from math import log1p
from typing import Any

from app.taxonomy import (
    BODY_REGION_TERMS,
    CLINICAL_FAMILY_TERMS,
    CLINICALLY_RELATED_FAMILY_PAIRS,
    CONTRAST_TERMS,
    GROUP_TERMS,
    LATERALITY_TERMS,
    MODALITY_PREFIXES,
)


TOKEN_PATTERN = re.compile(r"[a-z0-9]+")


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
    current_clinical_families: tuple[str, ...]
    prior_clinical_families: tuple[str, ...]
    shared_clinical_family_count: int
    clinically_related_families: bool


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def _tokens(description: str) -> set[str]:
    return set(TOKEN_PATTERN.findall(_normalize(description)))


def _contains_phrase(normalized: str, phrase: str) -> bool:
    return bool(re.search(rf"(?<![a-z0-9]){re.escape(phrase)}(?![a-z0-9])", normalized))


def _has_term(normalized: str, tokens: set[str], term: str) -> bool:
    if " " in term or "/" in term or "-" in term:
        return _contains_phrase(normalized, term)
    return term in tokens


def _has_any_term(normalized: str, tokens: set[str], terms: tuple[str, ...]) -> bool:
    return any(_has_term(normalized, tokens, term) for term in terms)


def _modality(description: str) -> str:
    normalized = _normalize(description)
    for modality, prefixes in MODALITY_PREFIXES.items():
        if normalized.startswith(prefixes):
            return modality
    return "other"


def _first_matching_group(description: str, groups: GROUP_TERMS, default: str) -> str:
    normalized = _normalize(description)
    tokens = _tokens(normalized)
    for group_name, terms in groups.items():
        if _has_any_term(normalized, tokens, terms):
            return group_name
    return default


def _body_region(description: str) -> str:
    return _first_matching_group(description, BODY_REGION_TERMS, "other")


def _laterality(description: str) -> str:
    return _first_matching_group(description, LATERALITY_TERMS, "unspecified")


def _contrast(description: str) -> str:
    return _first_matching_group(description, CONTRAST_TERMS, "unspecified")


def _clinical_families(description: str) -> tuple[str, ...]:
    normalized = _normalize(description)
    tokens = _tokens(normalized)
    families: list[str] = []
    for family_name, terms in CLINICAL_FAMILY_TERMS.items():
        if _has_any_term(normalized, tokens, terms):
            families.append(family_name)
    return tuple(families)


def _clinically_related(
    current_families: tuple[str, ...],
    prior_families: tuple[str, ...],
) -> bool:
    current_family_set = set(current_families)
    prior_family_set = set(prior_families)
    if current_family_set & prior_family_set:
        return True

    for current_family in current_family_set:
        for prior_family in prior_family_set:
            if frozenset((current_family, prior_family)) in CLINICALLY_RELATED_FAMILY_PAIRS:
                return True
    return False


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
    current_clinical_families = _clinical_families(current_desc)
    prior_clinical_families = _clinical_families(prior_desc)
    shared_clinical_families = set(current_clinical_families) & set(prior_clinical_families)
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
        current_clinical_families=current_clinical_families,
        prior_clinical_families=prior_clinical_families,
        shared_clinical_family_count=len(shared_clinical_families),
        clinically_related_families=_clinically_related(
            current_clinical_families,
            prior_clinical_families,
        ),
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
    feature_row = {
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
        "shared_clinical_family_count": features.shared_clinical_family_count,
        "clinically_related_families": features.clinically_related_families,
    }
    for family in CLINICAL_FAMILY_TERMS:
        feature_row[f"current_family={family}"] = family in features.current_clinical_families
        feature_row[f"prior_family={family}"] = family in features.prior_clinical_families
        feature_row[f"shared_family={family}"] = (
            family in features.current_clinical_families
            and family in features.prior_clinical_families
        )
    return feature_row
