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

CLINICAL_FAMILY_TERMS = {
    "breast_imaging": (
        "mam",
        "mammography",
        "mammo",
        "breast",
        "screen",
        "screening",
        "screener",
        "diagnostic",
        "diag",
        "tomo",
        "cad",
        "stereo",
        "biopsy",
        "bx",
        "localization",
        "ultrasound",
        "target",
    ),
    "cardiac_coronary": (
        "echo",
        "tte",
        "myo",
        "myocardial",
        "perfusion",
        "spect",
        "stress",
        "coronary",
        "cardiac",
        "calcium",
        "calc",
    ),
    "vascular_angio": (
        "angio",
        "angiogram",
        "cta",
        "carotid",
        "arterial",
        "venous",
        "vascular",
        "doppler",
    ),
    "pet_oncology": ("pet", "piflu", "skullthigh", "skull", "thigh", "f18"),
    "chest_lung": ("chest", "lung", "lungs", "pulmonary", "thorax", "ribs", "rib"),
    "ribs_thoracic_spine": ("rib", "ribs", "thoracic", "t-spine", "tspine", "sternum"),
    "spine": ("spine", "spinal", "cervical", "cervicl", "thoracic", "lumbar"),
    "head_neck_neuro": (
        "head",
        "brain",
        "stroke",
        "cerebral",
        "neck",
        "carotid",
        "cervical",
        "cervicl",
    ),
    "abdomen_pelvis": (
        "abdomen",
        "abdominal",
        "abd",
        "pelvis",
        "pelvic",
        "pel",
        "enterography",
        "urogram",
    ),
    "renal_urinary": ("renal", "kidney", "kidneys", "bladder", "urogram", "hematuria", "uro"),
    "hepatobiliary": ("liver", "hepatic", "biliary", "cholangiography", "gallbladder"),
    "lower_extremity": ("leg", "hip", "femur", "knee", "ankle", "foot", "pelvis"),
}

CLINICALLY_RELATED_FAMILY_PAIRS = {
    frozenset(("breast_imaging",)),
    frozenset(("cardiac_coronary", "vascular_angio")),
    frozenset(("cardiac_coronary", "chest_lung")),
    frozenset(("pet_oncology", "chest_lung")),
    frozenset(("pet_oncology", "abdomen_pelvis")),
    frozenset(("pet_oncology", "breast_imaging")),
    frozenset(("chest_lung", "ribs_thoracic_spine")),
    frozenset(("chest_lung", "spine")),
    frozenset(("ribs_thoracic_spine", "spine")),
    frozenset(("head_neck_neuro", "vascular_angio")),
    frozenset(("head_neck_neuro", "spine")),
    frozenset(("abdomen_pelvis", "renal_urinary")),
    frozenset(("abdomen_pelvis", "hepatobiliary")),
    frozenset(("abdomen_pelvis", "lower_extremity")),
    frozenset(("renal_urinary", "hepatobiliary")),
    frozenset(("vascular_angio", "lower_extremity")),
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
    current_clinical_families: tuple[str, ...]
    prior_clinical_families: tuple[str, ...]
    shared_clinical_family_count: int
    clinically_related_families: bool


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


def _clinical_families(description: str) -> tuple[str, ...]:
    normalized = _normalize(description)
    tokens = _tokens(normalized)
    families: list[str] = []
    for family_name, terms in CLINICAL_FAMILY_TERMS.items():
        if any(term in tokens or term in normalized for term in terms):
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
