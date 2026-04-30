from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import date
from pathlib import Path
from typing import Any

from app.features import extract_features


DEFAULT_PAYLOAD_PATH = Path("relevant_priors_public.json")
DEFAULT_LIMIT = 20


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


def _print_counter(title: str, counter: Counter[str], total: int, limit: int) -> None:
    print()
    print(title)
    for value, count in counter.most_common(limit):
        share = count / total if total else 0
        print(f"  {count:5d}  {share:7.2%}  {value}")


def analyze(payload_path: Path, limit: int) -> None:
    payload = _load_payload(payload_path)
    cases = payload.get("cases", [])
    total_priors = sum(len(case["prior_studies"]) for case in cases)

    current_modality_counts: Counter[str] = Counter()
    prior_modality_counts: Counter[str] = Counter()
    modality_pair_counts: Counter[str] = Counter()
    region_pair_counts: Counter[str] = Counter()
    age_bucket_counts: Counter[str] = Counter()
    overlap_bucket_counts: Counter[str] = Counter()
    risky_shift_counts: Counter[str] = Counter()
    current_description_counts: Counter[str] = Counter()
    prior_description_counts: Counter[str] = Counter()
    case_prior_counts: Counter[str] = Counter()

    for case in cases:
        current_study = case["current_study"]
        current_date = date.fromisoformat(current_study["study_date"])
        case_prior_count = len(case["prior_studies"])
        if case_prior_count == 0:
            case_prior_counts["0"] += 1
        elif case_prior_count <= 5:
            case_prior_counts["1-5"] += 1
        elif case_prior_count <= 20:
            case_prior_counts["6-20"] += 1
        elif case_prior_count <= 50:
            case_prior_counts["21-50"] += 1
        else:
            case_prior_counts["51+"] += 1

        current_description_counts[current_study["study_description"].strip().lower()] += 1
        for prior_study in case["prior_studies"]:
            prior_date = date.fromisoformat(prior_study["study_date"])
            features = extract_features(
                current_study_description=current_study["study_description"],
                current_study_date=current_date,
                prior_study_description=prior_study["study_description"],
                prior_study_date=prior_date,
            )
            current_modality_counts[features.current_modality] += 1
            prior_modality_counts[features.prior_modality] += 1
            modality_pair_counts[f"{features.current_modality}->{features.prior_modality}"] += 1
            region_pair_counts[f"{features.current_body_region}->{features.prior_body_region}"] += 1
            age_bucket_counts[_age_bucket(features.days_between_studies)] += 1
            overlap_bucket_counts[f"{features.token_jaccard:.1f}"] += 1
            prior_description_counts[prior_study["study_description"].strip().lower()] += 1

            if not features.body_region_matches and features.token_jaccard == 0:
                risky_shift_counts["region_mismatch_zero_overlap"] += 1
            if features.current_modality == "mam" or features.prior_modality == "mam":
                risky_shift_counts["mammography_involved"] += 1
            if features.current_body_region == "chest_cardiac" or features.prior_body_region == "chest_cardiac":
                risky_shift_counts["chest_or_cardiac_involved"] += 1
            if features.days_between_studies > 365 * 5:
                risky_shift_counts["older_than_5_years"] += 1
            if features.laterality_matches is False:
                risky_shift_counts["laterality_mismatch"] += 1

    print(f"payload: {payload_path}")
    print(f"cases: {len(cases)}")
    print(f"previous examinations: {total_priors}")
    print(f"average priors per case: {total_priors / len(cases):.2f}" if cases else "average priors per case: 0")

    _print_counter("Case prior-count buckets:", case_prior_counts, len(cases), limit)
    _print_counter("Current modality distribution:", current_modality_counts, total_priors, limit)
    _print_counter("Prior modality distribution:", prior_modality_counts, total_priors, limit)
    _print_counter("Modality pair distribution:", modality_pair_counts, total_priors, limit)
    _print_counter("Body-region pair distribution:", region_pair_counts, total_priors, limit)
    _print_counter("Age bucket distribution:", age_bucket_counts, total_priors, limit)
    _print_counter("Token Jaccard bucket distribution:", overlap_bucket_counts, total_priors, limit)
    _print_counter("Risky distribution markers:", risky_shift_counts, total_priors, limit)
    _print_counter("Top current descriptions:", current_description_counts, len(cases), limit)
    _print_counter("Top prior descriptions:", prior_description_counts, total_priors, limit)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize challenge payload distribution without labels.")
    parser.add_argument("--payload", type=Path, default=DEFAULT_PAYLOAD_PATH)
    parser.add_argument("--limit", type=int, default=DEFAULT_LIMIT)
    args = parser.parse_args()
    analyze(args.payload, args.limit)


if __name__ == "__main__":
    main()
