from __future__ import annotations

import unittest
from datetime import date

from app.features import extract_features, extract_model_features


class FeatureExtractionTests(unittest.TestCase):
    def test_laterality_match_distinguishes_left_and_right(self) -> None:
        features = extract_features(
            current_study_description="MAM diagnostic RT with tomo",
            current_study_date=date(2023, 2, 27),
            prior_study_description="MAM screen LT with tomo",
            prior_study_date=date(2021, 8, 27),
        )

        self.assertFalse(features.laterality_matches)
        self.assertEqual(features.current_body_region, "breast")
        self.assertEqual(features.prior_body_region, "breast")

    def test_contrast_parsing_distinguishes_with_and_without(self) -> None:
        features = extract_features(
            current_study_description="CT CHEST WITH CNTRST",
            current_study_date=date(2023, 1, 1),
            prior_study_description="CT CHEST WITHOUT CNTRST",
            prior_study_date=date(2022, 1, 1),
        )

        self.assertFalse(features.contrast_matches)
        self.assertTrue(features.modality_matches)

    def test_clinical_family_links_textually_different_breast_studies(self) -> None:
        features = extract_features(
            current_study_description="MAM SCREEN 3D",
            current_study_date=date(2021, 6, 15),
            prior_study_description="ULTRASOUND LT DIAG TARGET",
            prior_study_date=date(2020, 12, 21),
        )

        self.assertIn("breast_imaging", features.current_clinical_families)
        self.assertIn("breast_imaging", features.prior_clinical_families)
        self.assertTrue(features.clinically_related_families)

    def test_clinical_family_links_cardiac_stress_and_coronary_ct(self) -> None:
        feature_row = extract_model_features(
            current_study_description="NM myo perf SPECT rest & str",
            current_study_date=date(2022, 11, 7),
            prior_study_description="CT coronary calc screening",
            prior_study_date=date(2021, 7, 17),
        )

        self.assertTrue(feature_row["current_family=cardiac_coronary"])
        self.assertTrue(feature_row["prior_family=cardiac_coronary"])
        self.assertTrue(feature_row["clinically_related_families"])

    def test_body_region_parsing_handles_renal_and_pelvis_terms(self) -> None:
        features = extract_features(
            current_study_description="US kidneys and bladder",
            current_study_date=date(2021, 5, 3),
            prior_study_description="CT ABD PELVIS WO/W CTS",
            prior_study_date=date(2012, 1, 18),
        )

        self.assertEqual(features.current_body_region, "abdomen_pelvis")
        self.assertEqual(features.prior_body_region, "abdomen_pelvis")
        self.assertTrue(features.clinically_related_families)


if __name__ == "__main__":
    unittest.main()
