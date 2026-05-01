from __future__ import annotations

import unittest

from fastapi.testclient import TestClient

from app.main import app


class ApiContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = TestClient(app)

    def test_health_exposes_loaded_model_metadata(self) -> None:
        response = self.client.get("/health")

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["status"], "ok")
        self.assertEqual(body["model"]["name"], "random_forest_relevant_prior_v1")
        self.assertEqual(body["model"]["model_worker_count"], 1)

    def test_predict_returns_one_prediction_per_prior(self) -> None:
        payload = {
            "challenge_id": "relevant-priors-v1",
            "schema_version": 1,
            "generated_at": "2026-04-16T12:00:00.000Z",
            "cases": [
                {
                    "case_id": "1001016",
                    "patient_id": "606707",
                    "patient_name": "Andrews, Micheal",
                    "current_study": {
                        "study_id": "3100042",
                        "study_description": "MRI BRAIN STROKE LIMITED WITHOUT CONTRAST",
                        "study_date": "2026-03-08",
                    },
                    "prior_studies": [
                        {
                            "study_id": "2453245",
                            "study_description": "MRI BRAIN STROKE LIMITED WITHOUT CONTRAST",
                            "study_date": "2020-03-08",
                        },
                        {
                            "study_id": "992654",
                            "study_description": "CT HEAD WITHOUT CNTRST",
                            "study_date": "2021-03-08",
                        },
                    ],
                }
            ],
        }

        response = self.client.post("/predict", json=payload)

        self.assertEqual(response.status_code, 200)
        predictions = response.json()["predictions"]
        self.assertEqual(len(predictions), 2)
        self.assertEqual(
            {(prediction["case_id"], prediction["study_id"]) for prediction in predictions},
            {("1001016", "2453245"), ("1001016", "992654")},
        )
        self.assertTrue(all(isinstance(prediction["predicted_is_relevant"], bool) for prediction in predictions))

    def test_predict_trailing_slash_is_supported(self) -> None:
        payload = {
            "challenge_id": "relevant-priors-v1",
            "schema_version": 1,
            "generated_at": "2026-04-16T12:00:00.000Z",
            "cases": [],
        }

        response = self.client.post("/predict/", json=payload)

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"predictions": []})


if __name__ == "__main__":
    unittest.main()
