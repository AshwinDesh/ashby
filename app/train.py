from __future__ import annotations

from pathlib import Path

import joblib


def train_and_save() -> None:
    # Placeholder artifact for local development.
    serialized_model = {"name": "rule_based_relevant_prior_v1"}
    output_path = Path("models/model.joblib")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(serialized_model, output_path)
    print(f"Saved model artifact to {output_path}")


if __name__ == "__main__":
    train_and_save()
