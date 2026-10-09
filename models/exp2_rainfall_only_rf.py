"""Experiment 2: Random Forest, rainfall features only.

Must use the shared dataset, preprocessing pipeline
(src/preprocessing.py), the shared event-based train/test split, and
the shared evaluation protocol (src/evaluation.py) — identical to
every other experiment in models/. Only the feature set and/or model
type below may differ.
"""

CONFIG = {
    "experiment_name": "exp2_rainfall_only_rf",
    "feature_sets": ["rainfall"],
    "model": {"type": "RandomForestClassifier", "params": {}},
}

if __name__ == "__main__":
    raise NotImplementedError("Model training not yet implemented.")
