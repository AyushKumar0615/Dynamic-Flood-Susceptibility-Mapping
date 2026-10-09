"""Experiment 3: Random Forest, combined terrain + rainfall features
(static baseline, no interaction terms).

Must use the shared dataset, preprocessing pipeline
(src/preprocessing.py), the shared event-based train/test split, and
the shared evaluation protocol (src/evaluation.py) — identical to
every other experiment in models/. Only the feature set and/or model
type below may differ.
"""

CONFIG = {
    "experiment_name": "exp3_terrain_rainfall_rf",
    "feature_sets": ["terrain", "rainfall"],
    "model": {"type": "RandomForestClassifier", "params": {}},
}

if __name__ == "__main__":
    raise NotImplementedError("Model training not yet implemented.")
