"""Experiment 1: Random Forest, terrain features only.

Must use the shared dataset, preprocessing pipeline
(src/preprocessing.py), the shared event-based train/test split, and
the shared evaluation protocol (src/evaluation.py) — identical to
every other experiment in models/. Only the feature set and/or model
type below may differ.
"""

CONFIG = {
    "experiment_name": "exp1_terrain_only_rf",
    "feature_sets": ["terrain"],
    "model": {"type": "RandomForestClassifier", "params": {}},
}

if __name__ == "__main__":
    raise NotImplementedError("Model training not yet implemented.")
