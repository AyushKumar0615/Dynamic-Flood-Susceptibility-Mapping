"""Experiment 4: Random Forest, terrain + rainfall + interaction
features (the "dynamic" susceptibility signal).

Must use the shared dataset, preprocessing pipeline
(src/preprocessing.py), the shared event-based train/test split, and
the shared evaluation protocol (src/evaluation.py) — identical to
every other experiment in models/. Only the feature set and/or model
type below may differ.
"""

CONFIG = {
    "experiment_name": "exp4_terrain_rainfall_interaction_rf",
    "feature_sets": ["terrain", "rainfall", "interaction"],
    "model": {"type": "RandomForestClassifier", "params": {}},
}

if __name__ == "__main__":
    raise NotImplementedError("Model training not yet implemented.")
