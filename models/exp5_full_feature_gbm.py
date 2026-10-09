"""Experiment 5: Gradient boosting, full feature stack (terrain +
rainfall + interaction), to compare model family against exp4's
Random Forest on identical features.

Must use the shared dataset, preprocessing pipeline
(src/preprocessing.py), the shared event-based train/test split, and
the shared evaluation protocol (src/evaluation.py) — identical to
every other experiment in models/. Only the feature set and/or model
type below may differ.
"""

CONFIG = {
    "experiment_name": "exp5_full_feature_gbm",
    "feature_sets": ["terrain", "rainfall", "interaction"],
    "model": {"type": "XGBClassifier", "params": {}},
}

if __name__ == "__main__":
    raise NotImplementedError("Model training not yet implemented.")
