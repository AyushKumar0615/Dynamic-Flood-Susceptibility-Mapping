"""Experiment 3: Random Forest, terrain + rainfall memory features.

Feature set: conventional (non-calculus) terrain features combined
with the team's multi-timescale rainfall features and the antecedent
rainfall (ARI) memory feature. This experiment intentionally excludes
calculus-derived terrain features (gradients, curvature, flow
convergence) and terrain-rainfall interaction terms -- those are
reserved for Experiments 4 and 5 per docs/methodology.md.

Shares the same dataset, preprocessing pipeline (src/preprocessing.py),
train/test split, Random Forest hyperparameters, and evaluation
protocol (src/evaluation.py) as every other experiment in models/.
Only the feature subset below differs.

Known drift between configs/config.py and the processed dataset
(data/processed/flood_dataset.csv), NOT fixed here -- see the task
summary for why this script works around it instead of editing the
shared config:
  - TARGET_COL in config.py is "flood_label"; the actual label column
    in the processed dataset is "label". Resolved below at runtime
    with a printed warning rather than editing the shared config.
  - RAINFALL_MEMORY_FEATURES in config.py (`rain_1h`, `rain_24h`,
    `antecedent_rain_index`, ...) does not match the processed
    dataset's real columns (`precip_1h`, `precip_24h`, `ari_memory`).
    The real column names are used explicitly below.
  - CONVENTIONAL_TERRAIN_FEATURES includes "aspect", which is not a
    column in the processed dataset and is only ever produced inside
    src/terrain_features.compute_terrain_derivatives(), the calculus-
    derived module this experiment must avoid. "aspect" is dropped
    rather than fabricated or recomputed.

ari_memory provenance -- READ THIS BEFORE TRUSTING THIS FEATURE:
  src/rainfall_features.compute_antecedent_rainfall_index() defines
  ARI_t = sum_{i=1..24} exp(-0.1 * i) * P_{t-i}, applied to a column
  named "rain_24h" and written out as "antecedent_rain_index". That
  function is never called anywhere in this repository (checked every
  .py file), and neither its input nor output column name matches the
  processed dataset's actual "ari_memory" column. There is no evidence
  this formula produced the values this script consumes.

  Nothing in this repository establishes that "ari_memory" is strictly
  antecedent (computed only from rainfall before the prediction
  period): data/raw/ is empty, notebooks/01_data_harmonization.py is an
  unimplemented placeholder, and the processed CSV has no timestamp or
  event identifier to audit. Empirically, ari_memory correlates ~0.9997
  with precip_24h at a near-constant ratio (~0.91), which looks more
  like a rescaling of precip_24h than independently computed rainfall
  history.

  This script uses "ari_memory" as the team's designated rainfall-
  memory column AS PROVIDED, without claiming it is verified
  antecedent or leak-free. Flagged for the dataset owner to confirm;
  not resolved here.
"""

import json
import os
import sys

# Make the repo root importable when run directly as
# `python models/exp3_terrain_rainfall_rf.py`, mirroring the BASE_DIR
# pattern already used in configs/config.py.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from configs.config import (
    RANDOM_SEED,
    TEST_SIZE,
    RF_PARAMS,
    METRICS_DIR,
    RESULTS_DIR,
    TARGET_COL,
)
from src.preprocessing import build_feature_table, get_train_test_split
from src.evaluation import evaluate_model

EXPERIMENT_NAME = "exp3_terrain_rainfall_rf"

# Conventional terrain features present in the processed dataset.
TERRAIN_FEATURES = ["elevation", "slope"]

# Multi-timescale rainfall + antecedent rainfall memory (ARI) features,
# using the processed dataset's real column names.
RAINFALL_FEATURES = ["precip_1h", "precip_3h", "precip_6h", "precip_24h", "ari_memory"]

FEATURE_COLUMNS = TERRAIN_FEATURES + RAINFALL_FEATURES


def run_experiment_3():
    df = build_feature_table(None)

    missing = [c for c in FEATURE_COLUMNS if c not in df.columns]
    if missing:
        raise KeyError(
            f"{EXPERIMENT_NAME}: expected feature columns missing from the "
            f"processed dataset: {missing}"
        )

    target_col = TARGET_COL if TARGET_COL in df.columns else "label"
    if target_col != TARGET_COL:
        print(
            f"[{EXPERIMENT_NAME}] WARNING: configs.config.TARGET_COL="
            f"{TARGET_COL!r} not found in the processed dataset; using "
            f"{target_col!r} instead. configs/config.py should be updated "
            f"to match data/processed/flood_dataset.csv."
        )

    print(
        f"[{EXPERIMENT_NAME}] NOTE: 'ari_memory' is consumed as provided "
        f"by the processed dataset; its generation is not reproducible "
        f"from this repository and its temporal antecedence relative to "
        f"'{target_col}' has not been verified (see module docstring)."
    )

    # Central, shared split (stratified random split from
    # src/preprocessing.get_train_test_split) -- no separate split logic.
    X_train, X_test, y_train, y_test = get_train_test_split(
        df, FEATURE_COLUMNS, target_col=target_col
    )

    model = RandomForestClassifier(**RF_PARAMS)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = evaluate_model(y_test, y_pred, y_prob)

    os.makedirs(METRICS_DIR, exist_ok=True)
    metrics_path = os.path.join(METRICS_DIR, "model_3_metrics.json")
    output = {
        "experiment_name": EXPERIMENT_NAME,
        "feature_columns": FEATURE_COLUMNS,
        "target_col": target_col,
        "random_seed": RANDOM_SEED,
        "test_size": TEST_SIZE,
        "rf_params": RF_PARAMS,
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "metrics": metrics,
    }
    with open(metrics_path, "w") as f:
        json.dump(output, f, indent=4)

    print(f"[{EXPERIMENT_NAME}] metrics saved to {metrics_path}")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    # --- Export full-dataset prediction probabilities for later visualization ---
    # Uses the same fitted model and the same FEATURE_COLUMNS, over the full
    # processed dataset in its original row order (the order build_feature_table
    # returned it in -- no shuffling happens before this point). This does NOT
    # reshape predictions, invent coordinates, or produce a geographic map: these
    # are model probabilities per dataset row, nothing more.
    full_prob = model.predict_proba(df[FEATURE_COLUMNS])[:, 1]
    predictions_dir = os.path.join(RESULTS_DIR, "predictions")
    os.makedirs(predictions_dir, exist_ok=True)
    predictions_path = os.path.join(predictions_dir, "model_3_predictions.csv")
    predictions_df = pd.DataFrame(
        {
            "row_index": df.index,
            "flood_susceptibility_probability": full_prob,
            # Observed target (same column resolved above), included for
            # reference only -- this is the ground-truth label, not a prediction.
            target_col: df[target_col].values,
        }
    )
    predictions_df.to_csv(predictions_path, index=False)
    print(f"[{EXPERIMENT_NAME}] predictions saved to {predictions_path}")

    return output


if __name__ == "__main__":
    run_experiment_3()
