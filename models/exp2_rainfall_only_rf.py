"""Model 1: Random Forest, rainfall features only.

Uses the shared processed table, the shared stratified split in
src.preprocessing.get_train_test_split, RF_PARAMS, and
src.evaluation.evaluate_model. Only the feature subset differs.

The rainfall columns below are the same block Experiment 3 trains on,
without terrain, so Model 1 vs Model 3 is a terrain ablation.
configs/config.py names (rain_1h, antecedent_rain_index, flood_label)
do not match data/processed/flood_dataset.csv, so this script uses the
real column names and falls back to the label column that is present.
"""

import json
import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix

from configs.config import METRICS_DIR, RANDOM_SEED, RF_PARAMS, TARGET_COL, TEST_SIZE
from src.evaluation import evaluate_model
from src.preprocessing import build_feature_table, get_train_test_split

EXPERIMENT_NAME = "model_1_rainfall"
FEATURE_COLUMNS = [
    "precip_1h",
    "precip_3h",
    "precip_6h",
    "precip_24h",
    "ari_memory",
]


def run_model_1():
    df = build_feature_table(None)

    missing = [column for column in FEATURE_COLUMNS if column not in df.columns]
    if missing:
        raise KeyError(f"{EXPERIMENT_NAME}: missing columns {missing}")

    target_col = TARGET_COL if TARGET_COL in df.columns else "label"
    if target_col != TARGET_COL:
        print(
            f"[{EXPERIMENT_NAME}] WARNING: TARGET_COL={TARGET_COL!r} is not in "
            f"the processed dataset; using {target_col!r}."
        )

    X_train, X_test, y_train, y_test = get_train_test_split(
        df, FEATURE_COLUMNS, target_col=target_col
    )

    model = RandomForestClassifier(**RF_PARAMS)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    metrics = evaluate_model(y_test, y_pred, y_prob)
    labels = sorted(int(value) for value in y_test.unique())
    cm = confusion_matrix(y_test, y_pred, labels=labels).tolist()

    os.makedirs(METRICS_DIR, exist_ok=True)
    metrics_path = os.path.join(METRICS_DIR, "model_1_rainfall_metrics.json")
    output = {
        "experiment_name": EXPERIMENT_NAME,
        "script": "models/exp2_rainfall_only_rf.py",
        "feature_columns": FEATURE_COLUMNS,
        "target_col": target_col,
        "random_seed": RANDOM_SEED,
        "test_size": TEST_SIZE,
        "rf_params": RF_PARAMS,
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "metrics": metrics,
        "confusion_matrix": {"labels": labels, "matrix": cm},
    }
    with open(metrics_path, "w") as handle:
        json.dump(output, handle, indent=4)

    print(f"[{EXPERIMENT_NAME}] metrics saved to {metrics_path}")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}" if isinstance(value, float) else f"  {key}: {value}")
    print(f"  confusion_matrix labels={labels} matrix={cm}")
    return output


if __name__ == "__main__":
    run_model_1()
