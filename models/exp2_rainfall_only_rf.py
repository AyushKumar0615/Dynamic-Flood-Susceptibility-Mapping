"""Model 1: rainfall only.

Features are precip_1h, precip_3h, precip_6h, and precip_24h. ari_memory
is left out because it copies precip_24h in this CSV.

Two classifiers are trained with the shared RF_PARAMS and XGB_PARAMS,
the same pair Model 4 and Model 5 use. Each one gets an India map:
longitude 68-97 E, latitude 8-37 N, cells outside the India boundary
hidden. The shared split matches the other experiments. The spatial
block split is the stricter check.
"""

import json
import os
import sys

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from configs.config import (
    FIGURES_DIR,
    MAPS_DIR,
    METRICS_DIR,
    RANDOM_SEED,
    RF_PARAMS,
    TARGET_COL,
    TEST_SIZE,
    XGB_PARAMS,
)
from src.evaluation import evaluate_model, save_confusion_matrix, save_roc_curves
from src.mapping import plot_india_susceptibility_map
from src.preprocessing import build_feature_table, get_train_test_split

EXPERIMENT_NAME = "model_1_rainfall"
FEATURE_COLUMNS = ["precip_1h", "precip_3h", "precip_6h", "precip_24h"]
BLOCK = 10
MAP_FILES = {
    "RandomForest": "model_1_rf_susceptibility_map.png",
    "XGBoost": "model_1_xgb_susceptibility_map.png",
}
MAP_TITLES = {
    "RandomForest": "India-Wide Flood Susceptibility Map\nModel 1: Random Forest (Rainfall)",
    "XGBoost": "India-Wide Flood Susceptibility Map\nModel 1: XGBoost (Rainfall)",
}


def make_models():
    return {
        "RandomForest": RandomForestClassifier(**RF_PARAMS),
        "XGBoost": XGBClassifier(**XGB_PARAMS),
    }


def spatial_block_indices(n_rows, y, block=BLOCK):
    """Hold out whole blocks. Row i sits at (i // side, i % side)."""
    side = int(np.sqrt(n_rows))
    if side * side != n_rows or side % block != 0:
        raise ValueError(f"expected a square grid divisible by {block}, got {n_rows} rows")
    index = np.arange(n_rows)
    rows, cols = index // side, index % side
    blocks_on_side = side // block
    block_id = (rows // block) * blocks_on_side + (cols // block)
    blocks = np.unique(block_id)
    majority = [int(y[block_id == block_i].mean() >= 0.5) for block_i in blocks]
    train_blocks, test_blocks = train_test_split(
        blocks,
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED,
        stratify=majority,
    )
    test_mask = np.isin(block_id, test_blocks)
    return np.where(~test_mask)[0], np.where(test_mask)[0]


def score_split(df, train_index, test_index, target_col):
    x_train = df.iloc[train_index][FEATURE_COLUMNS]
    y_train = df.iloc[train_index][target_col]
    x_test = df.iloc[test_index][FEATURE_COLUMNS]
    y_test = df.iloc[test_index][target_col]
    scores = {}
    fitted = {}
    for name, model in make_models().items():
        model.fit(x_train, y_train)
        probability = model.predict_proba(x_test)[:, 1]
        prediction = (probability >= 0.5).astype(int)
        labels = sorted(int(value) for value in y_test.unique())
        matrix = confusion_matrix(y_test, prediction, labels=labels).tolist()
        scores[name] = {
            "threshold": 0.5,
            "n_train": int(len(train_index)),
            "n_test": int(len(test_index)),
            "metrics": evaluate_model(y_test, prediction, probability),
            "confusion_matrix": {"labels": labels, "matrix": matrix},
            "y_true": np.asarray(y_test),
            "y_prob": np.asarray(probability),
        }
        fitted[name] = model
    return scores, fitted


def save_india_maps(df, fitted, target_col):
    """Score every cell with each fitted classifier and draw an India map."""
    paths = {}
    features = df[FEATURE_COLUMNS]
    for name, model in fitted.items():
        probability = model.predict_proba(features)[:, 1]
        if len(probability) != 14400 or probability.min() < 0 or probability.max() > 1:
            raise ValueError(f"{name} map probabilities must be 14,400 values in [0, 1].")
        save_path = os.path.join(MAPS_DIR, MAP_FILES[name])
        os.makedirs(MAPS_DIR, exist_ok=True)
        plot_india_susceptibility_map(
            probability,
            save_path,
            title=MAP_TITLES[name],
            colorbar_label=f"Predicted Flood Susceptibility Probability (Model 1 {name})",
        )
        paths[name] = save_path
        print(f"[{EXPERIMENT_NAME}] {name} map saved to {save_path}")
    return paths


def run_model_1():
    df = build_feature_table(None).reset_index(drop=True)
    missing = [column for column in FEATURE_COLUMNS if column not in df.columns]
    if missing:
        raise KeyError(f"{EXPERIMENT_NAME}: missing columns {missing}")

    target_col = TARGET_COL if TARGET_COL in df.columns else "label"
    if target_col != TARGET_COL:
        print(
            f"[{EXPERIMENT_NAME}] WARNING: TARGET_COL={TARGET_COL!r} is not in "
            f"the processed dataset; using {target_col!r}."
        )

    shared_x_train, shared_x_test, _, _ = get_train_test_split(
        df, FEATURE_COLUMNS, target_col=target_col
    )
    shared_scores, shared_fitted = score_split(
        df, shared_x_train.index, shared_x_test.index, target_col
    )
    for name in shared_scores:
        shared_scores[name]["protocol"] = "src.preprocessing.get_train_test_split"

    train_index, test_index = spatial_block_indices(len(df), df[target_col].to_numpy())
    spatial_scores, _ = score_split(df, train_index, test_index, target_col)
    for name in spatial_scores:
        spatial_scores[name]["protocol"] = f"{BLOCK}x{BLOCK} spatial blocks, row-major grid"
        spatial_scores[name]["block"] = BLOCK

    roc_curves = []
    confusion_figures = {}
    models_out = {}
    for name in ("RandomForest", "XGBoost"):
        shared = shared_scores[name]
        spatial = spatial_scores[name]
        roc_curves.append((f"{name} shared", shared.pop("y_true"), shared.pop("y_prob")))
        roc_curves.append((f"{name} spatial", spatial.pop("y_true"), spatial.pop("y_prob")))
        confusion_figures[name] = {}
        for split_name, split, split_label in (
            ("shared_split", shared, "shared split"),
            ("spatial_block_split", spatial, "spatial block split"),
        ):
            slug = "shared" if split_name == "shared_split" else "spatial"
            filename = f"model_1_{name.lower()}_confusion_matrix_{slug}.png"
            figure_path = os.path.join(FIGURES_DIR, filename)
            save_confusion_matrix(
                split["confusion_matrix"]["matrix"],
                split["confusion_matrix"]["labels"],
                figure_path,
                f"Model 1 {name} confusion matrix — {split_label}",
            )
            confusion_figures[name][split_name] = figure_path
        models_out[name] = {
            "shared_split": shared,
            "spatial_block_split": spatial,
        }

    roc_path = os.path.join(FIGURES_DIR, "model_1_roc_curve.png")
    save_roc_curves(roc_curves, roc_path, "Model 1 ROC curve — rainfall")
    map_paths = save_india_maps(df, shared_fitted, target_col)
    for name, path in map_paths.items():
        models_out[name]["map"] = path

    output = {
        "experiment_name": EXPERIMENT_NAME,
        "script": "models/exp2_rainfall_only_rf.py",
        "feature_columns": FEATURE_COLUMNS,
        "dropped_columns": ["ari_memory"],
        "target_col": target_col,
        "random_seed": RANDOM_SEED,
        "test_size": TEST_SIZE,
        "rf_params": RF_PARAMS,
        "xgb_params": XGB_PARAMS,
        "models": models_out,
        "figures": {
            "roc_curve": roc_path,
            "confusion_matrices": confusion_figures,
            "india_maps": map_paths,
        },
    }
    os.makedirs(METRICS_DIR, exist_ok=True)
    metrics_path = os.path.join(METRICS_DIR, "model_1_rainfall_metrics.json")
    with open(metrics_path, "w") as handle:
        json.dump(output, handle, indent=4)

    print(f"[{EXPERIMENT_NAME}] metrics saved to {metrics_path}")
    print(f"[{EXPERIMENT_NAME}] ROC curve saved to {roc_path}")
    for name, result in models_out.items():
        for split_name in ("shared_split", "spatial_block_split"):
            metrics = result[split_name]["metrics"]
            matrix = result[split_name]["confusion_matrix"]["matrix"]
            print(f"  {name} {split_name} roc_auc {metrics['roc_auc']:.4f}")
            print(f"    confusion_matrix {matrix}")
    return output


if __name__ == "__main__":
    run_model_1()
