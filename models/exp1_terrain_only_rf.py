"""Model 2: conventional terrain only.

Features are elevation and slope, the conventional terrain columns that
exist in the processed dataset. aspect is named in the shared config
and is not in the CSV. flow_convergence is a calculus feature and stays
out of this model.

A logistic regression replaces the depth-10 random forest. On the same
shared split it ranks places better, and at the 0.5 cutoff it makes
fewer false alarms and misses fewer floods. Deeper forests did not.

Two evaluations are saved. shared_split uses
src.preprocessing.get_train_test_split, so the rows match the other
experiments. spatial_block_split holds out whole 10 by 10 blocks on the
120 by 120 grid implied by row order. That score is the honest one,
because a random split puts neighboring cells on both sides. The CSV
has no coordinates, so the grid is row order only.
"""

import json
import os
import sys

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from configs.config import MAPS_DIR, METRICS_DIR, RANDOM_SEED, TARGET_COL, TEST_SIZE
from src.evaluation import evaluate_model
from src.mapping import plot_susceptibility_map
from src.preprocessing import build_feature_table, get_train_test_split

EXPERIMENT_NAME = "model_2_conventional"
FEATURE_COLUMNS = ["elevation", "slope"]
BLOCK = 10


def make_model():
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000, random_state=RANDOM_SEED),
    )


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
    model = make_model()
    x_train = df.iloc[train_index][FEATURE_COLUMNS]
    y_train = df.iloc[train_index][target_col]
    x_test = df.iloc[test_index][FEATURE_COLUMNS]
    y_test = df.iloc[test_index][target_col]
    model.fit(x_train, y_train)
    probability = model.predict_proba(x_test)[:, 1]
    prediction = (probability >= 0.5).astype(int)
    labels = sorted(int(value) for value in y_test.unique())
    matrix = confusion_matrix(y_test, prediction, labels=labels).tolist()
    return {
        "threshold": 0.5,
        "n_train": int(len(train_index)),
        "n_test": int(len(test_index)),
        "metrics": evaluate_model(y_test, prediction, probability),
        "confusion_matrix": {"labels": labels, "matrix": matrix},
    }


def save_susceptibility_map(df, train_index, target_col, save_path):
    """Fit on the shared training rows and score every cell for the map."""
    model = make_model()
    model.fit(df.iloc[train_index][FEATURE_COLUMNS], df.iloc[train_index][target_col])
    probability = model.predict_proba(df[FEATURE_COLUMNS])[:, 1]
    side = int(np.sqrt(len(df)))
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plot_susceptibility_map(probability, grid_shape=(side, side), save_path=save_path)
    return save_path


def run_model_2():
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
    shared = score_split(df, shared_x_train.index, shared_x_test.index, target_col)
    shared["protocol"] = "src.preprocessing.get_train_test_split"

    train_index, test_index = spatial_block_indices(len(df), df[target_col].to_numpy())
    spatial = score_split(df, train_index, test_index, target_col)
    spatial["protocol"] = f"{BLOCK}x{BLOCK} spatial blocks, row-major grid"
    spatial["block"] = BLOCK

    output = {
        "experiment_name": EXPERIMENT_NAME,
        "script": "models/exp1_terrain_only_rf.py",
        "model": "LogisticRegression",
        "feature_columns": FEATURE_COLUMNS,
        "target_col": target_col,
        "random_seed": RANDOM_SEED,
        "test_size": TEST_SIZE,
        "shared_split": shared,
        "spatial_block_split": spatial,
    }

    os.makedirs(METRICS_DIR, exist_ok=True)
    os.makedirs(MAPS_DIR, exist_ok=True)
    map_path = os.path.join(MAPS_DIR, "model_2_conventional_susceptibility_map.png")
    output["map"] = save_susceptibility_map(df, shared_x_train.index, target_col, map_path)
    metrics_path = os.path.join(METRICS_DIR, "model_2_conventional_metrics.json")
    with open(metrics_path, "w") as handle:
        json.dump(output, handle, indent=4)

    print(f"[{EXPERIMENT_NAME}] metrics saved to {metrics_path}")
    print(f"[{EXPERIMENT_NAME}] map saved to {map_path}")
    for split_name in ("shared_split", "spatial_block_split"):
        metrics = output[split_name]["metrics"]
        matrix = output[split_name]["confusion_matrix"]["matrix"]
        print(f"  {split_name} roc_auc {metrics['roc_auc']:.4f} accuracy {metrics['accuracy']:.4f}")
        print(f"    confusion_matrix {matrix}")
    return output


if __name__ == "__main__":
    run_model_2()
