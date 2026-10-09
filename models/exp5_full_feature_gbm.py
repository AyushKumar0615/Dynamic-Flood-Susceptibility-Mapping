"""Model 5: Full Feature Gradient Boosting Machine (GBM) Benchmark Script."""

import os
import json
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split

# Safe config imports
try:
    from configs.config import RANDOM_SEED, TEST_SIZE, PROCESSED_DATA_PATH
except ImportError:
    RANDOM_SEED = 42
    TEST_SIZE = 0.2
    PROCESSED_DATA_PATH = "data/processed/flood_dataset.csv"

# Safe feature interaction import
try:
    from src.interaction_features import compute_interaction_features
except ImportError:
    try:
        from src.interaction_features import interaction_features as compute_interaction_features
    except ImportError:
        def compute_interaction_features(df):
            df = df.copy()
            if 'elevation' in df.columns and 'precip_24h' in df.columns:
                df['topo_precip_ratio'] = df['precip_24h'] / (df['elevation'] + 1.0)
            if 'slope' in df.columns and 'flow_convergence' in df.columns:
                df['slope_flow_prod'] = df['slope'] * df['flow_convergence']
            return df

# Safe evaluation metric function import
try:
    from src.evaluation import evaluate_predictions
except ImportError:
    try:
        from src.evaluation import evaluate_model as evaluate_predictions
    except ImportError:
        def evaluate_predictions(y_true, y_pred, y_prob):
            return {
                "accuracy": float(accuracy_score(y_true, y_pred)),
                "precision": float(precision_score(y_true, y_pred, zero_division=0)),
                "recall": float(recall_score(y_true, y_pred, zero_division=0)),
                "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
                "roc_auc": float(roc_auc_score(y_true, y_prob)) if y_prob is not None else 0.0
            }

def build_feature_table(data):
    """Computes interaction features on top of existing raw dataset columns."""
    df_feat = compute_interaction_features(data)
    return df_feat

def run_experiment_5():
    print("Loading harmonized tabular dataset...")
    data_path = PROCESSED_DATA_PATH if os.path.exists(PROCESSED_DATA_PATH) else "data/processed/flood_dataset.csv"

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at {data_path}. Run harmonization first.")

    raw_df = pd.read_csv(data_path)

    print("Building full feature set including interaction terms...")
    df = build_feature_table(raw_df)

    X = df.drop(columns=['label'])
    y = df['label']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=y
    )

    print("Training Model 5 (XGBoost Benchmark)...")
    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=RANDOM_SEED
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = evaluate_predictions(y_test, y_pred, y_prob)

    os.makedirs('results', exist_ok=True)
    metrics_path = 'results/model_5_metrics.json'
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=4)

    print("\nModel 5 Execution Complete!")
    print(f"Metrics Saved to {metrics_path}:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

if __name__ == "__main__":
    run_experiment_5()
