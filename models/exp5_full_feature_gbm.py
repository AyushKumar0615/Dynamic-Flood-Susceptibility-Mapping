"""Model / Experiment 5: Full Proposed Feature Model.

Evaluates the primary research hypothesis using:
- Conventional Terrain Features
- Calculus-derived Flow Convergence / Curvature
- Multi-timescale Rainfall Memory
- Calculus x Rainfall Interaction Features
"""

import json
import os
import joblib
import pandas as pd
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier

from configs.config import (
    MODEL_5_ALL_FEATURES,
    XGB_PARAMS,
    RF_PARAMS,
    METRICS_DIR
)
from src.preprocessing import build_feature_table, get_train_test_split
from src.interaction_features import compute_interaction_terms
from src.evaluation import evaluate_model  # Assumes standard evaluation metric helper


def run_experiment_5():
    # 1. Load harmonized dataset
    df = build_feature_table()
    
    # 2. Derive calculus x rainfall interaction features
    df = compute_interaction_terms(df)
    
    # 3. Get consistent train-test split for Model 5 features
    X_train, X_test, y_train, y_test = get_train_test_split(
        df, 
        feature_cols=MODEL_5_ALL_FEATURES
    )
    
    # 4. Train XGBoost Model
    xgb_model = XGBClassifier(**XGB_PARAMS)
    xgb_model.fit(X_train, y_train)
    xgb_preds = xgb_model.predict(X_test)
    xgb_probs = xgb_model.predict_proba(X_test)[:, 1]
    
    # 5. Train Random Forest Model
    rf_model = RandomForestClassifier(**RF_PARAMS)
    rf_model.fit(X_train, y_train)
    rf_preds = rf_model.predict(X_test)
    rf_probs = rf_model.predict_proba(X_test)[:, 1]
    
    # 6. Evaluate and save metrics
    xgb_metrics = evaluate_model(y_test, xgb_preds, xgb_probs)
    rf_metrics = evaluate_model(y_test, rf_preds, rf_probs)
    
    results = {
        "experiment": "Model_5_Full_Proposed",
        "xgboost": xgb_metrics,
        "random_forest": rf_metrics
    }
    
    os.makedirs(METRICS_DIR, exist_ok=True)
    with open(os.path.join(METRICS_DIR, "model_5_metrics.json"), "w") as f:
        json.dump(results, f, indent=4)
        
    print("Model 5 evaluation complete. Metrics saved to results/metrics/model_5_metrics.json")

if __name__ == "__main__":
    run_experiment_5()
