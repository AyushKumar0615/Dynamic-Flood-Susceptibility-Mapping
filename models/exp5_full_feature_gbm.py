"""Model 5: Full Feature Gradient Boosting Machine (GBM) Benchmark Script.

Executes Model 5 using the full feature set (Calculus Terrain Derivatives + 
Antecedent Rainfall Index Decay Memory + Feature Interaction Terms).
"""

import os
import json
import pandas as pd
import numpy as np
import xgboost as xgb
from configs.config import PROCESSED_DATA_PATH, RANDOM_SEED, TEST_SIZE
from src.interaction_features import compute_interaction_features
from src.evaluation import evaluate_predictions

def build_feature_table(data):
    """Computes interaction features on top of existing raw dataset columns."""
    df_feat = compute_interaction_features(data)
    return df_feat

def run_experiment_5():
    print("Loading harmonized tabular dataset...")
    if not os.path.exists(PROCESSED_DATA_PATH):
        raise FileNotFoundError(f"Dataset not found at {PROCESSED_DATA_PATH}. Run harmonization first.")
    
    raw_df = pd.read_csv(PROCESSED_DATA_PATH)
    
    print("Building full feature set including interaction terms...")
    df = build_feature_table(raw_df)
    
    X = df.drop(columns=['label'])
    y = df['label']
    
    # Train-Test Split matching central configuration
    from sklearn.model_selection import train_test_split
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
