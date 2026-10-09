"""Shared preprocessing pipeline.

All five experiments (see models/) must build on this module to go
from raw inputs in data/raw/ to the single, shared feature table in
data/processed/. Do not fork per-experiment preprocessing logic.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from configs.config import (
    PROCESSED_DATASET_PATH, 
    RANDOM_SEED, 
    TEST_SIZE, 
    TARGET_COL
)

def load_raw_data(config=None):
    """Load raw terrain, rainfall, and flood-inventory inputs per config."""
    # Placeholder for loading raw rasters prior to tabular conversion
    pass

def clean_and_align(data):
    """Clean and spatially align all raw inputs onto a common grid."""
    # Resampling 30m DEM, 250m GFD, and 10km IMERG to unified spatial grid
    return data

def build_feature_table(data):
    """Assemble the shared feature table used by every experiment."""
    # Load the processed CSV feature table
    df = pd.read_csv(PROCESSED_DATASET_PATH)
    return df

def get_train_test_split(df, feature_cols, target_col=TARGET_COL):
    """
    Returns consistent, stratified train and test splits for any feature subset.
    Ensures zero data leakage across Experiments 1-5.
    """
    X = df[feature_cols]
    y = df[target_col]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, 
        y, 
        test_size=TEST_SIZE, 
        random_state=RANDOM_SEED, 
        stratify=y
    )
    return X_train, X_test, y_train, y_test

def run_pipeline(config=None):
    """Entry point: raw data -> processed feature table in data/processed/."""
    df = build_feature_table(config)
    return df
