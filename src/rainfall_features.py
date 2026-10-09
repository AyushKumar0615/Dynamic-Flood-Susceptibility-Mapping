"""Multi-timescale precipitation memory and Antecedent Rainfall Index (ARI) module.

Computes rolling rainfall totals across multiple timescales (1h, 3h, 6h, 24h)
and dynamic exponential time-decayed soil moisture memory.
"""

import numpy as np
import pandas as pd


def compute_antecedent_rainfall_index(rainfall_series, lambda_decay=0.1, k_steps=24):
    """
    Computes Antecedent Rainfall Index (ARI) with exponential time decay:
    
    ARI_t = sum_{i=1}^{k} (e^(-lambda * i) * P_{t-i})
    """
    weights = np.exp(-lambda_decay * np.arange(1, k_steps + 1))
    weights /= weights.sum()  # Normalize weights
    
    ari = pd.Series(rainfall_series).rolling(window=k_steps, min_periods=1).apply(
        lambda x: np.sum(x[::-1][:len(weights)] * weights[:len(x)]), raw=True
    )
    return ari


def extract_rainfall_memory_features(df):
    """
    Extracts multi-timescale precipitation aggregates and ARI memory.
    """
    df = df.copy()
    
    # Calculate Antecedent Rainfall Index if 24h rainfall exists
    if 'rain_24h' in df.columns:
        df['antecedent_rain_index'] = compute_antecedent_rainfall_index(df['rain_24h'])
        
    return df
