"""Calculus-derived terrain and rainfall interaction feature engineering.

Constructs explicit physics-inspired interaction metrics coupling continuous
topographical flow metrics with multi-timescale rainfall memory.
"""

import numpy as np
import pandas as pd


def compute_interaction_terms(df):
    """
    Computes terrain-rainfall interaction features for Model 5.
    
    Expected input columns in df:
        - 'flow_convergence': calculus flow convergence metric (a / tan(beta))
        - 'surface_divergence': Laplacian / curvature (del^2 z)
        - 'rain_24h': 24-hour rolling rainfall sum
        - 'antecedent_rain_index': dynamic time-decayed rainfall memory
    """
    df = df.copy()
    
    # 1. Flow Convergence x 24h Accumulation Interaction
    if 'flow_convergence' in df.columns and 'rain_24h' in df.columns:
        df['convergence_x_rain24h'] = df['flow_convergence'] * df['rain_24h']
        
    # 2. Curvature Gradient x Rainfall Memory Coupling
    if 'surface_divergence' in df.columns and 'antecedent_rain_index' in df.columns:
        df['gradient_x_rain_memory'] = df['surface_divergence'] * df['antecedent_rain_index']
        
    # 3. Dynamic Vulnerability Index (Non-linear combined metric)
    if 'flow_convergence' in df.columns and 'antecedent_rain_index' in df.columns:
        df['vulnerability_index'] = df['flow_convergence'] * np.log1p(df['antecedent_rain_index'])
        
    return df
