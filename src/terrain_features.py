"""Calculus-derived terrain feature extraction module.

Computes spatial terrain derivatives (Gradient, Slope, Aspect, Curvature)
and Topographic Wetness / Flow Convergence fields from elevation rasters (DEM).
"""

import numpy as np
import pandas as pd


def compute_terrain_derivatives(df, elevation_col='elevation'):
    """
    Computes spatial terrain gradients and convergence indicators.
    
    Formula Reference:
    - Terrain Gradient: grad(z) = (dz/dx, dz/dy)
    - Flow Convergence (TWI proxy): ln(a / tan(beta))
    """
    df = df.copy()
    
    if elevation_col in df.columns:
        # 1. First spatial derivatives (Gradients)
        df['terrain_gradient_x'] = np.gradient(df[elevation_col].values)
        df['terrain_gradient_y'] = np.gradient(df[elevation_col].values)
        
        # 2. Gradient magnitude (Slope proxy)
        df['slope'] = np.sqrt(df['terrain_gradient_x']**2 + df['terrain_gradient_y']**2)
        
        # 3. Slope orientation (Aspect in degrees)
        df['aspect'] = np.degrees(np.arctan2(df['terrain_gradient_y'], df['terrain_gradient_x'])) % 360
        
        # 4. Second spatial derivative (Laplacian / Surface Divergence)
        df['surface_divergence'] = np.gradient(df['terrain_gradient_x'])[0] + np.gradient(df['terrain_gradient_y'])[0]
        
        # 5. Topographic Flow Convergence Index: ln(a / tan(beta))
        tan_beta = np.maximum(np.tan(np.radians(df['slope'])), 1e-4)  # Prevent division by zero
        df['flow_convergence'] = np.log(np.maximum(df[elevation_col] / tan_beta, 1e-4))
        
    return df
