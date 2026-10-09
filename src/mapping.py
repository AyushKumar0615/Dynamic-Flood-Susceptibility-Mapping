"""Spatial flood susceptibility visualization and GeoTIFF exporter.

Converts 1D model predictions back into 2D spatial raster grids and plots
susceptibility heatmaps using standard risk color ramps.
"""

import numpy as np
import matplotlib.pyplot as plt


def plot_susceptibility_map(y_prob, coordinates=None, grid_shape=None, save_path=None):
    """
    Plots a 2D spatial susceptibility heatmap from model probability predictions.
    
    Color Ramp: Green (Low Risk) -> Yellow -> Orange -> Red (High Flood Risk)
    """
    if grid_shape is not None:
        grid_data = np.array(y_prob).reshape(grid_shape)
    else:
        # Fallback: Approximate square spatial grid
        side = int(np.sqrt(len(y_prob)))
        grid_data = np.array(y_prob[:side*side]).reshape((side, side))
        
    plt.figure(figsize=(10, 8))
    plt.imshow(grid_data, cmap='RdYlGn_r', vmin=0.0, vmax=1.0)
    plt.colorbar(label='Predicted Flood Susceptibility Index')
    plt.title('Dynamic Flood Susceptibility Map (CIE3 Ablation Framework)')
    plt.xlabel('Grid Column (X)')
    plt.ylabel('Grid Row (Y)')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
