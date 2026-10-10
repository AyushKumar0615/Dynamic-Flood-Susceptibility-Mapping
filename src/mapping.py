"""Spatial flood susceptibility visualization and GeoTIFF exporter.

Converts 1D model predictions back into 2D spatial raster grids and plots
susceptibility heatmaps using standard risk color ramps.
"""

import json
import os
import urllib.request

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path
from scipy.ndimage import gaussian_filter

# Same India window as the Model 5 notebook plotting cell.
LON_MIN, LON_MAX = 68.0, 97.0
LAT_MIN, LAT_MAX = 8.0, 37.0
INDIA_BOUNDARY_URL = (
    "https://raw.githubusercontent.com/datasets/geo-countries/master/data/countries.geojson"
)


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


def _cache_countries_geojson(cache_path):
    if os.path.exists(cache_path):
        return cache_path
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    request = urllib.request.Request(INDIA_BOUNDARY_URL, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(request, timeout=60) as response:
        payload = response.read()
    with open(cache_path, "wb") as handle:
        handle.write(payload)
    return cache_path


def _india_polygons(cache_path):
    """Return India's polygons as lists of rings, exterior first, then holes."""
    with open(cache_path) as handle:
        data = json.load(handle)
    for feature in data["features"]:
        props = {str(key).lower(): value for key, value in feature.get("properties", {}).items()}
        name = str(props.get("admin") or props.get("name") or "")
        if name.lower() != "india":
            continue
        geometry = feature["geometry"]
        if geometry["type"] == "Polygon":
            return [geometry["coordinates"]]
        if geometry["type"] == "MultiPolygon":
            return geometry["coordinates"]
        raise ValueError(f"Unsupported India geometry type {geometry['type']}")
    raise ValueError("India was not found in the country boundary file.")


def _india_mask(polygons, lon, lat):
    points = np.column_stack([lon.ravel(), lat.ravel()])
    inside = np.zeros(points.shape[0], dtype=bool)
    for rings in polygons:
        contained = Path(np.asarray(rings[0])).contains_points(points)
        for hole in rings[1:]:
            if len(hole) >= 3:
                contained &= ~Path(np.asarray(hole)).contains_points(points)
        inside |= contained
    return inside.reshape(lon.shape)


def plot_india_susceptibility_map(y_prob, save_path, title, colorbar_label, cache_path=None):
    """Draw probabilities on the India window used by the Model 5 notebook.

    The 120 by 120 grid is placed on longitude 68-97 E and latitude 8-37 N.
    Row 0 is the southern edge. Cells outside the India boundary are hidden.
    A light smooth (sigma 1.8) matches the notebook plotting cell.
    """
    grid = np.asarray(y_prob, dtype=float).reshape((120, 120))
    smoothed = gaussian_filter(grid, sigma=1.8)

    if cache_path is None:
        cache_path = os.path.join(os.path.dirname(save_path), ".countries_cache.geojson")
    _cache_countries_geojson(cache_path)
    polygons = _india_polygons(cache_path)

    lons = np.linspace(LON_MIN, LON_MAX, 120)
    lats = np.linspace(LAT_MIN, LAT_MAX, 120)
    lon, lat = np.meshgrid(lons, lats)
    masked = np.where(_india_mask(polygons, lon, lat), smoothed, np.nan)

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_facecolor("#e8f4f8")
    for rings in polygons:
        for ring in rings:
            coords = np.asarray(ring)
            ax.plot(coords[:, 0], coords[:, 1], color="black", linewidth=1.0, zorder=3)
    image = ax.imshow(
        masked,
        cmap="YlOrRd",
        vmin=0.0,
        vmax=1.0,
        origin="lower",
        extent=[LON_MIN, LON_MAX, LAT_MIN, LAT_MAX],
        zorder=2,
        alpha=0.9,
    )
    ax.set_xlim(LON_MIN, LON_MAX)
    ax.set_ylim(LAT_MIN, LAT_MAX)
    ax.set_title(title, fontsize=13, fontweight="bold", pad=14)
    ax.set_xlabel("Longitude (°E)", fontsize=11)
    ax.set_ylabel("Latitude (°N)", fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.3, color="gray")
    colorbar = fig.colorbar(image, ax=ax, fraction=0.035, pad=0.04)
    colorbar.set_label(colorbar_label, fontsize=10)
    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return save_path
