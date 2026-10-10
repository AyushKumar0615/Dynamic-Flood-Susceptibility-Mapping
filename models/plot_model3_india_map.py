"""India-wide heatmap of Model 3 flood-susceptibility probabilities.

Grid specification used (supplied by teammate, NOT independently verified
against any code or data in this repository):
    - 120 x 120 = 14,400 cells
    - Latitude 8 deg N to 37 deg N
    - Longitude 68 deg E to 97 deg E

IMPORTANT, read before trusting this map's geographic placement:
This repository's own data-generation notebook
(1RUA25CSE0116_model_5_execution.ipynb, cell index 2 -- the cell BEFORE the
one whose boundary/plotting logic this script reuses) shows the 120x120
grid underlying data/processed/flood_dataset.csv was built by resampling a
single DEM clipped to bounds=(89.5, 26.0, 90.5, 27.0) -- a ~1x1 degree
patch near the Assam/Bhutan border -- not an India-wide extent. Cell index 3
of that notebook (the one this script's boundary-fetch and plotting style
is modeled on) uses a SEPARATE, unconnected 250x250 mesh over 68-97E/8-37N
with an entirely synthetic probability field (hand-authored Gaussian bumps
keyed to place names), never derived from any model's predict_proba().
Neither cell establishes that flood_dataset.csv's 14,400 rows, in their
on-disk order, correspond to a row-major scan of the 68-97E/8-37N box.

This script nonetheless renders that India-wide placement because it was
explicitly requested, with the following EXPLICIT, UNVERIFIED assumptions,
stated here and again in the output report:
    1. Row order in model_3_predictions.csv (== flood_dataset.csv's row
       order; both already verified to align by position) is assumed to be
       a row-major (C-order) scan of a 120x120 grid.
    2. Grid row 0 is assumed to be the southernmost row (lat=8N) and grid
       row 119 the northernmost (lat=37N); i.e. latitude increases with
       increasing row index. This is an assumption, not a verified fact.
    3. Grid column 0 is assumed to be the westernmost column (lon=68E) and
       column 119 the easternmost (lon=97E).
No source code or data in this repository confirms assumptions 1-3.

India boundary: fetched live from the public "geo-countries" dataset
(https://raw.githubusercontent.com/datasets/geo-countries/master/data/countries.geojson),
the same source used in cell index 3 of 1RUA25CSE0116_model_5_execution.ipynb.
This is a genuine, real India boundary (not invented), used here only to
mask/outline -- it has no bearing on whether the probability grid itself is
correctly placed within it.

Does not modify Model 3's training script, Model 5, the processed dataset,
shared config, or any existing Model 5 output. Reads only
results/predictions/model_3_predictions.csv.
"""

import json
import os
import urllib.request

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import geopandas as gpd
from shapely.geometry import Point
from scipy.ndimage import gaussian_filter

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

PREDICTIONS_PATH = os.path.join(
    _REPO_ROOT, "results", "predictions", "model_3_predictions.csv"
)
DATASET_PATH = os.path.join(_REPO_ROOT, "data", "processed", "flood_dataset.csv")
OUTPUT_PATH = os.path.join(
    _REPO_ROOT, "results", "maps", "model_3_india_susceptibility_map.png"
)
BOUNDARY_CACHE_PATH = os.path.join(_REPO_ROOT, "results", "maps", ".countries_cache.geojson")
BOUNDARY_URL = (
    "https://raw.githubusercontent.com/datasets/geo-countries/master/data/countries.geojson"
)

PROB_COLUMN = "flood_susceptibility_probability"
EXPECTED_N = 14400
GRID_SIDE = 120  # per teammate's specification; see module docstring.
LON_MIN, LON_MAX = 68.0, 97.0
LAT_MIN, LAT_MAX = 8.0, 37.0


def load_and_validate_probabilities():
    pred_df = pd.read_csv(PREDICTIONS_PATH)
    ds_df = pd.read_csv(DATASET_PATH)

    if len(ds_df) != EXPECTED_N:
        raise ValueError(
            f"Expected {EXPECTED_N} rows in {DATASET_PATH}, found {len(ds_df)}."
        )

    if PROB_COLUMN not in pred_df.columns:
        raise KeyError(
            f"Expected column {PROB_COLUMN!r} not found in {PREDICTIONS_PATH}. "
            f"Columns present: {list(pred_df.columns)}"
        )

    if len(pred_df) != EXPECTED_N:
        raise ValueError(
            f"Expected {EXPECTED_N} predictions, found {len(pred_df)} in "
            f"{PREDICTIONS_PATH}."
        )

    # row_index alignment check (predictions already established to be in
    # the same on-disk row order as the dataset; re-verify here).
    if "row_index" in pred_df.columns:
        if not (pred_df["row_index"].to_numpy() == np.arange(EXPECTED_N)).all():
            raise ValueError(
                "model_3_predictions.csv row_index is not a clean 0..14399 "
                "sequence in order; refusing to assume row-major alignment."
            )

    probs = pred_df[PROB_COLUMN].to_numpy(dtype=float)

    if not np.isfinite(probs).all():
        n_bad = int((~np.isfinite(probs)).sum())
        raise ValueError(f"{n_bad} non-finite value(s) in {PROB_COLUMN!r}; refusing to plot.")

    if (probs < 0.0).any() or (probs > 1.0).any():
        raise ValueError(
            f"Probabilities out of [0, 1] range (min={probs.min()}, "
            f"max={probs.max()}) in {PROB_COLUMN!r}; refusing to plot."
        )

    return probs


def load_india_boundary():
    """Fetch the real India boundary polygon (same source as the notebook's
    cell index 3). Caches the raw GeoJSON under results/maps/ so repeat runs
    don't re-download; this cache file is not part of the dataset/model and
    is safe to delete.
    """
    if not os.path.exists(BOUNDARY_CACHE_PATH):
        os.makedirs(os.path.dirname(BOUNDARY_CACHE_PATH), exist_ok=True)
        req = urllib.request.Request(BOUNDARY_URL, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = resp.read()
        with open(BOUNDARY_CACHE_PATH, "wb") as f:
            f.write(data)

    world = gpd.read_file(BOUNDARY_CACHE_PATH)
    name_col = next((c for c in ["ADMIN", "NAME", "name", "admin"] if c in world.columns), None)
    if name_col is None:
        raise KeyError("Could not find a country-name column in the fetched boundary dataset.")

    india = world[world[name_col].astype(str).str.lower() == "india"]
    if len(india) == 0:
        raise ValueError("India not found in the fetched boundary dataset.")
    return india


def build_grid_and_mask(india_gdf):
    lons = np.linspace(LON_MIN, LON_MAX, GRID_SIDE)
    lats = np.linspace(LAT_MIN, LAT_MAX, GRID_SIDE)
    LON, LAT = np.meshgrid(lons, lats)  # LAT increases with row index (row 0 = LAT_MIN)

    points = [Point(x, y) for x, y in zip(LON.flatten(), LAT.flatten())]
    grid_gdf = gpd.GeoDataFrame(geometry=points, crs="EPSG:4326")

    joined = gpd.sjoin(grid_gdf, india_gdf, how="inner", predicate="intersects")

    # BUG FIX: gpd.sjoin(how="inner") emits one output ROW per (left, right)
    # match pair. If india_gdf ever contains more than one matching row for
    # India (e.g. a boundary source that stores disputed-territory sub-
    # features or duplicate entries as separate rows), a single grid point
    # that intersects more than one of those rows appears multiple times in
    # `joined`, all under the SAME left (grid_gdf) index. Taking the raw,
    # un-deduplicated index therefore overcounts -- this is exactly how a
    # "cells inside India" count greater than the true number of grid cells
    # (here, more than GRID_SIDE*GRID_SIDE = 14,400) can occur. Deduplicate
    # to the set of distinct grid-cell indices, and hard-fail rather than
    # silently report an impossible count if it ever happens again.
    inside_flat_indices = np.unique(joined.index.to_numpy())

    max_possible = GRID_SIDE * GRID_SIDE
    if len(inside_flat_indices) > max_possible:
        raise ValueError(
            f"Grid-mask calculation produced {len(inside_flat_indices)} "
            f"'inside India' cell indices, which exceeds the total number "
            f"of grid cells ({max_possible}). This indicates india_gdf has "
            f"multiple overlapping/duplicate rows feeding gpd.sjoin; refusing "
            f"to plot with a corrupted count."
        )

    return LON, LAT, inside_flat_indices


def plot_india_map(probs, LON, LAT, inside_flat_indices, india_gdf, output_path):
    # ASSUMPTION (unverified -- see module docstring): row-major reshape,
    # row 0 = southernmost (lat=8N), column 0 = westernmost (lon=68E).
    grid = probs.reshape((GRID_SIDE, GRID_SIDE))

    # Reused verbatim from cell index 3 of 1RUA25CSE0116_model_5_execution.ipynb
    # ("# Spatial Gaussian Smoothing Filter to remove any grid artifacting",
    # smooth_prob_matrix = gaussian_filter(prob_matrix, sigma=1.8)). This is a
    # PRESENTATION-ONLY smoothing of Model 3's real probabilities -- it blurs
    # neighboring grid cells together for a less blocky image. It does not
    # verify, improve, or change the row/column-to-coordinate assumption
    # stated above; smoothing is applied, as in the notebook, before masking.
    smoothed_grid = gaussian_filter(grid, sigma=1.8)

    flat = smoothed_grid.flatten()
    masked = np.full(flat.shape, np.nan)
    masked[inside_flat_indices] = flat[inside_flat_indices]
    masked_grid = masked.reshape((GRID_SIDE, GRID_SIDE))

    fig, ax = plt.subplots(figsize=(10, 10))

    india_gdf.boundary.plot(ax=ax, color="black", linewidth=1.2, zorder=3)

    im = ax.imshow(
        masked_grid,
        cmap="YlOrRd",
        vmin=0.0,
        vmax=1.0,
        origin="lower",  # row 0 at the bottom == assumed southernmost row
        extent=[LON_MIN, LON_MAX, LAT_MIN, LAT_MAX],
        zorder=2,
        alpha=0.9,
    )
    ax.set_facecolor("#e8f4f8")

    ax.set_title(
        "India-Wide Flood Susceptibility Map\n"
        "Model 3: Random Forest (Terrain + Rainfall Memory)",
        fontsize=13,
        fontweight="bold",
        pad=14,
    )
    ax.set_xlabel("Longitude (°E)", fontsize=11)
    ax.set_ylabel("Latitude (°N)", fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.3, color="gray")

    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.04)
    cbar.set_label("Predicted Flood Susceptibility Probability (Model 3)", fontsize=10)

    fig.text(
        0.5,
        0.01,
        "Row-order-to-coordinate mapping is an assumption, not verified by repository source code or data.",
        ha="center",
        va="bottom",
        fontsize=8.5,
        color="firebrick",
        wrap=True,
    )

    fig.tight_layout(rect=(0, 0.03, 1, 1))

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    probs = load_and_validate_probabilities()
    print(f"Validation OK: {len(probs)} finite probabilities in [0, 1].")

    india_gdf = load_india_boundary()
    print("India boundary loaded (source: geo-countries GeoJSON).")

    LON, LAT, inside_flat_indices = build_grid_and_mask(india_gdf)
    print(f"Grid: {GRID_SIDE}x{GRID_SIDE}; {len(inside_flat_indices)} of {EXPECTED_N} cells fall inside India's polygon.")

    plot_india_map(probs, LON, LAT, inside_flat_indices, india_gdf, OUTPUT_PATH)
    print(os.path.abspath(OUTPUT_PATH))


if __name__ == "__main__":
    main()
