import os

# ==========================================
# 1. REPRODUCIBILITY & SPLIT CONFIG
# ==========================================
RANDOM_SEED = 42
TEST_SIZE = 0.20  # 80% Train, 20% Test
TARGET_COL = "flood_label"

# ==========================================
# 2. PATH CONFIGURATIONS
# ==========================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")

# Master processed dataset path
PROCESSED_DATASET_PATH = os.path.join(PROCESSED_DATA_DIR, "flood_dataset.csv")

# Output directory for results
RESULTS_DIR = os.path.join(BASE_DIR, "results")
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")
METRICS_DIR = os.path.join(RESULTS_DIR, "metrics")
MAPS_DIR = os.path.join(RESULTS_DIR, "maps")

# ==========================================
# 3. FEATURE CATEGORIES FOR ABLATION MODELS
# ==========================================
CONVENTIONAL_TERRAIN_FEATURES = [
    "elevation",
    "slope",
    "aspect"
]

CALCULUS_TERRAIN_FEATURES = [
    "elevation",
    "slope",
    "aspect",
    "terrain_gradient_x",
    "terrain_gradient_y",
    "flow_convergence",
    "surface_divergence"
]

RAINFALL_MEMORY_FEATURES = [
    "rain_1h",
    "rain_3h",
    "rain_6h",
    "rain_24h",
    "antecedent_rain_index"
]

INTERACTION_FEATURES = [
    "convergence_x_rain24h",
    "gradient_x_rain_memory",
    "vulnerability_index"
]

# Model 5 uses ALL features
MODEL_5_ALL_FEATURES = (
    CALCULUS_TERRAIN_FEATURES + 
    RAINFALL_MEMORY_FEATURES + 
    INTERACTION_FEATURES
)

# ==========================================
# 4. MODEL HYPERPARAMETERS
# ==========================================
XGB_PARAMS = {
    "n_estimators": 200,
    "max_depth": 6,
    "learning_rate": 0.05,
    "random_state": RANDOM_SEED,
    "eval_metric": "logloss"
}

RF_PARAMS = {
    "n_estimators": 200,
    "max_depth": 10,
    "random_state": RANDOM_SEED,
    "n_jobs": -1
}
