# Methodology

All five experiments in `models/` share one pipeline. Only the feature
set and/or model type may differ between them — never the data,
preprocessing, split, or evaluation.

1. **Preprocessing** (`src/preprocessing.py`) — load, clean, and
   spatially align raw terrain/rainfall/flood-inventory data into one
   feature table in `data/processed/`.
2. **Labeling** (`src/labels.py`) — construct the shared flood /
   no-flood labels.
3. **Feature engineering**
   - Terrain features — `src/terrain_features.py`
   - Rainfall features — `src/rainfall_features.py`
   - Terrain × rainfall interaction features — `src/interaction_features.py`
4. **Train/test split** — a single **event-based** split
   (`src/evaluation.py::event_based_split`), not a random row-wise
   split, to prevent leakage between spatially/temporally related
   samples.
5. **Model training** — each `models/exp*.py` config consumes the same
   processed features and split; configs vary only in feature subset
   and/or model type.
6. **Evaluation** (`src/evaluation.py`) — identical metric set and
   reporting for every experiment, saved to `results/metrics/`.
7. **Mapping** (`src/mapping.py`) — convert predictions into
   susceptibility maps, saved to `results/maps/`.

## Experiments
Document each experiment's configuration and rationale here as they
are finalized (see `models/`).
