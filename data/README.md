# Data

## data/raw/
Place raw, unmodified source data here (DEMs, rainfall records, flood
inventories, administrative boundaries, etc.). Not tracked in Git — see
the root `.gitignore`.

## data/processed/
Output of `src/preprocessing.py`: the single, shared feature table used
by every experiment in `models/`. Not tracked in Git.

## Obtaining the data
Document dataset download sources, versions, and placement instructions
here once finalized. See `docs/data_sources.md` for the source registry.
