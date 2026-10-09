# Dynamic Flood Susceptibility Mapping

A machine learning project for flood susceptibility mapping using terrain characteristics, rainfall history, and terrain–rainfall interactions.

## Objectives
- Build a shared geospatial preprocessing pipeline.
- Engineer terrain and rainfall features.
- Compare five model and feature configurations.
- Generate evaluation metrics, figures, and susceptibility maps.

## Project Structure
- `data/`: Raw and processed dataset locations and instructions.
- `configs/`: Shared experiment configuration.
- `notebooks/`: Exploratory data analysis.
- `src/`: Preprocessing, labels, features, evaluation, and mapping.
- `models/`: Five experiment configuration scripts.
- `results/`: Metrics, figures, and maps.
- `docs/`: Data sources, methodology, and team responsibilities.

## Collaboration Rules
- Keep `main` stable.
- Use separate feature branches for individual work.
- Use the same dataset, preprocessing pipeline, event-based split, and evaluation protocol for all five experiments.
- Do not upload large raw raster datasets or secrets to GitHub.

## Setup
Dependencies and execution instructions will be documented as the common pipeline is established.

## Dataset
See `data/README.md` for dataset acquisition and placement instructions.
