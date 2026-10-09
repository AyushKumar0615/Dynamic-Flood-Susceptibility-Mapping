"""Shared preprocessing pipeline.

All five experiments (see models/) must build on this module to go
from raw inputs in data/raw/ to the single, shared feature table in
data/processed/. Do not fork per-experiment preprocessing logic.
"""


def load_raw_data(config):
    """Load raw terrain, rainfall, and flood-inventory inputs per config."""
    raise NotImplementedError


def clean_and_align(data):
    """Clean and spatially align all raw inputs onto a common grid."""
    raise NotImplementedError


def build_feature_table(data):
    """Assemble the shared feature table used by every experiment."""
    raise NotImplementedError


def run_pipeline(config):
    """Entry point: raw data -> processed feature table in data/processed/."""
    raise NotImplementedError
