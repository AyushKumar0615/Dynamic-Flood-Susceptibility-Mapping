"""Terrain x rainfall interaction features — the core "dynamic" signal
(e.g. TWI x rainfall intensity, slope x antecedent rainfall).
"""


def build_interaction_terms(terrain_features, rainfall_features, config):
    """Construct interaction terms between terrain and rainfall features."""
    raise NotImplementedError


def build_interaction_feature_set(config):
    """Assemble the interaction feature block for the shared feature table."""
    raise NotImplementedError
