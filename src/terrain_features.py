"""Static terrain-derived features (elevation, slope, curvature, flow
accumulation, topographic wetness index, distance to drainage, etc.).
"""


def compute_elevation_derivatives(dem):
    """Slope, aspect, and curvature from a DEM."""
    raise NotImplementedError


def compute_hydrological_indices(dem):
    """Flow accumulation, TWI, and distance to the stream network."""
    raise NotImplementedError


def build_terrain_feature_set(config):
    """Assemble the terrain feature block for the shared feature table."""
    raise NotImplementedError
