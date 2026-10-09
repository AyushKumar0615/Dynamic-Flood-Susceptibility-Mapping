"""Susceptibility map generation.

Converts model predictions back into a spatial raster/figure for
results/maps/.
"""


def predictions_to_raster(predictions, reference_grid, config):
    """Reproject flat predictions back onto the spatial grid."""
    raise NotImplementedError


def save_susceptibility_map(raster, experiment_name, config):
    """Write the susceptibility map to results/maps/."""
    raise NotImplementedError
