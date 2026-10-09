"""Flood / no-flood label construction.

Defines the single labeling scheme shared by all experiments so that
positive and negative samples are identical across models.
"""


def load_flood_inventory(config):
    """Load historical flood event records (the positive-class source)."""
    raise NotImplementedError


def generate_negative_samples(config):
    """Sample non-flooded locations/times to build the negative class."""
    raise NotImplementedError


def build_labels(config):
    """Combine positive and negative samples into the shared label set."""
    raise NotImplementedError
