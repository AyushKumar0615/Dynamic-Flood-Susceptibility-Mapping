"""Dynamic rainfall-derived features (event totals, intensity,
antecedent rainfall, return period, etc.) that vary by event.
"""


def compute_event_rainfall_stats(rainfall_series, event_window):
    """Total, max intensity, and duration for a rainfall event window."""
    raise NotImplementedError


def compute_antecedent_rainfall(rainfall_series, lookback_days):
    """Cumulative rainfall in the days preceding an event."""
    raise NotImplementedError


def build_rainfall_feature_set(config):
    """Assemble the rainfall feature block for the shared feature table."""
    raise NotImplementedError
