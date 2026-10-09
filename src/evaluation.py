"""Shared evaluation protocol.

All five experiments must be scored with this module using the same
event-based train/test split and the same metrics so results are
directly comparable.
"""


def event_based_split(samples, config):
    """Split samples by flood event (not row-wise) to avoid leakage."""
    raise NotImplementedError


def compute_metrics(y_true, y_pred_proba, config):
    """Compute the shared metric set (e.g. AUC, F1, precision/recall)."""
    raise NotImplementedError


def save_metrics(metrics, experiment_name, config):
    """Write metrics to results/metrics/ in a consistent format."""
    raise NotImplementedError
