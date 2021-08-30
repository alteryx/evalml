"""Utilities for data checks."""
from .data_checks import DataChecks


class EmptyDataChecks(DataChecks):
    """An empty collection of data checks.

    Args:
        data_checks (list (DataCheck)): Ignored.
    """

    def __init__(self, data_checks=None):
        self.data_checks = []
