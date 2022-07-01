"""Component that removes trends from time series and returns the decomposed components."""
from abc import ABCMeta, abstractmethod

from evalml.pipelines.components.transformers.transformer import Transformer


class Detrender(Transformer):
    """Component that removes trends from time series and returns the decomposed components.

    Args:
        random_seed (int): Seed for the random number generator. Defaults to 0.
    """

    name = "Detrender"
    hyperparameter_ranges = None
    modifies_features = False
    modifies_target = True

    @abstractmethod
    def get_trend_dataframe(self, y):
        """Return a dataframe with 3 columns: trend, seasonality, residual"""

    @abstractmethod
    def inverse_transform(self, y):
        """Add the trend + seasonality back to y"""
