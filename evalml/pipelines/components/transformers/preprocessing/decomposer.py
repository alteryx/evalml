"""Component that removes trends from time series and returns the decomposed components."""
from abc import abstractmethod

from evalml.pipelines.components.transformers.transformer import Transformer


class Decomposer(Transformer):
    """Component that removes trends and seasonality from time series and returns the decomposed components.

    Args:
        random_seed (int): Seed for the random number generator. Defaults to 0.
    """

    name = "Decomposer"
    hyperparameter_ranges = None
    modifies_features = False
    modifies_target = True

    @abstractmethod
    def get_trend_dataframe(self, y):
        """Return a list of dataframes, each with 3 columns: trend, seasonality, residual."""

    @abstractmethod
    def inverse_transform(self, y):
        """Add the trend + seasonality back to y."""
