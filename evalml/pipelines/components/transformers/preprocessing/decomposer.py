"""Component that removes trends from time series and returns the decomposed components."""
from abc import abstractmethod

from evalml.pipelines.components.transformers.transformer import Transformer


class Decomposer(Transformer):
    """Component that removes trends and seasonality from time series and returns the decomposed components.

    Args:
        parameters (dict): Dictionary of parameters to pass to component object.
        component_obj (class) : Instance of a detrender/deseasonalizer class.
        random_seed (int): Seed for the random number generator. Defaults to 0.
    """

    name = "Decomposer"
    hyperparameter_ranges = None
    modifies_features = False
    modifies_target = True

    def __init__(self, parameters=None, component_obj=None, random_seed=0, **kwargs):
        super().__init__(
            parameters=parameters,
            component_obj=component_obj,
            random_seed=random_seed,
            **kwargs,
        )

    @abstractmethod
    def get_trend_dataframe(self, y):
        """Return a list of dataframes, each with 3 columns: trend, seasonality, residual."""

    @abstractmethod
    def inverse_transform(self, y):
        """Add the trend + seasonality back to y."""
