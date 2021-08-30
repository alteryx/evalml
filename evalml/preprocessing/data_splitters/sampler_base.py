"""Base class for all custom samplers."""
from abc import ABC, abstractmethod


class SamplerBase(ABC):
    """Base class for all custom samplers.

    Args:
        random_seed (int): The seed to use for random sampling. Defaults to 0.
    """

    def __init__(self, random_seed=0):
        self.random_seed = random_seed

    @abstractmethod
    def fit_resample(self, X, y):
        """Resample the input data with this sampling strategy.

        Args:
            X (pd.DataFrame): Training data to fit and resample.
            y (pd.Series): Training data targets to fit and resample.

        Returns:
            Tuple(pd.DataFrame, pd.Series) or list: resampled X and y data for oversampling or indices to keep for undersampling.
        """
