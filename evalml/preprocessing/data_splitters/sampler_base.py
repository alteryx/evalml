from abc import ABC, abstractmethod


class SamplerBase(ABC):
    """Base class for all custom samplers"""

    def __init__(self, random_seed=0):
        self.random_seed = random_seed

    @abstractmethod
    def fit_resample(self, X, y):
        """Resample the input data with this sampling strategy.

        Arguments:
            X (pd.DataFrame): Training data to fit and resample
            y (pd.Series): Training data targets to fit and resample

        Returns:
            pd.DataFrame, pd.Series: resampled X and y data
        """
