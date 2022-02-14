"""Base class for all data checks."""
from abc import ABC, abstractmethod

from evalml.utils import classproperty


class DataCheck(ABC):
    """Base class for all data checks.

    Data checks are a set of heuristics used to determine if there are
    problems with input data.
    """

    @classproperty
    def name(cls):
        """Return a name describing the data check."""
        return str(cls.__name__)

    @abstractmethod
    def validate(self, X, y=None):
        """Inspect and validate the input data, runs any necessary calculations or algorithms, and returns a list of warnings and errors if applicable.

        Args:
            X (pd.DataFrame): The input data of shape [n_samples, n_features]
            y (pd.Series, optional): The target data of length [n_samples]

        Returns:
            dict (DataCheckMessage): Dictionary of DataCheckError and DataCheckWarning messages
        """
