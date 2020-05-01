from abc import ABC, abstractmethod

from evalml.utils import classproperty


class DataCheck(ABC):
    """Base class for all data checks."""

    @classproperty
    def name(cls):
        """Returns a name describing the data check."""
        return str(cls.__name__)

    @abstractmethod
    def validate(self, X, y=None):
        """
        Inspects and validates the input data, runs any necessary calculations or algorithms, and returns a list of warnings and errors if applicable.

        Arguments:
            X (pd.DataFrame): the input data of shape [n_samples, n_features]
            y (pd.Series, optional): the target data of length [n_samples]

        Returns:
            list (DataCheckMessage): list of DataCheckError and DataCheckWarning objects
        """
