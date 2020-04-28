from abc import ABC, abstractmethod


class DataCheck(ABC):
    """Base class for all data checks."""

    @abstractmethod
    def validate(self, X, y, verbose=True):
        """
        Inspects and validates the input data, run any necessary calculations or algorithms, and return a list of warnings and errors if applicable.

        Arguments:
            X (pd.DataFrame): the input data of shape [n_samples, n_features]
            y (pd.Series): the target labels of length [n_samples]
            verbose (bool): Controls verbosity of output. If True, prints to console.

        Returns:
            list (DataCheckError), list (DataCheckWarning): returns a list of DataCheckError and DataCheckWarning objects
        """
