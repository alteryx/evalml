from abc import ABC, abstractmethod


class DataCheck(ABC):
    """Base class for all data checks."""

    @abstractmethod
    def validate(self, X, y=None, verbose=True):
        """
        Inspects and validates the input data, run any necessary calculations or algorithms, and return a list of warnings and errors if applicable.

        Arguments:
            X (pd.DataFrame): the input data of shape [n_samples, n_features]
            y (pd.Series, optional): the target data of length [n_samples]
            verbose (bool): If False, disables logging. Defaults to True.

        Returns:
            list (DataCheckError), list (DataCheckWarning): returns a list of DataCheckError and DataCheckWarning objects
        """
