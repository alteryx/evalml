from abc import ABC, abstractmethod

from evalml.data_checks.data_check_message_type import DataCheckMessageType
from evalml.utils import classproperty


class DataCheck(ABC):
    """Base class for all data checks. Data checks are a set of heuristics used to determine if there are problems with input data."""

    @classproperty
    def name(cls):
        """Returns a name describing the data check."""
        return str(cls.__name__)

    @abstractmethod
    def validate(self, X, y=None):
        """
        Inspects and validates the input data, runs any necessary calculations or algorithms, and returns a list of warnings and errors if applicable.

        Arguments:
            X (ww.DataTable, pd.DataFrame): The input data of shape [n_samples, n_features]
            y (ww.DataColumn, pd.Series, optional): The target data of length [n_samples]

        Returns:
            dict (DataCheckMessage): Dictionary of DataCheckError and DataCheckWarning messages
        """

    @staticmethod
    def _add_message(message, results):
        if message.message_type == DataCheckMessageType.ERROR:
            results["errors"].append(message.to_dict())
        elif message.message_type == DataCheckMessageType.WARNING:
            results["warnings"].append(message.to_dict())
