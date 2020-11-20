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
            X (pd.DataFrame): the input data of shape [n_samples, n_features]
            y (pd.Series, optional): the target data of length [n_samples]

        Returns:
            dict (DataCheckMessage): Dictionary of DataCheckError and DataCheckWarning messages
        """

    @staticmethod
    def _add_message(message, messages):
        if message.message_type == DataCheckMessageType.ERROR:
            messages["errors"].append(message.to_dict())
        elif message.message_type == DataCheckMessageType.WARNING:
            messages["warnings"].append(message.to_dict())
