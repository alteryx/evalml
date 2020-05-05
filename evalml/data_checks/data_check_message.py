from .data_check_message_type import DataCheckMessageType


class DataCheckMessage:
    """Base class for all DataCheckMessages."""

    message_type = None

    def __init__(self, message, data_check_name):
        """
        Message returned by a DataCheck, tagged by name"

        Arguments:
            message (str): message string
            data_check_name (str): name of data check
        """
        self.message = message
        self.data_check_name = data_check_name

    def __str__(self):
        """String representation of data check message, eqivalent to self.message attribute."""
        return self.message

    def __eq__(self, other):
        """Checks for equality. Two DataCheckMessage objs are considered equivalent if their message type and message are equivalent."""
        return (self.message_type == other.message_type and
                self.message == other.message and
                self.data_check_name == other.data_check_name)


class DataCheckError(DataCheckMessage):
    """DataCheckMessage subclass for errors returned by data checks."""
    message_type = DataCheckMessageType.ERROR


class DataCheckWarning(DataCheckMessage):
    """DataCheckMessage subclass for warnings returned by data checks."""
    message_type = DataCheckMessageType.WARNING
