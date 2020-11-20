from .data_check_message_type import DataCheckMessageType


class DataCheckMessage:
    """Base class for all DataCheckMessages."""

    message_type = None

    def __init__(self, message, data_check_name, message_code=None, details=None):
        """
        Message returned by a DataCheck, tagged by name.

        Arguments:
            message (str): Message string
            data_check_name (str): Name of data check
            message_code (DataCheckMessageCode, optional): Message code associated with message.
            details (dict, optional): Additional useful information associated with the message
        """
        self.message = message
        self.data_check_name = data_check_name
        self.message_code = message_code
        self.details = details

    def __str__(self):
        """String representation of data check message, equivalent to self.message attribute."""
        return self.message

    def __eq__(self, other):
        """Checks for equality. Two DataCheckMessage objs are considered equivalent if their message type and message are equivalent."""
        return (self.message_type == other.message_type and
                self.message == other.message and
                self.data_check_name == other.data_check_name and
                self.message_code == other.message_code)

    def to_dict(self):
        return {
            "message": self.message,
            "data_check_name": self.data_check_name,
            "code": self.message_code,
            "level": self.message_type.value
        }


class DataCheckError(DataCheckMessage):
    """DataCheckMessage subclass for errors returned by data checks."""
    message_type = DataCheckMessageType.ERROR


class DataCheckWarning(DataCheckMessage):
    """DataCheckMessage subclass for warnings returned by data checks."""
    message_type = DataCheckMessageType.WARNING
