from enum import Enum


class DataCheckMessageType(Enum):
    """Enum for type of data check message: WARNING or ERROR."""
    WARNING = "warning"
    """Warning message returned by a data check."""

    ERROR = "error"
    """Error message returned by a data check."""
