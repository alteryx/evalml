from enum import Enum


class DataCheckMessageType(Enum):
    """Enum for type of data check message: WARNING or ERROR"""
    WARNING = "warning"
    ERROR = "error"
