from enum import Enum


class DataCheckMessageType(Enum):
    """Enum for type of data check message"""
    WARNING = 'warning'
    ERROR = 'error'
