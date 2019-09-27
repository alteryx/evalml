from enum import Enum


class ComponentTypes(Enum):
    """Enum for type of component in our ML Pipeline"""
    CLASSIFIER = 'classifier'
    ENCODER = 'encoder'
    IMPUTER = 'imputer'
    FEATURE_SELECTION = 'feature_selection'
    REGRESSOR = 'regressor'
    SCALER = 'scaler'
