from enum import Enum


class ComponentTypes(Enum):
    """Enum for type of component in our ML Pipeline"""
    CLASSIFIER = 'classifier'
    CATEGORICAL_ENCODER = 'categorical_encoder'
    IMPUTER = 'imputer'
    FEATURE_SELECTION = 'feature_selection'
    FEATURE_SELECTION_CLASSIFIER = 'feature_selection_classifier'
    FEATURE_SELECTION_REGRESSOR = 'feature_selection_regressor'
    REGRESSOR = 'regressor'
    SCALER = 'scaler'
