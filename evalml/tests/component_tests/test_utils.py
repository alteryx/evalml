import pytest

from evalml.pipelines.components import (
    ComponentTypes,
    LinearRegressor,
    LogisticRegressionClassifier,
    OneHotEncoder,
    RFClassifierSelectFromModel,
    RFRegressorSelectFromModel,
    SimpleImputer,
    StandardScaler,
    handle_component,
    str_to_component_type
)

from evalml.pipelines.components.utils import __COMPONENTS, __components_dict


def test_components_dict():
    assert len(__components_dict()) == 10
    assert len(__COMPONENTS) == 10


def test_handle_component():
    component_strs = [
        'Linear Regressor', 'Logistic Regression Classifier', 'One Hot Encoder', 'RF Classifier Select From Model', 
        'RF Regressor Select From Model', 'Simple Imputer', 'Standard Scaler'
    ]
    components = [
        LinearRegressor, LogisticRegressionClassifier, OneHotEncoder, RFClassifierSelectFromModel, 
        RFRegressorSelectFromModel, SimpleImputer, StandardScaler
    ]

    for component_str, component in zip(component_strs, components):
        assert isinstance(handle_component(component_str), component)

    bad_str = 'Select From Model'
    with pytest.raises(ValueError):
        handle_component(bad_str)


def test_default_component():
    component_type_strs = ['classifier', 'categorical_encoder', 'imputer', 'regressor', 'scaler', 'feature_selection']
    components = [LogisticRegressionClassifier, OneHotEncoder, SimpleImputer, LinearRegressor, StandardScaler, RFClassifierSelectFromModel]
    for component_type_str, component in zip(component_type_strs, components):
        assert isinstance(handle_component(component_type_str), component)


@pytest.fixture
def correct_component_types():
    correct_component_types = [
        ComponentTypes.CLASSIFIER,
        ComponentTypes.CATEGORICAL_ENCODER,
        ComponentTypes.FEATURE_SELECTION,
        ComponentTypes.IMPUTER,
        ComponentTypes.REGRESSOR,
        ComponentTypes.SCALER
    ]
    return correct_component_types


def test_handle_string(correct_component_types):
    component_types = ['classifier', 'categorical_encoder', 'feature_selection', 'imputer', 'regressor', 'scaler']
    for component_type, correct_component_type in zip(component_types, correct_component_types):
        assert str_to_component_type(component_type) == correct_component_type

    component_type = 'fake'
    error_msg = 'Component type \'fake\' does not exist'
    with pytest.raises(ValueError, match=error_msg):
        str_to_component_type(component_type) == ComponentTypes.SCALER


def test_handle_component_types(correct_component_types):
    for component_type in correct_component_types:
        assert str_to_component_type(component_type) == component_type
