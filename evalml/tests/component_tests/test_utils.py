import pytest

from evalml.pipelines.components import (
    COMPONENTS,
    ComponentTypes,
    LinearRegressor,
    LogisticRegressionClassifier,
    OneHotEncoder,
    RFSelectFromModel,
    SimpleImputer,
    StandardScaler,
    components_dict,
    handle_component,
    handle_component_types
)


def test_components_dict():
    assert len(components_dict()) == 9
    assert len(COMPONENTS) == 9


def test_handle_component():
    component_strs = ['Linear Regressor', 'Logistic Regression Classifier', 'One Hot Encoder', 'RF Select From Model', 'Simple Imputer', 'Standard Scaler']
    components = [LinearRegressor, LogisticRegressionClassifier, OneHotEncoder, RFSelectFromModel, SimpleImputer, StandardScaler]

    for component_str, component in zip(component_strs, components):
        assert isinstance(handle_component(component_str), component)

    bad_str = 'Select From Model'
    with pytest.raises(ValueError):
        c = handle_component(bad_str)


def test_default_component():
    component_type_strs = ['classifier', 'encoder', 'imputer', 'regressor', 'scaler', 'feature_selection']
    components = [LogisticRegressionClassifier, OneHotEncoder, SimpleImputer, LinearRegressor, StandardScaler, RFSelectFromModel]
    for component_type_str, component in zip(component_type_strs, components):
        assert isinstance(handle_component(component_type_str), component)


@pytest.fixture
def correct_component_types():
    correct_component_types = [
        ComponentTypes.CLASSIFIER,
        ComponentTypes.ENCODER,
        ComponentTypes.FEATURE_SELECTION,
        ComponentTypes.IMPUTER,
        ComponentTypes.REGRESSOR,
        ComponentTypes.SCALER
    ]
    return correct_component_types


def test_handle_string(correct_component_types):
    component_types = ['classifier', 'encoder', 'feature_selection', 'imputer', 'regressor', 'scaler']
    for component_type, correct_component_type in zip(component_types, correct_component_types):
        assert handle_component_types(component_type) == correct_component_type

    component_type = 'fake'
    error_msg = 'Component type \'fake\' does not exist'
    with pytest.raises(ValueError, match=error_msg):
        handle_component_types(component_type) == ComponentTypes.SCALER


def test_handle_component_types(correct_component_types):
    for component_type in correct_component_types:
        assert handle_component_types(component_type) == component_type
