import pytest

from evalml.pipelines.components import (
    COMPONENTS,
    ComponentTypes,
    LinearRegressor,
    LogisticRegressionClassifier,
    OneHotEncoder,
    SimpleImputer,
    StandardScaler,
    components_dict,
    handle_component,
    handle_component_types
)


def test_components_dict():
    assert len(components_dict()) == 8
    assert len(COMPONENTS) == 8


def test_handle_component():
    component_strs = ['Linear Regressor', 'Simple Imputer']
    components = [LinearRegressor, SimpleImputer]

    for c in zip(component_strs, components):
        assert isinstance(handle_component(c[0]), c[1])

    bad_str = 'RF Select From Model'
    with pytest.raises(ValueError):
        c = handle_component(bad_str)


def test_default_component():
    component_type_strs = ['classifier', 'encoder', 'imputer', 'regressor', 'scaler']
    components = [LogisticRegressionClassifier, OneHotEncoder, SimpleImputer, LinearRegressor, StandardScaler]
    for c in zip(component_type_strs, components):
        assert isinstance(handle_component(c[0]), c[1])


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
    yield correct_component_types


def test_handle_string(correct_component_types):
    component_types = ['classifier', 'encoder', 'feature_selection', 'imputer', 'regressor', 'scaler']
    for component_type in zip(component_types, correct_component_types):
        assert handle_component_types(component_type[0]) == component_type[1]

    component_type = 'fake'
    error_msg = 'Component type \'fake\' does not exist'
    with pytest.raises(ValueError, match=error_msg):
        handle_component_types(component_type) == ComponentTypes.SCALER


def test_handle_component_types(correct_component_types):
    for component_type in correct_component_types:
        assert handle_component_types(component_type) == component_type
