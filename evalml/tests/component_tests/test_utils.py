import pytest

from evalml.pipelines.components import (
    COMPONENTS,
    LinearRegressor,
    SimpleImputer,
    components_dict,
    handle_component
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
