import inspect
from importlib import import_module
from unittest.mock import patch

import pytest

from evalml.exceptions import MissingComponentError
from evalml.pipelines.components import (
    ComponentBase,
    all_components,
    handle_component
)


def test_all_components(has_minimal_dependencies):
    if has_minimal_dependencies:
        assert len(all_components()) == 17
    else:
        assert len(all_components()) == 21


def make_mock_import_module(libs_to_blacklist):
    def _import_module(library):
        if library in libs_to_blacklist:
            raise ImportError("Cannot import {}; blacklisted by mock muahahaha".format(library))
        return import_module(library)
    return _import_module


@patch('importlib.import_module', make_mock_import_module({'xgboost', 'catboost'}))
def test_all_components_core_dependencies_mock():
    assert len(all_components()) == 17


def test_handle_component_names():
    for name, cls in all_components().items():
        cls_ret = handle_component(cls)
        assert inspect.isclass(cls_ret)
        assert issubclass(cls_ret, ComponentBase)
        name_ret = handle_component(name)
        assert inspect.isclass(name_ret)
        assert issubclass(name_ret, ComponentBase)

    for name, cls in all_components().items():
        obj = cls()
        with pytest.raises(ValueError, match='component_graph may only contain str or ComponentBase subclasses, not'):
            handle_component(obj)

    invalid_name = 'This Component Does Not Exist'
    with pytest.raises(MissingComponentError, match='Component "This Component Does Not Exist" was not found'):
        handle_component(invalid_name)

    class NonComponent:
        pass
    with pytest.raises(ValueError):
        handle_component(NonComponent())


def test_all_components_names():
    components = all_components()
    for component_name, component_class in components.items():
        print('Inspecting component {}'.format(component_class.name))
        assert component_class.name == component_name
