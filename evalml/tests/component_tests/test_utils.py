from importlib import import_module
from unittest.mock import patch

import pytest

from evalml.exceptions import MissingComponentError
from evalml.pipelines.components import all_components, handle_component


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
        assert isinstance(handle_component(cls()), cls)
        assert isinstance(handle_component(name), cls)

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
