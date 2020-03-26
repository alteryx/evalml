from importlib import import_module
from unittest.mock import patch

from evalml.pipelines.components import all_components


def test_all_components(has_minimal_dependencies):
    if has_minimal_dependencies:
        assert len(all_components()) == 9
    else:
        print(all_components())
        assert len(all_components()) == 12


def make_mock_import_module(libs_to_blacklist):
    def _import_module(library):
        if library in libs_to_blacklist:
            raise ImportError("Cannot import {}; blacklisted by mock muahahaha".format(library))
        return import_module(library)
    return _import_module


@patch('importlib.import_module', make_mock_import_module({'xgboost', 'catboost'}))
def test_all_components_core_dependencies_mock():
    assert len(all_components()) == 9
