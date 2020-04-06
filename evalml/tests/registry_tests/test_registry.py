from importlib import import_module
from unittest.mock import patch

import pytest

from evalml import AutoClassificationSearch, Registry
from evalml.model_family import ModelFamily
from evalml.pipelines import PipelineBase
from evalml.problem_types import ProblemTypes


@pytest.fixture
def clean_registry():
    Registry._other_pipelines = []


@pytest.fixture
def mock_pipeline():
    class MockPipeline(PipelineBase):
        component_graph = ["Logistic Regression Classifier"]
        supported_problem_types = ['binary']

    return MockPipeline


def test_register(clean_registry, mock_pipeline):
    Registry.register(mock_pipeline)
    assert mock_pipeline in Registry.all_pipelines()


def test_register_from_components(clean_registry):
    component_graph = ["Logistic Regression Classifier"]
    supported_problem_types = ['binary']
    name = "MockPipeline"

    Registry.register_from_components(component_graph, supported_problem_types, name)
    assert Registry.find_pipeline("Mock Pipeline")


def test_registry_with_automl(clean_registry, mock_pipeline, X_y):
    Registry.register(mock_pipeline)
    automl = AutoClassificationSearch(objective="precision", allowed_model_families=['linear_model'])

    assert mock_pipeline in automl.possible_pipelines


def test_list_model_families(has_minimal_dependencies):
    expected_model_families_binary = set([ModelFamily.RANDOM_FOREST, ModelFamily.LINEAR_MODEL])
    expected_model_families_regression = set([ModelFamily.RANDOM_FOREST, ModelFamily.LINEAR_MODEL])
    if not has_minimal_dependencies:
        expected_model_families_binary.add(ModelFamily.XGBOOST)
        expected_model_families_binary.add(ModelFamily.CATBOOST)
        expected_model_families_regression.add(ModelFamily.CATBOOST)
    assert set(Registry.list_model_families(ProblemTypes.BINARY)) == expected_model_families_binary
    assert set(Registry.list_model_families(ProblemTypes.REGRESSION)) == expected_model_families_regression


def test_default_pipelines(has_minimal_dependencies):
    if has_minimal_dependencies:
        assert len(Registry.default_pipelines) == 4
    else:
        assert len(Registry.default_pipelines) == 7


def make_mock_import_module(libs_to_blacklist):
    def _import_module(library):
        if library in libs_to_blacklist:
            raise ImportError("Cannot import {}; blacklisted by mock muahahaha".format(library))
        return import_module(library)
    return _import_module


@patch('importlib.import_module', make_mock_import_module({'xgboost', 'catboost'}))
def test_default_pipelines_core_dependencies_mock():
    assert len(Registry.default_pipelines) == 4


def test_get_pipelines(has_minimal_dependencies, clean_registry):
    if has_minimal_dependencies:
        assert len(Registry.get_pipelines(problem_type=ProblemTypes.BINARY)) == 2
        assert len(Registry.get_pipelines(problem_type=ProblemTypes.BINARY, model_families=[ModelFamily.LINEAR_MODEL])) == 1
        assert len(Registry.get_pipelines(problem_type=ProblemTypes.MULTICLASS)) == 2
        assert len(Registry.get_pipelines(problem_type=ProblemTypes.REGRESSION)) == 2
    else:
        assert len(Registry.get_pipelines(problem_type=ProblemTypes.BINARY)) == 4
        assert len(Registry.get_pipelines(problem_type=ProblemTypes.BINARY, model_families=[ModelFamily.LINEAR_MODEL])) == 1
        assert len(Registry.get_pipelines(problem_type=ProblemTypes.MULTICLASS)) == 4
        assert len(Registry.get_pipelines(problem_type=ProblemTypes.REGRESSION)) == 3

    with pytest.raises(RuntimeError, match="Unrecognized model type for problem type"):
        Registry.get_pipelines(problem_type=ProblemTypes.REGRESSION, model_families=["random_forest", "xgboost"])
    with pytest.raises(KeyError):
        Registry.get_pipelines(problem_type="Not A Valid Problem Type")


@patch('importlib.import_module', make_mock_import_module({'xgboost', 'catboost'}))
def test_get_pipelines_core_dependencies_mock(clean_registry):
    assert len(Registry.get_pipelines(problem_type=ProblemTypes.BINARY)) == 2
    assert len(Registry.get_pipelines(problem_type=ProblemTypes.BINARY, model_families=[ModelFamily.LINEAR_MODEL])) == 1
    assert len(Registry.get_pipelines(problem_type=ProblemTypes.MULTICLASS)) == 2
    assert len(Registry.get_pipelines(problem_type=ProblemTypes.REGRESSION)) == 2
    with pytest.raises(RuntimeError, match="Unrecognized model type for problem type"):
        Registry.get_pipelines(problem_type=ProblemTypes.REGRESSION, model_families=["random_forest", "xgboost"])
    with pytest.raises(KeyError):
        Registry.get_pipelines(problem_type="Not A Valid Problem Type")
