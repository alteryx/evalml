import pytest

from evalml import AutoClassificationSearch, Registry
from evalml.pipelines import PipelineBase


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
    X, y = X_y
    automl = AutoClassificationSearch(objective="precision", allowed_model_families=['linear_model'])
    Registry.register(mock_pipeline)

    automl.search(X, y)
    assert mock_pipeline.name in automl.rankings['pipeline_name'].values
    assert not automl.rankings.isnull().values.any()
