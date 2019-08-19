import pytest
from sklearn import datasets

from evalml.pipelines import get_pipelines, list_model_types


@pytest.fixture
def data():
    X, y = datasets.make_classification(n_samples=100, n_features=20,
                                        n_informative=2, n_redundant=2, random_state=0)

    return X, y


def test_list_model_types():
    assert set(list_model_types()) == set(["random_forest", "xgboost", "linear_model"])


def test_get_pipelines():
    assert len(get_pipelines(problem_type="classification")) == 3
    assert len(get_pipelines(problem_type="classification", model_types=["linear_model"])) == 1
    assert len(get_pipelines(problem_type="regression")) == 1
