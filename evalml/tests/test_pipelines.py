import pytest
from sklearn import datasets

from evalml.pipelines import list_model_types


@pytest.fixture
def data():
    X, y = datasets.make_classification(n_samples=100, n_features=20,
                                        n_informative=2, n_redundant=2, random_state=0)

    return X, y


def test_list_model_types():
    assert set(list_model_types()) == set(["random_forest", "xgboost", "linear_model"])



# def test_all_pipelines(data):
#     X, y = data
#     # TODO figure how to get default parameters for pipeline
#     for p in get_pipelines():
#         p
