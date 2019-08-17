import pandas as pd
import pytest
from sklearn import datasets

from evalml import AutoClassifier
from evalml.pipelines import PipelineBase, get_pipelines


@pytest.fixture
def X_y():
    X, y = datasets.make_classification(n_samples=100, n_features=20,
                                        n_informative=2, n_redundant=2, random_state=0)

    return X, y


def test_init(X_y):
    X, y = X_y

    clf = AutoClassifier()

    # check loads all pipelines
    assert get_pipelines() == clf.possible_pipelines

    clf.fit(X, y)

    assert isinstance(clf.rankings, pd.DataFrame)

    assert isinstance(clf.best_model, PipelineBase)


def test_init_select_model_types():
    model_types = ["random_forest"]
    clf = AutoClassifier(model_types=model_types)

    # todo also test get_pipelines
    assert get_pipelines(model_types=model_types) == clf.possible_pipelines


def test_max_models(X_y):
    X, y = X_y
    max_models = 3
    clf = AutoClassifier(max_models=max_models)

    clf.fit(X, y)

    assert len(clf.rankings) == max_models


def test_best_model(X_y):
    X, y = X_y
    max_models = 3
    clf = AutoClassifier(max_models=max_models)

    clf.fit(X, y)

    assert len(clf.rankings) == max_models
