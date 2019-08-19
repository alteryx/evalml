import pandas as pd
import pytest
from sklearn.model_selection import StratifiedKFold

from evalml import AutoClassifier
from evalml.objectives import Precision
from evalml.pipelines import PipelineBase, get_pipelines


@pytest.fixture
def trained_model(X_y):
    X, y = X_y

    clf = AutoClassifier()

    clf.fit(X, y)

    return clf


def test_init(X_y):
    X, y = X_y

    clf = AutoClassifier()

    # check loads all pipelines
    assert get_pipelines(problem_type="classification") == clf.possible_pipelines

    clf.fit(X, y)

    assert isinstance(clf.rankings, pd.DataFrame)

    assert isinstance(clf.best_pipeline, PipelineBase)

    # test with datafarmes
    clf.fit(pd.DataFrame(X), pd.Series(y))

    assert isinstance(clf.rankings, pd.DataFrame)

    assert isinstance(clf.best_pipeline, PipelineBase)

    assert isinstance(clf.get_pipeline(0), PipelineBase)

    clf.describe_pipeline(0)


def test_cv(X_y):
    X, y = X_y
    cv_folds = 5
    clf = AutoClassifier(cv=StratifiedKFold(cv_folds), max_pipelines=1)

    clf.fit(X, y)

    assert isinstance(clf.rankings, pd.DataFrame)

    assert len(clf.results[0]["scores"]) == cv_folds


def test_init_select_model_types():
    model_types = ["random_forest"]
    clf = AutoClassifier(model_types=model_types)

    assert get_pipelines(problem_type="classification", model_types=model_types) == clf.possible_pipelines
    assert model_types == clf.possible_model_types


def test_max_pipelines(X_y):
    X, y = X_y
    max_pipelines = 3
    clf = AutoClassifier(max_pipelines=max_pipelines)

    clf.fit(X, y)

    assert len(clf.rankings) == max_pipelines


def test_best_pipeline(X_y):
    X, y = X_y
    max_pipelines = 3
    clf = AutoClassifier(max_pipelines=max_pipelines)

    clf.fit(X, y)

    assert len(clf.rankings) == max_pipelines


def test_specify_objective(X_y):
    X, y = X_y
    clf = AutoClassifier(objective=Precision(), max_pipelines=1)
    clf.fit(X, y)

# def test_serialization(trained_model)
