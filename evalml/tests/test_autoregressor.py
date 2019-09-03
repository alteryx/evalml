import pandas as pd
import pytest

from evalml import AutoRegressor
from evalml.demos import load_diabetes
from evalml.pipelines import PipelineBase, get_pipelines


@pytest.fixture
def X_y():
    return load_diabetes()


def test_init(X_y):
    X, y = X_y

    clf = AutoRegressor(objective="R2", max_pipelines=3)

    # check loads all pipelines
    assert get_pipelines(problem_type="regression") == clf.possible_pipelines

    clf.fit(X, y)

    assert isinstance(clf.rankings, pd.DataFrame)

    assert isinstance(clf.best_pipeline, PipelineBase)
    assert isinstance(clf.best_pipeline.feature_importances, pd.DataFrame)

    # test with datafarmes
    clf.fit(pd.DataFrame(X), pd.Series(y))

    assert isinstance(clf.rankings, pd.DataFrame)

    assert isinstance(clf.best_pipeline, PipelineBase)

    assert isinstance(clf.get_pipeline(0), PipelineBase)

    clf.describe_pipeline(0)


def test_random_state(X_y):
    X, y = X_y
    clf = AutoRegressor(objective="R2", max_pipelines=5, random_state=0)
    clf.fit(X, y)

    clf_1 = AutoRegressor(objective="R2", max_pipelines=5, random_state=0)
    clf_1.fit(X, y)

    assert clf.rankings.equals(clf_1.rankings)
