import pandas as pd
import pytest

from evalml import AutoRegressor
from evalml.demos import load_diabetes
from evalml.pipelines import PipelineBase, get_pipelines
from evalml.problem_types import ProblemTypes


@pytest.fixture
def X_y():
    return load_diabetes()


def test_init(X_y):
    X, y = X_y

    clf = AutoRegressor(objective="R2", max_pipelines=3)

    # check loads all pipelines
    assert get_pipelines(problem_types=[ProblemTypes.REGRESSION]) == clf.possible_pipelines

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
