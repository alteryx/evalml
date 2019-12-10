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
    assert get_pipelines(problem_type=ProblemTypes.REGRESSION) == clf.possible_pipelines

    clf.fit(X, y, raise_errors=True)

    assert isinstance(clf.rankings, pd.DataFrame)

    assert isinstance(clf.best_pipeline, PipelineBase)
    assert isinstance(clf.best_pipeline.feature_importances, pd.DataFrame)

    # test with datafarmes
    clf.fit(pd.DataFrame(X), pd.Series(y), raise_errors=True)

    assert isinstance(clf.rankings, pd.DataFrame)

    assert isinstance(clf.best_pipeline, PipelineBase)

    assert isinstance(clf.get_pipeline(0), PipelineBase)

    clf.describe_pipeline(0)


def test_random_state(X_y):
    X, y = X_y
    clf = AutoRegressor(objective="R2", max_pipelines=5, random_state=0)
    clf.fit(X, y, raise_errors=True)

    clf_1 = AutoRegressor(objective="R2", max_pipelines=5, random_state=0)
    clf_1.fit(X, y, raise_errors=True)

    # need to use assert_frame_equal as R2 could be different at the 10+ decimal
    assert pd.testing.assert_frame_equal(clf.rankings, clf_1.rankings) is None


def test_categorical_regression(X_y_categorical_regression):
    X, y = X_y_categorical_regression
    clf = AutoRegressor(objective="R2", max_pipelines=5, random_state=0)
    clf.fit(X, y, raise_errors=True)
    assert not clf.rankings['score'].isnull().all()
    assert not clf.get_pipeline(0).feature_importances.isnull().all().all()


def test_callback(X_y):
    X, y = X_y

    counts = {
        "start_iteration_callback": 0,
        "add_result_callback": 0,
    }

    def start_iteration_callback(pipeline_class, parameters, counts=counts):
        counts["start_iteration_callback"] += 1

    def add_result_callback(results, trained_pipeline, counts=counts):
        counts["add_result_callback"] += 1

    max_pipelines = 3
    clf = AutoRegressor(objective="R2", max_pipelines=max_pipelines,
                        start_iteration_callback=start_iteration_callback,
                        add_result_callback=add_result_callback)
    clf.fit(X, y, raise_errors=True)

    assert counts["start_iteration_callback"] == max_pipelines
    assert counts["add_result_callback"] == max_pipelines


def test_early_stopping(capsys, X_y):
    tolerance = 0.005
    patience = 2

    X, y = X_y
    clf = AutoRegressor(objective='r2', max_pipelines=5, patience=patience, model_types=['linear_model'], random_state=0)
    clf.fit(X, y, raise_errors=True)

    out, _ = capsys.readouterr()
    assert "2 iterations without improvement. Stopping search early." in out

    clf = AutoRegressor(objective='r2', max_time='60 seconds', patience=patience, model_types=['linear_model'], random_state=0)
    clf.fit(X, y, raise_errors=True)

    out, _ = capsys.readouterr()
    assert "2 iterations without improvement. Stopping search early." in out

    clf = AutoRegressor(objective='mse', max_time='60 seconds', patience=patience, tolerance=tolerance, model_types=['linear_model'], random_state=0)
    clf.fit(X, y, raise_errors=True)

    num_without_improvement = 0
    best_score = None
    for id in clf.results['search_order']:
        score = clf.results['pipeline_results'][id]['score']
        if best_score is None:
            best_score = score
            continue
        if score < best_score:
            if abs((score - best_score) / best_score) > tolerance:
                num_without_improvement = 0
                best_score = score
        else:
            num_without_improvement += 1
    assert num_without_improvement == 2
