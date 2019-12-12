import time

import pandas as pd
import plotly.graph_objects as go
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


def test_early_stopping(capsys):
    tolerance = 0.005
    patience = 2
    clf = AutoRegressor(objective='mse', max_time='60 seconds', patience=patience, tolerance=tolerance, model_types=['linear_model'], random_state=0)

    mock_results = {
        'search_order': [0, 1, 2],
        'pipeline_results': {}
    }

    scores = [150, 200, 195]
    for id in mock_results['search_order']:
        mock_results['pipeline_results'][id] = {}
        mock_results['pipeline_results'][id]['score'] = scores[id]

    clf.results = mock_results
    clf._check_stopping_condition(time.time())
    out, _ = capsys.readouterr()
    assert "2 iterations without improvement. Stopping search early." in out


def test_plot_iterations_max_pipelines(X_y):
    X, y = X_y

    clf = AutoRegressor(max_pipelines=3)
    clf.fit(X, y)
    plot = clf.plot.search_iteration_plot()
    plot_data = plot.data[0]
    x = pd.Series(plot_data['x'])
    y = pd.Series(plot_data['y'])

    assert isinstance(plot, go.Figure)
    assert x.is_monotonic_increasing
    assert y.is_monotonic_increasing
    assert len(x) == 3
    assert len(y) == 3


def test_plot_iterations_max_time(X_y):
    X, y = X_y
    clf = AutoRegressor(max_time=10)
    clf.fit(X, y, show_iteration_plot=False)
    plot = clf.plot.search_iteration_plot()
    plot_data = plot.data[0]
    x = pd.Series(plot_data['x'])
    y = pd.Series(plot_data['y'])

    assert isinstance(plot, go.Figure)
    assert x.is_monotonic_increasing
    assert y.is_monotonic_increasing
    assert len(x) > 0
    assert len(y) > 0
