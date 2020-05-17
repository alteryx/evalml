import time

import pandas as pd
import pytest

from evalml import AutoRegressionSearch
from evalml.demos import load_diabetes
from evalml.pipelines import PipelineBase, get_pipelines
from evalml.problem_types import ProblemTypes


@pytest.fixture
def X_y():
    return load_diabetes()


def test_init(X_y):
    X, y = X_y

    automl = AutoRegressionSearch(objective="R2", max_pipelines=3, n_jobs=4)

    assert automl.n_jobs == 4

    # check loads all pipelines
    assert get_pipelines(problem_type=ProblemTypes.REGRESSION) == automl.possible_pipelines

    automl.search(X, y)

    assert isinstance(automl.rankings, pd.DataFrame)

    assert isinstance(automl.best_pipeline, PipelineBase)
    assert isinstance(automl.best_pipeline.feature_importances, pd.DataFrame)

    # test with datafarmes
    automl.search(pd.DataFrame(X), pd.Series(y))

    assert isinstance(automl.rankings, pd.DataFrame)

    assert isinstance(automl.best_pipeline, PipelineBase)

    assert isinstance(automl.get_pipeline(0), PipelineBase)

    automl.describe_pipeline(0)


def test_random_state(X_y):
    X, y = X_y
    automl = AutoRegressionSearch(objective="R2", max_pipelines=5, random_state=0)
    automl.search(X, y)

    automl_1 = AutoRegressionSearch(objective="R2", max_pipelines=5, random_state=0)
    automl_1.search(X, y)

    # need to use assert_frame_equal as R2 could be different at the 10+ decimal
    assert pd.testing.assert_frame_equal(automl.rankings, automl_1.rankings) is None


def test_categorical_regression(X_y_categorical_regression):
    X, y = X_y_categorical_regression
    automl = AutoRegressionSearch(objective="R2", max_pipelines=5, random_state=0)
    automl.search(X, y)
    assert not automl.rankings['score'].isnull().all()
    assert not automl.get_pipeline(0).feature_importances.isnull().all().all()


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
    automl = AutoRegressionSearch(objective="R2", max_pipelines=max_pipelines,
                                  start_iteration_callback=start_iteration_callback,
                                  add_result_callback=add_result_callback)
    automl.search(X, y)

    assert counts["start_iteration_callback"] == max_pipelines
    assert counts["add_result_callback"] == max_pipelines + 1


def test_early_stopping(capsys):
    tolerance = 0.005
    patience = 2
    automl = AutoRegressionSearch(objective='mse', max_time='60 seconds', patience=patience, tolerance=tolerance, allowed_model_families=['linear_model'], random_state=0)

    mock_results = {
        'search_order': [0, 1, 2],
        'pipeline_results': {}
    }

    scores = [150, 200, 195]
    for id in mock_results['search_order']:
        mock_results['pipeline_results'][id] = {}
        mock_results['pipeline_results'][id]['score'] = scores[id]

    automl.results = mock_results
    automl._check_stopping_condition(time.time())
    out, _ = capsys.readouterr()
    assert "2 iterations without improvement. Stopping search early." in out


def test_plot_disabled_missing_dependency(X_y, has_minimal_dependencies):
    X, y = X_y

    automl = AutoRegressionSearch(max_pipelines=3)
    if has_minimal_dependencies:
        with pytest.raises(AttributeError):
            automl.plot.search_iteration_plot
    else:
        automl.plot.search_iteration_plot


def test_plot_iterations_max_pipelines(X_y):
    go = pytest.importorskip('plotly.graph_objects', reason='Skipping plotting test because plotly not installed')
    X, y = X_y

    automl = AutoRegressionSearch(max_pipelines=3)
    automl.search(X, y)
    plot = automl.plot.search_iteration_plot()
    plot_data = plot.data[0]
    x = pd.Series(plot_data['x'])
    y = pd.Series(plot_data['y'])

    assert isinstance(plot, go.Figure)
    assert x.is_monotonic_increasing
    assert y.is_monotonic_increasing
    assert len(x) == 4  # max_pipelines + 1 baseline
    assert len(y) == 4  # max_pipelines + 1 baseline


def test_plot_iterations_max_time(X_y):
    go = pytest.importorskip('plotly.graph_objects', reason='Skipping plotting test because plotly not installed')
    X, y = X_y

    automl = AutoRegressionSearch(max_time=10)
    automl.search(X, y, show_iteration_plot=False)
    plot = automl.plot.search_iteration_plot()
    plot_data = plot.data[0]
    x = pd.Series(plot_data['x'])
    y = pd.Series(plot_data['y'])

    assert isinstance(plot, go.Figure)
    assert x.is_monotonic_increasing
    assert y.is_monotonic_increasing
    assert len(x) > 0
    assert len(y) > 0
