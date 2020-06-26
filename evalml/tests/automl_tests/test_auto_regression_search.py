import time
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from evalml import AutoRegressionSearch
from evalml.demos import load_diabetes
from evalml.exceptions import ObjectiveNotFoundError
from evalml.model_family import ModelFamily
from evalml.objectives import MeanSquaredLogError, RootMeanSquaredLogError
from evalml.pipelines import (
    LinearRegressionPipeline,
    MeanBaselineRegressionPipeline,
    PipelineBase
)
from evalml.pipelines.utils import get_estimators, make_pipeline
from evalml.problem_types import ProblemTypes


@pytest.fixture
def X_y():
    return load_diabetes()


def test_init(X_y):
    X, y = X_y

    automl = AutoRegressionSearch(objective="R2", max_pipelines=3, n_jobs=4)
    automl.search(X, y)

    assert automl.n_jobs == 4
    assert isinstance(automl.rankings, pd.DataFrame)
    assert isinstance(automl.best_pipeline, PipelineBase)
    assert isinstance(automl.best_pipeline.feature_importance, pd.DataFrame)

    # test with datafarmes
    automl.search(pd.DataFrame(X), pd.Series(y))

    assert isinstance(automl.rankings, pd.DataFrame)

    assert isinstance(automl.best_pipeline, PipelineBase)

    assert isinstance(automl.get_pipeline(0), PipelineBase)
    with pytest.raises(RuntimeError, match="Pipeline not found"):
        automl.get_pipeline(1000)

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
    assert not automl.get_pipeline(0).feature_importance.isnull().all().all()


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
    assert counts["add_result_callback"] == max_pipelines


def test_early_stopping(caplog):
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
        mock_results['pipeline_results'][id]['pipeline_class'] = LinearRegressionPipeline

    automl.results = mock_results
    automl._check_stopping_condition(time.time())
    out = caplog.text
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
    assert len(x) == 3
    assert len(y) == 3


def test_plot_iterations_max_time(X_y):
    go = pytest.importorskip('plotly.graph_objects', reason='Skipping plotting test because plotly not installed')
    X, y = X_y

    automl = AutoRegressionSearch(max_time=10, random_state=1)
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


def test_log_metrics_only_passed_directly():
    with pytest.raises(ObjectiveNotFoundError, match="Could not find the specified objective."):
        AutoRegressionSearch(additional_objectives=['RootMeanSquaredLogError', 'MeanSquaredLogError'])

    ar = AutoRegressionSearch(additional_objectives=[RootMeanSquaredLogError(), MeanSquaredLogError()])
    assert ar.additional_objectives[0].name == 'Root Mean Squared Log Error'
    assert ar.additional_objectives[1].name == 'Mean Squared Log Error'


def test_automl_allowed_pipelines_no_allowed_pipelines(X_y):
    X, y = X_y
    automl = AutoRegressionSearch(allowed_pipelines=None, allowed_model_families=[])
    assert automl.allowed_pipelines is None
    with pytest.raises(ValueError, match="No allowed pipelines to search"):
        automl.search(X, y)


@patch('evalml.pipelines.RegressionPipeline.score')
@patch('evalml.pipelines.RegressionPipeline.fit')
def test_automl_allowed_pipelines_specified_allowed_pipelines(mock_fit, mock_score, dummy_regression_pipeline_class, X_y):
    X, y = X_y
    automl = AutoRegressionSearch(allowed_pipelines=[dummy_regression_pipeline_class], allowed_model_families=None)
    expected_pipelines = [dummy_regression_pipeline_class]
    mock_score.return_value = {automl.objective.name: 1.0}
    assert automl.allowed_pipelines == expected_pipelines
    assert automl.allowed_model_families is None

    automl.search(X, y)
    mock_fit.assert_called()
    mock_score.assert_called()
    assert automl.allowed_pipelines == expected_pipelines
    assert automl.allowed_model_families == [ModelFamily.NONE]


@patch('evalml.pipelines.RegressionPipeline.score')
@patch('evalml.pipelines.RegressionPipeline.fit')
def test_automl_allowed_pipelines_specified_allowed_model_families(mock_fit, mock_score, X_y, assert_allowed_pipelines_equal_helper):
    X, y = X_y
    automl = AutoRegressionSearch(allowed_pipelines=None, allowed_model_families=[ModelFamily.RANDOM_FOREST])
    mock_score.return_value = {automl.objective.name: 1.0}
    expected_pipelines = [make_pipeline(X, y, estimator, ProblemTypes.REGRESSION) for estimator in get_estimators(ProblemTypes.REGRESSION, model_families=[ModelFamily.RANDOM_FOREST])]
    assert automl.allowed_pipelines is None

    automl.search(X, y)
    assert_allowed_pipelines_equal_helper(automl.allowed_pipelines, expected_pipelines)
    assert set(automl.allowed_model_families) == set([ModelFamily.RANDOM_FOREST])
    mock_fit.assert_called()
    mock_score.assert_called()

    mock_fit.reset_mock()
    mock_score.reset_mock()
    automl = AutoRegressionSearch(allowed_pipelines=None, allowed_model_families=['random_forest'])
    expected_pipelines = [make_pipeline(X, y, estimator, ProblemTypes.REGRESSION) for estimator in get_estimators(ProblemTypes.REGRESSION, model_families=[ModelFamily.RANDOM_FOREST])]
    assert automl.allowed_pipelines is None

    automl.search(X, y)
    assert_allowed_pipelines_equal_helper(automl.allowed_pipelines, expected_pipelines)
    assert set(automl.allowed_model_families) == set([ModelFamily.RANDOM_FOREST])
    mock_fit.assert_called()
    mock_score.assert_called()


@patch('evalml.pipelines.RegressionPipeline.score')
@patch('evalml.pipelines.RegressionPipeline.fit')
def test_automl_allowed_pipelines_init_allowed_both_not_specified(mock_fit, mock_score, X_y, assert_allowed_pipelines_equal_helper):
    X, y = X_y
    automl = AutoRegressionSearch(allowed_pipelines=None, allowed_model_families=None)
    mock_score.return_value = {automl.objective.name: 1.0}
    expected_pipelines = [make_pipeline(X, y, estimator, ProblemTypes.REGRESSION) for estimator in get_estimators(ProblemTypes.REGRESSION, model_families=None)]
    assert automl.allowed_pipelines is None

    automl.search(X, y)
    assert_allowed_pipelines_equal_helper(automl.allowed_pipelines, expected_pipelines)
    assert set(automl.allowed_model_families) == set([p.model_family for p in expected_pipelines])
    mock_fit.assert_called()
    mock_score.assert_called()


@patch('evalml.pipelines.RegressionPipeline.score')
@patch('evalml.pipelines.RegressionPipeline.fit')
def test_automl_allowed_pipelines_init_allowed_both_specified(mock_fit, mock_score, dummy_regression_pipeline_class, X_y, assert_allowed_pipelines_equal_helper):
    X, y = X_y
    automl = AutoRegressionSearch(allowed_pipelines=[dummy_regression_pipeline_class], allowed_model_families=[ModelFamily.RANDOM_FOREST])
    expected_pipelines = [dummy_regression_pipeline_class]
    assert automl.allowed_pipelines == expected_pipelines
    assert set(automl.allowed_model_families) == set([ModelFamily.RANDOM_FOREST])

    automl.search(X, y)
    assert_allowed_pipelines_equal_helper(automl.allowed_pipelines, expected_pipelines)
    assert set(automl.allowed_model_families) == set([p.model_family for p in expected_pipelines])
    mock_fit.assert_called()
    mock_score.assert_called()


@patch('evalml.pipelines.RegressionPipeline.score')
@patch('evalml.pipelines.RegressionPipeline.fit')
def test_automl_allowed_pipelines_search(mock_fit, mock_score, dummy_regression_pipeline_class, X_y):
    X, y = X_y
    mock_score.return_value = {'R2': 1.0}

    allowed_pipelines = [dummy_regression_pipeline_class]
    start_iteration_callback = MagicMock()
    automl = AutoRegressionSearch(max_pipelines=2, start_iteration_callback=start_iteration_callback,
                                  allowed_pipelines=allowed_pipelines)
    automl.search(X, y)

    assert start_iteration_callback.call_count == 2
    assert start_iteration_callback.call_args_list[0][0][0] == MeanBaselineRegressionPipeline
    assert start_iteration_callback.call_args_list[1][0][0] == dummy_regression_pipeline_class
