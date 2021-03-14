from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from evalml import AutoMLSearch
from evalml.exceptions import ObjectiveNotFoundError
from evalml.model_family import ModelFamily
from evalml.objectives import MeanSquaredLogError, RootMeanSquaredLogError
from evalml.pipelines import (
    MeanBaselineRegressionPipeline,
    PipelineBase,
    TimeSeriesBaselineRegressionPipeline
)
from evalml.pipelines.components.utils import get_estimators
from evalml.pipelines.utils import make_pipeline
from evalml.preprocessing import TimeSeriesSplit
from evalml.problem_types import ProblemTypes


def test_init(X_y_regression):
    X, y = X_y_regression

    automl = AutoMLSearch(X_train=X, y_train=y, problem_type='regression', objective="R2", max_iterations=3, n_jobs=1)
    automl.search()

    assert automl.n_jobs == 1
    assert isinstance(automl.rankings, pd.DataFrame)
    assert isinstance(automl.best_pipeline, PipelineBase)
    automl.best_pipeline.predict(X)

    # test with dataframes
    automl = AutoMLSearch(pd.DataFrame(X), pd.Series(y), problem_type='regression', objective="R2", max_iterations=3, n_jobs=1)
    automl.search()

    assert isinstance(automl.rankings, pd.DataFrame)
    assert isinstance(automl.full_rankings, pd.DataFrame)
    assert isinstance(automl.best_pipeline, PipelineBase)
    automl.best_pipeline.predict(X)
    assert isinstance(automl.get_pipeline(0), PipelineBase)


def test_random_seed(X_y_regression):
    X, y = X_y_regression
    automl = AutoMLSearch(X_train=X, y_train=y, problem_type='regression', objective="R2", max_iterations=5, random_seed=0,
                          n_jobs=1)
    automl.search()

    automl_1 = AutoMLSearch(X_train=X, y_train=y, problem_type='regression', objective="R2", max_iterations=5, random_seed=0,
                            n_jobs=1)
    automl_1.search()

    # need to use assert_frame_equal as R2 could be different at the 10+ decimal
    assert pd.testing.assert_frame_equal(automl.rankings, automl_1.rankings) is None


def test_categorical_regression(X_y_categorical_regression):
    X, y = X_y_categorical_regression
    automl = AutoMLSearch(X_train=X, y_train=y, problem_type='regression', objective="R2", max_iterations=5, random_seed=0,
                          n_jobs=1)
    automl.search()
    assert not automl.rankings['score'].isnull().all()


def test_callback(X_y_regression):
    X, y = X_y_regression

    counts = {
        "start_iteration_callback": 0,
        "add_result_callback": 0,
    }

    def start_iteration_callback(pipeline_class, parameters, automl_obj, counts=counts):
        counts["start_iteration_callback"] += 1

    def add_result_callback(results, trained_pipeline, automl_obj, counts=counts):
        counts["add_result_callback"] += 1

    max_iterations = 3
    automl = AutoMLSearch(X_train=X, y_train=y, problem_type='regression', objective="R2", max_iterations=max_iterations,
                          start_iteration_callback=start_iteration_callback,
                          add_result_callback=add_result_callback, n_jobs=1)
    automl.search()

    assert counts["start_iteration_callback"] == max_iterations
    assert counts["add_result_callback"] == max_iterations


def test_plot_disabled_missing_dependency(X_y_regression, has_minimal_dependencies):
    X, y = X_y_regression

    automl = AutoMLSearch(X_train=X, y_train=y, problem_type='regression', max_iterations=3)
    if has_minimal_dependencies:
        with pytest.raises(AttributeError):
            automl.plot.search_iteration_plot
    else:
        automl.plot.search_iteration_plot


def test_plot_iterations_max_iterations(X_y_regression):
    go = pytest.importorskip('plotly.graph_objects', reason='Skipping plotting test because plotly not installed')
    X, y = X_y_regression

    automl = AutoMLSearch(X_train=X, y_train=y, problem_type='regression', max_iterations=3, n_jobs=1)
    automl.search()
    plot = automl.plot.search_iteration_plot()
    plot_data = plot.data[0]
    x = pd.Series(plot_data['x'])
    y = pd.Series(plot_data['y'])

    assert isinstance(plot, go.Figure)
    assert x.is_monotonic_increasing
    assert y.is_monotonic_increasing
    assert len(x) == 3
    assert len(y) == 3


def test_plot_iterations_max_time(X_y_regression):
    go = pytest.importorskip('plotly.graph_objects', reason='Skipping plotting test because plotly not installed')
    X, y = X_y_regression

    automl = AutoMLSearch(X_train=X, y_train=y, problem_type='regression', max_time=10, random_seed=1, n_jobs=1)
    automl.search(show_iteration_plot=False)
    plot = automl.plot.search_iteration_plot()
    plot_data = plot.data[0]
    x = pd.Series(plot_data['x'])
    y = pd.Series(plot_data['y'])

    assert isinstance(plot, go.Figure)
    assert x.is_monotonic_increasing
    assert y.is_monotonic_increasing
    assert len(x) > 0
    assert len(y) > 0


def test_log_metrics_only_passed_directly(X_y_regression):
    X, y = X_y_regression
    with pytest.raises(ObjectiveNotFoundError, match="RootMeanSquaredLogError is not a valid Objective!"):
        AutoMLSearch(X_train=X, y_train=y, problem_type='regression', additional_objectives=['RootMeanSquaredLogError', 'MeanSquaredLogError'])

    ar = AutoMLSearch(X_train=X, y_train=y, problem_type='regression', additional_objectives=[RootMeanSquaredLogError(), MeanSquaredLogError()])
    assert ar.additional_objectives[0].name == 'Root Mean Squared Log Error'
    assert ar.additional_objectives[1].name == 'Mean Squared Log Error'


def test_automl_allowed_pipelines_no_allowed_pipelines(X_y_regression):
    X, y = X_y_regression
    with pytest.raises(ValueError, match="No allowed pipelines to search"):
        AutoMLSearch(X_train=X, y_train=y, problem_type='regression', allowed_pipelines=None, allowed_model_families=[])


@patch('evalml.pipelines.RegressionPipeline.score')
@patch('evalml.pipelines.RegressionPipeline.fit')
def test_automl_allowed_pipelines_specified_allowed_pipelines(mock_fit, mock_score, dummy_regression_pipeline_class, X_y_regression):
    X, y = X_y_regression
    automl = AutoMLSearch(X_train=X, y_train=y, problem_type='regression', allowed_pipelines=[dummy_regression_pipeline_class], allowed_model_families=None)
    mock_score.return_value = {automl.objective.name: 1.0}
    expected_pipelines = [dummy_regression_pipeline_class]
    mock_score.return_value = {automl.objective.name: 1.0}
    assert automl.allowed_pipelines == expected_pipelines
    assert automl.allowed_model_families == [ModelFamily.NONE]

    automl.search()
    mock_fit.assert_called()
    mock_score.assert_called()
    assert automl.allowed_pipelines == expected_pipelines
    assert automl.allowed_model_families == [ModelFamily.NONE]


@patch('evalml.pipelines.RegressionPipeline.score')
@patch('evalml.pipelines.RegressionPipeline.fit')
def test_automl_allowed_pipelines_specified_allowed_model_families(mock_fit, mock_score, X_y_regression, assert_allowed_pipelines_equal_helper):
    X, y = X_y_regression
    automl = AutoMLSearch(X_train=X, y_train=y, problem_type='regression', allowed_pipelines=None, allowed_model_families=[ModelFamily.RANDOM_FOREST])
    mock_score.return_value = {automl.objective.name: 1.0}
    expected_pipelines = [make_pipeline(X, y, estimator, ProblemTypes.REGRESSION) for estimator in get_estimators(ProblemTypes.REGRESSION, model_families=[ModelFamily.RANDOM_FOREST])]
    assert_allowed_pipelines_equal_helper(automl.allowed_pipelines, expected_pipelines)
    assert set(automl.allowed_model_families) == set([ModelFamily.RANDOM_FOREST])
    automl.search()
    mock_fit.assert_called()
    mock_score.assert_called()

    mock_fit.reset_mock()
    mock_score.reset_mock()
    automl = AutoMLSearch(X_train=X, y_train=y, problem_type='regression', allowed_pipelines=None, allowed_model_families=['random_forest'])
    expected_pipelines = [make_pipeline(X, y, estimator, ProblemTypes.REGRESSION) for estimator in get_estimators(ProblemTypes.REGRESSION, model_families=[ModelFamily.RANDOM_FOREST])]
    assert_allowed_pipelines_equal_helper(automl.allowed_pipelines, expected_pipelines)
    assert set(automl.allowed_model_families) == set([ModelFamily.RANDOM_FOREST])
    automl.search()
    mock_fit.assert_called()
    mock_score.assert_called()


@patch('evalml.pipelines.RegressionPipeline.score')
@patch('evalml.pipelines.RegressionPipeline.fit')
def test_automl_allowed_pipelines_init_allowed_both_not_specified(mock_fit, mock_score, X_y_regression, assert_allowed_pipelines_equal_helper):
    X, y = X_y_regression
    automl = AutoMLSearch(X_train=X, y_train=y, problem_type='regression', allowed_pipelines=None, allowed_model_families=None)
    mock_score.return_value = {automl.objective.name: 1.0}
    expected_pipelines = [make_pipeline(X, y, estimator, ProblemTypes.REGRESSION) for estimator in get_estimators(ProblemTypes.REGRESSION, model_families=None)]
    assert_allowed_pipelines_equal_helper(automl.allowed_pipelines, expected_pipelines)
    assert set(automl.allowed_model_families) == set([p.model_family for p in expected_pipelines])
    automl.search()
    mock_fit.assert_called()
    mock_score.assert_called()


@patch('evalml.pipelines.RegressionPipeline.score')
@patch('evalml.pipelines.RegressionPipeline.fit')
def test_automl_allowed_pipelines_init_allowed_both_specified(mock_fit, mock_score, dummy_regression_pipeline_class, X_y_regression, assert_allowed_pipelines_equal_helper):
    X, y = X_y_regression
    automl = AutoMLSearch(X_train=X, y_train=y, problem_type='regression', allowed_pipelines=[dummy_regression_pipeline_class], allowed_model_families=[ModelFamily.RANDOM_FOREST])
    mock_score.return_value = {automl.objective.name: 1.0}
    expected_pipelines = [dummy_regression_pipeline_class]
    assert_allowed_pipelines_equal_helper(automl.allowed_pipelines, expected_pipelines)
    assert set(automl.allowed_model_families) == set([p.model_family for p in expected_pipelines])
    automl.search()
    mock_fit.assert_called()
    mock_score.assert_called()


@patch('evalml.pipelines.RegressionPipeline.score')
@patch('evalml.pipelines.RegressionPipeline.fit')
def test_automl_allowed_pipelines_search(mock_fit, mock_score, dummy_regression_pipeline_class, X_y_regression):
    X, y = X_y_regression
    mock_score.return_value = {'R2': 1.0}

    allowed_pipelines = [dummy_regression_pipeline_class]
    start_iteration_callback = MagicMock()
    automl = AutoMLSearch(X_train=X, y_train=y, problem_type='regression', max_iterations=2, start_iteration_callback=start_iteration_callback,
                          allowed_pipelines=allowed_pipelines)
    automl.search()

    assert start_iteration_callback.call_count == 2
    assert start_iteration_callback.call_args_list[0][0][0] == MeanBaselineRegressionPipeline
    assert start_iteration_callback.call_args_list[1][0][0] == dummy_regression_pipeline_class


def test_automl_regression_nonlinear_pipeline_search(nonlinear_regression_pipeline_class, X_y_regression):
    X, y = X_y_regression

    allowed_pipelines = [nonlinear_regression_pipeline_class]
    start_iteration_callback = MagicMock()
    automl = AutoMLSearch(X_train=X, y_train=y, problem_type='regression', max_iterations=2, start_iteration_callback=start_iteration_callback,
                          allowed_pipelines=allowed_pipelines, n_jobs=1)
    automl.search()

    assert start_iteration_callback.call_count == 2
    assert start_iteration_callback.call_args_list[0][0][0] == MeanBaselineRegressionPipeline
    assert start_iteration_callback.call_args_list[1][0][0] == nonlinear_regression_pipeline_class


@patch('evalml.pipelines.TimeSeriesRegressionPipeline.score', return_value={"R2": 0.3})
@patch('evalml.pipelines.TimeSeriesRegressionPipeline.fit')
def test_automl_supports_time_series_regression(mock_fit, mock_score, X_y_regression):
    X, y = X_y_regression

    configuration = {"gap": 0, "max_delay": 0, 'delay_target': False, 'delay_features': True}

    automl = AutoMLSearch(X_train=X, y_train=y, problem_type="time series regression", problem_configuration=configuration,
                          max_batches=2)
    automl.search()
    assert isinstance(automl.data_splitter, TimeSeriesSplit)
    for result in automl.results['pipeline_results'].values():
        if result["id"] == 0:
            assert result['pipeline_class'] == TimeSeriesBaselineRegressionPipeline
            continue

        assert result['parameters']['Delayed Feature Transformer'] == configuration
        assert result['parameters']['pipeline'] == configuration
