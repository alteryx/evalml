import time
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import StratifiedKFold, TimeSeriesSplit

from evalml import AutoMLSearch
from evalml.automl.pipeline_search_plots import SearchIterationPlot
from evalml.exceptions import PipelineNotFoundError
from evalml.model_family import ModelFamily
from evalml.objectives import (
    FraudCost,
    Precision,
    PrecisionMicro,
    Recall,
    get_objective,
    get_objectives
)
from evalml.pipelines import ModeBaselineBinaryPipeline, PipelineBase
from evalml.pipelines.components.utils import get_estimators
from evalml.pipelines.utils import make_pipeline
from evalml.problem_types import ProblemTypes


def test_init(X_y_binary):
    X, y = X_y_binary

    automl = AutoMLSearch(problem_type='binary', max_pipelines=1, n_jobs=4)
    automl.search(X, y)

    assert automl.n_jobs == 4
    assert isinstance(automl.rankings, pd.DataFrame)
    assert isinstance(automl.best_pipeline, PipelineBase)

    # test with dataframes
    automl = AutoMLSearch(problem_type='binary', max_pipelines=1, n_jobs=4)
    automl.search(pd.DataFrame(X), pd.Series(y))

    assert isinstance(automl.rankings, pd.DataFrame)
    assert isinstance(automl.full_rankings, pd.DataFrame)
    assert isinstance(automl.best_pipeline, PipelineBase)
    assert isinstance(automl.get_pipeline(0), PipelineBase)
    assert automl.objective.name == 'Log Loss Binary'


def test_init_objective():
    automl = AutoMLSearch(problem_type='binary', objective=Precision(), max_pipelines=1)
    assert isinstance(automl.objective, Precision)
    automl = AutoMLSearch(problem_type='binary', objective='Precision', max_pipelines=1)
    assert isinstance(automl.objective, Precision)


def test_get_pipeline_none():
    automl = AutoMLSearch(problem_type='binary')
    with pytest.raises(PipelineNotFoundError, match="Pipeline not found"):
        automl.describe_pipeline(0)


def test_data_split(X_y_binary):
    X, y = X_y_binary
    cv_folds = 5
    automl = AutoMLSearch(problem_type='binary', data_split=StratifiedKFold(cv_folds), max_pipelines=1)
    automl.search(X, y)

    assert isinstance(automl.rankings, pd.DataFrame)
    assert len(automl.results['pipeline_results'][0]["cv_data"]) == cv_folds

    automl = AutoMLSearch(problem_type='binary', data_split=TimeSeriesSplit(cv_folds), max_pipelines=1)
    automl.search(X, y)

    assert isinstance(automl.rankings, pd.DataFrame)
    assert len(automl.results['pipeline_results'][0]["cv_data"]) == cv_folds


def test_max_pipelines(X_y_binary):
    X, y = X_y_binary
    max_pipelines = 5
    automl = AutoMLSearch(problem_type='binary', max_pipelines=max_pipelines)
    automl.search(X, y)
    assert len(automl.full_rankings) == max_pipelines


def test_recall_error(X_y_binary):
    X, y = X_y_binary
    # Recall is a valid objective but it's not allowed in AutoML so a ValueError is expected
    error_msg = 'Recall is not allowed in AutoML!'
    with pytest.raises(ValueError, match=error_msg):
        AutoMLSearch(problem_type='binary', objective='recall', max_pipelines=1)


def test_recall_object(X_y_binary):
    X, y = X_y_binary
    automl = AutoMLSearch(problem_type='binary', objective=Recall(), max_pipelines=1)
    automl.search(X, y)
    assert len(automl.full_rankings) > 0
    assert automl.objective.name == 'Recall'


def test_binary_auto(X_y_binary):
    X, y = X_y_binary
    automl = AutoMLSearch(problem_type='binary', objective="Log Loss Binary", max_pipelines=5)
    automl.search(X, y)

    best_pipeline = automl.best_pipeline
    best_pipeline.fit(X, y)
    y_pred = best_pipeline.predict(X)
    assert len(np.unique(y_pred)) == 2


def test_multi_auto(X_y_multi):
    X, y = X_y_multi
    objective = PrecisionMicro()
    automl = AutoMLSearch(problem_type='multiclass', objective=objective, max_pipelines=5)
    automl.search(X, y)
    best_pipeline = automl.best_pipeline
    best_pipeline.fit(X, y)
    y_pred = best_pipeline.predict(X)
    assert len(np.unique(y_pred)) == 3

    expected_additional_objectives = [obj() for obj in get_objectives('multiclass') if obj not in automl._objectives_not_allowed_in_automl]
    objective_in_additional_objectives = next((obj for obj in expected_additional_objectives if obj.name == objective.name), None)
    expected_additional_objectives.remove(objective_in_additional_objectives)

    for expected, additional in zip(expected_additional_objectives, automl.additional_objectives):
        assert type(additional) is type(expected)


def test_multi_objective(X_y_multi):
    automl = AutoMLSearch(problem_type='binary', objective="Log Loss Binary")
    assert automl.problem_type == ProblemTypes.BINARY

    automl = AutoMLSearch(problem_type='multiclass', objective="Log Loss Multiclass")
    assert automl.problem_type == ProblemTypes.MULTICLASS

    automl = AutoMLSearch(problem_type='multiclass', objective='AUC Micro')
    assert automl.problem_type == ProblemTypes.MULTICLASS

    automl = AutoMLSearch(problem_type='binary', objective='AUC')
    assert automl.problem_type == ProblemTypes.BINARY

    automl = AutoMLSearch(problem_type='multiclass')
    assert automl.problem_type == ProblemTypes.MULTICLASS

    automl = AutoMLSearch(problem_type='binary')
    assert automl.problem_type == ProblemTypes.BINARY


def test_categorical_classification(X_y_categorical_classification):
    X, y = X_y_categorical_classification
    automl = AutoMLSearch(problem_type='binary', objective="precision", max_pipelines=5)
    automl.search(X, y)
    assert not automl.rankings['score'].isnull().all()


def test_random_state(X_y_binary):
    X, y = X_y_binary

    automl = AutoMLSearch(problem_type='binary', objective=Precision(), max_pipelines=5, random_state=0)
    automl.search(X, y)

    automl_1 = AutoMLSearch(problem_type='binary', objective=Precision(), max_pipelines=5, random_state=0)
    automl_1.search(X, y)
    assert automl.rankings.equals(automl_1.rankings)


def test_callback(X_y_binary):
    X, y = X_y_binary

    counts = {
        "start_iteration_callback": 0,
        "add_result_callback": 0,
    }

    def start_iteration_callback(pipeline_class, parameters, automl_obj, counts=counts):
        counts["start_iteration_callback"] += 1

    def add_result_callback(results, trained_pipeline, automl_obj, counts=counts):
        counts["add_result_callback"] += 1

    max_pipelines = 3
    automl = AutoMLSearch(problem_type='binary', objective=Precision(), max_pipelines=max_pipelines,
                          start_iteration_callback=start_iteration_callback,
                          add_result_callback=add_result_callback)
    automl.search(X, y)

    assert counts["start_iteration_callback"] == max_pipelines
    assert counts["add_result_callback"] == max_pipelines


def test_additional_objectives(X_y_binary):
    X, y = X_y_binary

    objective = FraudCost(retry_percentage=.5,
                          interchange_fee=.02,
                          fraud_payout_percentage=.75,
                          amount_col=10)
    automl = AutoMLSearch(problem_type='binary', objective='F1', max_pipelines=2, additional_objectives=[objective])
    automl.search(X, y)

    results = automl.describe_pipeline(0, return_dict=True)
    assert 'Fraud Cost' in list(results["cv_data"][0]["all_objective_scores"].keys())


@patch('evalml.objectives.BinaryClassificationObjective.optimize_threshold')
@patch('evalml.pipelines.BinaryClassificationPipeline.predict_proba')
@patch('evalml.pipelines.BinaryClassificationPipeline.score')
@patch('evalml.pipelines.BinaryClassificationPipeline.fit')
def test_optimizable_threshold_enabled(mock_fit, mock_score, mock_predict_proba, mock_optimize_threshold, X_y_binary, caplog):
    mock_optimize_threshold.return_value = 0.8
    X, y = X_y_binary
    automl = AutoMLSearch(problem_type='binary', objective='precision', max_pipelines=1, optimize_thresholds=True)
    mock_score.return_value = {automl.objective.name: 1.0}
    automl.search(X, y)
    mock_fit.assert_called()
    mock_score.assert_called()
    mock_predict_proba.assert_called()
    mock_optimize_threshold.assert_called()
    assert automl.best_pipeline.threshold is None
    assert automl.results['pipeline_results'][0]['cv_data'][0].get('binary_classification_threshold') == 0.8
    assert automl.results['pipeline_results'][0]['cv_data'][1].get('binary_classification_threshold') == 0.8
    assert automl.results['pipeline_results'][0]['cv_data'][2].get('binary_classification_threshold') == 0.8

    automl.describe_pipeline(0)
    out = caplog.text
    assert "Objective to optimize binary classification pipeline thresholds for" in out


@patch('evalml.objectives.BinaryClassificationObjective.optimize_threshold')
@patch('evalml.pipelines.BinaryClassificationPipeline.predict_proba')
@patch('evalml.pipelines.BinaryClassificationPipeline.score')
@patch('evalml.pipelines.BinaryClassificationPipeline.fit')
def test_optimizable_threshold_disabled(mock_fit, mock_score, mock_predict_proba, mock_optimize_threshold, X_y_binary):
    mock_optimize_threshold.return_value = 0.8
    X, y = X_y_binary
    automl = AutoMLSearch(problem_type='binary', objective='precision', max_pipelines=1, optimize_thresholds=False)
    mock_score.return_value = {automl.objective.name: 1.0}
    automl.search(X, y)
    mock_fit.assert_called()
    mock_score.assert_called()
    assert not mock_predict_proba.called
    assert not mock_optimize_threshold.called
    assert automl.best_pipeline.threshold is None
    assert automl.results['pipeline_results'][0]['cv_data'][0].get('binary_classification_threshold') == 0.5
    assert automl.results['pipeline_results'][0]['cv_data'][1].get('binary_classification_threshold') == 0.5
    assert automl.results['pipeline_results'][0]['cv_data'][2].get('binary_classification_threshold') == 0.5


@patch('evalml.pipelines.BinaryClassificationPipeline.score')
@patch('evalml.pipelines.BinaryClassificationPipeline.fit')
def test_non_optimizable_threshold(mock_fit, mock_score, X_y_binary):
    mock_score.return_value = {"AUC": 1.0}
    X, y = X_y_binary
    automl = AutoMLSearch(problem_type='binary', objective='AUC', max_pipelines=1)
    automl.search(X, y)
    mock_fit.assert_called()
    mock_score.assert_called()
    assert automl.best_pipeline.threshold is None
    assert automl.results['pipeline_results'][0]['cv_data'][0].get('binary_classification_threshold') == 0.5
    assert automl.results['pipeline_results'][0]['cv_data'][1].get('binary_classification_threshold') == 0.5
    assert automl.results['pipeline_results'][0]['cv_data'][2].get('binary_classification_threshold') == 0.5


def test_describe_pipeline_objective_ordered(X_y_binary, caplog):
    X, y = X_y_binary
    automl = AutoMLSearch(problem_type='binary', objective='AUC', max_pipelines=2)
    automl.search(X, y)

    automl.describe_pipeline(0)
    out = caplog.text
    out_stripped = " ".join(out.split())

    objectives = [get_objective(obj) for obj in automl.additional_objectives]
    objectives_names = [obj.name for obj in objectives]
    expected_objective_order = " ".join(objectives_names)

    assert expected_objective_order in out_stripped


def test_max_time_units():
    str_max_time = AutoMLSearch(problem_type='binary', objective='F1', max_time='60 seconds')
    assert str_max_time.max_time == 60

    hour_max_time = AutoMLSearch(problem_type='binary', objective='F1', max_time='1 hour')
    assert hour_max_time.max_time == 3600

    min_max_time = AutoMLSearch(problem_type='binary', objective='F1', max_time='30 mins')
    assert min_max_time.max_time == 1800

    min_max_time = AutoMLSearch(problem_type='binary', objective='F1', max_time='30 s')
    assert min_max_time.max_time == 30

    with pytest.raises(AssertionError, match="Invalid unit. Units must be hours, mins, or seconds. Received 'year'"):
        AutoMLSearch(problem_type='binary', objective='F1', max_time='30 years')

    with pytest.raises(TypeError, match="max_time must be a float, int, or string. Received a <class 'tuple'>."):
        AutoMLSearch(problem_type='binary', objective='F1', max_time=(30, 'minutes'))


def test_early_stopping(caplog, logistic_regression_binary_pipeline_class):
    with pytest.raises(ValueError, match='patience value must be a positive integer.'):
        automl = AutoMLSearch(problem_type='binary', objective='AUC', max_pipelines=5, allowed_model_families=['linear_model'], patience=-1, random_state=0)

    with pytest.raises(ValueError, match='tolerance value must be'):
        automl = AutoMLSearch(problem_type='binary', objective='AUC', max_pipelines=5, allowed_model_families=['linear_model'], patience=1, tolerance=1.5, random_state=0)

    automl = AutoMLSearch(problem_type='binary', objective='AUC', max_pipelines=5, allowed_model_families=['linear_model'], patience=2, tolerance=0.05, random_state=0)
    mock_results = {
        'search_order': [0, 1, 2],
        'pipeline_results': {}
    }

    scores = [0.95, 0.84, 0.96]  # 0.96 is only 1% greater so it doesn't trigger patience due to tolerance
    for id in mock_results['search_order']:
        mock_results['pipeline_results'][id] = {}
        mock_results['pipeline_results'][id]['score'] = scores[id]
        mock_results['pipeline_results'][id]['pipeline_class'] = logistic_regression_binary_pipeline_class

    automl._results = mock_results
    automl._check_stopping_condition(time.time())
    out = caplog.text
    assert "2 iterations without improvement. Stopping search early." in out


def test_plot_disabled_missing_dependency(X_y_binary, has_minimal_dependencies):
    X, y = X_y_binary

    automl = AutoMLSearch(problem_type='binary', max_pipelines=3)
    if has_minimal_dependencies:
        with pytest.raises(AttributeError):
            automl.plot.search_iteration_plot
    else:
        automl.plot.search_iteration_plot


def test_plot_iterations_max_pipelines(X_y_binary):
    go = pytest.importorskip('plotly.graph_objects', reason='Skipping plotting test because plotly not installed')
    X, y = X_y_binary

    automl = AutoMLSearch(problem_type='binary', objective="f1", max_pipelines=3)
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


def test_plot_iterations_max_time(X_y_binary):
    go = pytest.importorskip('plotly.graph_objects', reason='Skipping plotting test because plotly not installed')
    X, y = X_y_binary

    automl = AutoMLSearch(problem_type='binary', objective="f1", max_time=10)
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


@patch('IPython.display.display')
def test_plot_iterations_ipython_mock(mock_ipython_display, X_y_binary):
    pytest.importorskip('IPython.display', reason='Skipping plotting test because ipywidgets not installed')
    pytest.importorskip('plotly.graph_objects', reason='Skipping plotting test because plotly not installed')
    X, y = X_y_binary

    automl = AutoMLSearch(problem_type='binary', objective="f1", max_pipelines=3)
    automl.search(X, y)
    plot = automl.plot.search_iteration_plot(interactive_plot=True)
    assert isinstance(plot, SearchIterationPlot)
    assert isinstance(plot.data, AutoMLSearch)
    mock_ipython_display.assert_called_with(plot.best_score_by_iter_fig)


@patch('IPython.display.display')
def test_plot_iterations_ipython_mock_import_failure(mock_ipython_display, X_y_binary):
    pytest.importorskip('IPython.display', reason='Skipping plotting test because ipywidgets not installed')
    go = pytest.importorskip('plotly.graph_objects', reason='Skipping plotting test because plotly not installed')
    X, y = X_y_binary

    automl = AutoMLSearch(problem_type='binary', objective="f1", max_pipelines=3)
    automl.search(X, y)

    mock_ipython_display.side_effect = ImportError('KABOOOOOOMMMM')
    plot = automl.plot.search_iteration_plot(interactive_plot=True)
    mock_ipython_display.assert_called_once()

    assert isinstance(plot, go.Figure)
    assert isinstance(plot.data, tuple)
    plot_data = plot.data[0]
    x = pd.Series(plot_data['x'])
    y = pd.Series(plot_data['y'])
    assert x.is_monotonic_increasing
    assert y.is_monotonic_increasing
    assert len(x) == 3
    assert len(y) == 3


def test_max_time(X_y_binary):
    X, y = X_y_binary
    clf = AutoMLSearch(problem_type='binary', max_time=1e-16)
    clf.search(X, y)
    # search will always run at least one pipeline
    assert len(clf.results['pipeline_results']) == 1


@pytest.mark.parametrize("automl_type", [ProblemTypes.BINARY, ProblemTypes.MULTICLASS])
def test_automl_allowed_pipelines_no_allowed_pipelines(automl_type, X_y_binary, X_y_multi):
    is_multiclass = automl_type == ProblemTypes.MULTICLASS
    X, y = X_y_multi if is_multiclass else X_y_binary
    problem_type = 'multiclass' if is_multiclass else 'binary'
    automl = AutoMLSearch(problem_type=problem_type, allowed_pipelines=None, allowed_model_families=[])
    assert automl.allowed_pipelines is None
    with pytest.raises(ValueError, match="No allowed pipelines to search"):
        automl.search(X, y)


@patch('evalml.pipelines.BinaryClassificationPipeline.score')
@patch('evalml.pipelines.BinaryClassificationPipeline.fit')
def test_automl_allowed_pipelines_specified_allowed_pipelines_binary(mock_fit, mock_score, dummy_binary_pipeline_class, X_y_binary):
    X, y = X_y_binary
    automl = AutoMLSearch(problem_type='binary', allowed_pipelines=[dummy_binary_pipeline_class], allowed_model_families=None)
    expected_pipelines = [dummy_binary_pipeline_class]
    mock_score.return_value = {automl.objective.name: 1.0}
    assert automl.allowed_pipelines == expected_pipelines
    assert automl.allowed_model_families is None

    automl.search(X, y)
    mock_fit.assert_called()
    mock_score.assert_called()
    assert automl.allowed_pipelines == expected_pipelines
    assert automl.allowed_model_families == [ModelFamily.NONE]


@patch('evalml.pipelines.MulticlassClassificationPipeline.score')
@patch('evalml.pipelines.MulticlassClassificationPipeline.fit')
def test_automl_allowed_pipelines_specified_allowed_pipelines_multi(mock_fit, mock_score, dummy_multiclass_pipeline_class, X_y_multi):
    X, y = X_y_multi
    automl = AutoMLSearch(problem_type='multiclass', allowed_pipelines=[dummy_multiclass_pipeline_class], allowed_model_families=None)
    expected_pipelines = [dummy_multiclass_pipeline_class]
    mock_score.return_value = {automl.objective.name: 1.0}
    assert automl.allowed_pipelines == expected_pipelines
    assert automl.allowed_model_families is None

    automl.search(X, y)
    mock_fit.assert_called()
    mock_score.assert_called()
    assert automl.allowed_pipelines == expected_pipelines
    assert automl.allowed_model_families == [ModelFamily.NONE]


@patch('evalml.pipelines.BinaryClassificationPipeline.score')
@patch('evalml.pipelines.BinaryClassificationPipeline.fit')
def test_automl_allowed_pipelines_specified_allowed_model_families_binary(mock_fit, mock_score, X_y_binary, assert_allowed_pipelines_equal_helper):
    X, y = X_y_binary
    automl = AutoMLSearch(problem_type='binary', allowed_pipelines=None, allowed_model_families=[ModelFamily.RANDOM_FOREST])
    mock_score.return_value = {automl.objective.name: 1.0}
    expected_pipelines = [make_pipeline(X, y, estimator, ProblemTypes.BINARY) for estimator in get_estimators(ProblemTypes.BINARY, model_families=[ModelFamily.RANDOM_FOREST])]
    assert automl.allowed_pipelines is None

    automl.search(X, y)
    assert_allowed_pipelines_equal_helper(automl.allowed_pipelines, expected_pipelines)
    assert set(automl.allowed_model_families) == set([ModelFamily.RANDOM_FOREST])
    mock_fit.assert_called()
    mock_score.assert_called()

    mock_fit.reset_mock()
    mock_score.reset_mock()
    automl = AutoMLSearch(problem_type='binary', allowed_pipelines=None, allowed_model_families=['random_forest'])
    expected_pipelines = [make_pipeline(X, y, estimator, ProblemTypes.BINARY) for estimator in get_estimators(ProblemTypes.BINARY, model_families=[ModelFamily.RANDOM_FOREST])]
    assert automl.allowed_pipelines is None
    automl.search(X, y)
    assert_allowed_pipelines_equal_helper(automl.allowed_pipelines, expected_pipelines)
    assert set(automl.allowed_model_families) == set([ModelFamily.RANDOM_FOREST])
    mock_fit.assert_called()
    mock_score.assert_called()


@patch('evalml.pipelines.MulticlassClassificationPipeline.score')
@patch('evalml.pipelines.MulticlassClassificationPipeline.fit')
def test_automl_allowed_pipelines_specified_allowed_model_families_multi(mock_fit, mock_score, X_y_multi, assert_allowed_pipelines_equal_helper):
    X, y = X_y_multi
    automl = AutoMLSearch(problem_type='multiclass', allowed_pipelines=None, allowed_model_families=[ModelFamily.RANDOM_FOREST])
    mock_score.return_value = {automl.objective.name: 1.0}
    expected_pipelines = [make_pipeline(X, y, estimator, ProblemTypes.MULTICLASS) for estimator in get_estimators(ProblemTypes.MULTICLASS, model_families=[ModelFamily.RANDOM_FOREST])]
    assert automl.allowed_pipelines is None

    automl.search(X, y)
    assert_allowed_pipelines_equal_helper(automl.allowed_pipelines, expected_pipelines)
    assert set(automl.allowed_model_families) == set([ModelFamily.RANDOM_FOREST])
    mock_fit.assert_called()
    mock_score.assert_called()

    mock_fit.reset_mock()
    mock_score.reset_mock()
    automl = AutoMLSearch(problem_type='multiclass', allowed_pipelines=None, allowed_model_families=['random_forest'])
    expected_pipelines = [make_pipeline(X, y, estimator, ProblemTypes.MULTICLASS) for estimator in get_estimators(ProblemTypes.MULTICLASS, model_families=[ModelFamily.RANDOM_FOREST])]
    assert automl.allowed_pipelines is None
    automl.search(X, y)
    assert_allowed_pipelines_equal_helper(automl.allowed_pipelines, expected_pipelines)
    assert set(automl.allowed_model_families) == set([ModelFamily.RANDOM_FOREST])
    mock_fit.assert_called()
    mock_score.assert_called()


@patch('evalml.pipelines.BinaryClassificationPipeline.score')
@patch('evalml.pipelines.BinaryClassificationPipeline.fit')
def test_automl_allowed_pipelines_init_allowed_both_not_specified_binary(mock_fit, mock_score, X_y_binary, assert_allowed_pipelines_equal_helper):
    X, y = X_y_binary
    automl = AutoMLSearch(problem_type='binary', allowed_pipelines=None, allowed_model_families=None)
    mock_score.return_value = {automl.objective.name: 1.0}
    expected_pipelines = [make_pipeline(X, y, estimator, ProblemTypes.BINARY) for estimator in get_estimators(ProblemTypes.BINARY, model_families=None)]
    assert automl.allowed_pipelines is None

    automl.search(X, y)
    assert_allowed_pipelines_equal_helper(automl.allowed_pipelines, expected_pipelines)
    assert set(automl.allowed_model_families) == set([p.model_family for p in expected_pipelines])
    mock_fit.assert_called()
    mock_score.assert_called()


@patch('evalml.pipelines.MulticlassClassificationPipeline.score')
@patch('evalml.pipelines.MulticlassClassificationPipeline.fit')
def test_automl_allowed_pipelines_init_allowed_both_not_specified_multi(mock_fit, mock_score, X_y_multi, assert_allowed_pipelines_equal_helper):
    X, y = X_y_multi
    automl = AutoMLSearch(problem_type='multiclass', allowed_pipelines=None, allowed_model_families=None)
    mock_score.return_value = {automl.objective.name: 1.0}
    expected_pipelines = [make_pipeline(X, y, estimator, ProblemTypes.MULTICLASS) for estimator in get_estimators(ProblemTypes.MULTICLASS, model_families=None)]
    assert automl.allowed_pipelines is None

    automl.search(X, y)
    assert_allowed_pipelines_equal_helper(automl.allowed_pipelines, expected_pipelines)
    assert set(automl.allowed_model_families) == set([p.model_family for p in expected_pipelines])
    mock_fit.assert_called()
    mock_score.assert_called()


@patch('evalml.pipelines.BinaryClassificationPipeline.score')
@patch('evalml.pipelines.BinaryClassificationPipeline.fit')
def test_automl_allowed_pipelines_init_allowed_both_specified_binary(mock_fit, mock_score, dummy_binary_pipeline_class, X_y_binary, assert_allowed_pipelines_equal_helper):
    X, y = X_y_binary
    automl = AutoMLSearch(problem_type='binary', allowed_pipelines=[dummy_binary_pipeline_class], allowed_model_families=[ModelFamily.RANDOM_FOREST])
    mock_score.return_value = {automl.objective.name: 1.0}
    expected_pipelines = [dummy_binary_pipeline_class]
    assert automl.allowed_pipelines == expected_pipelines
    assert set(automl.allowed_model_families) == set([ModelFamily.RANDOM_FOREST])

    automl.search(X, y)
    assert_allowed_pipelines_equal_helper(automl.allowed_pipelines, expected_pipelines)
    assert set(automl.allowed_model_families) == set([p.model_family for p in expected_pipelines])
    mock_fit.assert_called()
    mock_score.assert_called()


@patch('evalml.pipelines.MulticlassClassificationPipeline.score')
@patch('evalml.pipelines.MulticlassClassificationPipeline.fit')
def test_automl_allowed_pipelines_init_allowed_both_specified_multi(mock_fit, mock_score, dummy_multiclass_pipeline_class, X_y_multi, assert_allowed_pipelines_equal_helper):
    X, y = X_y_multi
    automl = AutoMLSearch(problem_type='multiclass', allowed_pipelines=[dummy_multiclass_pipeline_class], allowed_model_families=[ModelFamily.RANDOM_FOREST])
    mock_score.return_value = {automl.objective.name: 1.0}
    expected_pipelines = [dummy_multiclass_pipeline_class]
    assert automl.allowed_pipelines == expected_pipelines
    assert set(automl.allowed_model_families) == set([ModelFamily.RANDOM_FOREST])

    automl.search(X, y)
    assert_allowed_pipelines_equal_helper(automl.allowed_pipelines, expected_pipelines)
    assert set(automl.allowed_model_families) == set([p.model_family for p in expected_pipelines])
    mock_fit.assert_called()
    mock_score.assert_called()


@patch('evalml.pipelines.BinaryClassificationPipeline.score')
@patch('evalml.pipelines.BinaryClassificationPipeline.fit')
def test_automl_allowed_pipelines_search(mock_fit, mock_score, dummy_binary_pipeline_class, X_y_binary):
    X, y = X_y_binary
    mock_score.return_value = {'Log Loss Binary': 1.0}

    allowed_pipelines = [dummy_binary_pipeline_class]
    start_iteration_callback = MagicMock()
    automl = AutoMLSearch(problem_type='binary', max_pipelines=2, start_iteration_callback=start_iteration_callback,
                          allowed_pipelines=allowed_pipelines)
    automl.search(X, y)

    assert start_iteration_callback.call_count == 2
    assert start_iteration_callback.call_args_list[0][0][0] == ModeBaselineBinaryPipeline
    assert start_iteration_callback.call_args_list[1][0][0] == dummy_binary_pipeline_class
