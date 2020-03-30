import time
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import StratifiedKFold, TimeSeriesSplit

from evalml import AutoClassificationSearch
from evalml.automl.pipeline_search_plots import SearchIterationPlot
from evalml.model_family import ModelFamily
from evalml.objectives import (
    FraudCost,
    Precision,
    PrecisionMicro,
    get_objective,
    get_objectives
)
from evalml.pipelines import PipelineBase, get_pipelines
from evalml.problem_types import ProblemTypes


def test_init(X_y):
    X, y = X_y

    automl = AutoClassificationSearch(multiclass=False, max_pipelines=1, n_jobs=4)

    assert automl.n_jobs == 4

    # check loads all pipelines
    assert get_pipelines(problem_type=ProblemTypes.BINARY) == automl.possible_pipelines

    automl.search(X, y, raise_errors=True)

    assert isinstance(automl.rankings, pd.DataFrame)
    assert isinstance(automl.best_pipeline, PipelineBase)
    assert isinstance(automl.best_pipeline.feature_importances, pd.DataFrame)
    # test with datafarmes
    automl.search(pd.DataFrame(X), pd.Series(y))

    assert isinstance(automl.rankings, pd.DataFrame)
    assert isinstance(automl.best_pipeline, PipelineBase)
    assert isinstance(automl.get_pipeline(0), PipelineBase)

    automl.describe_pipeline(0)


def test_cv(X_y):
    X, y = X_y
    cv_folds = 5
    automl = AutoClassificationSearch(cv=StratifiedKFold(cv_folds), max_pipelines=1)
    automl.search(X, y, raise_errors=True)

    assert isinstance(automl.rankings, pd.DataFrame)
    assert len(automl.results['pipeline_results'][0]["cv_data"]) == cv_folds

    automl = AutoClassificationSearch(cv=TimeSeriesSplit(cv_folds), max_pipelines=1)
    automl.search(X, y, raise_errors=True)

    assert isinstance(automl.rankings, pd.DataFrame)
    assert len(automl.results['pipeline_results'][0]["cv_data"]) == cv_folds


def test_init_select_model_families():
    model_families = [ModelFamily.RANDOM_FOREST]
    automl = AutoClassificationSearch(allowed_model_families=model_families)

    assert get_pipelines(problem_type=ProblemTypes.BINARY, model_families=model_families) == automl.possible_pipelines
    assert model_families == automl.possible_model_families


def test_max_pipelines(X_y):
    X, y = X_y
    max_pipelines = 5
    automl = AutoClassificationSearch(max_pipelines=max_pipelines)
    automl.search(X, y, raise_errors=True)

    assert len(automl.rankings) == max_pipelines


def test_best_pipeline(X_y):
    X, y = X_y
    max_pipelines = 5
    automl = AutoClassificationSearch(max_pipelines=max_pipelines)
    automl.search(X, y, raise_errors=True)

    assert len(automl.rankings) == max_pipelines


def test_specify_objective(X_y):
    X, y = X_y
    automl = AutoClassificationSearch(objective=Precision(), max_pipelines=1)
    automl.search(X, y, raise_errors=True)
    assert isinstance(automl.objective, Precision)
    assert automl.best_pipeline.threshold is not None


def test_binary_auto(X_y):
    X, y = X_y
    automl = AutoClassificationSearch(objective="log_loss_binary", multiclass=False, max_pipelines=5)
    automl.search(X, y, raise_errors=True)
    y_pred = automl.best_pipeline.predict(X)

    assert len(np.unique(y_pred)) == 2


def test_multi_error(X_y_multi):
    X, y = X_y_multi
    error_automls = [AutoClassificationSearch(objective='recall'), AutoClassificationSearch(objective='recall_micro', additional_objectives=['recall'], multiclass=True)]
    error_msg = 'not compatible with a multiclass problem.'
    for automl in error_automls:
        with pytest.raises(ValueError, match=error_msg):
            automl.search(X, y)


def test_multi_auto(X_y_multi):
    X, y = X_y_multi
    automl = AutoClassificationSearch(objective="recall_micro", multiclass=True, max_pipelines=5)
    automl.search(X, y, raise_errors=True)
    y_pred = automl.best_pipeline.predict(X)
    assert len(np.unique(y_pred)) == 3

    objective = PrecisionMicro()
    automl = AutoClassificationSearch(objective=objective, multiclass=True, max_pipelines=5)
    automl.search(X, y, raise_errors=True)
    y_pred = automl.best_pipeline.predict(X)
    assert len(np.unique(y_pred)) == 3

    expected_additional_objectives = get_objectives('multiclass')
    objective_in_additional_objectives = next((obj for obj in expected_additional_objectives if obj.name == objective.name), None)
    expected_additional_objectives.remove(objective_in_additional_objectives)

    for expected, additional in zip(expected_additional_objectives, automl.additional_objectives):
        assert type(additional) is type(expected)


def test_multi_objective(X_y_multi):
    automl = AutoClassificationSearch(objective="log_loss_binary")
    assert automl.problem_type == ProblemTypes.BINARY

    automl = AutoClassificationSearch(objective="log_loss_multi")
    assert automl.problem_type == ProblemTypes.MULTICLASS

    automl = AutoClassificationSearch(objective='recall_micro')
    assert automl.problem_type == ProblemTypes.MULTICLASS

    automl = AutoClassificationSearch(objective='recall')
    assert automl.problem_type == ProblemTypes.BINARY

    automl = AutoClassificationSearch(multiclass=True)
    assert automl.problem_type == ProblemTypes.MULTICLASS

    automl = AutoClassificationSearch()
    assert automl.problem_type == ProblemTypes.BINARY


def test_categorical_classification(X_y_categorical_classification):
    X, y = X_y_categorical_classification
    automl = AutoClassificationSearch(objective="recall", max_pipelines=5, multiclass=False)
    automl.search(X, y, raise_errors=True)
    assert not automl.rankings['score'].isnull().all()
    assert not automl.get_pipeline(0).feature_importances.isnull().all().all()


def test_random_state(X_y):
    X, y = X_y

    fc = FraudCost(retry_percentage=.5,
                   interchange_fee=.02,
                   fraud_payout_percentage=.75,
                   amount_col=10)

    automl = AutoClassificationSearch(objective=Precision(), max_pipelines=5, random_state=0)
    automl.search(X, y, raise_errors=True)

    automl_1 = AutoClassificationSearch(objective=Precision(), max_pipelines=5, random_state=0)
    automl_1.search(X, y, raise_errors=True)
    assert automl.rankings.equals(automl_1.rankings)

    # test an objective that requires fitting
    automl = AutoClassificationSearch(objective=fc, max_pipelines=5, random_state=30)
    automl.search(X, y, raise_errors=True)

    automl_1 = AutoClassificationSearch(objective=fc, max_pipelines=5, random_state=30)
    automl_1.search(X, y, raise_errors=True)

    assert automl.rankings.equals(automl_1.rankings)


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
    automl = AutoClassificationSearch(objective=Precision(), max_pipelines=max_pipelines,
                                      start_iteration_callback=start_iteration_callback,
                                      add_result_callback=add_result_callback)
    automl.search(X, y, raise_errors=True)

    assert counts["start_iteration_callback"] == max_pipelines
    assert counts["add_result_callback"] == max_pipelines


def test_additional_objectives(X_y):
    X, y = X_y

    objective = FraudCost(retry_percentage=.5,
                          interchange_fee=.02,
                          fraud_payout_percentage=.75,
                          amount_col=10)
    automl = AutoClassificationSearch(objective='F1', max_pipelines=2, additional_objectives=[objective])
    automl.search(X, y, raise_errors=True)

    results = automl.describe_pipeline(0, return_dict=True)
    assert 'Fraud Cost' in list(results["cv_data"][0]["all_objective_scores"].keys())


def test_non_optimizable_threshold(X_y):
    X, y = X_y
    automl = AutoClassificationSearch(objective='AUC', max_pipelines=1)
    automl.search(X, y, raise_errors=True)
    assert automl.best_pipeline.threshold == 0.5


def test_describe_pipeline_objective_ordered(X_y, capsys):
    X, y = X_y
    automl = AutoClassificationSearch(objective='AUC', max_pipelines=2)
    automl.search(X, y, raise_errors=True)

    automl.describe_pipeline(0)
    out, err = capsys.readouterr()
    out_stripped = " ".join(out.split())

    objectives = [get_objective(obj) for obj in automl.additional_objectives]
    objectives_names = [obj.name for obj in objectives]
    expected_objective_order = " ".join(objectives_names)

    assert err == ''
    assert expected_objective_order in out_stripped


def test_model_families_as_list():
    with pytest.raises(TypeError, match="model_families parameter is not a list."):
        AutoClassificationSearch(objective='AUC', allowed_model_families='linear_model', max_pipelines=2)


def test_max_time_units():
    str_max_time = AutoClassificationSearch(objective='F1', max_time='60 seconds')
    assert str_max_time.max_time == 60

    hour_max_time = AutoClassificationSearch(objective='F1', max_time='1 hour')
    assert hour_max_time.max_time == 3600

    min_max_time = AutoClassificationSearch(objective='F1', max_time='30 mins')
    assert min_max_time.max_time == 1800

    min_max_time = AutoClassificationSearch(objective='F1', max_time='30 s')
    assert min_max_time.max_time == 30

    with pytest.raises(AssertionError, match="Invalid unit. Units must be hours, mins, or seconds. Received 'year'"):
        AutoClassificationSearch(objective='F1', max_time='30 years')

    with pytest.raises(TypeError, match="max_time must be a float, int, or string. Received a <class 'tuple'>."):
        AutoClassificationSearch(objective='F1', max_time=(30, 'minutes'))


def test_early_stopping(capsys):
    with pytest.raises(ValueError, match='patience value must be a positive integer.'):
        automl = AutoClassificationSearch(objective='AUC', max_pipelines=5, allowed_model_families=['linear_model'], patience=-1, random_state=0)

    with pytest.raises(ValueError, match='tolerance value must be'):
        automl = AutoClassificationSearch(objective='AUC', max_pipelines=5, allowed_model_families=['linear_model'], patience=1, tolerance=1.5, random_state=0)

    automl = AutoClassificationSearch(objective='AUC', max_pipelines=5, allowed_model_families=['linear_model'], patience=2, tolerance=0.05, random_state=0)
    mock_results = {
        'search_order': [0, 1, 2],
        'pipeline_results': {}
    }

    scores = [0.95, 0.84, 0.96]  # 0.96 is only 1% greater so it doesn't trigger patience due to tolerance
    for id in mock_results['search_order']:
        mock_results['pipeline_results'][id] = {}
        mock_results['pipeline_results'][id]['score'] = scores[id]

    automl.results = mock_results
    automl._check_stopping_condition(time.time())
    out, _ = capsys.readouterr()
    assert "2 iterations without improvement. Stopping search early." in out


def test_plot_disabled_missing_dependency(X_y, has_minimal_dependencies):
    X, y = X_y

    automl = AutoClassificationSearch(max_pipelines=3)
    if has_minimal_dependencies:
        with pytest.raises(AttributeError):
            automl.plot.search_iteration_plot
    else:
        automl.plot.search_iteration_plot


def test_plot_iterations_max_pipelines(X_y):
    go = pytest.importorskip('plotly.graph_objects', reason='Skipping plotting test because plotly not installed')
    X, y = X_y

    automl = AutoClassificationSearch(objective="f1", max_pipelines=3)
    automl.search(X, y, raise_errors=True)
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

    automl = AutoClassificationSearch(objective="f1", max_time=10)
    automl.search(X, y, show_iteration_plot=False, raise_errors=True)
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
def test_plot_iterations_ipython_mock(mock_ipython_display, X_y):
    pytest.importorskip('IPython.display', reason='Skipping plotting test because ipywidgets not installed')
    pytest.importorskip('plotly.graph_objects', reason='Skipping plotting test because plotly not installed')
    X, y = X_y

    automl = AutoClassificationSearch(objective="f1", max_pipelines=3)
    automl.search(X, y, raise_errors=True)
    plot = automl.plot.search_iteration_plot(interactive_plot=True)
    assert isinstance(plot, SearchIterationPlot)
    assert isinstance(plot.data, AutoClassificationSearch)
    mock_ipython_display.assert_called_with(plot.best_score_by_iter_fig)


@patch('IPython.display.display')
def test_plot_iterations_ipython_mock_import_failure(mock_ipython_display, X_y):
    pytest.importorskip('IPython.display', reason='Skipping plotting test because ipywidgets not installed')
    go = pytest.importorskip('plotly.graph_objects', reason='Skipping plotting test because plotly not installed')
    X, y = X_y

    automl = AutoClassificationSearch(objective="f1", max_pipelines=3)
    automl.search(X, y, raise_errors=True)

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
