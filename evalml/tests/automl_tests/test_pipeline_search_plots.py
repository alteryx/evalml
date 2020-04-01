import pandas as pd
import pytest
from sklearn.model_selection import StratifiedKFold

from evalml.automl.auto_base import AutoBase
from evalml.automl.pipeline_search_plots import (
    PipelineSearchPlots,
    SearchIterationPlot
)
from evalml.objectives import ROC, ConfusionMatrix
from evalml.pipelines import LogisticRegressionBinaryPipeline
from evalml.problem_types import ProblemTypes


def test_generate_roc(X_y):
    go = pytest.importorskip('plotly.graph_objects', reason='Skipping plotting test because plotly not installed')
    X, y = X_y

    # Make mock class and generate mock results

    class MockAuto(AutoBase):
        def __init__(self):
            self.results = {}
            self.results['pipeline_results'] = {}
            self.problem_type = ProblemTypes.BINARY

        def search(self):
            parameters = {
                'Simple Imputer': {
                    'impute_strategy': 'mean'
                },
                'Logistic Regression Classifier': {
                    'penalty': 'l2',
                    'C': 0.5,
                }
            }
            pipeline = LogisticRegressionBinaryPipeline(parameters=parameters, random_state=0)
            cv = StratifiedKFold(n_splits=5, random_state=0)
            plot_data = []
            for train, test in cv.split(X, y):
                if isinstance(X, pd.DataFrame):
                    X_train, X_test = X.iloc[train], X.iloc[test]
                else:
                    X_train, X_test = X[train], X[test]
                if isinstance(y, pd.Series):
                    y_train, y_test = y.iloc[train], y.iloc[test]
                else:
                    y_train, y_test = y[train], y[test]

                pipeline.fit(X_train, y_train)
                plot_data.append(pipeline.get_plot_data(X_test, y_test, [ROC()]))

            self.results['pipeline_results'].update({0: {"plot_data": plot_data,
                                                         "pipeline_name": pipeline.name}})

            self.results['pipeline_results'].update({0: {"plot_data": plot_data, "pipeline_name": pipeline.name}})

    mock_automl = MockAuto()
    search_plots = PipelineSearchPlots(mock_automl)
    with pytest.raises(RuntimeError, match="You must first call search"):
        search_plots.get_roc_data(0)
    with pytest.raises(RuntimeError, match="You must first call search"):
        search_plots.generate_roc_plot(0)

    mock_automl.search()

    with pytest.raises(RuntimeError, match="Pipeline 1 not found"):
        search_plots.get_roc_data(1)
    with pytest.raises(RuntimeError, match="Pipeline 1 not found"):
        search_plots.generate_roc_plot(1)

    roc_data = search_plots.get_roc_data(0)
    assert len(roc_data["fpr_tpr_data"]) == 5
    assert len(roc_data["roc_aucs"]) == 5

    fig = search_plots.generate_roc_plot(0)
    assert isinstance(fig, type(go.Figure()))


def test_generate_roc_multi_raises_errors(X_y):
    pytest.importorskip('plotly.graph_objects', reason='Skipping plotting test because plotly not installed')

    class MockAutoMulti(AutoBase):
        def __init__(self):
            self.results = {}
            self.results['pipeline_results'] = {}
            self.problem_type = ProblemTypes.MULTICLASS

    mock_automl = MockAutoMulti()
    search_plots = PipelineSearchPlots(mock_automl)

    with pytest.raises(RuntimeError, match="ROC plots can only be generated for binary classification problems."):
        search_plots.get_roc_data(0)
    with pytest.raises(RuntimeError, match="ROC plots can only be generated for binary classification problems."):
        search_plots.generate_roc_plot(0)


def test_generate_confusion_matrix(X_y):
    go = pytest.importorskip('plotly.graph_objects', reason='Skipping plotting test because plotly not installed')
    X, y = X_y
    y_test_lens = []

    # Make mock class and generate mock results
    class MockAutoClassificationSearch(AutoBase):
        def __init__(self):
            self.results = {}
            self.results['pipeline_results'] = {}
            self.problem_type = ProblemTypes.BINARY

        def search(self):
            parameters = {
                'Simple Imputer': {
                    'impute_strategy': 'mean'
                },
                'Logistic Regression Classifier': {
                    'penalty': 'l2',
                    'C': 0.5,
                }
            }
            pipeline = LogisticRegressionBinaryPipeline(parameters=parameters, random_state=0)
            cv = StratifiedKFold(n_splits=5, random_state=0)
            plot_data = []
            for train, test in cv.split(X, y):
                if isinstance(X, pd.DataFrame):
                    X_train, X_test = X.iloc[train], X.iloc[test]
                else:
                    X_train, X_test = X[train], X[test]
                if isinstance(y, pd.Series):
                    y_train, y_test = y.iloc[train], y.iloc[test]
                else:
                    y_train, y_test = y[train], y[test]

                pipeline.fit(X_train, y_train)
                plot_data.append(pipeline.get_plot_data(X_test, y_test, [ConfusionMatrix()]))
                # store information for testing purposes later
                y_test_lens.append(len(y_test))
            self.results['pipeline_results'].update({0: {"plot_data": plot_data,
                                                         "pipeline_name": pipeline.name}})

    mock_automl = MockAutoClassificationSearch()
    search_plots = PipelineSearchPlots(mock_automl)
    with pytest.raises(RuntimeError, match="You must first call search"):
        search_plots.get_confusion_matrix_data(0)
    with pytest.raises(RuntimeError, match="You must first call search"):
        search_plots.generate_confusion_matrix(0)

    mock_automl.search()

    with pytest.raises(RuntimeError, match="Pipeline 1 not found"):
        search_plots.get_confusion_matrix_data(1)
    with pytest.raises(RuntimeError, match="Pipeline 1 not found"):
        search_plots.generate_confusion_matrix(1)

    cm_data = search_plots.get_confusion_matrix_data(0)

    for i, cm in enumerate(cm_data):
        labels = cm.columns
        assert all(label in y for label in labels)
        assert (cm.to_numpy().sum() == y_test_lens[i])

    cm_data_normalized = search_plots.get_confusion_matrix_data(0, normalize='true')
    for i, cm_normalized in enumerate(cm_data_normalized):
        labels = cm_normalized.columns
        assert all(label in y for label in labels)
        assert all(cm_normalized.sum(axis=1) == 1.0)

    fig = search_plots.generate_confusion_matrix(0)
    assert isinstance(fig, type(go.Figure()))

    fig = search_plots.generate_confusion_matrix(0, normalize=True)
    assert isinstance(fig, type(go.Figure()))


def test_confusion_matrix_regression_throws_error():
    pytest.importorskip('plotly.graph_objects', reason='Skipping plotting test because plotly not installed')

    # Make mock class and generate mock results
    class MockAutoRegressionSearch(AutoBase):
        def __init__(self):
            self.results = {}
            self.results['pipeline_results'] = {}
            self.problem_type = ProblemTypes.REGRESSION

    mock_automl = MockAutoRegressionSearch()
    search_plots = PipelineSearchPlots(mock_automl)

    with pytest.raises(RuntimeError, match="Confusion matrix plots can only be generated for classification problems"):
        search_plots.get_confusion_matrix_data(0)
    with pytest.raises(RuntimeError, match="Confusion matrix plots can only be generated for classification problems."):
        search_plots.generate_confusion_matrix(0)


def test_search_iteration_plot_class(X_y):
    pytest.importorskip('plotly.graph_objects', reason='Skipping plotting test because plotly not installed')

    class MockObjective:
        def __init__(self):
            self.name = 'Test Objective'
            self.greater_is_better = True

    class MockResults:
        def __init__(self):
            self.objective = MockObjective()
            self.results = {
                'pipeline_results': {
                    2: {
                        'score': 0.50
                    },
                    0: {
                        'score': 0.60
                    },
                    1: {
                        'score': 0.75
                    },
                },
                'search_order': [1, 2, 0]
            }
            self.rankings = pd.DataFrame({
                'score': [0.75, 0.60, 0.50]
            })

    mock_data = MockResults()
    plot = SearchIterationPlot(mock_data)

    # Check best score trace
    plot_data = plot.best_score_by_iter_fig.data[0]
    x = list(plot_data['x'])
    y = list(plot_data['y'])

    assert isinstance(plot, SearchIterationPlot)
    assert x == [0, 1, 2]
    assert y == [0.60, 0.75, 0.75]

    # Check current score trace
    plot_data = plot.best_score_by_iter_fig.data[1]
    x = list(plot_data['x'])
    y = list(plot_data['y'])

    assert isinstance(plot, SearchIterationPlot)
    assert x == [0, 1, 2]
    assert y == [0.60, 0.75, 0.50]
