import os
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from skopt.space import Real

import graphviz

from evalml.objectives import get_objectives
from evalml.pipelines import (
    BinaryClassificationPipeline,
    LinearRegressionPipeline,
    LogisticRegressionBinaryPipeline,
    LogisticRegressionMulticlassPipeline
)
from evalml.pipelines.graph_utils import (
    calculate_permutation_importance,
    graph_permutation_importance
)
from evalml.problem_types import ProblemTypes


@pytest.fixture
def test_pipeline():
    class TestPipeline(BinaryClassificationPipeline):
        component_graph = ['Simple Imputer', 'One Hot Encoder', 'Standard Scaler', 'Logistic Regression Classifier']

        hyperparameters = {
            "penalty": ["l2"],
            "C": Real(.01, 10),
            "impute_strategy": ["mean", "median", "most_frequent"],
        }

        def __init__(self, parameters):
            super().__init__(parameters=parameters)

        @property
        def feature_importance(self):
            importance = [1.0, 0.2, 0.0002, 0.0, 0.0, -1.0]
            feature_names = range(len(importance))
            f_i = list(zip(feature_names, importance))
            df = pd.DataFrame(f_i, columns=["feature", "importance"])
            return df

    return TestPipeline(parameters={})


@patch('graphviz.Digraph.pipe')
def test_backend(mock_func, test_pipeline):
    mock_func.side_effect = graphviz.backend.ExecutableNotFound('Not Found')
    clf = test_pipeline
    with pytest.raises(RuntimeError):
        clf.graph()


def test_returns_digraph_object(test_pipeline):
    clf = test_pipeline
    graph = clf.graph()
    assert isinstance(graph, graphviz.Digraph)


def test_saving_png_file(tmpdir, test_pipeline):
    filepath = os.path.join(str(tmpdir), 'pipeline.png')
    pipeline = test_pipeline
    pipeline.graph(filepath=filepath)
    assert os.path.isfile(filepath)


def test_missing_file_extension(tmpdir, test_pipeline):
    filepath = os.path.join(str(tmpdir), 'test1')
    pipeline = test_pipeline
    with pytest.raises(ValueError, match="Unknown format"):
        pipeline.graph(filepath=filepath)


def test_invalid_format(tmpdir, test_pipeline):
    filepath = os.path.join(str(tmpdir), 'test1.xyz')
    pipeline = test_pipeline
    with pytest.raises(ValueError, match="Unknown format"):
        pipeline.graph(filepath=filepath)


def test_invalid_path(tmpdir, test_pipeline):
    filepath = os.path.join(str(tmpdir), 'invalid', 'path', 'pipeline.png')
    assert not os.path.exists(filepath)
    pipeline = test_pipeline
    with pytest.raises(ValueError, match="Specified filepath is not writeable"):
        pipeline.graph(filepath=filepath)
    assert not os.path.exists(filepath)


def test_graph_feature_importance(X_y_binary, test_pipeline):
    go = pytest.importorskip('plotly.graph_objects', reason='Skipping plotting test because plotly not installed')
    X, y = X_y_binary
    clf = test_pipeline
    clf.fit(X, y)
    assert isinstance(clf.graph_feature_importance(), go.Figure)


def test_graph_feature_importance_show_all_features(X_y_binary, test_pipeline):
    go = pytest.importorskip('plotly.graph_objects', reason='Skipping plotting test because plotly not installed')
    X, y = X_y_binary
    clf = test_pipeline
    clf.fit(X, y)
    figure = clf.graph_feature_importance()
    assert isinstance(figure, go.Figure)

    data = figure.data[0]
    assert (np.all(data['x']))

    figure = clf.graph_feature_importance(show_all_features=True)
    data = figure.data[0]
    assert (np.any(data['x'] == 0.0))


def test_get_permutation_importance_invalid_objective(X_y_regression):
    X, y = X_y_regression
    pipeline = LinearRegressionPipeline(parameters={}, random_state=np.random.RandomState(42))
    with pytest.raises(ValueError, match=f"Given objective 'MCC Multiclass' cannot be used with '{pipeline.name}'"):
        calculate_permutation_importance(pipeline, X, y, "mcc_multi")


@pytest.mark.parametrize("data_type", ['np', 'pd'])
def test_get_permutation_importance_binary(X_y_binary, data_type):
    X, y = X_y_binary
    if data_type == 'pd':
        X = pd.DataFrame(X)
        y = pd.Series(y)
    pipeline = LogisticRegressionBinaryPipeline(parameters={}, random_state=np.random.RandomState(42))
    pipeline.fit(X, y)
    for objective in get_objectives(ProblemTypes.BINARY):
        permutation_importance = calculate_permutation_importance(pipeline, X, y, objective)
        assert list(permutation_importance.columns) == ["feature", "importance"]
        assert not permutation_importance.isnull().all().all()


def test_get_permutation_importance_multiclass(X_y_multi):
    X, y = X_y_multi
    pipeline = LogisticRegressionMulticlassPipeline(parameters={}, random_state=np.random.RandomState(42))
    pipeline.fit(X, y)
    for objective in get_objectives(ProblemTypes.MULTICLASS):
        permutation_importance = calculate_permutation_importance(pipeline, X, y, objective)
        assert list(permutation_importance.columns) == ["feature", "importance"]
        assert not permutation_importance.isnull().all().all()


def test_get_permutation_importance_regression(X_y_regression):
    X, y = X_y_regression
    pipeline = LinearRegressionPipeline(parameters={}, random_state=np.random.RandomState(42))
    pipeline.fit(X, y)
    for objective in get_objectives(ProblemTypes.REGRESSION):
        permutation_importance = calculate_permutation_importance(pipeline, X, y, objective)
        assert list(permutation_importance.columns) == ["feature", "importance"]
        assert not permutation_importance.isnull().all().all()


def test_get_permutation_importance_correlated_features():
    y = pd.Series([1, 0, 1, 1])
    X = pd.DataFrame()
    X["correlated"] = y * 2
    X["not correlated"] = [-1, -1, -1, 0]
    y = y.astype(bool)
    pipeline = LogisticRegressionBinaryPipeline(parameters={}, random_state=np.random.RandomState(42))
    pipeline.fit(X, y)
    importance = calculate_permutation_importance(pipeline, X, y, objective="log_loss_binary", random_state=0)
    assert list(importance.columns) == ["feature", "importance"]
    assert not importance.isnull().all().all()
    correlated_importance_val = importance["importance"][importance.index[importance["feature"] == "correlated"][0]]
    not_correlated_importance_val = importance["importance"][importance.index[importance["feature"] == "not correlated"][0]]
    assert correlated_importance_val > not_correlated_importance_val


def test_graph_permutation_importance(X_y_binary, test_pipeline):
    go = pytest.importorskip('plotly.graph_objects', reason='Skipping plotting test because plotly not installed')
    X, y = X_y_binary
    clf = test_pipeline
    clf.fit(X, y)
    fig = graph_permutation_importance(test_pipeline, X, y, "log_loss_binary", show_all_features=True)
    assert isinstance(fig, go.Figure)
    fig_dict = fig.to_dict()
    assert fig_dict['layout']['title']['text'] == "Permutation Importance<br><sub>"\
                                                  "The relative importance of each input feature's overall "\
                                                  "influence on the pipelines' predictions, computed using the "\
                                                  "permutation importance algorithm.</sub>"
    assert len(fig_dict['data']) == 1

    perm_importance_data = calculate_permutation_importance(clf, X, y, "log_loss_binary")
    assert np.array_equal(fig_dict['data'][0]['x'][::-1], perm_importance_data['importance'].values)
    assert np.array_equal(fig_dict['data'][0]['y'][::-1], perm_importance_data['feature'])


@patch('evalml.pipelines.graph_utils.calculate_permutation_importance')
def test_graph_permutation_importance_show_all_features(mock_perm_importance):
    go = pytest.importorskip('plotly.graph_objects', reason='Skipping plotting test because plotly not installed')
    mock_perm_importance.return_value = pd.DataFrame({"feature": ["f1", "f2"], "importance": [0.0, 0.6]})
    figure = graph_permutation_importance(test_pipeline, pd.DataFrame(), pd.Series(), "log_loss_binary")
    assert isinstance(figure, go.Figure)

    data = figure.data[0]
    assert (np.all(data['x']))

    figure = graph_permutation_importance(test_pipeline, pd.DataFrame(), pd.Series(), "log_loss_binary", show_all_features=True)
    data = figure.data[0]
    assert (np.any(data['x'] == 0.0))
