import os
from unittest.mock import patch

import graphviz
import numpy as np
import pandas as pd
import pytest
from skopt.space import Real

from evalml.pipelines import BinaryClassificationPipeline
from evalml.pipelines.utils import (
    calculate_permutation_importances,
    graph_permutation_importances
)


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
        def feature_importances(self):
            importances = [1.0, 0.2, 0.0002, 0.0, 0.0, -1.0]
            feature_names = range(len(importances))
            f_i = list(zip(feature_names, importances))
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


def test_graph_feature_importances(X_y, test_pipeline):
    go = pytest.importorskip('plotly.graph_objects', reason='Skipping plotting test because plotly not installed')
    X, y = X_y
    clf = test_pipeline
    clf.fit(X, y)
    assert isinstance(clf.graph_feature_importance(), go.Figure)


def test_graph_feature_importances_show_all_features(X_y, test_pipeline):
    go = pytest.importorskip('plotly.graph_objects', reason='Skipping plotting test because plotly not installed')
    X, y = X_y
    clf = test_pipeline
    clf.fit(X, y)
    figure = clf.graph_feature_importance()
    assert isinstance(figure, go.Figure)

    data = figure.data[0]
    assert (np.all(data['x']))

    figure = clf.graph_feature_importance(show_all_features=True)
    data = figure.data[0]
    assert (np.any(data['x'] == 0.0))


def test_graph_permutation_importances(X_y, test_pipeline):
    go = pytest.importorskip('plotly.graph_objects', reason='Skipping plotting test because plotly not installed')
    X, y = X_y
    clf = test_pipeline
    clf.fit(X, y)
    fig = graph_permutation_importances(test_pipeline, X, y, "log_loss_binary")
    assert isinstance(fig, go.Figure)
    fig_dict = fig.to_dict()
    assert fig_dict['layout']['title']['text'] == "Permutation Importance<br><sub>"\
                                                  "The relative importance of each input feature's overall "\
                                                  "influence on the pipelines' predictions, computed using the "\
                                                  "permutation importance algorithm.</sub>"
    assert len(fig_dict['data']) == 1

    perm_importance_data = calculate_permutation_importances(clf, X, y, "log_loss_binary")
    assert np.array_equal(fig_dict['data'][0]['x'][::-1], perm_importance_data['importance'].values)
    assert np.array_equal(fig_dict['data'][0]['y'][::-1], perm_importance_data['feature'])


@patch('evalml.pipelines.utils.calculate_permutation_importances')
def test_graph_permutation_importances_show_all_features(mock_perm_importances):
    go = pytest.importorskip('plotly.graph_objects', reason='Skipping plotting test because plotly not installed')
    mock_perm_importances.return_value = pd.DataFrame({"feature": ["f1", "f2"], "importance": [0.0, 0.6]})
    figure = graph_permutation_importances(test_pipeline, pd.DataFrame(), pd.Series(), "log_loss_binary")
    assert isinstance(figure, go.Figure)

    data = figure.data[0]
    assert (np.all(data['x']))

    figure = graph_permutation_importances(test_pipeline, pd.DataFrame(), pd.Series(), "log_loss_binary", show_all_features=True)
    data = figure.data[0]
    assert (np.any(data['x'] == 0.0))
