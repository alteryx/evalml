import os
from unittest.mock import patch

import graphviz
import numpy as np
import pandas as pd
import pytest
from skopt.space import Real

from evalml.pipelines import BinaryClassificationPipeline


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
    figure = clf.graph_feature_importance(feature_threshold=0.001)
    assert isinstance(figure, go.Figure)

    data = figure.data[0]
    assert (np.all(data['x']))

    figure = clf.graph_feature_importance(feature_threshold=0)
    data = figure.data[0]
    assert (np.any(data['x'] == 0.0))


def test_graph_feature_importance_feature_threshold(X_y_binary, test_pipeline):
    go = pytest.importorskip('plotly.graph_objects', reason='Skipping plotting test because plotly not installed')
    X, y = X_y_binary
    clf = test_pipeline
    clf.fit(X, y)
    with pytest.raises(ValueError):
        figure = clf.graph_feature_importance(feature_threshold=-0.0001)
    figure = clf.graph_feature_importance(feature_threshold=0.5)
    assert isinstance(figure, go.Figure)

    data = figure.data[0]
    assert (np.all(data['x'] >= 0.5))
