import os

import graphviz
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest
from skopt.space import Real

from evalml.model_types import ModelTypes
from evalml.pipelines import PipelineBase


@pytest.fixture
def test_pipeline():
    class TestPipeline(PipelineBase):
        model_type = ModelTypes.LINEAR_MODEL
        component_graph = ['Simple Imputer', 'One Hot Encoder', 'Standard Scaler', 'Logistic Regression Classifier']
        problem_types = ['binary', 'multiclass']

        hyperparameters = {
            "penalty": ["l2"],
            "C": Real(.01, 10),
            "impute_strategy": ["mean", "median", "most_frequent"],
        }

        def __init__(self, objective, parameters):
            super().__init__(objective=objective,
                             parameters=parameters)

        @property
        def feature_importances(self):
            importances = [1.0, 0.2, 0.0002, 0.0, 0.0, -1.0]
            feature_names = range(len(importances))
            f_i = list(zip(feature_names, importances))
            df = pd.DataFrame(f_i, columns=["feature", "importance"])
            return df

    return TestPipeline(objective='precision', parameters={})


def test_returns_digraph_object(test_pipeline):
    clf = test_pipeline
    graph = test_pipeline.graph()
    assert isinstance(graph, graphviz.Digraph)


def test_saving_png_file(tmpdir, test_pipeline):
    filepath = os.path.join(str(tmpdir), 'pipeline.png')
    pipeline = test_pipeline
    pipeline.graph(filepath=filepath)
    assert os.path.isfile(filepath)


def test_missing_file_extension(test_pipeline):
    filepath = "test1"
    pipeline = test_pipeline
    with pytest.raises(ValueError, match="Unknown format"):
        pipeline.graph(filepath=filepath)


def test_invalid_format(test_pipeline):
    filepath = "test1.xzy"
    pipeline = test_pipeline
    with pytest.raises(ValueError, match="Unknown format"):
        pipeline.graph(filepath=filepath)


def test_invalid_path(tmpdir, test_pipeline):
    filepath = os.path.join(str(tmpdir), 'invalid', 'path', 'pipeline.png')
    pipeline = test_pipeline
    with pytest.raises(ValueError, match="Specified parent directory does not exist"):
        pipeline.graph(filepath=filepath)


def test_feature_importance_plot(X_y, test_pipeline):
    X, y = X_y
    clf = test_pipeline
    clf.fit(X, y)
    assert isinstance(clf.feature_importance_graph(), go.Figure)


def test_feature_importance_plot_show_all_features(X_y, test_pipeline):
    X, y = X_y
    clf = test_pipeline
    clf.fit(X, y)
    figure = clf.feature_importance_graph()
    assert isinstance(figure, go.Figure)

    data = figure.data[0]
    assert (np.all(data['x']))

    figure = clf.feature_importance_graph(show_all_features=True)
    data = figure.data[0]
    assert (np.any(data['x'] == 0.0))
