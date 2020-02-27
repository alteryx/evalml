import os

import graphviz
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest

from evalml.pipelines import PipelineBase


def test_returns_digraph_object():
    clf = PipelineBase('precision', component_list=['Simple Imputer', 'One Hot Encoder', 'Standard Scaler', 'Logistic Regression Classifier'], n_jobs=-1, random_state=0)
    graph = clf.graph()
    assert isinstance(graph, graphviz.Digraph)


def test_saving_png_file(tmpdir):
    filepath = os.path.join(str(tmpdir), 'pipeline.png')
    pipeline = PipelineBase('precision', component_list=['Simple Imputer', 'One Hot Encoder', 'Standard Scaler', 'Logistic Regression Classifier'], n_jobs=-1, random_state=0)
    pipeline.graph(filepath=filepath)
    assert os.path.isfile(filepath)


def test_missing_file_extension():
    filepath = "test1"
    pipeline = PipelineBase('precision', component_list=['Simple Imputer', 'One Hot Encoder', 'Standard Scaler', 'Logistic Regression Classifier'], n_jobs=-1, random_state=0)
    with pytest.raises(ValueError, match="Please use a file extension"):
        pipeline.graph(filepath=filepath)


def test_invalid_format():
    filepath = "test1.xzy"
    pipeline = PipelineBase('precision', component_list=['Simple Imputer', 'One Hot Encoder', 'Standard Scaler', 'Logistic Regression Classifier'], n_jobs=-1, random_state=0)
    with pytest.raises(ValueError, match="Unknown format"):
        pipeline.graph(filepath=filepath)


def test_feature_importance_plot(X_y):
    X, y = X_y
    clf = PipelineBase('precision', component_list=['Simple Imputer', 'One Hot Encoder', 'Standard Scaler', 'Logistic Regression Classifier'], n_jobs=-1, random_state=0)
    clf.fit(X, y)
    assert isinstance(clf.feature_importance_graph(), go.Figure)


def test_feature_importance_plot_show_all_features(X_y):

    class MockPipeline(PipelineBase):
        name = "Mock Pipeline"

        def __init__(self):
            objective = "Precision"
            component_list = ['Logistic Regression Classifier']
            n_jobs = 1
            random_state = 0
            super().__init__(objective=objective, component_list=component_list, n_jobs=n_jobs, random_state=random_state)

        @property
        def feature_importances(self):
            importances = [1.0, 0.2, 0.0002, 0.0, 0.0, -1.0]
            feature_names = range(len(importances))
            f_i = list(zip(feature_names, importances))
            df = pd.DataFrame(f_i, columns=["feature", "importance"])
            return df

    X, y = X_y
    clf = MockPipeline()
    clf.fit(X, y)
    figure = clf.feature_importance_graph()
    assert isinstance(figure, go.Figure)

    data = figure.data[0]
    assert (np.all(data['x']))

    figure = clf.feature_importance_graph(show_all_features=True)
    data = figure.data[0]
    assert (np.any(data['x'] == 0.0))
