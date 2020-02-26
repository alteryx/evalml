import os

import graphviz
import plotly.graph_objects as go
import pytest

from evalml.pipelines import PipelineBase


def test_returns_digraph_object():
    clf = PipelineBase(objective='precision',
                       component_graph=['Simple Imputer', 'One Hot Encoder', 'Standard Scaler', 'Logistic Regression Classifier'], problem_types=['binary', 'multiclass'],
                       parameters={})
    plot = clf.plot()
    assert isinstance(plot, graphviz.Digraph)


def test_saving_png_file(tmpdir):
    path = os.path.join(str(tmpdir), 'pipeline.png')
    pipeline = PipelineBase(objective='precision',
                            component_graph=['Simple Imputer', 'One Hot Encoder', 'Standard Scaler', 'Logistic Regression Classifier'], problem_types=['binary', 'multiclass'],
                            parameters={})
    pipeline.plot(to_file=path)
    assert os.path.isfile(path)


def test_missing_file_extension():
    path = "test1"
    pipeline = PipelineBase(objective='precision',
                            component_graph=['Simple Imputer', 'One Hot Encoder', 'Standard Scaler', 'Logistic Regression Classifier'], problem_types=['binary', 'multiclass'],
                            parameters={})
    with pytest.raises(ValueError, match="Please use a file extension"):
        pipeline.plot(to_file=path)


def test_invalid_format():
    path = "test1.xzy"
    pipeline = PipelineBase(objective='precision',
                            component_graph=['Simple Imputer', 'One Hot Encoder', 'Standard Scaler', 'Logistic Regression Classifier'], problem_types=['binary', 'multiclass'],
                            parameters={})
    with pytest.raises(ValueError, match="Unknown format"):
        pipeline.plot(to_file=path)


def test_feature_importance_plot(X_y):
    X, y = X_y
    clf = PipelineBase(objective='precision',
                       component_graph=['Simple Imputer', 'One Hot Encoder', 'Standard Scaler', 'Logistic Regression Classifier'], problem_types=['binary', 'multiclass'],
                       parameters={})
    clf.fit(X, y)
    assert isinstance(clf.plot.feature_importances(), go.Figure)
