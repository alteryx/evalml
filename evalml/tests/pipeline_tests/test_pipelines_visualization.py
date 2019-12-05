import os

import graphviz
import pytest

from evalml.pipelines import PipelineBase


def test_returns_digraph_object():
    clf = PipelineBase('precision', component_list=['Simple Imputer', 'categorical_encoder', 'Standard Scaler', 'Logistic Regression Classifier'], n_jobs=-1, random_state=0)
    plot = clf.visualize()
    assert isinstance(plot, graphviz.Digraph)


def test_saving_png_file(tmpdir):
    path = os.path.join(str(tmpdir), 'pipeline.png')
    pipeline = PipelineBase('precision', component_list=['Simple Imputer', 'categorical_encoder', 'Standard Scaler', 'Logistic Regression Classifier'], n_jobs=-1, random_state=0)
    pipeline.visualize(to_file=path)
    assert os.path.isfile(path)


def test_missing_file_extension():
    path = "test1"
    pipeline = PipelineBase('precision', component_list=['Simple Imputer', 'categorical_encoder', 'Standard Scaler', 'Logistic Regression Classifier'], n_jobs=-1, random_state=0)
    with pytest.raises(ValueError, match="Please use a file extension"):
        pipeline.visualize(to_file=path)


def test_invalid_format():
    path = "test1.xzy"
    pipeline = PipelineBase('precision', component_list=['Simple Imputer', 'categorical_encoder', 'Standard Scaler', 'Logistic Regression Classifier'], n_jobs=-1, random_state=0)
    with pytest.raises(ValueError, match="Unknown format"):
        pipeline.visualize(to_file=path)
