import json
import os
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from evalml.pipelines import BinaryClassificationPipeline, ComponentGraph


@pytest.fixture
def test_pipeline():
    class TestPipeline(BinaryClassificationPipeline):
        component_graph = [
            "Simple Imputer",
            "One Hot Encoder",
            "Standard Scaler",
            "Logistic Regression Classifier",
        ]

        def __init__(self, parameters, random_seed=0):
            super().__init__(
                component_graph=self.component_graph, parameters=parameters
            )

        @property
        def feature_importance(self):
            importance = [1.0, 0.2, 0.0002, 0.0, 0.0, -1.0]
            feature_names = range(len(importance))
            f_i = list(zip(feature_names, importance))
            df = pd.DataFrame(f_i, columns=["feature", "importance"])
            return df

    return TestPipeline(parameters={"Logistic Regression Classifier": {"n_jobs": 1}})


@pytest.fixture
def test_component_graph(example_graph):
    component_graph = ComponentGraph(example_graph)
    return component_graph


def test_backend(test_pipeline):
    graphviz = pytest.importorskip(
        "graphviz", reason="Skipping plotting test because graphviz not installed"
    )
    with patch("graphviz.Digraph.pipe") as mock_func:
        mock_func.side_effect = graphviz.backend.ExecutableNotFound("Not Found")
        clf = test_pipeline
        with pytest.raises(RuntimeError):
            clf.graph()


def test_returns_digraph_object(test_pipeline):
    graphviz = pytest.importorskip(
        "graphviz", reason="Skipping plotting test because graphviz not installed"
    )
    clf = test_pipeline
    graph = clf.graph()
    assert isinstance(graph, graphviz.Digraph)


def test_backend_comp_graph(test_component_graph):
    graphviz = pytest.importorskip(
        "graphviz", reason="Skipping plotting test because graphviz not installed"
    )
    with patch("graphviz.Digraph.pipe") as mock_func:
        mock_func.side_effect = graphviz.backend.ExecutableNotFound("Not Found")
        comp = test_component_graph
        with pytest.raises(RuntimeError):
            comp.graph()


def test_saving_png_file(tmpdir, test_pipeline):
    pytest.importorskip(
        "graphviz", reason="Skipping plotting test because graphviz not installed"
    )
    filepath = os.path.join(str(tmpdir), "pipeline.png")
    pipeline = test_pipeline
    pipeline.graph(filepath=filepath)
    assert os.path.isfile(filepath)


def test_returns_digraph_object_comp_graph(test_component_graph):
    graphviz = pytest.importorskip(
        "graphviz", reason="Skipping plotting test because graphviz not installed"
    )
    comp = test_component_graph
    graph = comp.graph("test", "png")
    assert isinstance(graph, graphviz.Digraph)


def test_returns_digraph_object_comp_graph_with_params(test_component_graph):
    graphviz = pytest.importorskip(
        "graphviz", reason="Skipping plotting test because graphviz not installed"
    )
    comp = test_component_graph
    parameters = {
        "OneHot_RandomForest": {"top_n": 3},
        "OneHot_ElasticNet": {"top_n": 5},
        "Elastic Net": {"max_iter": 100},
    }
    comp.instantiate(parameters)
    graph = comp.graph("test", "png")
    assert isinstance(graph, graphviz.Digraph)
    assert "top_n : 3" in graph.source
    assert "top_n : 5" in graph.source
    assert "max_iter : 100" in graph.source


def test_missing_file_extension(tmpdir, test_pipeline):
    pytest.importorskip(
        "graphviz", reason="Skipping plotting test because graphviz not installed"
    )
    filepath = os.path.join(str(tmpdir), "test1")
    pipeline = test_pipeline
    with pytest.raises(ValueError, match="Unknown format"):
        pipeline.graph(filepath=filepath)


def test_invalid_format(tmpdir, test_pipeline):
    pytest.importorskip(
        "graphviz", reason="Skipping plotting test because graphviz not installed"
    )
    filepath = os.path.join(str(tmpdir), "test1.xyz")
    pipeline = test_pipeline
    with pytest.raises(ValueError, match="Unknown format"):
        pipeline.graph(filepath=filepath)


def test_invalid_path(tmpdir, test_pipeline):
    pytest.importorskip(
        "graphviz", reason="Skipping plotting test because graphviz not installed"
    )
    filepath = os.path.join(str(tmpdir), "invalid", "path", "pipeline.png")
    assert not os.path.exists(filepath)
    pipeline = test_pipeline
    with pytest.raises(ValueError, match="Specified filepath is not writeable"):
        pipeline.graph(filepath=filepath)
    assert not os.path.exists(filepath)


def test_graph_feature_importance(X_y_binary, test_pipeline):
    go = pytest.importorskip(
        "plotly.graph_objects",
        reason="Skipping plotting test because plotly not installed",
    )
    X, y = X_y_binary
    clf = test_pipeline
    clf.fit(X, y)
    assert isinstance(clf.graph_feature_importance(), go.Figure)


def test_graph_feature_importance_show_all_features(X_y_binary, test_pipeline):
    go = pytest.importorskip(
        "plotly.graph_objects",
        reason="Skipping plotting test because plotly not installed",
    )
    X, y = X_y_binary
    clf = test_pipeline
    clf.fit(X, y)

    figure = clf.graph_feature_importance()
    assert isinstance(figure, go.Figure)

    data = figure.data[0]
    assert np.any(data["x"] == 0.0)


def test_graph_feature_importance_threshold(X_y_binary, test_pipeline):
    go = pytest.importorskip(
        "plotly.graph_objects",
        reason="Skipping plotting test because plotly not installed",
    )
    X, y = X_y_binary
    clf = test_pipeline
    clf.fit(X, y)

    with pytest.raises(
        ValueError,
        match="Provided importance threshold of -0.0001 must be greater than or equal to 0",
    ):
        figure = clf.graph_feature_importance(importance_threshold=-0.0001)
    figure = clf.graph_feature_importance(importance_threshold=0.5)
    assert isinstance(figure, go.Figure)

    data = figure.data[0]
    assert np.all(data["x"] >= 0.5)


@patch("evalml.pipelines.pipeline_base.jupyter_check")
@patch("evalml.pipelines.pipeline_base.import_or_raise")
def test_jupyter_graph_check(import_check, jupyter_check, X_y_binary, test_pipeline):
    pytest.importorskip(
        "plotly.graph_objects",
        reason="Skipping plotting test because plotly not installed",
    )
    X, y = X_y_binary
    clf = test_pipeline
    clf.fit(X, y)
    jupyter_check.return_value = False
    with pytest.warns(None) as graph_valid:
        clf.graph_feature_importance()
        assert len(graph_valid) == 0

    jupyter_check.return_value = True
    with pytest.warns(None) as graph_valid:
        clf.graph_feature_importance()
        import_check.assert_called_with("ipywidgets", warning=True)


@pytest.mark.parametrize("graph_type", ["graph", "list"])
def test_component_as_json(
    graph_type,
    linear_regression_pipeline_class,
    nonlinear_binary_with_target_pipeline_class,
):
    pipeline_ = linear_regression_pipeline_class({})
    if graph_type == "graph":
        pipeline_ = nonlinear_binary_with_target_pipeline_class({})

    pipeline_parameters = pipeline_.parameters
    expected_nodes = pipeline_.component_graph.component_dict
    dag_str = pipeline_.graph_json()

    assert isinstance(dag_str, str)
    dag_json = json.loads(dag_str)
    assert isinstance(dag_json, dict)
    assert dag_json["x_edges"][0][0] == "X"
    assert len(expected_nodes.keys()) == len(dag_json["Nodes"].keys()) - 2
    assert dag_json["Nodes"].keys() - expected_nodes.keys() == {"X", "y"}
    x_edges_set = set()
    y_edges_set = set()
    for node_, graph_ in expected_nodes.items():
        assert node_ in dag_json["Nodes"]
        comp_name = graph_[0].name if graph_type == "list" else graph_[0]
        assert comp_name == dag_json["Nodes"][node_]["Name"]
        for comp_ in graph_[1:]:
            if comp_ == "X":
                x_edges_set.add(("X", node_))
            elif comp_.endswith(".x"):
                x_edges_set.add((comp_[:-2], node_))
            elif comp_ == "y":
                y_edges_set.add(("y", node_))
            else:
                y_edges_set.add((comp_[:-2], node_))
    for node_, params_ in pipeline_parameters.items():
        for key_, val_ in params_.items():
            assert (
                dag_json["Nodes"][node_]["Attributes"][key_]
                == pipeline_parameters[node_][key_]
            )
    assert x_edges_set == set(tuple(edge_) for edge_ in dag_json["x_edges"])
    assert y_edges_set == set(tuple(edge_) for edge_ in dag_json["y_edges"])
