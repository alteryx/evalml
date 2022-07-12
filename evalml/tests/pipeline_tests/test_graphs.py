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
                component_graph=self.component_graph,
                parameters=parameters,
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


def test_backend(test_pipeline, graphviz):

    with patch("graphviz.Digraph.pipe") as mock_func:
        mock_func.side_effect = graphviz.backend.ExecutableNotFound("Not Found")
        clf = test_pipeline
        with pytest.raises(RuntimeError):
            clf.graph()


def test_returns_digraph_object(test_pipeline, graphviz):

    clf = test_pipeline
    graph = clf.graph()
    assert isinstance(graph, graphviz.Digraph)


def test_backend_comp_graph(test_component_graph, graphviz):

    with patch("graphviz.Digraph.pipe") as mock_func:
        mock_func.side_effect = graphviz.backend.ExecutableNotFound("Not Found")
        comp = test_component_graph
        with pytest.raises(RuntimeError):
            comp.graph()


def test_saving_png_file(tmpdir, test_pipeline):
    filepath = os.path.join(str(tmpdir), "pipeline.png")
    pipeline = test_pipeline
    pipeline.graph(filepath=filepath)
    assert os.path.isfile(filepath)


def test_returns_digraph_object_comp_graph(test_component_graph, graphviz):

    comp = test_component_graph
    graph = comp.graph("test", "png")
    assert isinstance(graph, graphviz.Digraph)


def test_returns_digraph_object_comp_graph_with_params(test_component_graph, graphviz):

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
    filepath = os.path.join(str(tmpdir), "test1")
    pipeline = test_pipeline
    with pytest.raises(ValueError, match="Unknown format"):
        pipeline.graph(filepath=filepath)


def test_invalid_format(tmpdir, test_pipeline):
    filepath = os.path.join(str(tmpdir), "test1.xyz")
    pipeline = test_pipeline
    with pytest.raises(ValueError, match="Unknown format"):
        pipeline.graph(filepath=filepath)


def test_invalid_path(tmpdir, test_pipeline):
    filepath = os.path.join(str(tmpdir), "invalid", "path", "pipeline.png")
    assert not os.path.exists(filepath)
    pipeline = test_pipeline
    with pytest.raises(ValueError, match="Specified filepath is not writeable"):
        pipeline.graph(filepath=filepath)
    assert not os.path.exists(filepath)


def test_graph_feature_importance(X_y_binary, test_pipeline, go):
    X, y = X_y_binary
    clf = test_pipeline
    clf.fit(X, y)
    assert isinstance(clf.graph_feature_importance(), go.Figure)


def test_graph_feature_importance_show_all_features(X_y_binary, test_pipeline, go):
    X, y = X_y_binary
    clf = test_pipeline
    clf.fit(X, y)

    figure = clf.graph_feature_importance()
    assert isinstance(figure, go.Figure)

    data = figure.data[0]
    assert np.any(data["x"] == 0.0)


def test_graph_feature_importance_threshold(X_y_binary, test_pipeline, go):
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
    linear_regression_pipeline,
    nonlinear_binary_with_target_pipeline,
):
    pipeline_ = linear_regression_pipeline
    if graph_type == "graph":
        pipeline_ = nonlinear_binary_with_target_pipeline

    pipeline_parameters = pipeline_.parameters
    expected_nodes = pipeline_.component_graph.component_dict
    dag_dict = pipeline_.graph_dict()

    assert isinstance(dag_dict, dict)
    assert dag_dict["x_edges"][0]["from"] == "X"
    assert len(expected_nodes.keys()) == len(dag_dict["Nodes"].keys()) - 2
    assert dag_dict["Nodes"].keys() - expected_nodes.keys() == {"X", "y"}
    x_edges_expected = []
    y_edges_expected = []
    for node_, graph_ in expected_nodes.items():
        assert node_ in dag_dict["Nodes"]
        comp_name = graph_[0].name if graph_type == "list" else graph_[0]
        assert comp_name == dag_dict["Nodes"][node_]["Name"]
        for comp_ in graph_[1:]:
            if comp_ == "X":
                x_edges_expected.append({"from": "X", "to": node_})
            elif comp_.endswith(".x"):
                x_edges_expected.append({"from": comp_[:-2], "to": node_})
            elif comp_ == "y":
                y_edges_expected.append({"from": "y", "to": node_})
            else:
                y_edges_expected.append({"from": comp_[:-2], "to": node_})
    for node_, params_ in pipeline_parameters.items():
        for key_, val_ in params_.items():
            assert (
                dag_dict["Nodes"][node_]["Parameters"][key_]
                == pipeline_parameters[node_][key_]
            )
    assert len(x_edges_expected) == len(dag_dict["x_edges"])
    assert [edge in dag_dict["x_edges"] for edge in x_edges_expected]
    assert len(y_edges_expected) == len(dag_dict["y_edges"])
    assert [edge in dag_dict["y_edges"] for edge in y_edges_expected]


def test_ensemble_as_json():
    component_graph = {
        "Label Encoder": ["Label Encoder", "X", "y"],
        "Random Forest Pipeline - Label Encoder": [
            "Label Encoder",
            "X",
            "Label Encoder.y",
        ],
        "Random Forest Pipeline - Imputer": [
            "Imputer",
            "X",
            "Random Forest Pipeline - Label Encoder.y",
        ],
        "Random Forest Pipeline - Random Forest Classifier": [
            "Random Forest Classifier",
            "Random Forest Pipeline - Imputer.x",
            "Random Forest Pipeline - Label Encoder.y",
        ],
        "Decision Tree Pipeline - Label Encoder": [
            "Label Encoder",
            "X",
            "Label Encoder.y",
        ],
        "Decision Tree Pipeline - Imputer": [
            "Imputer",
            "X",
            "Decision Tree Pipeline - Label Encoder.y",
        ],
        "Decision Tree Pipeline - Decision Tree Classifier": [
            "Decision Tree Classifier",
            "Decision Tree Pipeline - Imputer.x",
            "Decision Tree Pipeline - Label Encoder.y",
        ],
        "Stacked Ensemble Classifier": [
            "Stacked Ensemble Classifier",
            "Random Forest Pipeline - Random Forest Classifier.x",
            "Decision Tree Pipeline - Decision Tree Classifier.x",
            "Decision Tree Pipeline - Label Encoder.y",
        ],
    }
    parameters = {
        "Random Forest Pipeline - Random Forest Classifier": {"max_depth": np.int64(7)},
    }
    pipeline = BinaryClassificationPipeline(component_graph, parameters=parameters)
    dag_dict = pipeline.graph_dict()

    assert list(dag_dict["Nodes"].keys()) == list(component_graph.keys()) + ["X", "y"]
