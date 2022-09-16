import re
import warnings
from datetime import datetime, timedelta
from unittest.mock import patch

import featuretools as ft
import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal, assert_index_equal, assert_series_equal
from woodwork.logical_types import Boolean, Categorical, Double, EmailAddress, Integer

from evalml.demos import load_diabetes
from evalml.exceptions import (
    MethodPropertyNotFoundError,
    MissingComponentError,
    ParameterNotUsedWarning,
    PipelineError,
    PipelineErrorCodeEnum,
)
from evalml.pipelines import ComponentGraph
from evalml.pipelines.components import (
    DateTimeFeaturizer,
    DropRowsTransformer,
    ElasticNetClassifier,
    Estimator,
    Imputer,
    LogisticRegressionClassifier,
    NaturalLanguageFeaturizer,
    OneHotEncoder,
    RandomForestClassifier,
    SelectColumns,
    StandardScaler,
    TargetImputer,
    Transformer,
    Undersampler,
)
from evalml.problem_types import is_classification
from evalml.utils import infer_feature_types


class DummyTransformer(Transformer):
    name = "Dummy Transformer"

    def __init__(self, parameters=None, random_seed=0):
        parameters = parameters or {}
        super().__init__(
            parameters=parameters,
            component_obj=None,
            random_seed=random_seed,
        )

    def fit(self, X, y):
        return self

    def transform(self, X, y=None):
        return X


class TransformerA(DummyTransformer):
    """copy class"""


class TransformerB(DummyTransformer):
    """copy class"""


class TransformerC(DummyTransformer):
    """copy class"""


class DummyEstimator(Estimator):
    name = "Dummy Estimator"
    model_family = None
    supported_problem_types = None

    def __init__(self, parameters=None, random_seed=0):
        parameters = parameters or {}
        super().__init__(
            parameters=parameters,
            component_obj=None,
            random_seed=random_seed,
        )

    def fit(self, X, y):
        return self


class EstimatorA(DummyEstimator):
    """copy class"""


class EstimatorB(DummyEstimator):
    """copy class"""


class EstimatorC(DummyEstimator):
    """copy class"""


@pytest.fixture
def dummy_components():
    return TransformerA, TransformerB, TransformerC, EstimatorA, EstimatorB, EstimatorC


def test_init(example_graph):
    comp_graph = ComponentGraph()
    assert len(comp_graph.component_dict) == 0

    graph = example_graph
    comp_graph = ComponentGraph(graph)
    assert len(comp_graph.component_dict) == 6

    expected_order = [
        "Imputer",
        "OneHot_ElasticNet",
        "Elastic Net",
        "OneHot_RandomForest",
        "Random Forest",
        "Logistic Regression Classifier",
    ]
    assert comp_graph.compute_order == expected_order


def test_init_str_components():
    graph = {
        "Imputer": ["Imputer", "X", "y"],
        "OneHot_RandomForest": ["One Hot Encoder", "Imputer.x", "y"],
        "OneHot_ElasticNet": ["One Hot Encoder", "Imputer.x", "y"],
        "Random Forest": ["Random Forest Classifier", "OneHot_RandomForest.x", "y"],
        "Elastic Net": ["Elastic Net Classifier", "OneHot_ElasticNet.x", "y"],
        "Logistic Regression Classifier": [
            "Logistic Regression Classifier",
            "Random Forest.x",
            "Elastic Net.x",
            "y",
        ],
    }
    comp_graph = ComponentGraph(graph)
    assert len(comp_graph.component_dict) == 6

    expected_order = [
        "Imputer",
        "OneHot_ElasticNet",
        "Elastic Net",
        "OneHot_RandomForest",
        "Random Forest",
        "Logistic Regression Classifier",
    ]
    assert comp_graph.compute_order == expected_order


def test_init_instantiated():
    graph = {
        "Imputer": [
            Imputer(numeric_impute_strategy="constant", numeric_fill_value=0),
            "X",
            "y",
        ],
    }
    component_graph = ComponentGraph(graph)
    component_graph.instantiate(
        {"Imputer": {"numeric_fill_value": 10, "categorical_fill_value": "Fill"}},
    )
    cg_imputer = component_graph.get_component("Imputer")
    assert graph["Imputer"][0] == cg_imputer
    assert cg_imputer.parameters["numeric_fill_value"] == 0
    assert cg_imputer.parameters["categorical_fill_value"] is None


def test_invalid_init():
    invalid_graph = {"Imputer": [Imputer, "X", "y"], "OHE": OneHotEncoder}
    with pytest.raises(
        ValueError,
        match="All component information should be passed in as a list",
    ):
        ComponentGraph(invalid_graph)

    graph = {
        "Imputer": [
            None,
            "X",
            "y",
        ],
    }
    with pytest.raises(
        ValueError,
        match="may only contain str or ComponentBase subclasses",
    ):
        ComponentGraph(graph)

    graph = {
        "Fake": ["Fake Component", "X", "y"],
        "Estimator": [ElasticNetClassifier, "Fake.x", "y"],
    }
    with pytest.raises(MissingComponentError):
        ComponentGraph(graph)


def test_init_bad_graphs():
    graph_with_cycle = {
        "Imputer": [Imputer, "X", "y"],
        "OHE": [OneHotEncoder, "Imputer.x", "Estimator.x", "y"],
        "Estimator": [RandomForestClassifier, "OHE.x", "y"],
    }
    with pytest.raises(ValueError, match="given graph contains a cycle"):
        ComponentGraph(graph_with_cycle)

    graph_with_more_than_one_final_component = {
        "Imputer": ["Imputer", "X", "y"],
        "OneHot_RandomForest": ["One Hot Encoder", "Imputer.x", "y"],
        "OneHot_ElasticNet": ["One Hot Encoder", "Imputer.x", "y"],
        "Random Forest": ["Random Forest Classifier", "OneHot_RandomForest.x", "y"],
        "Elastic Net": ["Elastic Net Classifier", "X", "y"],
        "Logistic Regression Classifier": [
            "Logistic Regression Classifier",
            "Random Forest.x",
            "Elastic Net.x",
            "y",
        ],
    }
    with pytest.raises(ValueError, match="graph has more than one final"):
        ComponentGraph(graph_with_more_than_one_final_component)

    graph_with_unconnected_imputer = {
        "Imputer": ["Imputer", "X", "y"],
        "DateTime": ["DateTime Featurizer", "X", "y"],
        "Logistic Regression Classifier": [
            "Logistic Regression Classifier",
            "DateTime.x",
            "y",
        ],
    }
    with pytest.raises(ValueError, match="The given graph is not completely connected"):
        ComponentGraph(graph_with_unconnected_imputer)


def test_order_x_and_y():
    graph = {
        "Imputer": [Imputer, "X", "y"],
        "OHE": [OneHotEncoder, "Imputer.x", "y"],
        "Random Forest": [RandomForestClassifier, "OHE.x", "y"],
    }
    component_graph = ComponentGraph(graph).instantiate()
    assert component_graph.compute_order == ["Imputer", "OHE", "Random Forest"]


def test_list_raises_error():
    component_list = ["Imputer", "One Hot Encoder", RandomForestClassifier]
    with pytest.raises(
        ValueError,
        match="component_dict must be a dictionary which specifies the components and edges between components",
    ):
        ComponentGraph(component_list)


def test_instantiate_with_parameters(example_graph):
    graph = example_graph
    component_graph = ComponentGraph(graph)

    assert not isinstance(component_graph.get_component("Imputer"), Imputer)
    assert not isinstance(
        component_graph.get_component("Elastic Net"),
        ElasticNetClassifier,
    )

    parameters = {
        "OneHot_RandomForest": {"top_n": 3},
        "OneHot_ElasticNet": {"top_n": 5},
        "Elastic Net": {"max_iter": 100},
    }
    component_graph.instantiate(parameters)

    expected_order = [
        "Imputer",
        "OneHot_ElasticNet",
        "Elastic Net",
        "OneHot_RandomForest",
        "Random Forest",
        "Logistic Regression Classifier",
    ]
    assert component_graph.compute_order == expected_order

    assert isinstance(component_graph.get_component("Imputer"), Imputer)
    assert isinstance(
        component_graph.get_component("Random Forest"),
        RandomForestClassifier,
    )
    assert isinstance(
        component_graph.get_component("Logistic Regression Classifier"),
        LogisticRegressionClassifier,
    )
    assert component_graph.get_component("OneHot_RandomForest").parameters["top_n"] == 3
    assert component_graph.get_component("OneHot_ElasticNet").parameters["top_n"] == 5
    assert component_graph.get_component("Elastic Net").parameters["max_iter"] == 100


@pytest.mark.parametrize("parameters", [None, {}])
def test_instantiate_without_parameters(parameters, example_graph):
    graph = example_graph
    component_graph = ComponentGraph(graph)
    if parameters is not None:
        component_graph.instantiate(parameters)
    else:
        component_graph.instantiate()
    assert (
        component_graph.get_component("OneHot_RandomForest").parameters["top_n"] == 10
    )
    assert component_graph.get_component("OneHot_ElasticNet").parameters["top_n"] == 10
    assert component_graph.get_component(
        "OneHot_RandomForest",
    ) is not component_graph.get_component("OneHot_ElasticNet")

    expected_order = [
        "Imputer",
        "OneHot_ElasticNet",
        "Elastic Net",
        "OneHot_RandomForest",
        "Random Forest",
        "Logistic Regression Classifier",
    ]
    assert component_graph.compute_order == expected_order


def test_reinstantiate(example_graph):
    component_graph = ComponentGraph(example_graph)
    component_graph.instantiate()
    with pytest.raises(ValueError, match="Cannot reinstantiate a component graph"):
        component_graph.instantiate({"OneHot": {"top_n": 7}})


def test_bad_instantiate_can_reinstantiate(example_graph):
    component_graph = ComponentGraph(example_graph)
    with pytest.raises(ValueError, match="Error received when instantiating component"):
        component_graph.instantiate(
            parameters={"Elastic Net": {"max_iter": 100, "fake_param": None}},
        )

    component_graph.instantiate({"Elastic Net": {"max_iter": 22}})
    assert component_graph.get_component("Elastic Net").parameters["max_iter"] == 22


def test_get_component(example_graph):
    graph = example_graph
    component_graph = ComponentGraph(graph)

    assert component_graph.get_component("OneHot_ElasticNet") == OneHotEncoder
    assert (
        component_graph.get_component("Logistic Regression Classifier")
        == LogisticRegressionClassifier
    )

    with pytest.raises(ValueError, match="not in the graph"):
        component_graph.get_component("Fake Component")

    component_graph.instantiate(
        {
            "OneHot_RandomForest": {"top_n": 3},
            "Random Forest": {"max_depth": 4, "n_estimators": 50},
        },
    )
    assert component_graph.get_component("OneHot_ElasticNet") == OneHotEncoder()
    assert component_graph.get_component("OneHot_RandomForest") == OneHotEncoder(
        top_n=3,
    )
    assert component_graph.get_component("Random Forest") == RandomForestClassifier(
        n_estimators=50,
        max_depth=4,
    )


def test_get_estimators(example_graph):
    component_graph = ComponentGraph(example_graph)
    with pytest.raises(ValueError, match="Cannot get estimators until"):
        component_graph.get_estimators()

    component_graph.instantiate()
    assert component_graph.get_estimators() == [
        RandomForestClassifier(),
        ElasticNetClassifier(),
        LogisticRegressionClassifier(),
    ]

    component_graph = ComponentGraph({"Imputer": ["Imputer", "X", "y"]})
    component_graph.instantiate()
    assert component_graph.get_estimators() == []


def test_parents(example_graph):
    graph = example_graph
    component_graph = ComponentGraph(graph)

    assert component_graph.get_inputs("Imputer") == ["X", "y"]
    assert component_graph.get_inputs("OneHot_RandomForest") == ["Imputer.x", "y"]
    assert component_graph.get_inputs("OneHot_ElasticNet") == ["Imputer.x", "y"]
    assert component_graph.get_inputs("Random Forest") == ["OneHot_RandomForest.x", "y"]
    assert component_graph.get_inputs("Elastic Net") == ["OneHot_ElasticNet.x", "y"]
    assert component_graph.get_inputs("Logistic Regression Classifier") == [
        "Random Forest.x",
        "Elastic Net.x",
        "y",
    ]

    with pytest.raises(ValueError, match="not in the graph"):
        component_graph.get_inputs("Fake component")

    component_graph.instantiate()
    assert component_graph.get_inputs("Imputer") == ["X", "y"]
    assert component_graph.get_inputs("OneHot_RandomForest") == ["Imputer.x", "y"]
    assert component_graph.get_inputs("OneHot_ElasticNet") == ["Imputer.x", "y"]
    assert component_graph.get_inputs("Random Forest") == ["OneHot_RandomForest.x", "y"]
    assert component_graph.get_inputs("Elastic Net") == ["OneHot_ElasticNet.x", "y"]
    assert component_graph.get_inputs("Logistic Regression Classifier") == [
        "Random Forest.x",
        "Elastic Net.x",
        "y",
    ]

    with pytest.raises(ValueError, match="not in the graph"):
        component_graph.get_inputs("Fake component")


def test_get_last_component(example_graph):
    component_graph = ComponentGraph()
    with pytest.raises(
        ValueError,
        match="Cannot get last component from edgeless graph",
    ):
        component_graph.get_last_component()

    component_graph = ComponentGraph(example_graph)
    assert component_graph.get_last_component() == LogisticRegressionClassifier

    component_graph.instantiate()
    assert component_graph.get_last_component() == LogisticRegressionClassifier()

    component_graph = ComponentGraph({"Imputer": [Imputer, "X", "y"]})
    assert component_graph.get_last_component() == Imputer

    component_graph = ComponentGraph(
        {"Imputer": [Imputer, "X", "y"], "OneHot": [OneHotEncoder, "Imputer.x", "y"]},
    )
    assert component_graph.get_last_component() == OneHotEncoder


@patch("evalml.pipelines.components.Transformer.fit_transform")
@patch("evalml.pipelines.components.Estimator.fit")
@patch("evalml.pipelines.components.Estimator.predict_proba")
def test_fit_component_graph(
    mock_predict_proba,
    mock_fit,
    mock_fit_transform,
    example_graph,
    X_y_binary,
):
    X, y = X_y_binary
    mock_fit_transform.return_value = pd.DataFrame(X)
    mock_predict_proba.return_value = y
    component_graph = ComponentGraph(example_graph).instantiate()
    component_graph.fit(X, y)

    assert mock_fit_transform.call_count == 3
    assert mock_fit.call_count == 3
    assert mock_predict_proba.call_count == 2


@patch("evalml.pipelines.components.TargetImputer.fit_transform")
@patch("evalml.pipelines.components.OneHotEncoder.fit_transform")
def test_fit_correct_inputs(
    mock_ohe_fit_transform,
    mock_imputer_fit_transform,
    X_y_binary,
):
    X, y = X_y_binary
    graph = {
        "Target Imputer": [TargetImputer, "X", "y"],
        "OHE": [OneHotEncoder, "Target Imputer.x", "Target Imputer.y"],
    }
    expected_x = pd.DataFrame(index=X.index, columns=X.columns).fillna(1.0)
    expected_x.ww.init()

    expected_y = pd.Series(index=y.index).fillna(0)
    mock_imputer_fit_transform.return_value = tuple((expected_x, expected_y))
    mock_ohe_fit_transform.return_value = expected_x
    component_graph = ComponentGraph(graph).instantiate()
    component_graph.fit(X, y)
    assert_frame_equal(expected_x, mock_ohe_fit_transform.call_args[0][0])
    assert_series_equal(expected_y, mock_ohe_fit_transform.call_args[0][1])


@patch("evalml.pipelines.components.Transformer.fit_transform", autospec=True)
@patch("evalml.pipelines.components.OneHotEncoder.transform", autospec=True)
@patch("evalml.pipelines.components.Imputer.transform", autospec=True)
@patch("evalml.pipelines.components.LabelEncoder.transform", autospec=True)
def test_component_graph_fit_transform(
    mock_label_encoder_transform,
    mock_imputer_transform,
    mock_ohe_transform,
    mock_fit_transform,
    example_graph,
    example_graph_with_transformer_last_component,
    X_y_binary,
):
    X, y = X_y_binary
    component_graph = ComponentGraph(example_graph)
    component_graph.instantiate()
    with pytest.raises(
        ValueError,
        match=re.escape(
            "Cannot call fit_transform() on a component graph because the final component is an Estimator. Use fit_and_transform_all_but_final instead.",
        ),
    ):
        component_graph.fit_transform(X, y)

    component_graph = ComponentGraph(example_graph_with_transformer_last_component)
    component_graph.instantiate()
    ones_df = pd.DataFrame(np.ones(pd.DataFrame(X).shape))

    def fit_transform_side_effect(self, X, y):
        self._is_fitted = True
        return ones_df

    mock_fit_transform.side_effect = fit_transform_side_effect
    mock_label_encoder_transform.return_value = ones_df
    mock_imputer_transform.return_value = ones_df
    mock_ohe_transform.return_value = ones_df

    component_graph.fit_transform(X, y)

    assert mock_label_encoder_transform.call_count == 2
    assert mock_imputer_transform.call_count == 1
    assert mock_ohe_transform.call_count == 1
    assert mock_fit_transform.call_count == 2


@patch("evalml.pipelines.components.Transformer.fit_transform")
@patch("evalml.pipelines.components.Estimator.fit")
@patch("evalml.pipelines.components.Estimator.predict_proba")
def test_component_graph_fit_and_transform_all_but_final(
    mock_predict_proba,
    mock_fit,
    mock_fit_transform,
    example_graph,
    X_y_binary,
):
    X, y = X_y_binary
    component_graph = ComponentGraph(example_graph)
    component_graph.instantiate()

    mock_X_t = pd.DataFrame(np.ones(pd.DataFrame(X).shape))
    mock_fit_transform.return_value = mock_X_t
    mock_fit.return_value = Estimator
    mock_predict_proba.return_value = y

    component_graph.fit_and_transform_all_but_final(X, y)

    assert mock_fit_transform.call_count == 3
    assert mock_fit.call_count == 2
    assert mock_predict_proba.call_count == 2


@patch("evalml.pipelines.components.Estimator.fit")
@patch("evalml.pipelines.components.Estimator.predict_proba")
@patch("evalml.pipelines.components.Estimator.predict")
def test_predict(mock_predict, mock_predict_proba, mock_fit, example_graph, X_y_binary):
    X, y = X_y_binary
    mock_predict_proba.return_value = y
    mock_predict.return_value = pd.Series(y)
    component_graph = ComponentGraph(example_graph).instantiate()
    component_graph.fit(X, y)

    component_graph.predict(X)
    assert (
        mock_predict_proba.call_count == 4
    )  # Called twice when fitting pipeline, twice when predicting
    assert mock_predict.call_count == 1  # Called once during predict

    assert mock_fit.call_count == 3  # Only called during fit, not predict


@patch("evalml.pipelines.components.Estimator.fit")
@patch("evalml.pipelines.components.Estimator.predict_proba")
@patch("evalml.pipelines.components.Estimator.predict")
def test_predict_multiclass(
    mock_predict,
    mock_predict_proba,
    mock_fit,
    example_graph,
    X_y_multi,
):
    X, y = X_y_multi
    mock_predict_proba.return_value = pd.DataFrame(
        {
            0: np.full(X.shape[0], 0.33),
            1: np.full(X.shape[0], 0.33),
            2: np.full(X.shape[0], 0.33),
        },
    )
    mock_predict_proba.return_value.ww.init()
    mock_predict.return_value = pd.Series(y)
    component_graph = ComponentGraph(example_graph).instantiate()
    component_graph.fit(X, y)
    final_estimator_input = component_graph.transform_all_but_final(X, y)
    assert final_estimator_input.columns.to_list() == [
        "Col 0 Random Forest.x",
        "Col 1 Random Forest.x",
        "Col 2 Random Forest.x",
        "Col 0 Elastic Net.x",
        "Col 1 Elastic Net.x",
        "Col 2 Elastic Net.x",
    ]
    for col in final_estimator_input:
        assert np.array_equal(
            final_estimator_input[col].to_numpy(),
            np.full(X.shape[0], 0.33),
        )
    component_graph.predict(X)
    assert (
        mock_predict_proba.call_count == 6
    )  # Called twice when fitting pipeline, twice to compute final features, and twice when predicting
    assert mock_predict.call_count == 1  # Called once during predict
    assert mock_fit.call_count == 3  # Only called during fit, not predict


@patch("evalml.pipelines.components.Estimator.fit")
@patch("evalml.pipelines.components.Estimator.predict_proba")
@patch("evalml.pipelines.components.Estimator.predict")
def test_predict_regression(
    mock_predict,
    mock_predict_proba,
    mock_fit,
    example_regression_graph,
    X_y_multi,
):
    X, y = X_y_multi
    mock_predict.return_value = pd.Series(y)
    mock_predict_proba.side_effect = MethodPropertyNotFoundError
    component_graph = ComponentGraph(example_regression_graph).instantiate()
    component_graph.fit(X, y)
    final_estimator_input = component_graph.transform_all_but_final(X, y)
    assert final_estimator_input.columns.to_list() == [
        "Random Forest.x",
        "Elastic Net.x",
    ]
    component_graph.predict(X)
    assert (
        mock_predict_proba.call_count == 6
    )  # Called twice when fitting pipeline, twice to compute final features, and twice when predicting
    assert (
        mock_predict.call_count == 7
    )  # Called because `predict_proba` does not exist for regresssions
    assert mock_fit.call_count == 3  # Only called during fit, not predict


@patch("evalml.pipelines.components.Estimator.fit")
@patch("evalml.pipelines.components.Estimator.predict_proba")
@patch("evalml.pipelines.components.Estimator.predict")
def test_predict_repeat_estimator(
    mock_predict,
    mock_predict_proba,
    mock_fit,
    X_y_binary,
):
    X, y = X_y_binary
    mock_predict_proba.return_value = y
    mock_predict.return_value = pd.Series(y)
    graph = {
        "Imputer": [Imputer, "X", "y"],
        "OneHot_RandomForest": [OneHotEncoder, "Imputer.x", "y"],
        "OneHot_Logistic": [OneHotEncoder, "Imputer.x", "y"],
        "Random Forest": [RandomForestClassifier, "OneHot_RandomForest.x", "y"],
        "Logistic Regression Classifier": [
            LogisticRegressionClassifier,
            "OneHot_Logistic.x",
            "y",
        ],
        "Final Estimator": [
            LogisticRegressionClassifier,
            "Random Forest.x",
            "Logistic Regression Classifier.x",
            "y",
        ],
    }
    component_graph = ComponentGraph(graph)
    component_graph.instantiate()
    component_graph.fit(X, y)

    assert (
        not component_graph.get_component(
            "Logistic Regression Classifier",
        )._component_obj
        == component_graph.get_component("Final Estimator")._component_obj
    )

    component_graph.predict(X)
    assert mock_predict_proba.call_count == 4
    assert mock_predict.call_count == 1
    assert mock_fit.call_count == 3


@patch("evalml.pipelines.components.Imputer.transform")
@patch("evalml.pipelines.components.OneHotEncoder.transform")
@patch("evalml.pipelines.components.RandomForestClassifier.predict_proba")
@patch("evalml.pipelines.components.ElasticNetClassifier.predict_proba")
def test_transform_all_but_final(
    mock_en_predict_proba,
    mock_rf_predict_proba,
    mock_ohe,
    mock_imputer,
    example_graph,
    X_y_binary,
):
    X, y = X_y_binary
    mock_imputer.return_value = pd.DataFrame(X)
    mock_ohe.return_value = pd.DataFrame(X)
    mock_en_predict_proba.return_value = pd.DataFrame(
        ({0: np.zeros(X.shape[0]), 1: np.ones(X.shape[0])}),
    )
    mock_en_predict_proba.return_value.ww.init()
    mock_rf_predict_proba.return_value = pd.DataFrame(
        ({0: np.ones(X.shape[0]), 1: np.zeros(X.shape[0])}),
    )
    mock_rf_predict_proba.return_value.ww.init()
    X_expected = pd.DataFrame(
        {
            "Col 1 Random Forest.x": np.zeros(X.shape[0]),
            "Col 1 Elastic Net.x": np.ones(X.shape[0]),
        },
    )
    component_graph = ComponentGraph(example_graph).instantiate()
    component_graph.fit(X, y)

    X_t = component_graph.transform_all_but_final(X)
    assert_frame_equal(X_expected, X_t)
    assert mock_imputer.call_count == 2
    assert mock_ohe.call_count == 4


@patch(f"{__name__}.DummyTransformer.transform")
def test_transform_all_but_final_single_component(mock_transform, X_y_binary):
    X, y = X_y_binary
    X = pd.DataFrame(X)
    mock_transform.return_value = X
    component_graph = ComponentGraph(
        {"Dummy Component": [DummyTransformer, "X", "y"]},
    ).instantiate()
    component_graph.fit(X, y)

    X_t = component_graph.transform_all_but_final(X)
    assert_frame_equal(X, X_t)


@patch("evalml.pipelines.components.Imputer.fit_transform")
def test_fit_y_parent(mock_fit_transform, X_y_binary):
    X, y = X_y_binary
    graph = {
        "Imputer": [Imputer, "X", "y"],
        "OHE": [OneHotEncoder, "Imputer.x", "y"],
        "Random Forest": [RandomForestClassifier, "OHE.x", "y"],
    }
    component_graph = ComponentGraph(graph).instantiate()
    mock_fit_transform.return_value = tuple((pd.DataFrame(X), pd.Series(y)))

    component_graph.fit(X, y)
    mock_fit_transform.assert_called_once()


def test_predict_empty_graph(X_y_binary):
    X, y = X_y_binary
    X = pd.DataFrame(X)
    component_graph = ComponentGraph()
    component_graph.instantiate()

    component_graph.fit(X, y)
    X_t = component_graph.transform(X, y)
    assert_frame_equal(X, X_t)

    X_pred = component_graph.predict(X)
    assert_frame_equal(X, X_pred)


def test_no_instantiate_before_fit(X_y_binary):
    X, y = X_y_binary
    graph = {
        "Imputer": [Imputer, "X", "y"],
        "OHE": [OneHotEncoder, "Imputer.x", "y"],
        "Estimator": [RandomForestClassifier, "OHE.x", "y"],
    }
    component_graph = ComponentGraph(graph)
    with pytest.raises(
        ValueError,
        match="All components must be instantiated before fitting or predicting",
    ):
        component_graph.fit(X, y)


def test_multiple_y_parents():
    graph = {
        "Imputer": [Imputer, "X", "y"],
        "TargetImputer": [Imputer, "Imputer.x", "y"],
        "Estimator": [RandomForestClassifier, "Imputer.x", "y", "TargetImputer.y"],
    }
    with pytest.raises(ValueError, match="All components must have exactly one target"):
        ComponentGraph(graph)


def test_component_graph_order(example_graph):
    component_graph = ComponentGraph(example_graph)
    expected_order = [
        "Imputer",
        "OneHot_ElasticNet",
        "Elastic Net",
        "OneHot_RandomForest",
        "Random Forest",
        "Logistic Regression Classifier",
    ]
    assert expected_order == component_graph.compute_order

    component_graph = ComponentGraph({"Imputer": [Imputer, "X", "y"]})
    expected_order = ["Imputer"]
    assert expected_order == component_graph.compute_order


@pytest.mark.parametrize(
    "index",
    [
        list(range(-5, 0)),
        list(range(100, 105)),
        [f"row_{i}" for i in range(5)],
        pd.date_range("2020-09-08", periods=5),
    ],
)
@pytest.mark.parametrize("with_estimator_last_component", [True, False])
def test_component_graph_transform_and_predict_with_custom_index(
    index,
    with_estimator_last_component,
    example_graph,
    example_graph_with_transformer_last_component,
):
    X = pd.DataFrame(
        {"categories": [f"cat_{i}" for i in range(5)], "numbers": np.arange(5)},
        index=index,
    )
    y = pd.Series([1, 2, 1, 2, 1], index=index)
    X.ww.init(logical_types={"categories": "categorical"})

    graph_to_use = (
        example_graph
        if with_estimator_last_component
        else example_graph_with_transformer_last_component
    )
    component_graph = ComponentGraph(graph_to_use)
    component_graph.instantiate()
    component_graph.fit(X, y)

    if with_estimator_last_component:
        predictions = component_graph.predict(X)
        assert_index_equal(predictions.index, X.index)
        assert not predictions.isna().any(axis=None)
    else:
        X_t = component_graph.transform(X)
        assert_index_equal(X_t.index, X.index)
        assert not X_t.isna().any(axis=None)

        y_in = pd.Series([0, 1, 0, 1, 0], index=index)
        y_inv = component_graph.inverse_transform(y_in)
        assert_index_equal(y_inv.index, y.index)
        assert not y_inv.isna().any(axis=None)


@patch(f"{__name__}.EstimatorC.predict")
@patch(f"{__name__}.EstimatorB.predict")
@patch(f"{__name__}.EstimatorA.predict")
@patch(f"{__name__}.TransformerC.transform")
@patch(f"{__name__}.TransformerB.transform")
@patch(f"{__name__}.TransformerA.transform")
def test_component_graph_evaluation_plumbing(
    mock_transform_a,
    mock_transform_b,
    mock_transform_c,
    mock_predict_a,
    mock_predict_b,
    mock_predict_c,
    dummy_components,
):
    (
        TransformerA,
        TransformerB,
        TransformerC,
        EstimatorA,
        EstimatorB,
        EstimatorC,
    ) = dummy_components
    mock_transform_a.return_value = pd.DataFrame(
        {"feature trans": [1, 0, 0, 0, 0, 0], "feature a": np.ones(6)},
    )
    mock_transform_b.return_value = pd.DataFrame({"feature b": np.ones(6) * 2})
    mock_transform_c.return_value = pd.DataFrame({"feature c": np.ones(6) * 3})
    mock_predict_a.return_value = pd.Series([0, 0, 0, 1, 0, 0])
    mock_predict_b.return_value = pd.Series([0, 0, 0, 0, 1, 0])
    mock_predict_c.return_value = pd.Series([0, 0, 0, 0, 0, 1])
    graph = {
        "transformer a": [TransformerA, "X", "y"],
        "transformer b": [TransformerB, "transformer a.x", "y"],
        "transformer c": [TransformerC, "transformer a.x", "transformer b.x", "y"],
        "estimator a": [EstimatorA, "X", "y"],
        "estimator b": [EstimatorB, "transformer a.x", "y"],
        "estimator c": [
            EstimatorC,
            "transformer a.x",
            "estimator a.x",
            "transformer b.x",
            "estimator b.x",
            "transformer c.x",
            "y",
        ],
    }
    component_graph = ComponentGraph(graph)
    component_graph.instantiate()
    X = pd.DataFrame({"feature1": np.zeros(6), "feature2": np.zeros(6)})
    y = pd.Series(np.zeros(6))
    component_graph.fit(X, y)
    predict_out = component_graph.predict(X)

    assert_frame_equal(mock_transform_a.call_args[0][0], X)
    assert_frame_equal(
        mock_transform_b.call_args[0][0],
        pd.DataFrame(
            {
                "feature trans": pd.Series([1, 0, 0, 0, 0, 0], dtype="int64"),
                "feature a": np.ones(6),
            },
            columns=["feature trans", "feature a"],
        ),
    )
    assert_frame_equal(
        mock_transform_c.call_args[0][0],
        pd.DataFrame(
            {
                "feature trans": pd.Series([1, 0, 0, 0, 0, 0], dtype="int64"),
                "feature a": np.ones(6),
                "feature b": np.ones(6) * 2,
            },
            columns=["feature trans", "feature a", "feature b"],
        ),
    )
    assert_frame_equal(mock_predict_a.call_args[0][0], X)
    assert_frame_equal(
        mock_predict_b.call_args[0][0],
        pd.DataFrame(
            {
                "feature trans": pd.Series([1, 0, 0, 0, 0, 0], dtype="int64"),
                "feature a": np.ones(6),
            },
            columns=["feature trans", "feature a"],
        ),
    )
    assert_frame_equal(
        mock_predict_c.call_args[0][0],
        pd.DataFrame(
            {
                "feature trans": pd.Series([1, 0, 0, 0, 0, 0], dtype="int64"),
                "feature a": np.ones(6),
                "estimator a.x": pd.Series([0, 0, 0, 1, 0, 0], dtype="int64"),
                "feature b": np.ones(6) * 2,
                "estimator b.x": pd.Series([0, 0, 0, 0, 1, 0], dtype="int64"),
                "feature c": np.ones(6) * 3,
            },
            columns=[
                "feature trans",
                "feature a",
                "estimator a.x",
                "feature b",
                "estimator b.x",
                "feature c",
            ],
        ),
    )
    assert_series_equal(pd.Series([0, 0, 0, 0, 0, 1], dtype="int64"), predict_out)


def test_input_feature_names(example_graph):
    X = pd.DataFrame(
        {
            "column_1": ["a", "b", "c", "d", "a", "a", "b", "c", "b"],
            "column_2": [1, 2, 3, 4, 5, 6, 5, 4, 3],
        },
    )
    y = pd.Series([1, 0, 1, 0, 1, 1, 0, 0, 0])
    X.ww.init(logical_types={"column_1": "categorical"})

    component_graph = ComponentGraph(example_graph)
    component_graph.instantiate(
        {"OneHot_RandomForest": {"top_n": 2}, "OneHot_ElasticNet": {"top_n": 3}},
    )
    assert component_graph.input_feature_names == {}
    component_graph.fit(X, y)

    input_feature_names = component_graph.input_feature_names
    assert input_feature_names["Imputer"] == ["column_1", "column_2"]
    assert input_feature_names["OneHot_RandomForest"] == ["column_1", "column_2"]
    assert input_feature_names["OneHot_ElasticNet"] == ["column_1", "column_2"]
    assert input_feature_names["Random Forest"] == [
        "column_2",
        "column_1_a",
        "column_1_b",
    ]
    assert input_feature_names["Elastic Net"] == [
        "column_2",
        "column_1_a",
        "column_1_b",
        "column_1_c",
    ]
    assert input_feature_names["Logistic Regression Classifier"] == [
        "Col 1 Random Forest.x",
        "Col 1 Elastic Net.x",
    ]


def test_iteration(example_graph):
    component_graph = ComponentGraph(example_graph)

    expected = [
        Imputer,
        OneHotEncoder,
        ElasticNetClassifier,
        OneHotEncoder,
        RandomForestClassifier,
        LogisticRegressionClassifier,
    ]
    iteration = [component for component in component_graph]
    assert iteration == expected

    component_graph.instantiate({"OneHot_RandomForest": {"top_n": 32}})
    expected = [
        Imputer(),
        OneHotEncoder(),
        ElasticNetClassifier(),
        OneHotEncoder(top_n=32),
        RandomForestClassifier(),
        LogisticRegressionClassifier(),
    ]
    iteration = [component for component in component_graph]
    assert iteration == expected


def test_custom_input_feature_types(example_graph):
    X = pd.DataFrame(
        {
            "column_1": ["a", "a", "a", "b", "b", "b", "c", "c", "d"],
            "column_2": [1, 2, 3, 3, 4, 4, 5, 5, 6],
        },
    )
    y = pd.Series([1, 0, 1, 0, 1, 1, 0, 0, 0])
    X = infer_feature_types(X, {"column_1": "categorical", "column_2": "categorical"})

    component_graph = ComponentGraph(example_graph)
    component_graph.instantiate(
        {"OneHot_RandomForest": {"top_n": 2}, "OneHot_ElasticNet": {"top_n": 3}},
    )
    assert component_graph.input_feature_names == {}
    component_graph.fit(X, y)

    input_feature_names = component_graph.input_feature_names
    assert input_feature_names["Imputer"] == ["column_1", "column_2"]
    assert input_feature_names["OneHot_RandomForest"] == ["column_1", "column_2"]
    assert input_feature_names["OneHot_ElasticNet"] == ["column_1", "column_2"]
    assert input_feature_names["Random Forest"] == [
        "column_1_a",
        "column_1_b",
        "column_2_4",
        "column_2_5",
    ]
    assert input_feature_names["Elastic Net"] == [
        "column_1_a",
        "column_1_b",
        "column_1_c",
        "column_2_3",
        "column_2_4",
        "column_2_5",
    ]
    assert input_feature_names["Logistic Regression Classifier"] == [
        "Col 1 Random Forest.x",
        "Col 1 Elastic Net.x",
    ]


def test_component_graph_dataset_with_different_types():
    # Checks that types are converted correctly by Woodwork. Specifically, the standard scaler
    # should convert column_3 to float, so our code to try to convert back to the original boolean type
    # will catch the TypeError thrown and not convert the column.
    # Also, column_4 will be treated as a datetime feature, but the identical column_5 set as natural language
    # should be treated as natural language, not as datetime.
    graph = {
        "Text": [NaturalLanguageFeaturizer, "X", "y"],
        "Imputer": [Imputer, "Text.x", "y"],
        "OneHot": [OneHotEncoder, "Imputer.x", "y"],
        "DateTime": [DateTimeFeaturizer, "OneHot.x", "y"],
        "Scaler": [StandardScaler, "DateTime.x", "y"],
        "Random Forest": [RandomForestClassifier, "Scaler.x", "y"],
        "Elastic Net": [ElasticNetClassifier, "Scaler.x", "y"],
        "Logistic Regression Classifier": [
            LogisticRegressionClassifier,
            "Random Forest.x",
            "Elastic Net.x",
            "y",
        ],
    }

    X = pd.DataFrame(
        {
            "column_1": ["a", "b", "c", "d", "a", "a", "b", "c", "b"],
            "column_2": [1, 2, 3, 4, 5, 6, 5, 4, 3],
            "column_3": [True, False, True, False, True, False, True, False, False],
        },
    )
    X["column_4"] = [
        str((datetime(2021, 5, 21, 12, 0, 0) + timedelta(minutes=5 * x)))
        for x in range(len(X))
    ]
    X["column_5"] = X["column_4"]

    y = pd.Series([1, 0, 1, 0, 1, 1, 0, 0, 0])
    X = infer_feature_types(
        X,
        {
            "column_1": "categorical",
            "column_2": "categorical",
            "column_5": "NaturalLanguage",
        },
    )

    component_graph = ComponentGraph(graph)
    component_graph.instantiate()
    assert component_graph.input_feature_names == {}
    component_graph.fit(X, y)

    def check_feature_names(input_feature_names):
        assert input_feature_names["Text"] == [
            "column_1",
            "column_2",
            "column_3",
            "column_4",
            "column_5",
        ]
        text_columns = [
            "DIVERSITY_SCORE(column_5)",
            "MEAN_CHARACTERS_PER_WORD(column_5)",
            "NUM_CHARACTERS(column_5)",
            "NUM_WORDS(column_5)",
            "POLARITY_SCORE(column_5)",
            "LSA(column_5)[0]",
            "LSA(column_5)[1]",
        ]

        assert (
            input_feature_names["Imputer"]
            == [
                "column_1",
                "column_2",
                "column_3",
                "column_4",
            ]
            + text_columns
        )
        assert (
            input_feature_names["OneHot"]
            == [
                "column_1",
                "column_2",
                "column_3",
                "column_4",
            ]
            + text_columns
        )
        assert sorted(input_feature_names["DateTime"]) == sorted(
            [
                "column_3",
                "column_4",
                "column_1_a",
                "column_1_b",
                "column_1_c",
                "column_1_d",
                "column_2_1",
                "column_2_2",
                "column_2_3",
                "column_2_4",
                "column_2_5",
                "column_2_6",
            ]
            + text_columns,
        )
        assert sorted(input_feature_names["Scaler"]) == sorted(
            (
                [
                    "column_3",
                    "column_1_a",
                    "column_1_b",
                    "column_1_c",
                    "column_1_d",
                    "column_2_1",
                    "column_2_2",
                    "column_2_3",
                    "column_2_4",
                    "column_2_5",
                    "column_2_6",
                    "column_4_year",
                    "column_4_month",
                    "column_4_day_of_week",
                    "column_4_hour",
                ]
                + text_columns
            ),
        )
        assert sorted(input_feature_names["Random Forest"]) == sorted(
            (
                [
                    "column_3",
                    "column_1_a",
                    "column_1_b",
                    "column_1_c",
                    "column_1_d",
                    "column_2_1",
                    "column_2_2",
                    "column_2_3",
                    "column_2_4",
                    "column_2_5",
                    "column_2_6",
                    "column_4_year",
                    "column_4_month",
                    "column_4_day_of_week",
                    "column_4_hour",
                ]
                + text_columns
            ),
        )
        assert sorted(input_feature_names["Elastic Net"]) == sorted(
            (
                [
                    "column_3",
                    "column_1_a",
                    "column_1_b",
                    "column_1_c",
                    "column_1_d",
                    "column_2_1",
                    "column_2_2",
                    "column_2_3",
                    "column_2_4",
                    "column_2_5",
                    "column_2_6",
                    "column_4_year",
                    "column_4_month",
                    "column_4_day_of_week",
                    "column_4_hour",
                ]
                + text_columns
            ),
        )
        assert input_feature_names["Logistic Regression Classifier"] == [
            "Col 1 Random Forest.x",
            "Col 1 Elastic Net.x",
        ]

    check_feature_names(component_graph.input_feature_names)
    component_graph.input_feature_names = {}
    component_graph.predict(X)
    check_feature_names(component_graph.input_feature_names)


@patch("evalml.pipelines.components.RandomForestClassifier.fit")
def test_component_graph_types_merge_mock(mock_rf_fit):
    graph = {
        "Select numeric col_2": [SelectColumns, "X", "y"],
        "Imputer numeric col_2": [Imputer, "Select numeric col_2.x", "y"],
        "Scaler col_2": [StandardScaler, "Imputer numeric col_2.x", "y"],
        "Select categorical col_1": [SelectColumns, "X", "y"],
        "Imputer categorical col_1": [Imputer, "Select categorical col_1.x", "y"],
        "OneHot col_1": [OneHotEncoder, "Imputer categorical col_1.x", "y"],
        "Pass through col_3": [SelectColumns, "X", "y"],
        "Random Forest": [
            RandomForestClassifier,
            "Scaler col_2.x",
            "OneHot col_1.x",
            "Pass through col_3.x",
            "y",
        ],
    }

    X = pd.DataFrame(
        {
            "column_1": ["a", "b", "c", "d", "a", "a", "b", "c", "b"],
            "column_2": [1, 2, 3, 4, 5, 6, 5, 4, 3],
            "column_3": [True, False, True, False, True, False, True, False, False],
        },
    )
    y = pd.Series([1, 0, 1, 0, 1, 1, 0, 0, 0])
    # woodwork would infer this as boolean by default -- convert to a numeric type
    X = infer_feature_types(X, {"column_1": "categorical", "column_3": "integer"})

    component_graph = ComponentGraph(graph)
    # we don't have feature type selectors defined yet, so in order for the above graph to work we have to
    # specify the types to select here.
    # if the user-specified woodwork types are being respected, we should see the pass-through column_3 staying as numeric,
    # meaning it won't cause a modeling error when it reaches the estimator
    component_graph.instantiate(
        {
            "Select numeric col_2": {"columns": ["column_2"]},
            "Select categorical col_1": {"columns": ["column_1"]},
            "Pass through col_3": {"columns": ["column_3"]},
        },
    )
    assert component_graph.input_feature_names == {}
    component_graph.fit(X, y)

    input_feature_names = component_graph.input_feature_names
    assert input_feature_names["Random Forest"] == (
        ["column_2", "column_1_a", "column_1_b", "column_1_c", "column_1_d", "column_3"]
    )
    assert isinstance(mock_rf_fit.call_args[0][0].ww.logical_types["column_3"], Integer)
    assert isinstance(mock_rf_fit.call_args[0][0].ww.logical_types["column_2"], Double)


def test_component_graph_preserves_ltypes_created_during_pipeline_evaluation():

    # This test checks that the component graph preserves logical types created during pipeline evaluation
    # The other tests ensure that logical types set before pipeline evaluation are preserved

    class ZipCodeExtractor(Transformer):
        name = "Zip Code Extractor"

        def fit(self, X, y=None):
            return self

        def transform(self, X, y=None):
            X = pd.DataFrame({"zip_code": pd.Series(["02101", "02139", "02152"] * 3)})
            X.ww.init(logical_types={"zip_code": "PostalCode"})
            return X

    class ZipCodeToAveragePrice(Transformer):
        name = "Check Zip Code Preserved"

        def fit(self, X, y=None):
            return self

        def transform(self, X, y=None):
            X = infer_feature_types(X)
            original_columns = list(X.columns)
            X = X.ww.select(["PostalCode"])
            # This would make the test fail if the componant graph
            assert len(X.columns) > 0, "No Zip Code!"
            X.ww["average_apartment_price"] = pd.Series([1000, 2000, 3000] * 3)
            X = X.ww.drop(original_columns)
            return X

    graph = {
        "Select non address": [SelectColumns, "X", "y"],
        "OneHot": [OneHotEncoder, "Select non address.x", "y"],
        "Select address": [SelectColumns, "X", "y"],
        "Extract ZipCode": [ZipCodeExtractor, "Select address.x", "y"],
        "Average Price From ZipCode": [ZipCodeToAveragePrice, "Extract ZipCode.x", "y"],
        "Random Forest": [
            RandomForestClassifier,
            "OneHot.x",
            "Average Price From ZipCode.x",
            "y",
        ],
    }

    X = pd.DataFrame(
        {
            "column_1": ["a", "b", "c", "d", "a", "a", "b", "c", "b"],
            "column_2": [1, 2, 3, 4, 5, 6, 5, 4, 3],
            "column_3": [True, False, True, False, True, False, True, False, False],
            "address": [f"address-{i}" for i in range(9)],
        },
    )
    y = pd.Series([1, 0, 1, 0, 1, 1, 0, 0, 0])

    # woodwork would infer this as boolean by default -- convert to a numeric type
    X.ww.init(
        logical_types={"column_1": "categorical"},
        semantic_tags={"address": "address"},
    )

    component_graph = ComponentGraph(graph)
    # we don't have feature type selectors defined yet, so in order for the above graph to work we have to
    # specify the types to select here.
    # if the user-specified woodwork types are being respected, we should see the pass-through column_3 staying as numeric,
    # meaning it won't cause a modeling error when it reaches the estimator
    component_graph.instantiate(
        {
            "Select non address": {"columns": ["column_1", "column_2", "column_3"]},
            "Select address": {"columns": ["address"]},
        },
    )
    assert component_graph.input_feature_names == {}
    component_graph.fit(X, y)

    input_feature_names = component_graph.input_feature_names
    assert sorted(input_feature_names["Random Forest"]) == sorted(
        [
            "column_2",
            "column_1_a",
            "column_1_b",
            "column_1_c",
            "column_1_d",
            "column_3",
            "average_apartment_price",
        ],
    )


def test_component_graph_types_merge():
    graph = {
        "Select numeric": [SelectColumns, "X", "y"],
        "Imputer numeric": [Imputer, "Select numeric.x", "y"],
        "Select text": [SelectColumns, "X", "y"],
        "Text": [NaturalLanguageFeaturizer, "Select text.x", "y"],
        "Imputer text": [Imputer, "Text.x", "y"],
        "Scaler": [StandardScaler, "Imputer numeric.x", "y"],
        "Select categorical": [SelectColumns, "X", "y"],
        "Imputer categorical": [Imputer, "Select categorical.x", "y"],
        "OneHot": [OneHotEncoder, "Imputer categorical.x", "y"],
        "Select datetime": [SelectColumns, "X", "y"],
        "Imputer datetime": [Imputer, "Select datetime.x", "y"],
        "DateTime": [DateTimeFeaturizer, "Imputer datetime.x", "y"],
        "Select pass through": [SelectColumns, "X", "y"],
        "Random Forest": [
            RandomForestClassifier,
            "Scaler.x",
            "OneHot.x",
            "DateTime.x",
            "Imputer text.x",
            "Select pass through.x",
            "y",
        ],
    }

    X = pd.DataFrame(
        {
            "column_1": ["a", "b", "c", "d", "a", "a", "b", "c", "b"],
            "column_2": [1, 2, 3, 4, 5, 6, 5, 4, 3],
            "column_3": [True, False, True, False, True, False, True, False, False],
        },
    )
    X["column_4"] = [
        str((datetime(2021, 5, 21, 12, 0, 0) + timedelta(minutes=5 * x)))
        for x in range(len(X))
    ]
    X["column_5"] = X["column_4"]
    X["column_6"] = [42.0] * len(X)
    y = pd.Series([1, 0, 1, 0, 1, 1, 0, 0, 0])
    X = infer_feature_types(
        X,
        {"column_1": "categorical", "column_5": "NaturalLanguage"},
    )

    component_graph = ComponentGraph(graph)
    # we don't have feature type selectors defined yet, so in order for the above graph to work we have to
    # specify the types to select here.
    component_graph.instantiate(
        {
            "Select numeric": {"columns": ["column_2"]},
            "Select categorical": {"columns": ["column_1", "column_3"]},
            "Select datetime": {"columns": ["column_4"]},
            "Select text": {"columns": ["column_5"]},
            "Select pass through": {"columns": ["column_6"]},
        },
    )
    assert component_graph.input_feature_names == {}
    component_graph.fit(X, y)

    input_feature_names = component_graph.input_feature_names
    assert input_feature_names["Random Forest"] == (
        [
            "column_2",
            "column_3",
            "column_1_a",
            "column_1_b",
            "column_1_c",
            "column_1_d",
            "column_4_year",
            "column_4_month",
            "column_4_day_of_week",
            "column_4_hour",
            "DIVERSITY_SCORE(column_5)",
            "MEAN_CHARACTERS_PER_WORD(column_5)",
            "NUM_CHARACTERS(column_5)",
            "NUM_WORDS(column_5)",
            "POLARITY_SCORE(column_5)",
            "LSA(column_5)[0]",
            "LSA(column_5)[1]",
            "column_6",
        ]
    )


def test_component_graph_get_inputs_with_sampler():
    graph = {
        "Imputer": [Imputer, "X", "y"],
        "OneHot": [OneHotEncoder, "Imputer.x", "y"],
        "Undersampler": [Undersampler, "OneHot.x", "y"],
        "Random Forest": [RandomForestClassifier, "Undersampler.x", "Undersampler.y"],
        "Elastic Net": [ElasticNetClassifier, "Undersampler.x", "Undersampler.y"],
        "Logistic Regression Classifier": [
            LogisticRegressionClassifier,
            "Random Forest.x",
            "Elastic Net.x",
            "Undersampler.y",
        ],
    }

    component_graph = ComponentGraph(graph)
    component_graph.instantiate()
    assert component_graph.get_inputs("Imputer") == ["X", "y"]
    assert component_graph.get_inputs("OneHot") == ["Imputer.x", "y"]
    assert component_graph.get_inputs("Undersampler") == ["OneHot.x", "y"]
    assert component_graph.get_inputs("Random Forest") == [
        "Undersampler.x",
        "Undersampler.y",
    ]
    assert component_graph.get_inputs("Elastic Net") == [
        "Undersampler.x",
        "Undersampler.y",
    ]
    assert component_graph.get_inputs("Logistic Regression Classifier") == [
        "Random Forest.x",
        "Elastic Net.x",
        "Undersampler.y",
    ]


def test_component_graph_dataset_with_target_imputer():
    X = pd.DataFrame(
        {
            "column_1": ["a", "b", "c", "d", "a", "a", "b", "c", "b"],
            "column_2": [1, 2, 3, 4, 5, 6, 5, 4, 3],
        },
    )
    y = pd.Series([1, 0, 1, 0, 1, 1, 0, 0, np.nan])
    X = infer_feature_types(X, {"column_1": "categorical"})
    graph = {
        "Target Imputer": [TargetImputer, "X", "y"],
        "OneHot": [OneHotEncoder, "Target Imputer.x", "Target Imputer.y"],
        "Random Forest": [RandomForestClassifier, "OneHot.x", "Target Imputer.y"],
        "Elastic Net": [ElasticNetClassifier, "OneHot.x", "Target Imputer.y"],
        "Logistic Regression Classifier": [
            LogisticRegressionClassifier,
            "Random Forest.x",
            "Elastic Net.x",
            "Target Imputer.y",
        ],
    }

    component_graph = ComponentGraph(graph)
    component_graph.instantiate()
    assert component_graph.get_inputs("Target Imputer") == ["X", "y"]
    assert component_graph.get_inputs("OneHot") == [
        "Target Imputer.x",
        "Target Imputer.y",
    ]
    assert component_graph.get_inputs("Random Forest") == [
        "OneHot.x",
        "Target Imputer.y",
    ]
    assert component_graph.get_inputs("Elastic Net") == [
        "OneHot.x",
        "Target Imputer.y",
    ]

    component_graph.fit(X, y)
    predictions = component_graph.predict(X)
    assert not pd.isnull(predictions).any()


@patch("evalml.pipelines.components.estimators.LogisticRegressionClassifier.fit")
def test_component_graph_sampler_y_passes(mock_estimator_fit):
    # makes sure the y value from oversampler gets passed to the estimator
    X = pd.DataFrame({"a": [i for i in range(100)], "b": [i % 3 for i in range(100)]})
    y = pd.Series([0] * 90 + [1] * 10)
    component_graph = {
        "Imputer": ["Imputer", "X", "y"],
        "Oversampler": ["Oversampler", "Imputer.x", "y"],
        "Standard Scaler": [
            "Standard Scaler",
            "Oversampler.x",
            "Oversampler.y",
        ],
        "Logistic Regression Classifier": [
            "Logistic Regression Classifier",
            "Standard Scaler.x",
            "Oversampler.y",
        ],
    }

    component_graph = ComponentGraph(component_graph)
    component_graph.instantiate()
    component_graph.fit(X, y)
    assert len(mock_estimator_fit.call_args[0][0]) == len(
        mock_estimator_fit.call_args[0][1],
    )
    assert len(mock_estimator_fit.call_args[0][0]) == int(1.25 * 90)


def test_component_graph_equality(example_graph):
    different_graph = {
        "Target Imputer": [TargetImputer, "X", "y"],
        "OneHot": [OneHotEncoder, "Target Imputer.x", "Target Imputer.y"],
        "Random Forest": [RandomForestClassifier, "OneHot.x", "Target Imputer.y"],
        "Elastic Net": [ElasticNetClassifier, "OneHot.x", "Target Imputer.y"],
        "Logistic Regression Classifier": [
            LogisticRegressionClassifier,
            "Random Forest.x",
            "Elastic Net.x",
            "Target Imputer.y",
        ],
    }

    same_graph_different_order = {
        "Imputer": [Imputer, "X", "y"],
        "OneHot_ElasticNet": [OneHotEncoder, "Imputer.x", "y"],
        "OneHot_RandomForest": [OneHotEncoder, "Imputer.x", "y"],
        "Random Forest": [RandomForestClassifier, "OneHot_RandomForest.x", "y"],
        "Elastic Net": [ElasticNetClassifier, "OneHot_ElasticNet.x", "y"],
        "Logistic Regression Classifier": [
            LogisticRegressionClassifier,
            "Random Forest.x",
            "Elastic Net.x",
            "y",
        ],
    }

    component_graph = ComponentGraph(example_graph, random_seed=0)
    component_graph_eq = ComponentGraph(example_graph, random_seed=0)
    component_graph_different_seed = ComponentGraph(example_graph, random_seed=5)
    component_graph_not_eq = ComponentGraph(different_graph, random_seed=0)
    component_graph_different_order = ComponentGraph(
        same_graph_different_order,
        random_seed=0,
    )

    component_graph.instantiate()
    component_graph_eq.instantiate()
    component_graph_different_seed.instantiate()
    component_graph_not_eq.instantiate()
    component_graph_different_order.instantiate()

    assert component_graph == component_graph
    assert component_graph == component_graph_eq

    assert component_graph != "not a component graph"
    assert component_graph != component_graph_different_seed
    assert component_graph != component_graph_not_eq
    assert component_graph != component_graph_different_order


def test_component_graph_equality_same_graph():
    # Same component nodes and edges, just specified in a different order in the input dictionary
    component_graph = ComponentGraph(
        {
            "Component B": [OneHotEncoder, "X", "y"],
            "Component A": [DateTimeFeaturizer, "Component B.x", "y"],
            "Random Forest": [
                RandomForestClassifier,
                "Component A.x",
                "Component B.x",
                "y",
            ],
        },
    )

    equal_component_graph = ComponentGraph(
        {
            "Component B": [OneHotEncoder, "X", "y"],
            "Component A": [DateTimeFeaturizer, "Component B.x", "y"],
            "Random Forest": [
                RandomForestClassifier,
                "Component B.x",
                "Component A.x",
                "y",
            ],
        },
    )
    component_graph == equal_component_graph


@pytest.mark.parametrize("return_dict", [True, False])
def test_describe_component_graph(return_dict, example_graph, caplog):
    component_graph = ComponentGraph(example_graph, random_seed=0)
    component_graph.instantiate()
    expected_component_graph_dict = {
        "Imputer": {
            "name": "Imputer",
            "parameters": {
                "categorical_impute_strategy": "most_frequent",
                "numeric_impute_strategy": "mean",
                "boolean_impute_strategy": "most_frequent",
                "categorical_fill_value": None,
                "numeric_fill_value": None,
                "boolean_fill_value": None,
            },
        },
        "One Hot Encoder": {
            "name": "One Hot Encoder",
            "parameters": {
                "top_n": 10,
                "features_to_encode": None,
                "categories": None,
                "drop": "if_binary",
                "handle_unknown": "ignore",
                "handle_missing": "error",
            },
        },
        "Random Forest Classifier": {
            "name": "Random Forest Classifier",
            "parameters": {"n_estimators": 100, "max_depth": 6, "n_jobs": -1},
        },
        "Elastic Net Classifier": {
            "name": "Elastic Net Classifier",
            "parameters": {
                "C": 1,
                "l1_ratio": 0.15,
                "n_jobs": -1,
                "solver": "saga",
                "penalty": "elasticnet",
                "multi_class": "auto",
            },
        },
        "Logistic Regression Classifier": {
            "name": "Logistic Regression Classifier",
            "parameters": {
                "penalty": "l2",
                "C": 1.0,
                "n_jobs": -1,
                "multi_class": "auto",
                "solver": "lbfgs",
            },
        },
    }
    component_graph_dict = component_graph.describe(return_dict=return_dict)
    if return_dict:
        assert component_graph_dict == expected_component_graph_dict
    else:
        assert component_graph_dict is None

    out = caplog.text
    for component in component_graph.component_instances.values():
        if component.hyperparameter_ranges:
            for parameter in component.hyperparameter_ranges:
                assert parameter in out
        assert component.name in out


def test_describe_component_graph_value_error(example_graph):
    cg_with_estimators = ComponentGraph(example_graph)
    with pytest.raises(ValueError):
        cg_with_estimators.describe()


class LogTransform(Transformer):
    name = "Log Transform"
    modifies_features = False
    modifies_target = True

    def __init__(self, parameters=None, random_seed=0):
        super().__init__(parameters={}, component_obj=None, random_seed=random_seed)

    def fit(self, X, y):
        return self

    def transform(self, X, y=None):
        if y is None:
            return X, y
        y = infer_feature_types(y)
        return X, infer_feature_types(np.log(y))

    def inverse_transform(self, y):
        y = infer_feature_types(y)
        return infer_feature_types(np.exp(y))


class DoubleTransform(Transformer):
    name = "Double Transform"
    modifies_features = False
    modifies_target = True

    def __init__(self, parameters=None, random_seed=0):
        super().__init__(parameters={}, component_obj=None, random_seed=random_seed)

    def fit(self, X, y):
        return self

    def transform(self, X, y=None):
        if y is None:
            return X, y
        y = infer_feature_types(y)
        return X, infer_feature_types(y * 2)

    def inverse_transform(self, y):
        y = infer_feature_types(y)
        return infer_feature_types(y / 2)


class SubsetData(Transformer):
    """To simulate a transformer that modifies the target but is not a target transformer, e.g. a sampler."""

    name = "Subset Data"
    modifies_target = True

    def __init__(self, parameters=None, random_seed=0):
        super().__init__(parameters={}, component_obj=None, random_seed=random_seed)

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X_new = X.iloc[:50]
        y_new = None
        if y is not None:
            y_new = y.iloc[:50]
        return X_new, y_new


@pytest.mark.parametrize(
    "component_graph,answer_func",
    [
        (
            ComponentGraph(
                {
                    "Imputer": [Imputer, "X", "y"],
                    "Log": [LogTransform, "X", "y"],
                    "Random Forest": ["Random Forest Regressor", "Imputer.x", "Log.y"],
                },
            ),
            lambda y: infer_feature_types(np.exp(y)),
        ),
        (
            ComponentGraph(
                {
                    "Imputer": [Imputer, "X", "y"],
                    "Log": [LogTransform, "X", "y"],
                    "Double": [DoubleTransform, "X", "Log.y"],
                    "Random Forest": [
                        "Random Forest Regressor",
                        "Imputer.x",
                        "Double.y",
                    ],
                },
            ),
            lambda y: infer_feature_types(np.exp(y / 2)),
        ),
        (
            ComponentGraph(
                {
                    "Imputer": [Imputer, "X", "y"],
                    "Log": [LogTransform, "Imputer.x", "y"],
                    "Double": [DoubleTransform, "Log.x", "Log.y"],
                    "Random Forest": [
                        "Random Forest Regressor",
                        "Double.x",
                        "Double.y",
                    ],
                },
            ),
            lambda y: infer_feature_types(np.exp(y / 2)),
        ),
        (
            ComponentGraph(
                {
                    "Imputer": [Imputer, "X", "y"],
                    "OneHot": [OneHotEncoder, "Imputer.x", "y"],
                    "DateTime": [DateTimeFeaturizer, "OneHot.x", "y"],
                    "Log": [LogTransform, "X", "y"],
                    "Double": [DoubleTransform, "DateTime.x", "Log.y"],
                    "Random Forest": [
                        "Random Forest Regressor",
                        "DateTime.x",
                        "Double.y",
                    ],
                },
            ),
            lambda y: infer_feature_types(np.exp(y / 2)),
        ),
        (
            ComponentGraph(
                {
                    "Imputer": [Imputer, "X", "y"],
                    "OneHot": [OneHotEncoder, "Imputer.x", "y"],
                    "DateTime": [DateTimeFeaturizer, "OneHot.x", "y"],
                    "Log": [LogTransform, "X", "y"],
                    "Double": [DoubleTransform, "X", "Log.y"],
                    "Double2": [DoubleTransform, "X", "Double.y"],
                    "Random Forest": [
                        "Random Forest Regressor",
                        "DateTime.x",
                        "Double2.y",
                    ],
                },
            ),
            lambda y: infer_feature_types(np.exp(y / 4)),
        ),
        (
            ComponentGraph(
                {
                    "Imputer": ["Imputer", "X", "y"],
                    "Double": [DoubleTransform, "X", "y"],
                    "DateTime 1": [
                        "DateTime Featurizer",
                        "Imputer.x",
                        "y",
                    ],
                    "ET": ["Extra Trees Regressor", "DateTime 1.x", "Double.y"],
                    "Double 2": [DoubleTransform, "X", "y"],
                    "DateTime 2": [
                        "DateTime Featurizer",
                        "Imputer.x",
                        "y",
                    ],
                    "Double 3": [DoubleTransform, "X", "Double 2.y"],
                    "RandomForest": [
                        "Random Forest Regressor",
                        "DateTime 2.x",
                        "Double 3.y",
                    ],
                    "DateTime 3": [
                        "DateTime Featurizer",
                        "Imputer.x",
                        "y",
                    ],
                    "Double 4": [DoubleTransform, "X", "y"],
                    "Catboost": [
                        "Random Forest Regressor",
                        "DateTime 3.x",
                        "Double 4.y",
                    ],
                    "Logistic Regression Classifier": [
                        "Linear Regressor",
                        "Catboost.x",
                        "RandomForest.x",
                        "ET.x",
                        "Double 3.y",
                    ],
                },
            ),
            lambda y: infer_feature_types(y / 4),
        ),
        (
            ComponentGraph(
                {
                    "Imputer": [Imputer, "X", "y"],
                    "OneHot": [OneHotEncoder, "Imputer.x", "y"],
                    "DateTime": [DateTimeFeaturizer, "OneHot.x", "y"],
                    "Log": [LogTransform, "X", "y"],
                    "Double": [DoubleTransform, "X", "Log.y"],
                    "Double2": [DoubleTransform, "X", "Double.y"],
                    "Subset": [SubsetData, "DateTime.x", "Double2.y"],
                    "Random Forest": [
                        "Random Forest Regressor",
                        "Subset.x",
                        "Subset.y",
                    ],
                },
            ),
            lambda y: infer_feature_types(np.exp(y / 4)),
        ),
        (
            ComponentGraph(
                {
                    "Imputer": [Imputer, "X", "y"],
                    "Random Forest": ["Random Forest Regressor", "Imputer.x", "y"],
                },
            ),
            lambda y: y,
        ),
        (
            ComponentGraph(
                {
                    "Imputer": [Imputer, "X", "y"],
                    "DateTime": [DateTimeFeaturizer, "Imputer.x", "y"],
                    "OneHot": [OneHotEncoder, "DateTime.x", "y"],
                    "Random Forest": ["Random Forest Regressor", "OneHot.x", "y"],
                },
            ),
            lambda y: y,
        ),
        (
            ComponentGraph({"Random Forest": ["Random Forest Regressor", "X", "y"]}),
            lambda y: y,
        ),
        (
            ComponentGraph(
                {
                    "Imputer": ["Imputer", "X", "y"],
                    "Double": [DoubleTransform, "X", "y"],
                    "DateTime 1": [
                        "DateTime Featurizer",
                        "Imputer.x",
                        "y",
                    ],
                    "ET": ["Extra Trees Regressor", "DateTime 1.x", "Double.y"],
                    "Double 2": [DoubleTransform, "X", "y"],
                    "DateTime 2": [
                        "DateTime Featurizer",
                        "Imputer.x",
                        "y",
                    ],
                    "Double 3": [DoubleTransform, "X", "Double 2.y"],
                    "RandomForest": [
                        "Random Forest Regressor",
                        "DateTime 2.x",
                        "Double 3.y",
                    ],
                    "DateTime 3": [
                        "DateTime Featurizer",
                        "Imputer.x",
                        "y",
                    ],
                    "Double 4": [DoubleTransform, "X", "y"],
                    "Linear": ["Linear Regressor", "DateTime 3.x", "Double 4.y"],
                    "Logistic Regression Classifier": [
                        "Linear Regressor",
                        "Linear.x",
                        "RandomForest.x",
                        "ET.x",
                        "y",
                    ],
                },
            ),
            lambda y: y,
        ),
    ],
)
def test_component_graph_inverse_transform(
    component_graph,
    answer_func,
    X_y_regression,
):
    X, y = X_y_regression
    y = pd.Series(np.abs(y))
    X = pd.DataFrame(X)
    component_graph.instantiate()
    component_graph.fit(X, y)
    predictions = component_graph.predict(X)
    answer = component_graph.inverse_transform(predictions)
    expected = answer_func(predictions)
    pd.testing.assert_series_equal(answer, expected)


def test_final_component_features_does_not_have_target():
    X = pd.DataFrame(
        {
            "column_1": ["a", "b", "c", "d", "a", "a", "b", "c", "b"],
            "column_2": [1, 2, 3, 4, 5, 6, 5, 4, 3],
        },
    )
    y = pd.Series([1, 0, 1, 0, 1, 1, 0, 0, 0])
    X.ww.init(logical_types={"column_1": "categorical"})

    cg = ComponentGraph(
        {
            "Imputer": ["Imputer", "X", "y"],
            "OneHot": ["One Hot Encoder", "Imputer.x", "y"],
            "TargetImputer": ["Target Imputer", "OneHot.x", "y"],
            "Logistic Regression Classifier": [
                "Logistic Regression Classifier",
                "TargetImputer.x",
                "TargetImputer.y",
            ],
        },
    )
    cg.instantiate()
    cg.fit(X, y)

    final_features = cg.transform_all_but_final(X, y)
    assert "TargetImputer.y" not in final_features.columns


@patch("evalml.pipelines.components.Imputer.fit_transform")
def test_component_graph_with_X_y_inputs_X(mock_fit):
    class DummyColumnNameTransformer(Transformer):
        name = "Dummy Column Name Transform"

        def __init__(self, parameters=None, random_seed=0):
            super().__init__(parameters={}, component_obj=None, random_seed=random_seed)

        def fit(self, X, y):
            return self

        def transform(self, X, y=None):
            return X.rename(columns=lambda x: x + "_new", inplace=False)

    X = pd.DataFrame(
        {
            "column_1": [0, 2, 3, 1, 5, 6, 5, 4, 3],
            "column_2": [1, 2, 3, 4, 5, 6, 5, 4, 3],
        },
    )

    y = pd.Series([1, 0, 1, 0, 1, 1, 0, 0, 0])
    graph = {
        "DummyColumnNameTransformer": [DummyColumnNameTransformer, "X", "y"],
        "Imputer": ["Imputer", "DummyColumnNameTransformer.x", "X", "y"],
        "Random Forest": ["Random Forest Classifier", "Imputer.x", "y"],
    }

    component_graph = ComponentGraph(graph)
    component_graph.instantiate()
    mock_fit.return_value = X
    assert component_graph.get_inputs("DummyColumnNameTransformer") == ["X", "y"]
    assert component_graph.get_inputs("Imputer") == [
        "DummyColumnNameTransformer.x",
        "X",
        "y",
    ]

    component_graph.fit(X, y)

    # Check that we have columns from both the output of DummyColumnNameTransformer as well as the original columns since "X" was specified
    assert list(mock_fit.call_args[0][0].columns) == [
        "column_1_new",
        "column_2_new",
        "column_1",
        "column_2",
    ]


@patch("evalml.pipelines.components.Imputer.fit_transform")
@patch("evalml.pipelines.components.Estimator.fit")
def test_component_graph_with_X_y_inputs_y(mock_fit, mock_fit_transform):
    X = pd.DataFrame(
        {
            "column_1": [0, 2, 3, 1, 5, 6, 5, 4, 3],
            "column_2": [1, 2, 3, 4, 5, 6, 5, 4, 3],
        },
    )
    y = pd.Series([1, 0, 1, 0, 1, 1, 0, 0, 0])
    graph = {
        "Log": [LogTransform, "X", "y"],
        "Imputer": ["Imputer", "Log.x", "y"],
        "Random Forest": ["Random Forest Classifier", "Imputer.x", "Log.y"],
    }
    mock_fit_transform.return_value = X
    component_graph = ComponentGraph(graph)
    component_graph.instantiate()
    assert component_graph.get_inputs("Log") == ["X", "y"]
    assert component_graph.get_inputs("Imputer") == ["Log.x", "y"]
    assert component_graph.get_inputs("Random Forest") == ["Imputer.x", "Log.y"]

    component_graph.fit(X, y)
    # Check that we use "y" for Imputer, not "Log.y"
    assert_series_equal(mock_fit_transform.call_args[0][1], y)
    # Check that we use "Log.y" for RF
    assert_series_equal(mock_fit.call_args[0][1], infer_feature_types(np.log(y)))


def test_component_graph_does_not_define_all_edges():
    # Graph does not define an X edge
    with pytest.raises(
        ValueError,
        match="All components must have at least one input feature",
    ):
        ComponentGraph(
            {
                "Imputer": [Imputer, "y"],  # offending line
                "One Hot Encoder": [OneHotEncoder, "Imputer.x", "y"],
                "Target Imputer": [TargetImputer, "One Hot Encoder.x", "y"],
                "Random Forest Classifier": [
                    RandomForestClassifier,
                    "One Hot Encoder.x",
                    "Target Imputer.y",
                ],
            },
        )
    # Graph does not define a y edge
    with pytest.raises(ValueError, match="All components must have exactly one target"):
        ComponentGraph(
            {
                "Imputer": [Imputer, "X"],  # offending line
                "One Hot Encoder": [OneHotEncoder, "Imputer.x", "y"],
                "Target Imputer": [TargetImputer, "One Hot Encoder.x", "y"],
                "Random Forest Classifier": [
                    RandomForestClassifier,
                    "One Hot Encoder.x",
                    "Target Imputer.y",
                ],
            },
        )
    # Graph does not define X and y edges
    with pytest.raises(
        ValueError,
        match="All components must have at least one input feature",
    ):
        ComponentGraph(
            {
                "Imputer": [Imputer],  # offending line
                "One Hot Encoder": [OneHotEncoder, "Imputer.x", "y"],
                "Target Imputer": [TargetImputer, "One Hot Encoder.x", "y"],
                "Random Forest Classifier": [
                    RandomForestClassifier,
                    "One Hot Encoder.x",
                    "Target Imputer.y",
                ],
            },
        )


def test_component_graph_defines_edges_with_bad_syntax():
    # Graph does not define an X edge
    with pytest.raises(
        ValueError,
        match="All edges must be specified as either an input feature",
    ):
        ComponentGraph(
            {
                "Imputer": [Imputer, "X", "y"],
                "One Hot Encoder": [OneHotEncoder, "Imputer.x", "y"],
                "Target Imputer": [
                    TargetImputer,
                    "Imputer",  # offending line: "Imputer" not allowed
                    "One Hot Encoder.x",
                    "y",
                ],
                "Random Forest Classifier": [
                    RandomForestClassifier,
                    "One Hot Encoder.x",
                    "Target Imputer.y",
                ],
            },
        )


def test_component_graph_defines_edge_with_invalid_syntax():
    # Graph does not define an X edge using .x
    with pytest.raises(
        ValueError,
        match="All components must have at least one input feature",
    ):
        ComponentGraph(
            {
                "Imputer": [Imputer, "X", "y"],
                "One Hot Encoder": [OneHotEncoder, "Imputer", "y"],  # offending line
                "Target Imputer": [TargetImputer, "One Hot Encoder.x", "y"],
                "Random Forest Classifier": [
                    RandomForestClassifier,
                    "One Hot Encoder.x",
                    "Target Imputer.y",
                ],
            },
        )


@pytest.mark.parametrize(
    "pipeline_parameters,set_values",
    [
        ({"Logistic Regression Classifier": {"penalty": "l1"}}, {}),
        (
            {"Logistic Regression": {"penalty": "l1"}},
            {"Logistic Regression"},
        ),
        (
            {"Random Forest Classifier": {"n_estimators": 10}},
            {"Random Forest Classifier"},
        ),
        (
            {
                "Imputer": {"numeric_impute_strategy": "mean"},
                "Random Forest Classifier": {"n_estimators": 10},
            },
            {"Random Forest Classifier"},
        ),
        (
            {
                "Undersampler": {"sampling_ratio": 0.05},
                "Random Forest Classifier": {"n_estimators": 10},
            },
            {"Random Forest Classifier", "Undersampler"},
        ),
    ],
)
def test_component_graph_instantiate_parameters(pipeline_parameters, set_values):
    graph = {
        "Imputer": ["Imputer", "X", "y"],
        "Scaler": ["Standard Scaler", "Imputer.x", "y"],
        "Logistic Regression Classifier": [
            "Logistic Regression Classifier",
            "Scaler.x",
            "y",
        ],
    }
    component_graph = ComponentGraph(graph)
    with warnings.catch_warnings(record=True) as w:
        warnings.filterwarnings(
            "always",
            category=ParameterNotUsedWarning,
        )
        component_graph.instantiate(pipeline_parameters)
    assert len(w) == (1 if len(set_values) else 0)
    if len(w):
        assert w[0].message.components == set_values


def test_component_graph_repr():
    # Test with component graph defined by strings
    component_dict = {
        "Imputer": ["Imputer", "X", "y"],
        "OHE": ["One Hot Encoder", "Imputer.x", "y"],
        "Random Forest Regressor": ["Random Forest Regressor", "OHE.x", "y"],
    }
    expected_repr = "{'Imputer': ['Imputer', 'X', 'y'], 'OHE': ['One Hot Encoder', 'Imputer.x', 'y'], 'Random Forest Regressor': ['Random Forest Regressor', 'OHE.x', 'y']}"
    component_graph = ComponentGraph(component_dict)
    assert repr(component_graph) == expected_repr

    # Test with component graph defined by strings and objects
    component_dict_with_objs = {
        "Imputer": [Imputer, "X", "y"],
        "OHE": [OneHotEncoder, "Imputer.x", "y"],
        "Random Forest Classifier": [RandomForestClassifier, "OHE.x", "y"],
    }
    expected_repr = "{'Imputer': ['Imputer', 'X', 'y'], 'OHE': ['One Hot Encoder', 'Imputer.x', 'y'], 'Random Forest Classifier': ['Random Forest Classifier', 'OHE.x', 'y']}"
    component_graph = ComponentGraph(component_dict_with_objs)
    assert repr(component_graph) == expected_repr


@patch("evalml.pipelines.components.estimators.LogisticRegressionClassifier.fit")
@pytest.mark.parametrize("sampler", ["Undersampler", "Oversampler"])
def test_component_graph_transform_all_but_final_with_sampler(
    mock_estimator_fit,
    sampler,
):
    expected_length = 750 if sampler == "Undersampler" else int(1.25 * 850)
    X = pd.DataFrame([[i] for i in range(1000)])
    y = pd.Series([0] * 150 + [1] * 850)
    component_graph = {
        sampler: [sampler, "X", "y"],
        "Logistic Regression Classifier": [
            "Logistic Regression Classifier",
            f"{sampler}.x",
            f"{sampler}.y",
        ],
    }

    component_graph = ComponentGraph(component_graph)
    component_graph.instantiate()
    component_graph.fit(X, y)
    assert len(mock_estimator_fit.call_args[0][0]) == len(
        mock_estimator_fit.call_args[0][1],
    )
    assert len(mock_estimator_fit.call_args[0][0]) == expected_length
    features_for_estimator = component_graph.transform_all_but_final(X, y)
    assert len(features_for_estimator) == len(y)


@patch("evalml.pipelines.components.Imputer.transform")
@patch("evalml.pipelines.components.OneHotEncoder.transform")
@patch("evalml.pipelines.components.RandomForestClassifier.predict")
def test_component_graph_transform(
    mock_rf_predict,
    mock_ohe_transform,
    mock_imputer_transform,
    X_y_binary,
    make_data_type,
):
    X, y = X_y_binary

    X = make_data_type("ww", X)
    y = make_data_type("ww", y)

    dummy_return_value = pd.DataFrame({"test df": [1, 2]})
    mock_imputer_transform.return_value = X
    mock_ohe_transform.return_value = dummy_return_value
    component_dict = {
        "Imputer": ["Imputer", "X", "y"],
        "OHE": ["One Hot Encoder", "Imputer.x", "y"],
    }

    mock_rf_predict.return_value = y

    component_graph = ComponentGraph(component_dict)
    component_graph.instantiate()
    component_graph.fit(X, y)
    transformed_X = component_graph.transform(X, y)
    assert_frame_equal(transformed_X, dummy_return_value)

    component_dict_with_estimator = {
        "Imputer": ["Imputer", "X", "y"],
        "Random Forest Classifier": ["Random Forest Classifier", "Imputer.x", "y"],
        "OHE": ["One Hot Encoder", "Random Forest Classifier.x", "y"],
    }
    component_graph = ComponentGraph(component_dict_with_estimator)
    component_graph.instantiate()
    component_graph.fit(X, y)
    transformed_X = component_graph.transform(X, y)
    assert_frame_equal(transformed_X, dummy_return_value)


@patch("evalml.pipelines.components.Imputer.transform")
@patch("evalml.pipelines.components.OneHotEncoder.transform")
@patch("evalml.pipelines.components.TargetImputer.transform")
def test_component_graph_transform_with_target_transformer(
    mock_target_imputer_transform,
    mock_ohe_transform,
    mock_imputer_transform,
    X_y_binary,
    make_data_type,
):
    X, y = X_y_binary
    X = make_data_type("ww", X)
    y = make_data_type("ww", y)

    component_dict = {
        "Imputer": ["Imputer", "X", "y"],
        "OHE": ["One Hot Encoder", "Imputer.x", "y"],
        "Target Imputer": ["Target Imputer", "OHE.x", "y"],
    }
    mock_imputer_transform.return_value = X
    mock_ohe_transform.return_value = X
    mock_target_imputer_transform.return_value = tuple([X, y])

    component_graph = ComponentGraph(component_dict)
    component_graph.instantiate()
    component_graph.fit(X, y)
    transformed = component_graph.transform(X, y)
    assert_frame_equal(transformed[0], X)
    assert_series_equal(transformed[1], y)


def test_component_graph_transform_with_estimator_end(X_y_binary):
    X, y = X_y_binary
    component_dict = {
        "Imputer": ["Imputer", "X", "y"],
        "OHE": ["One Hot Encoder", "Imputer.x", "y"],
        "RF": ["Random Forest Classifier", "OHE.x", "y"],
    }
    component_graph = ComponentGraph(component_dict)
    component_graph.instantiate()
    component_graph.fit(X, y)
    with pytest.raises(
        ValueError,
        match=re.escape(
            "Cannot call transform() on a component graph because the final component is not a Transformer.",
        ),
    ):
        component_graph.transform(X, y)


def test_component_graph_predict_with_transformer_end(X_y_binary):
    X, y = X_y_binary
    component_dict = {
        "Imputer": ["Imputer", "X", "y"],
        "OHE": ["One Hot Encoder", "Imputer.x", "y"],
    }
    component_graph = ComponentGraph(component_dict)
    component_graph.instantiate()
    component_graph.fit(X, y)
    with pytest.raises(
        ValueError,
        match=re.escape(
            "Cannot call predict() on a component graph because the final component is not an Estimator.",
        ),
    ):
        component_graph.predict(X)


def test_component_graph_with_invalid_y_edge(X_y_binary):
    X, y = X_y_binary
    component_dict = {
        "OHE": ["One Hot Encoder", "X", "y"],
        "RF": ["Random Forest Classifier", "OHE.x", "OHE.y"],
    }
    with pytest.raises(ValueError, match="OHE.y is not a valid input edge"):
        ComponentGraph(component_dict)


def test_training_only_component_in_component_graph_fit_and_transform_all_but_final(
    X_y_binary,
):
    # Test that calling fit_and_transform_all_but_final() will evaluate all training-only transformations
    X, y = X_y_binary
    component_dict = {
        "Imputer": ["Imputer", "X", "y"],
        "Drop Rows Transformer": [DropRowsTransformer, "Imputer.x", "y"],
        "RF": [
            "Random Forest Classifier",
            "Drop Rows Transformer.x",
            "Drop Rows Transformer.y",
        ],
    }
    component_graph = ComponentGraph(component_dict)
    component_graph.instantiate({"Drop Rows Transformer": {"indices_to_drop": [0, 9]}})
    transformed_X, transformed_y = component_graph.fit_and_transform_all_but_final(X, y)
    assert len(transformed_X) == len(X) - 2


def test_training_only_component_in_component_graph_transform_all_but_final(
    X_y_binary,
):
    # Test that calling transform_all_but_final() will not evaluate all training-only transformations
    X, y = X_y_binary
    component_dict = {
        "Imputer": ["Imputer", "X", "y"],
        "Drop Rows Transformer": [DropRowsTransformer, "Imputer.x", "y"],
        "RF": [
            "Random Forest Classifier",
            "Drop Rows Transformer.x",
            "Drop Rows Transformer.y",
        ],
    }
    component_graph = ComponentGraph(component_dict)
    component_graph.instantiate({"Drop Rows Transformer": {"indices_to_drop": [0, 9]}})
    component_graph.fit(X, y)
    transformed_X = component_graph.transform_all_but_final(X, y)
    assert len(transformed_X) == len(X)


@pytest.mark.parametrize("problem_type", ["binary", "multiclass", "regression"])
def test_fit_predict_different_types(
    problem_type,
    X_y_binary,
    X_y_multi,
    X_y_regression,
):
    if problem_type == "binary":
        X, y = X_y_binary
    elif problem_type == "multiclass":
        X, y = X_y_multi
    else:
        X, y = X_y_regression

    X = infer_feature_types(X)
    X.ww.set_types({0: "Double"})
    X2 = infer_feature_types(X.copy())
    X2.ww.set_types({0: "Categorical"})
    if is_classification(problem_type):
        component_dict = {
            "Imputer": ["Imputer", "X", "y"],
            "RF": [
                "Random Forest Classifier",
                "Imputer.x",
                "y",
            ],
        }
    else:
        component_dict = {
            "Imputer": ["Imputer", "X", "y"],
            "RF": [
                "Random Forest Regressor",
                "Imputer.x",
                "y",
            ],
        }
    component_graph = ComponentGraph(component_dict).instantiate({})
    component_graph.fit(X, y)
    with pytest.raises(
        PipelineError,
        match="Input X data types are different from the input types",
    ) as e:
        component_graph.predict(X2)
    assert e.value.code == PipelineErrorCodeEnum.PREDICT_INPUT_SCHEMA_UNEQUAL
    assert e.value.details["input_features_types"] is not None
    assert e.value.details["pipeline_features_types"] is not None


def test_fit_transform_different_types(X_y_binary):
    X, y = X_y_binary
    X = infer_feature_types(X)
    X.ww.set_types({0: "Double"})
    X2 = infer_feature_types(X.copy())
    X2.ww.set_types({0: "Categorical"})
    component_dict = {"Imputer": ["Imputer", "X", "y"]}
    component_graph = ComponentGraph(component_dict).instantiate({})
    component_graph.fit(X, y)
    with pytest.raises(
        PipelineError,
        match="Input X data types are different from the input types",
    ) as e:
        component_graph.transform(X2)
    assert e.value.code == PipelineErrorCodeEnum.PREDICT_INPUT_SCHEMA_UNEQUAL
    assert e.value.details["input_features_types"] is not None
    assert e.value.details["pipeline_features_types"] is not None


def test_component_graph_cache():
    X1 = pd.DataFrame({"a": [1, 2, 1, 0, 2, 2, 2], "b": [3, 2, 3, 1, 2, 1, 2]})
    X2 = pd.DataFrame({"a": [0, 2, 1, 0, 2, 0, 2], "b": [1, 0, 2, 1, 2, 3, 3]})
    y = pd.Series([0, 0, 1, 0, 1, 1, 0])
    comp = LogisticRegressionClassifier()
    comp.fit(X1, y)
    preds = comp.predict(X1)
    preds2 = comp.predict(X2)
    # define the cache
    hashes1 = hash(tuple(X1.index))
    hashes2 = hash(tuple(X2.index))
    # use the same component for both hashes
    # allows us to determine whether we are using the cached data
    cache = {
        hashes1: {"Logistic Regression Classifier": comp},
        hashes2: {"Logistic Regression Classifier": comp},
    }

    component_graph = {
        "Logistic Regression Classifier": ["Logistic Regression Classifier", "X", "y"],
    }
    cg_cache = ComponentGraph(component_graph, cached_data=cache).instantiate()
    cg_no_cache = ComponentGraph(component_graph).instantiate()

    cg_cache.fit(X1, y)
    preds_cg = cg_cache.predict(X1)
    pd.testing.assert_series_equal(preds, preds_cg)
    cg_no_cache.fit(X1, y)
    preds_cg = cg_no_cache.predict(X1)
    pd.testing.assert_series_equal(preds, preds_cg)

    # expect the same hashed component to be called
    cg_cache.fit(X2, y)
    preds_cg = cg_cache.predict(X2)
    pd.testing.assert_series_equal(preds2, preds_cg)

    # expect this is not the same as without cache
    cg_no_cache = ComponentGraph(component_graph).instantiate()
    cg_no_cache.fit(X2, y)
    preds_cg_no = cg_no_cache.predict(X2)
    assert not (preds2 == preds_cg_no).all()


@patch("evalml.pipelines.components.transformers.preprocessing.featuretools.dfs")
@patch(
    "evalml.pipelines.components.transformers.preprocessing.featuretools.calculate_feature_matrix",
)
def test_component_graph_handles_engineered_features(
    mock_calculate_feature_matrix,
    mock_dfs,
):
    X, y = load_diabetes()
    del X.ww

    X_fit = X.iloc[: X.shape[0] // 2]
    X_fit = X_fit.reset_index()
    X_fit.ww.init(name="X", index="index")

    es = ft.EntitySet()
    es = es.add_dataframe(
        dataframe_name="X",
        dataframe=X_fit,
        index="index",
        make_index=True,
    )
    feature_matrix, _ = ft.dfs(
        entityset=es,
        target_dataframe_name="X",
        trans_primitives=["absolute"],
    )

    graph = {"DFS Transformer": ["DFS Transformer", "X", "y"]}
    component_graph = ComponentGraph(graph)
    component_graph.instantiate()
    component_graph.fit(feature_matrix, y)

    base_features = [
        c
        for c in X_fit.ww.columns
        if X_fit.ww[c].ww.origin == "base" or X_fit.ww[c].ww.origin is None
    ]
    feature_matrix_base_only = X_fit.ww[base_features]
    feature_matrix_base_only = feature_matrix_base_only.ww.drop("index")

    component_graph.transform(X_fit.drop("index", axis=1))
    assert (
        component_graph._input_types.columns.keys()
        == feature_matrix_base_only.ww.schema.columns.keys()
    )


def test_get_component_input_logical_types():

    X = pd.DataFrame(
        {
            "cat": pd.Series(["a"] * 50 + ["b"] * 50 + ["c"] * 50),
            "numeric": pd.Series(range(150)),
            "email": pd.Series(["foo@gmail.com"] * 50 + ["bar@yahoo.com"] * 100),
        },
    )
    y = pd.Series(range(-300, -150))
    X.ww.init(logical_types={"email": "EmailAddress", "numeric": "Integer"})

    graph1 = ComponentGraph(
        component_dict={
            "Email": ["Email Featurizer", "X", "y"],
            "OHE": ["One Hot Encoder", "Email.x", "y"],
            "RF": [
                "Random Forest Classifier",
                "OHE.x",
                "y",
            ],
        },
    )
    graph1.instantiate()
    graph1.fit(X, y)
    assert graph1.get_component_input_logical_types("OHE") == {
        "numeric": Integer(),
        "cat": Categorical(),
        "EMAIL_ADDRESS_TO_DOMAIN(email)": Categorical(),
        "IS_FREE_EMAIL_DOMAIN(email)": Categorical(),
    }
    assert graph1.last_component_input_logical_types == {
        "numeric": Integer(),
        "cat_a": Boolean(),
        "cat_b": Boolean(),
        "cat_c": Boolean(),
        "EMAIL_ADDRESS_TO_DOMAIN(email)_gmail.com": Boolean(),
        "IS_FREE_EMAIL_DOMAIN(email)_True": Boolean(),
    }

    ensemble = ComponentGraph(
        component_dict={
            "Email": ["Email Featurizer", "X", "y"],
            "OHE": ["One Hot Encoder", "Email.x", "y"],
            "RF": [
                "Random Forest Regressor",
                "OHE.x",
                "y",
            ],
            "Email Catboost": ["Email Featurizer", "X", "y"],
            "Catboost": [
                "CatBoost Regressor",
                "Email Catboost.x",
                "y",
            ],
            "Estimator": ["Elastic Net Regressor", "Catboost.x", "RF.x", "y"],
        },
    )
    ensemble.instantiate()
    ensemble.fit(X, y)

    assert ensemble.last_component_input_logical_types == {
        "Catboost.x": Double(),
        "RF.x": Double(),
    }
    assert ensemble.get_component_input_logical_types("Catboost") == {
        "cat": Categorical(),
        "numeric": Integer(),
        "EMAIL_ADDRESS_TO_DOMAIN(email)": Categorical(),
        "IS_FREE_EMAIL_DOMAIN(email)": Categorical(),
    }

    no_estimator = ComponentGraph(
        {
            "Imputer": ["Imputer", "X", "y"],
            "OHE": ["One Hot Encoder", "Imputer.x", "y"],
        },
    )
    no_estimator.instantiate()

    with pytest.raises(ValueError, match="has not been fit"):
        _ = no_estimator.last_component_input_logical_types

    no_estimator.fit(X, y)

    with pytest.raises(ValueError, match="not in the graph"):
        _ = no_estimator.get_component_input_logical_types("Catboost")

    no_estimator.fit(X, y)
    assert no_estimator.last_component_input_logical_types == {
        "cat": Categorical(),
        "numeric": Double(),
        "email": EmailAddress(),
    }
