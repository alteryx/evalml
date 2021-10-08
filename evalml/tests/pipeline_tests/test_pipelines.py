import os
import re
from unittest.mock import patch

import cloudpickle
import numpy as np
import pandas as pd
import pytest
import woodwork as ww
from pandas.testing import assert_frame_equal
from skopt.space import Categorical, Integer

from evalml.exceptions import (
    MissingComponentError,
    ObjectiveCreationError,
    ObjectiveNotFoundError,
    PipelineNotYetFittedError,
    PipelineScoreError,
)
from evalml.model_family import ModelFamily
from evalml.objectives import (
    CostBenefitMatrix,
    FraudCost,
    Precision,
    get_objective,
)
from evalml.pipelines import (
    BinaryClassificationPipeline,
    MulticlassClassificationPipeline,
    PipelineBase,
    RegressionPipeline,
)
from evalml.pipelines.component_graph import ComponentGraph
from evalml.pipelines.components import (
    DropNullColumns,
    DropRowsTransformer,
    ElasticNetClassifier,
    Imputer,
    LogisticRegressionClassifier,
    OneHotEncoder,
    RandomForestClassifier,
    RandomForestRegressor,
    StandardScaler,
    TargetImputer,
    Transformer,
    Undersampler,
)
from evalml.pipelines.components.utils import (
    _all_estimators_used_in_search,
    allowed_model_families,
)
from evalml.pipelines.utils import _get_pipeline_base_class
from evalml.preprocessing.utils import is_classification
from evalml.problem_types import (
    ProblemTypes,
    is_binary,
    is_multiclass,
    is_time_series,
)


@pytest.mark.parametrize(
    "pipeline_class",
    [
        BinaryClassificationPipeline,
        MulticlassClassificationPipeline,
        RegressionPipeline,
    ],
)
def test_init_with_invalid_type_raises_error(pipeline_class):
    with pytest.raises(
        ValueError,
        match="component_graph must be a list, dict, or ComponentGraph object",
    ):
        pipeline_class(component_graph="this is not a valid component graph")


@pytest.mark.parametrize(
    "pipeline_class",
    [
        BinaryClassificationPipeline,
        MulticlassClassificationPipeline,
        RegressionPipeline,
    ],
)
def test_init_list_with_component_that_is_not_supported_by_list_API(pipeline_class):
    assert not TargetImputer._supported_by_list_API
    with pytest.raises(
        ValueError,
        match=f"{TargetImputer.name} cannot be defined in a list because edges may be ambiguous",
    ):
        pipeline_class(component_graph=["Target Imputer"])


def test_allowed_model_families(has_minimal_dependencies):
    families = [
        ModelFamily.RANDOM_FOREST,
        ModelFamily.LINEAR_MODEL,
        ModelFamily.EXTRA_TREES,
        ModelFamily.DECISION_TREE,
    ]
    expected_model_families_binary = set(families)
    expected_model_families_regression = set(families)
    if not has_minimal_dependencies:
        expected_model_families_binary.update(
            [ModelFamily.XGBOOST, ModelFamily.CATBOOST, ModelFamily.LIGHTGBM]
        )
        expected_model_families_regression.update(
            [ModelFamily.CATBOOST, ModelFamily.XGBOOST, ModelFamily.LIGHTGBM]
        )
    assert (
        set(allowed_model_families(ProblemTypes.BINARY))
        == expected_model_families_binary
    )
    assert (
        set(allowed_model_families(ProblemTypes.REGRESSION))
        == expected_model_families_regression
    )


def test_all_estimators(
    has_minimal_dependencies,
    is_running_py_39_or_above,
    is_using_conda,
    is_using_windows,
):
    if has_minimal_dependencies:
        assert len((_all_estimators_used_in_search())) == 9
    else:
        if is_using_conda:
            n_estimators = 15
        else:
            n_estimators = 16 if is_running_py_39_or_above or is_using_windows else 17
        assert len(_all_estimators_used_in_search()) == n_estimators


def test_required_fields():
    class TestPipelineWithoutComponentGraph(PipelineBase):
        pass

    with pytest.raises(TypeError):
        TestPipelineWithoutComponentGraph(parameters={})


def test_serialization(X_y_binary, tmpdir, logistic_regression_binary_pipeline_class):
    X, y = X_y_binary
    path = os.path.join(str(tmpdir), "pipe.pkl")
    pipeline = logistic_regression_binary_pipeline_class(
        parameters={"Logistic Regression Classifier": {"n_jobs": 1}}
    )
    pipeline.fit(X, y)
    pipeline.save(path)
    assert pipeline.score(X, y, ["precision"]) == PipelineBase.load(path).score(
        X, y, ["precision"]
    )


@patch("cloudpickle.dump")
def test_serialization_protocol(
    mock_cloudpickle_dump, tmpdir, logistic_regression_binary_pipeline_class
):
    path = os.path.join(str(tmpdir), "pipe.pkl")
    pipeline = logistic_regression_binary_pipeline_class(parameters={})

    pipeline.save(path)
    assert len(mock_cloudpickle_dump.call_args_list) == 1
    assert (
        mock_cloudpickle_dump.call_args_list[0][1]["protocol"]
        == cloudpickle.DEFAULT_PROTOCOL
    )

    mock_cloudpickle_dump.reset_mock()

    pipeline.save(path, pickle_protocol=42)
    assert len(mock_cloudpickle_dump.call_args_list) == 1
    assert mock_cloudpickle_dump.call_args_list[0][1]["protocol"] == 42


@pytest.fixture
def pickled_pipeline_path(
    X_y_binary, tmpdir, logistic_regression_binary_pipeline_class
):
    X, y = X_y_binary
    path = os.path.join(str(tmpdir), "pickled_pipe.pkl")
    pipeline = logistic_regression_binary_pipeline_class(
        parameters={"Logistic Regression Classifier": {"n_jobs": 1}}
    )
    pipeline.fit(X, y)
    pipeline.save(path)
    return path


def test_load_pickled_pipeline_with_custom_objective(
    X_y_binary, pickled_pipeline_path, logistic_regression_binary_pipeline_class
):
    X, y = X_y_binary
    # checks that class is not defined before loading in pipeline
    with pytest.raises(NameError):
        MockPrecision()  # noqa: F821: ignore flake8's "undefined name" error
    objective = Precision()
    pipeline = logistic_regression_binary_pipeline_class(
        parameters={"Logistic Regression Classifier": {"n_jobs": 1}}
    )
    pipeline.fit(X, y)
    assert PipelineBase.load(pickled_pipeline_path).score(
        X, y, [objective]
    ) == pipeline.score(X, y, [objective])


def test_reproducibility(X_y_binary, logistic_regression_binary_pipeline_class):
    X, y = X_y_binary
    objective = FraudCost(
        retry_percentage=0.5,
        interchange_fee=0.02,
        fraud_payout_percentage=0.75,
        amount_col=10,
    )

    parameters = {
        "Imputer": {
            "categorical_impute_strategy": "most_frequent",
            "numeric_impute_strategy": "mean",
        },
        "Logistic Regression Classifier": {"penalty": "l2", "C": 1.0, "n_jobs": 1},
    }

    clf = logistic_regression_binary_pipeline_class(parameters=parameters)
    clf.fit(X, y)

    clf_1 = logistic_regression_binary_pipeline_class(parameters=parameters)
    clf_1.fit(X, y)

    assert clf_1.score(X, y, [objective]) == clf.score(X, y, [objective])


def test_indexing(X_y_binary, logistic_regression_binary_pipeline_class):
    X, y = X_y_binary
    clf = logistic_regression_binary_pipeline_class(
        parameters={"Logistic Regression Classifier": {"n_jobs": 1}}
    )
    clf.fit(X, y)

    assert isinstance(clf[1], OneHotEncoder)
    assert isinstance(clf["Imputer"], Imputer)

    setting_err_msg = "Setting pipeline components is not supported."
    with pytest.raises(NotImplementedError, match=setting_err_msg):
        clf[1] = OneHotEncoder()

    slicing_err_msg = "Slicing pipelines is currently not supported."
    with pytest.raises(NotImplementedError, match=slicing_err_msg):
        clf[:1]


@pytest.mark.parametrize("is_linear", [True, False])
@pytest.mark.parametrize("is_fitted", [True, False])
@pytest.mark.parametrize("return_dict", [True, False])
def test_describe_pipeline(
    is_linear,
    is_fitted,
    return_dict,
    X_y_binary,
    caplog,
    logistic_regression_binary_pipeline_class,
    nonlinear_binary_pipeline_class,
):
    X, y = X_y_binary

    if is_linear:
        pipeline = logistic_regression_binary_pipeline_class(parameters={})
        name = "Logistic Regression Binary Pipeline"
        expected_pipeline_dict = {
            "name": name,
            "problem_type": ProblemTypes.BINARY,
            "model_family": ModelFamily.LINEAR_MODEL,
            "components": {
                "Imputer": {
                    "name": "Imputer",
                    "parameters": {
                        "categorical_impute_strategy": "most_frequent",
                        "numeric_impute_strategy": "mean",
                        "categorical_fill_value": None,
                        "numeric_fill_value": None,
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
                "Standard Scaler": {"name": "Standard Scaler", "parameters": {}},
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
            },
        }
    else:
        pipeline = nonlinear_binary_pipeline_class(parameters={})
        name = "Non Linear Binary Pipeline"
        expected_pipeline_dict = {
            "name": name,
            "problem_type": ProblemTypes.BINARY,
            "model_family": ModelFamily.LINEAR_MODEL,
            "components": {
                "Imputer": {
                    "name": "Imputer",
                    "parameters": {
                        "categorical_impute_strategy": "most_frequent",
                        "numeric_impute_strategy": "mean",
                        "categorical_fill_value": None,
                        "numeric_fill_value": None,
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
                "Random Forest Classifier": {
                    "name": "Random Forest Classifier",
                    "parameters": {"n_estimators": 100, "max_depth": 6, "n_jobs": -1},
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
            },
        }

    if is_fitted:
        pipeline.fit(X, y)

    pipeline_dict = pipeline.describe(return_dict=return_dict)
    if return_dict:
        assert pipeline_dict == expected_pipeline_dict
    else:
        assert pipeline_dict is None

    out = caplog.text
    assert name in out
    assert "Problem Type: binary" in out
    assert "Model Family: Linear" in out

    if is_fitted:
        assert "Number of features: " in out
    else:
        assert "Number of features: " not in out

    for component in pipeline:
        if component.hyperparameter_ranges:
            for parameter in component.hyperparameter_ranges:
                assert parameter in out
        assert component.name in out


def test_nonlinear_model_family(example_graph):
    class DummyNonlinearPipeline(BinaryClassificationPipeline):
        component_graph = example_graph

        def __init__(self, parameters, random_seed=0):
            super().__init__(
                self.component_graph,
                parameters=parameters,
                random_seed=random_seed,
            )

    nlbp = DummyNonlinearPipeline({})
    assert nlbp.model_family == ModelFamily.LINEAR_MODEL


def test_parameters(logistic_regression_binary_pipeline_class):
    parameters = {
        "Imputer": {
            "categorical_impute_strategy": "most_frequent",
            "numeric_impute_strategy": "median",
        },
        "Logistic Regression Classifier": {
            "penalty": "l2",
            "C": 3.0,
        },
    }
    lrp = logistic_regression_binary_pipeline_class(parameters=parameters)
    expected_parameters = {
        "Imputer": {
            "categorical_impute_strategy": "most_frequent",
            "numeric_impute_strategy": "median",
            "categorical_fill_value": None,
            "numeric_fill_value": None,
        },
        "One Hot Encoder": {
            "top_n": 10,
            "features_to_encode": None,
            "categories": None,
            "drop": "if_binary",
            "handle_unknown": "ignore",
            "handle_missing": "error",
        },
        "Logistic Regression Classifier": {
            "penalty": "l2",
            "C": 3.0,
            "n_jobs": -1,
            "multi_class": "auto",
            "solver": "lbfgs",
        },
    }
    assert lrp.parameters == expected_parameters


def test_parameters_nonlinear(nonlinear_binary_pipeline_class):
    parameters = {
        "Imputer": {
            "categorical_impute_strategy": "most_frequent",
            "numeric_impute_strategy": "median",
        },
        "Logistic Regression": {
            "penalty": "l2",
            "C": 3.0,
        },
    }
    nlbp = nonlinear_binary_pipeline_class(parameters=parameters)
    expected_parameters = {
        "Imputer": {
            "categorical_impute_strategy": "most_frequent",
            "numeric_impute_strategy": "median",
            "categorical_fill_value": None,
            "numeric_fill_value": None,
        },
        "OneHot_RandomForest": {
            "top_n": 10,
            "features_to_encode": None,
            "categories": None,
            "drop": "if_binary",
            "handle_unknown": "ignore",
            "handle_missing": "error",
        },
        "OneHot_ElasticNet": {
            "top_n": 10,
            "features_to_encode": None,
            "categories": None,
            "drop": "if_binary",
            "handle_unknown": "ignore",
            "handle_missing": "error",
        },
        "Random Forest": {"max_depth": 6, "n_estimators": 100, "n_jobs": -1},
        "Elastic Net": {
            "C": 1.0,
            "l1_ratio": 0.15,
            "multi_class": "auto",
            "solver": "saga",
            "n_jobs": -1,
            "penalty": "elasticnet",
        },
        "Logistic Regression": {
            "penalty": "l2",
            "C": 3.0,
            "n_jobs": -1,
            "multi_class": "auto",
            "solver": "lbfgs",
        },
    }
    assert nlbp.parameters == expected_parameters


def test_name():
    pipeline = BinaryClassificationPipeline(
        component_graph=["Logistic Regression Classifier"]
    )
    assert pipeline.name == "Logistic Regression Classifier"
    assert pipeline.custom_name is None

    pipeline_with_custom_name = BinaryClassificationPipeline(
        component_graph=["Logistic Regression Classifier"],
        custom_name="Cool Logistic Regression",
    )
    assert pipeline_with_custom_name.name == "Cool Logistic Regression"
    assert pipeline_with_custom_name.custom_name == "Cool Logistic Regression"

    pipeline_with_neat_name = BinaryClassificationPipeline(
        component_graph=["Logistic Regression Classifier"], custom_name="some_neat_name"
    )
    assert pipeline_with_neat_name.name == "some_neat_name"
    assert pipeline_with_neat_name.custom_name == "some_neat_name"


def test_multi_format_creation(X_y_binary):
    X, y = X_y_binary
    # Test that we can mix and match string and component classes

    component_graph = {
        "Imputer": ["Imputer", "X", "y"],
        "OneHot": ["One Hot Encoder", "Imputer.x", "y"],
        "Scaler": [StandardScaler, "OneHot.x", "y"],
        "Logistic Regression": ["Logistic Regression Classifier", "Scaler.x", "y"],
    }
    parameters = {
        "Imputer": {
            "categorical_impute_strategy": "most_frequent",
            "numeric_impute_strategy": "mean",
        },
        "Logistic Regression Classifier": {"penalty": "l2", "C": 1.0, "n_jobs": 1},
    }

    clf = BinaryClassificationPipeline(
        component_graph=component_graph, parameters=parameters
    )
    correct_components = [
        Imputer,
        OneHotEncoder,
        StandardScaler,
        LogisticRegressionClassifier,
    ]
    for component, correct_components in zip(clf, correct_components):
        assert isinstance(component, correct_components)
    assert clf.model_family == ModelFamily.LINEAR_MODEL

    clf.fit(X, y)
    clf.score(X, y, ["precision"])
    assert not clf.feature_importance.isnull().all().all()


def test_problem_types():
    with pytest.raises(
        ValueError,
        match="not valid for this component graph. Valid problem types include *.",
    ):
        BinaryClassificationPipeline(
            component_graph=["Random Forest Regressor"], parameters={}
        )


def make_mock_regression_pipeline():
    return RegressionPipeline(
        component_graph=["Random Forest Regressor"], parameters={}
    )


def make_mock_binary_pipeline():
    return BinaryClassificationPipeline(
        component_graph=["Random Forest Classifier"], parameters={}
    )


def make_mock_multiclass_pipeline():
    return MulticlassClassificationPipeline(
        component_graph=["Random Forest Classifier"], parameters={}
    )


@patch("evalml.pipelines.RegressionPipeline.fit")
@patch("evalml.pipelines.RegressionPipeline.predict")
def test_score_regression_single(mock_predict, mock_fit, X_y_regression):
    X, y = X_y_regression
    mock_predict.return_value = pd.Series(y)
    clf = make_mock_regression_pipeline()
    clf.fit(X, y)
    objective_names = ["r2"]
    scores = clf.score(X, y, objective_names)
    mock_predict.assert_called()
    assert scores == {"R2": 1.0}


@patch("evalml.pipelines.ComponentGraph.fit")
@patch("evalml.pipelines.RegressionPipeline.predict")
def test_score_nonlinear_regression(
    mock_predict, mock_fit, nonlinear_regression_pipeline_class, X_y_regression
):
    X, y = X_y_regression
    mock_predict.return_value = pd.Series(y)
    clf = nonlinear_regression_pipeline_class({})
    clf.fit(X, y)
    objective_names = ["r2"]
    scores = clf.score(X, y, objective_names)
    mock_predict.assert_called()
    assert scores == {"R2": 1.0}


@patch("evalml.pipelines.BinaryClassificationPipeline._encode_targets")
@patch("evalml.pipelines.BinaryClassificationPipeline.fit")
@patch("evalml.pipelines.components.Estimator.predict")
def test_score_binary_single(mock_predict, mock_fit, mock_encode, X_y_binary):
    X, y = X_y_binary
    mock_predict.return_value = y
    mock_encode.return_value = y
    clf = make_mock_binary_pipeline()
    clf.fit(X, y)
    objective_names = ["f1"]
    scores = clf.score(X, y, objective_names)
    mock_encode.assert_called()
    mock_fit.assert_called()
    mock_predict.assert_called()
    assert scores == {"F1": 1.0}


@patch("evalml.pipelines.MulticlassClassificationPipeline._encode_targets")
@patch("evalml.pipelines.MulticlassClassificationPipeline.fit")
@patch("evalml.pipelines.components.Estimator.predict")
def test_score_multiclass_single(mock_predict, mock_fit, mock_encode, X_y_binary):
    X, y = X_y_binary
    mock_predict.return_value = y
    mock_encode.return_value = y
    clf = make_mock_multiclass_pipeline()
    clf.fit(X, y)
    objective_names = ["f1 micro"]
    scores = clf.score(X, y, objective_names)
    mock_encode.assert_called()
    mock_fit.assert_called()
    mock_predict.assert_called()
    assert scores == {"F1 Micro": 1.0}


@patch("evalml.pipelines.MulticlassClassificationPipeline._encode_targets")
@patch("evalml.pipelines.MulticlassClassificationPipeline.fit")
@patch("evalml.pipelines.ComponentGraph.predict")
def test_score_nonlinear_multiclass(
    mock_predict, mock_fit, mock_encode, nonlinear_multiclass_pipeline_class, X_y_multi
):
    X, y = X_y_multi
    mock_predict.return_value = pd.Series(y)
    mock_encode.return_value = pd.Series(y)
    clf = nonlinear_multiclass_pipeline_class({})
    clf.fit(X, y)
    objective_names = ["f1 micro", "precision micro"]
    scores = clf.score(X, y, objective_names)
    mock_predict.assert_called()
    assert scores == {"F1 Micro": 1.0, "Precision Micro": 1.0}


@patch("evalml.pipelines.RegressionPipeline.fit")
@patch("evalml.pipelines.RegressionPipeline.predict")
def test_score_regression_list(mock_predict, mock_fit, X_y_binary):
    X, y = X_y_binary
    mock_predict.return_value = pd.Series(y)
    clf = make_mock_regression_pipeline()
    clf.fit(X, y)
    objective_names = ["r2", "mse"]
    scores = clf.score(X, y, objective_names)
    mock_predict.assert_called()
    assert scores == {"R2": 1.0, "MSE": 0.0}


@patch("evalml.pipelines.BinaryClassificationPipeline._encode_targets")
@patch("evalml.pipelines.BinaryClassificationPipeline.fit")
@patch("evalml.pipelines.components.Estimator.predict")
def test_score_binary_list(mock_predict, mock_fit, mock_encode, X_y_binary):
    X, y = X_y_binary
    mock_predict.return_value = y
    mock_encode.return_value = y
    clf = make_mock_binary_pipeline()
    clf.fit(X, y)
    objective_names = ["f1", "precision"]
    scores = clf.score(X, y, objective_names)
    mock_fit.assert_called()
    mock_encode.assert_called()
    mock_predict.assert_called()
    assert scores == {"F1": 1.0, "Precision": 1.0}


@patch("evalml.pipelines.MulticlassClassificationPipeline._encode_targets")
@patch("evalml.pipelines.MulticlassClassificationPipeline.fit")
@patch("evalml.pipelines.components.Estimator.predict")
def test_score_multi_list(mock_predict, mock_fit, mock_encode, X_y_binary):
    X, y = X_y_binary
    mock_predict.return_value = y
    mock_encode.return_value = y
    clf = make_mock_multiclass_pipeline()
    clf.fit(X, y)
    objective_names = ["f1 micro", "precision micro"]
    scores = clf.score(X, y, objective_names)
    mock_predict.assert_called()
    assert scores == {"F1 Micro": 1.0, "Precision Micro": 1.0}


@patch("evalml.objectives.R2.score")
@patch("evalml.pipelines.RegressionPipeline.fit")
@patch("evalml.pipelines.RegressionPipeline.predict")
def test_score_regression_objective_error(
    mock_predict, mock_fit, mock_objective_score, X_y_binary
):
    mock_objective_score.side_effect = Exception("finna kabooom 💣")
    X, y = X_y_binary
    mock_predict.return_value = pd.Series(y)
    clf = make_mock_regression_pipeline()
    clf.fit(X, y)
    objective_names = ["r2", "mse"]
    # Using pytest.raises to make sure we error if an error is not thrown.
    with pytest.raises(PipelineScoreError):
        _ = clf.score(X, y, objective_names)
    try:
        _ = clf.score(X, y, objective_names)
    except PipelineScoreError as e:
        assert e.scored_successfully == {"MSE": 0.0}
        assert "finna kabooom 💣" in e.message
        assert "R2" in e.exceptions


@patch("evalml.pipelines.BinaryClassificationPipeline._encode_targets")
@patch("evalml.objectives.F1.score")
@patch("evalml.pipelines.BinaryClassificationPipeline.fit")
@patch("evalml.pipelines.components.Estimator.predict")
def test_score_binary_objective_error(
    mock_predict, mock_fit, mock_objective_score, mock_encode, X_y_binary
):
    mock_objective_score.side_effect = Exception("finna kabooom 💣")
    X, y = X_y_binary
    mock_predict.return_value = y
    mock_encode.return_value = y
    clf = make_mock_binary_pipeline()
    clf.fit(X, y)
    objective_names = ["f1", "precision"]
    # Using pytest.raises to make sure we error if an error is not thrown.
    with pytest.raises(PipelineScoreError):
        _ = clf.score(X, y, objective_names)
    try:
        _ = clf.score(X, y, objective_names)
    except PipelineScoreError as e:
        assert e.scored_successfully == {"Precision": 1.0}
        assert "finna kabooom 💣" in e.message


@patch("evalml.pipelines.BinaryClassificationPipeline._encode_targets")
@patch("evalml.objectives.F1.score")
@patch("evalml.pipelines.BinaryClassificationPipeline.fit")
@patch("evalml.pipelines.ComponentGraph.predict")
def test_score_nonlinear_binary_objective_error(
    mock_predict,
    mock_fit,
    mock_objective_score,
    mock_encode,
    nonlinear_binary_pipeline_class,
    X_y_binary,
):
    mock_objective_score.side_effect = Exception("finna kabooom 💣")
    X, y = X_y_binary
    mock_predict.return_value = pd.Series(y)
    mock_encode.return_value = y
    clf = nonlinear_binary_pipeline_class({})
    clf.fit(X, y)
    objective_names = ["f1", "precision"]
    # Using pytest.raises to make sure we error if an error is not thrown.
    with pytest.raises(PipelineScoreError):
        _ = clf.score(X, y, objective_names)
    try:
        _ = clf.score(X, y, objective_names)
    except PipelineScoreError as e:
        assert e.scored_successfully == {"Precision": 1.0}
        assert "finna kabooom 💣" in e.message


@patch("evalml.pipelines.MulticlassClassificationPipeline._encode_targets")
@patch("evalml.objectives.F1Micro.score")
@patch("evalml.pipelines.MulticlassClassificationPipeline.fit")
@patch("evalml.pipelines.components.Estimator.predict")
def test_score_multiclass_objective_error(
    mock_predict, mock_fit, mock_objective_score, mock_encode, X_y_binary
):
    mock_objective_score.side_effect = Exception("finna kabooom 💣")
    X, y = X_y_binary
    mock_predict.return_value = y
    mock_encode.return_value = y
    clf = make_mock_multiclass_pipeline()
    clf.fit(X, y)
    objective_names = ["f1 micro", "precision micro"]
    # Using pytest.raises to make sure we error if an error is not thrown.
    with pytest.raises(PipelineScoreError):
        _ = clf.score(X, y, objective_names)
    try:
        _ = clf.score(X, y, objective_names)
    except PipelineScoreError as e:
        assert e.scored_successfully == {"Precision Micro": 1.0}
        assert "finna kabooom 💣" in e.message
        assert "F1 Micro" in e.exceptions


@patch("evalml.pipelines.components.Imputer.transform")
@patch("evalml.pipelines.components.OneHotEncoder.transform")
@patch("evalml.pipelines.components.StandardScaler.transform")
def test_compute_estimator_features(
    mock_scaler,
    mock_ohe,
    mock_imputer,
    X_y_binary,
    logistic_regression_binary_pipeline_class,
):
    X, y = X_y_binary
    X = pd.DataFrame(X)
    X_expected = pd.DataFrame(index=X.index, columns=X.columns).fillna(0)
    mock_imputer.return_value = X
    mock_ohe.return_value = X
    mock_scaler.return_value = X_expected
    X_expected = X_expected.astype("int64")

    pipeline = logistic_regression_binary_pipeline_class({})
    pipeline.fit(X, y)

    X_t = pipeline.compute_estimator_features(X)
    assert_frame_equal(X_expected, X_t)
    assert mock_imputer.call_count == 2
    assert mock_ohe.call_count == 2
    assert mock_scaler.call_count == 2


@patch("evalml.pipelines.components.Imputer.transform")
@patch("evalml.pipelines.components.OneHotEncoder.transform")
@patch("evalml.pipelines.components.RandomForestClassifier.predict_proba")
@patch("evalml.pipelines.components.ElasticNetClassifier.predict_proba")
@patch("evalml.pipelines.components.RandomForestClassifier.predict")
@patch("evalml.pipelines.components.ElasticNetClassifier.predict")
def test_compute_estimator_features_nonlinear(
    mock_en_predict,
    mock_rf_predict,
    mock_en_predict_proba,
    mock_rf_predict_proba,
    mock_ohe,
    mock_imputer,
    X_y_binary,
    nonlinear_binary_pipeline_class,
):
    X, y = X_y_binary
    mock_imputer.return_value = pd.DataFrame(X)
    mock_ohe.return_value = pd.DataFrame(X)
    mock_en_predict.return_value = pd.Series(np.ones(X.shape[0]))
    mock_rf_predict.return_value = pd.Series(np.zeros(X.shape[0]))

    mock_en_predict_proba_df = pd.DataFrame(
        {0: np.ones(X.shape[0]), 1: np.zeros(X.shape[0])}
    )
    mock_en_predict_proba_df.ww.init()
    mock_rf_predict_proba_df = pd.DataFrame(
        {0: np.zeros(X.shape[0]), 1: np.ones(X.shape[0])}
    )
    mock_rf_predict_proba_df.ww.init()
    mock_en_predict_proba.return_value = mock_en_predict_proba_df
    mock_rf_predict_proba.return_value = mock_rf_predict_proba_df

    X_expected_df = pd.DataFrame(
        {
            "Col 1 Random Forest.x": np.ones(X.shape[0]),
            "Col 1 Elastic Net.x": np.zeros(X.shape[0]),
        }
    )

    pipeline = nonlinear_binary_pipeline_class({})
    pipeline.fit(X, y)
    X_t = pipeline.compute_estimator_features(X)

    assert_frame_equal(X_expected_df, X_t)
    assert mock_imputer.call_count == 2
    assert mock_ohe.call_count == 4
    assert mock_en_predict_proba.call_count == 2
    assert mock_rf_predict_proba.call_count == 2


def test_no_default_parameters():
    class MockComponent(Transformer):
        name = "Mock Component"
        hyperparameter_ranges = {"a": [0, 1, 2]}

        def __init__(self, a, b=1, c="2", random_seed=0):
            self.a = a
            self.b = b
            self.c = c
            super().__init__()

        def transform(self, X, y=None):
            return X

    class TestPipeline(BinaryClassificationPipeline):
        component_graph = [MockComponent, "Logistic Regression Classifier"]

        def __init__(self, parameters, random_seed=0):
            super().__init__(self.component_graph, parameters=parameters)

    with pytest.raises(
        ValueError, match="Error received when instantiating component *."
    ):
        TestPipeline(parameters={})

    assert TestPipeline(parameters={"Mock Component": {"a": 42}})


def test_init_components_invalid_parameters():
    component_graph = [
        "RF Classifier Select From Model",
        "Logistic Regression Classifier",
    ]
    parameters = {"Logistic Regression Classifier": {"cool_parameter": "yes"}}

    with pytest.raises(ValueError, match="Error received when instantiating component"):
        BinaryClassificationPipeline(
            component_graph=component_graph, parameters=parameters
        )


def test_correct_parameters(logistic_regression_binary_pipeline_class):
    parameters = {
        "Imputer": {
            "categorical_impute_strategy": "most_frequent",
            "numeric_impute_strategy": "mean",
        },
        "Logistic Regression Classifier": {
            "penalty": "l2",
            "C": 3.0,
        },
    }
    lr_pipeline = logistic_regression_binary_pipeline_class(parameters=parameters)
    assert lr_pipeline.estimator.random_seed == 0
    assert lr_pipeline.estimator.parameters["C"] == 3.0
    assert (
        lr_pipeline["Imputer"].parameters["categorical_impute_strategy"]
        == "most_frequent"
    )
    assert lr_pipeline["Imputer"].parameters["numeric_impute_strategy"] == "mean"


def test_correct_nonlinear_parameters(nonlinear_binary_pipeline_class):
    parameters = {
        "Imputer": {
            "categorical_impute_strategy": "most_frequent",
            "numeric_impute_strategy": "mean",
        },
        "OneHot_RandomForest": {"top_n": 4},
        "Logistic Regression": {
            "penalty": "l2",
            "C": 3.0,
        },
    }
    nlb_pipeline = nonlinear_binary_pipeline_class(parameters=parameters)
    assert nlb_pipeline.estimator.random_seed == 0
    assert nlb_pipeline.estimator.parameters["C"] == 3.0
    assert (
        nlb_pipeline["Imputer"].parameters["categorical_impute_strategy"]
        == "most_frequent"
    )
    assert nlb_pipeline["Imputer"].parameters["numeric_impute_strategy"] == "mean"
    assert nlb_pipeline["OneHot_RandomForest"].parameters["top_n"] == 4
    assert nlb_pipeline["OneHot_ElasticNet"].parameters["top_n"] == 10


@patch("evalml.pipelines.components.Estimator.predict")
def test_score_with_objective_that_requires_predict_proba(
    mock_predict, dummy_regression_pipeline_class, X_y_binary
):
    X, y = X_y_binary
    mock_predict.return_value = pd.Series([1] * 100)
    # Using pytest.raises to make sure we error if an error is not thrown.
    with pytest.raises(PipelineScoreError):
        clf = dummy_regression_pipeline_class(parameters={})
        clf.fit(X, y)
        clf.score(X, y, ["precision", "auc"])
    try:
        clf = dummy_regression_pipeline_class(parameters={})
        clf.fit(X, y)
        clf.score(X, y, ["precision", "auc"])
    except PipelineScoreError as e:
        assert (
            "Invalid objective AUC specified for problem type regression" in e.message
        )
        assert (
            "Invalid objective Precision specified for problem type regression"
            in e.message
        )
    mock_predict.assert_called()


def test_score_auc(X_y_binary, logistic_regression_binary_pipeline_class):
    X, y = X_y_binary
    lr_pipeline = logistic_regression_binary_pipeline_class(
        parameters={"Logistic Regression Classifier": {"n_jobs": 1}}
    )
    lr_pipeline.fit(X, y)
    lr_pipeline.score(X, y, ["auc"])


def test_pipeline_summary():
    assert (
        BinaryClassificationPipeline(["Imputer", "One Hot Encoder"]).summary
        == "Pipeline w/ Imputer + One Hot Encoder"
    )
    assert BinaryClassificationPipeline(["Imputer"]).summary == "Pipeline w/ Imputer"
    assert (
        BinaryClassificationPipeline(["Random Forest Classifier"]).summary
        == "Random Forest Classifier"
    )
    assert BinaryClassificationPipeline([]).summary == "Empty Pipeline"
    assert (
        BinaryClassificationPipeline(
            ["Imputer", "One Hot Encoder", "Random Forest Classifier"]
        ).summary
        == "Random Forest Classifier w/ Imputer + One Hot Encoder"
    )


def test_nonlinear_pipeline_summary(
    nonlinear_binary_pipeline_class,
    nonlinear_multiclass_pipeline_class,
    nonlinear_regression_pipeline_class,
):
    assert (
        nonlinear_binary_pipeline_class({}).summary
        == "Logistic Regression Classifier w/ Imputer + One Hot Encoder + One Hot Encoder + Random Forest Classifier + Elastic Net Classifier"
    )
    assert (
        nonlinear_multiclass_pipeline_class({}).summary
        == "Logistic Regression Classifier w/ Imputer + One Hot Encoder + One Hot Encoder + Random Forest Classifier + Elastic Net Classifier"
    )
    assert (
        nonlinear_regression_pipeline_class({}).summary
        == "Linear Regressor w/ Imputer + One Hot Encoder + Random Forest Regressor + Elastic Net Regressor"
    )


def test_drop_columns_in_pipeline():
    parameters = {
        "Drop Columns Transformer": {"columns": ["column to drop"]},
        "Imputer": {
            "categorical_impute_strategy": "most_frequent",
            "numeric_impute_strategy": "mean",
        },
        "Logistic Regression Classifier": {"penalty": "l2", "C": 3.0, "n_jobs": 1},
    }
    pipeline_with_drop_col = BinaryClassificationPipeline(
        component_graph=[
            "Drop Columns Transformer",
            "Imputer",
            "Logistic Regression Classifier",
        ],
        parameters=parameters,
    )
    X = pd.DataFrame({"column to drop": [1, 0, 1, 3], "other col": [1, 2, 4, 1]})
    y = pd.Series([1, 0, 1, 0])
    pipeline_with_drop_col.fit(X, y)
    pipeline_with_drop_col.score(X, y, ["auc"])
    assert list(pipeline_with_drop_col.feature_importance["feature"]) == ["other col"]


@pytest.mark.parametrize("is_linear", [True, False])
def test_clone_init(
    is_linear, linear_regression_pipeline_class, nonlinear_regression_pipeline_class
):
    if is_linear:
        pipeline_class = linear_regression_pipeline_class
    else:
        pipeline_class = nonlinear_regression_pipeline_class
    parameters = {
        "Imputer": {
            "categorical_impute_strategy": "most_frequent",
            "numeric_impute_strategy": "mean",
        },
        "Linear Regressor": {
            "fit_intercept": True,
            "normalize": True,
        },
    }
    pipeline = pipeline_class(parameters=parameters, random_seed=42)
    pipeline_clone = pipeline.clone()
    assert pipeline.parameters == pipeline_clone.parameters
    assert pipeline.random_seed == pipeline_clone.random_seed


@pytest.mark.parametrize("is_linear", [True, False])
def test_clone_fitted(
    is_linear,
    X_y_binary,
    logistic_regression_binary_pipeline_class,
    nonlinear_binary_pipeline_class,
):
    X, y = X_y_binary
    if is_linear:
        pipeline_class = logistic_regression_binary_pipeline_class
    else:
        pipeline_class = nonlinear_binary_pipeline_class
    pipeline = pipeline_class(
        parameters={"Logistic Regression Classifier": {"n_jobs": 1}}, random_seed=42
    )
    pipeline.fit(X, y)
    X_t = pipeline.predict_proba(X)

    pipeline_clone = pipeline.clone()
    assert pipeline.parameters == pipeline_clone.parameters
    assert pipeline.random_seed == pipeline_clone.random_seed

    with pytest.raises(PipelineNotYetFittedError):
        pipeline_clone.predict(X)
    pipeline_clone.fit(X, y)

    X_t_clone = pipeline_clone.predict_proba(X)
    assert_frame_equal(X_t, X_t_clone)


def test_feature_importance_has_feature_names(
    X_y_binary, logistic_regression_binary_pipeline_class
):
    X, y = X_y_binary
    col_names = ["col_{}".format(i) for i in range(len(X[0]))]
    X = pd.DataFrame(X, columns=col_names)
    parameters = {
        "Imputer": {
            "categorical_impute_strategy": "most_frequent",
            "numeric_impute_strategy": "mean",
        },
        "RF Classifier Select From Model": {
            "percent_features": 1.0,
            "number_features": len(X.columns),
            "n_estimators": 20,
        },
        "Logistic Regression Classifier": {"penalty": "l2", "C": 1.0, "n_jobs": 1},
    }

    clf = logistic_regression_binary_pipeline_class(parameters=parameters)
    clf.fit(X, y)
    assert len(clf.feature_importance) == len(X.columns)
    assert not clf.feature_importance.isnull().all().all()
    assert sorted(clf.feature_importance["feature"]) == sorted(col_names)


def test_nonlinear_feature_importance_has_feature_names(
    X_y_binary, nonlinear_binary_pipeline_class
):
    X, y = X_y_binary
    col_names = ["col_{}".format(i) for i in range(len(X[0]))]
    X = pd.DataFrame(X, columns=col_names)
    parameters = {
        "Imputer": {
            "categorical_impute_strategy": "most_frequent",
            "numeric_impute_strategy": "mean",
        },
        "Logistic Regression Classifier": {"penalty": "l2", "C": 1.0, "n_jobs": 1},
    }

    clf = nonlinear_binary_pipeline_class(parameters=parameters)
    clf.fit(X, y)
    assert len(clf.feature_importance) == 2
    assert not clf.feature_importance.isnull().all().all()
    assert sorted(clf.feature_importance["feature"]) == [
        "Col 1 Elastic Net.x",
        "Col 1 Random Forest.x",
    ]


@pytest.mark.parametrize(
    "problem_type",
    [ProblemTypes.BINARY, ProblemTypes.MULTICLASS, ProblemTypes.REGRESSION],
)
def test_feature_importance_has_feature_names_xgboost(
    problem_type, has_minimal_dependencies, X_y_regression, X_y_binary, X_y_multi
):
    # Testing that we store the original feature names since we map to numeric values for XGBoost
    if has_minimal_dependencies:
        pytest.skip("Skipping because XGBoost not installed for minimal dependencies")
    if problem_type == ProblemTypes.REGRESSION:
        pipeline = RegressionPipeline(
            component_graph=["Simple Imputer", "XGBoost Regressor"],
            parameters={"XGBoost Regressor": {"nthread": 1}},
        )
        X, y = X_y_regression
    elif problem_type == ProblemTypes.BINARY:
        pipeline = BinaryClassificationPipeline(
            component_graph=["Simple Imputer", "XGBoost Classifier"],
            parameters={"XGBoost Classifier": {"nthread": 1}},
        )
        X, y = X_y_binary
    elif problem_type == ProblemTypes.MULTICLASS:
        pipeline = MulticlassClassificationPipeline(
            component_graph=["Simple Imputer", "XGBoost Classifier"],
            parameters={"XGBoost Classifier": {"nthread": 1}},
        )
        X, y = X_y_multi

    X = pd.DataFrame(X)
    X = X.rename(columns={col_name: f"<[{col_name}]" for col_name in X.columns.values})
    col_names = X.columns.values
    pipeline.fit(X, y)
    assert len(pipeline.feature_importance) == len(X.columns)
    assert not pipeline.feature_importance.isnull().all().all()
    assert sorted(pipeline.feature_importance["feature"]) == sorted(col_names)


def test_component_not_found():
    with pytest.raises(MissingComponentError, match="was not found"):
        BinaryClassificationPipeline(
            component_graph=[
                "Imputer",
                "One Hot Encoder",
                "This Component Does Not Exist",
                "Standard Scaler",
                "Logistic Regression Classifier",
            ]
        )


def test_get_default_parameters(logistic_regression_binary_pipeline_class):
    expected_defaults = {
        "Imputer": {
            "categorical_impute_strategy": "most_frequent",
            "numeric_impute_strategy": "mean",
            "categorical_fill_value": None,
            "numeric_fill_value": None,
        },
        "One Hot Encoder": {
            "top_n": 10,
            "features_to_encode": None,
            "categories": None,
            "drop": "if_binary",
            "handle_unknown": "ignore",
            "handle_missing": "error",
        },
        "Logistic Regression Classifier": {
            "penalty": "l2",
            "C": 1.0,
            "n_jobs": -1,
            "multi_class": "auto",
            "solver": "lbfgs",
        },
    }
    assert (
        logistic_regression_binary_pipeline_class({}).component_graph.default_parameters
        == expected_defaults
    )


@pytest.mark.parametrize("data_type", ["li", "np", "pd", "ww"])
@pytest.mark.parametrize("problem_type", [ProblemTypes.BINARY, ProblemTypes.MULTICLASS])
@pytest.mark.parametrize(
    "target_type",
    [
        "int16",
        "int32",
        "int64",
        "float16",
        "float32",
        "float64",
        "bool",
        "category",
        "object",
    ],
)
def test_targets_data_types_classification_pipelines(
    breast_cancer_local,
    wine_local,
    data_type,
    problem_type,
    target_type,
    all_binary_pipeline_classes,
    make_data_type,
    all_multiclass_pipeline_classes,
    helper_functions,
):
    if data_type == "np" and target_type in ["Int64", "boolean"]:
        pytest.skip(
            "Skipping test where data type is numpy and target type is nullable dtype"
        )

    if problem_type == ProblemTypes.BINARY:
        objective = "Log Loss Binary"
        pipeline_classes = all_binary_pipeline_classes
        X, y = breast_cancer_local
        if "bool" in target_type:
            y = y.map({"malignant": False, "benign": True})
    elif problem_type == ProblemTypes.MULTICLASS:
        if "bool" in target_type:
            pytest.skip(
                "Skipping test where problem type is multiclass but target type is boolean"
            )
        objective = "Log Loss Multiclass"
        pipeline_classes = all_multiclass_pipeline_classes
        X, y = wine_local

    # Update target types as necessary
    unique_vals = y.unique()

    if "int" in target_type.lower():
        unique_vals = y.unique()
        y = y.map({unique_vals[i]: int(i) for i in range(len(unique_vals))})
    elif "float" in target_type.lower():
        unique_vals = y.unique()
        y = y.map({unique_vals[i]: float(i) for i in range(len(unique_vals))})
    if target_type == "category":
        y = pd.Series(pd.Categorical(y))
    else:
        y = y.astype(target_type)
    unique_vals = y.unique()

    X = make_data_type(data_type, X)
    y = make_data_type(data_type, y)

    for pipeline_class in pipeline_classes:
        pipeline = helper_functions.safe_init_pipeline_with_njobs_1(pipeline_class)
        pipeline.fit(X, y)
        predictions = pipeline.predict(X, objective)
        assert set(predictions.unique()).issubset(unique_vals)
        predict_proba = pipeline.predict_proba(X)
        assert set(predict_proba.columns) == set(unique_vals)


@pytest.mark.parametrize(
    "problem_type",
    [ProblemTypes.BINARY, ProblemTypes.MULTICLASS, ProblemTypes.REGRESSION],
)
def test_pipeline_not_fitted_error(
    problem_type,
    X_y_binary,
    X_y_multi,
    X_y_regression,
    logistic_regression_binary_pipeline_class,
    logistic_regression_multiclass_pipeline_class,
    linear_regression_pipeline_class,
):
    if problem_type == ProblemTypes.BINARY:
        X, y = X_y_binary
        clf = logistic_regression_binary_pipeline_class(
            parameters={"Logistic Regression Classifier": {"n_jobs": 1}}
        )
    elif problem_type == ProblemTypes.MULTICLASS:
        X, y = X_y_multi
        clf = logistic_regression_multiclass_pipeline_class(
            parameters={"Logistic Regression Classifier": {"n_jobs": 1}}
        )
    elif problem_type == ProblemTypes.REGRESSION:
        X, y = X_y_regression
        clf = linear_regression_pipeline_class(
            parameters={"Linear Regressor": {"n_jobs": 1}}
        )

    with pytest.raises(PipelineNotYetFittedError):
        clf.predict(X)
    with pytest.raises(PipelineNotYetFittedError):
        clf.feature_importance

    if is_classification(problem_type):
        with pytest.raises(PipelineNotYetFittedError):
            clf.predict_proba(X)

    clf.fit(X, y)

    if is_classification(problem_type):
        to_patch = "evalml.pipelines.ClassificationPipeline._predict"
        if problem_type == ProblemTypes.BINARY:
            to_patch = "evalml.pipelines.BinaryClassificationPipeline._predict"
        with patch(to_patch) as mock_predict:
            clf.predict(X)
            mock_predict.assert_called()
            _, kwargs = mock_predict.call_args
            assert kwargs["objective"] is None

            mock_predict.reset_mock()
            clf.predict(X, "Log Loss Binary")
            mock_predict.assert_called()
            _, kwargs = mock_predict.call_args
            assert kwargs["objective"] is not None

            mock_predict.reset_mock()
            clf.predict(X, objective="Log Loss Binary")
            mock_predict.assert_called()
            _, kwargs = mock_predict.call_args
            assert kwargs["objective"] is not None

        clf.predict_proba(X)
    else:
        clf.predict(X)
    clf.feature_importance


@patch("evalml.pipelines.PipelineBase.fit")
@pytest.mark.parametrize(
    "problem_type",
    [ProblemTypes.BINARY, ProblemTypes.MULTICLASS, ProblemTypes.REGRESSION],
)
def test_nonlinear_pipeline_not_fitted_error(
    mock_fit,
    problem_type,
    X_y_binary,
    X_y_multi,
    X_y_regression,
    nonlinear_binary_pipeline_class,
    nonlinear_multiclass_pipeline_class,
    nonlinear_regression_pipeline_class,
):
    if problem_type == ProblemTypes.BINARY:
        X, y = X_y_binary
        clf = nonlinear_binary_pipeline_class(
            parameters={"Logistic Regression Classifier": {"n_jobs": 1}}
        )
    elif problem_type == ProblemTypes.MULTICLASS:
        X, y = X_y_multi
        clf = nonlinear_multiclass_pipeline_class(
            parameters={"Logistic Regression Classifier": {"n_jobs": 1}}
        )
    elif problem_type == ProblemTypes.REGRESSION:
        X, y = X_y_regression
        clf = nonlinear_regression_pipeline_class(
            parameters={"Linear Regressor": {"n_jobs": 1}}
        )

    with pytest.raises(PipelineNotYetFittedError):
        clf.predict(X)
    with pytest.raises(PipelineNotYetFittedError):
        clf.feature_importance

    if problem_type in [ProblemTypes.BINARY, ProblemTypes.MULTICLASS]:
        with pytest.raises(PipelineNotYetFittedError):
            clf.predict_proba(X)

    clf.fit(X, y)
    if problem_type in [ProblemTypes.BINARY, ProblemTypes.MULTICLASS]:
        with patch("evalml.pipelines.ClassificationPipeline.predict") as mock_predict:
            clf.predict(X)
            mock_predict.assert_called()
        with patch(
            "evalml.pipelines.ClassificationPipeline.predict_proba"
        ) as mock_predict_proba:
            clf.predict_proba(X)
            mock_predict_proba.assert_called()
    else:
        with patch("evalml.pipelines.RegressionPipeline.predict") as mock_predict:
            clf.predict(X)
            mock_predict.assert_called()
    clf.feature_importance


@pytest.mark.parametrize(
    "pipeline_class",
    [
        BinaryClassificationPipeline,
        MulticlassClassificationPipeline,
        RegressionPipeline,
    ],
)
def test_pipeline_equality_different_attributes(pipeline_class):
    # Tests that two classes which are equivalent are not equal
    if pipeline_class in [
        BinaryClassificationPipeline,
        MulticlassClassificationPipeline,
    ]:
        final_estimator = "Random Forest Classifier"
    else:
        final_estimator = "Random Forest Regressor"

    class MockPipeline(pipeline_class):
        custom_name = "Mock Pipeline"
        component_graph = ["Imputer", final_estimator]

        def __init__(self, parameters, random_seed=0):
            super().__init__(
                self.component_graph,
                parameters=parameters,
                custom_name=self.custom_name,
                random_seed=random_seed,
            )

    class MockPipelineWithADifferentClassName(pipeline_class):
        custom_name = "Mock Pipeline"
        component_graph = ["Imputer", final_estimator]

        def __init__(self, parameters, random_seed=0):
            super().__init__(
                self.component_graph,
                parameters=parameters,
                custom_name=self.custom_name,
                random_seed=random_seed,
            )

    assert MockPipeline(parameters={}) != MockPipelineWithADifferentClassName(
        parameters={}
    )


@pytest.mark.parametrize(
    "pipeline_class",
    [
        BinaryClassificationPipeline,
        MulticlassClassificationPipeline,
        RegressionPipeline,
    ],
)
def test_pipeline_equality_subclasses(pipeline_class):
    if pipeline_class in [
        BinaryClassificationPipeline,
        MulticlassClassificationPipeline,
    ]:
        final_estimator = "Random Forest Classifier"
    else:
        final_estimator = "Random Forest Regressor"

    class MockPipeline(pipeline_class):
        custom_name = "Mock Pipeline"
        component_graph = ["Imputer", final_estimator]

        def __init__(self, parameters, random_seed=0):
            super().__init__(
                self.component_graph,
                parameters=parameters,
                custom_name=self.custom_name,
                random_seed=random_seed,
            )

    class MockPipelineSubclass(MockPipeline):
        pass

    assert MockPipeline(parameters={}) != MockPipelineSubclass(parameters={})


@pytest.mark.parametrize(
    "pipeline_class",
    [
        BinaryClassificationPipeline,
        MulticlassClassificationPipeline,
        RegressionPipeline,
    ],
)
@patch("evalml.pipelines.ComponentGraph.fit")
def test_pipeline_equality(mock_fit, pipeline_class):
    if pipeline_class in [
        BinaryClassificationPipeline,
        MulticlassClassificationPipeline,
    ]:
        final_estimator = "Random Forest Classifier"
    else:
        final_estimator = "Random Forest Regressor"

    parameters = {
        "Imputer": {
            "categorical_impute_strategy": "most_frequent",
            "numeric_impute_strategy": "mean",
        }
    }

    different_parameters = {
        "Imputer": {
            "categorical_impute_strategy": "constant",
            "numeric_impute_strategy": "mean",
        }
    }

    class MockPipeline(pipeline_class):
        custom_name = "Mock Pipeline"
        component_graph = ["Imputer", final_estimator]

        def __init__(self, parameters, random_seed=0):
            super().__init__(
                self.component_graph,
                parameters=parameters,
                custom_name=self.custom_name,
                random_seed=random_seed,
            )

    # Test self-equality
    mock_pipeline = MockPipeline(parameters={})
    assert mock_pipeline == mock_pipeline

    # Test defaults
    assert MockPipeline(parameters={}) == MockPipeline(parameters={})

    # Test random_seed
    assert MockPipeline(parameters={}, random_seed=10) == MockPipeline(
        parameters={}, random_seed=10
    )
    assert MockPipeline(parameters={}, random_seed=10) != MockPipeline(
        parameters={}, random_seed=0
    )

    # Test parameters
    assert MockPipeline(parameters=parameters) != MockPipeline(
        parameters=different_parameters
    )

    # Test fitted equality
    X = pd.DataFrame({})
    y = pd.Series([])
    mock_pipeline.fit(X, y)
    assert mock_pipeline != MockPipeline(parameters={})

    mock_pipeline_equal = MockPipeline(parameters={})
    mock_pipeline_equal.fit(X, y)
    assert mock_pipeline == mock_pipeline_equal

    # Test fitted equality: same data but different target names are not equal
    mock_pipeline_different_target_name = MockPipeline(parameters={})
    mock_pipeline_different_target_name.fit(
        X, y=pd.Series([], name="target with a name")
    )
    assert mock_pipeline != mock_pipeline_different_target_name


@pytest.mark.parametrize(
    "pipeline_class",
    [
        BinaryClassificationPipeline,
        MulticlassClassificationPipeline,
        RegressionPipeline,
    ],
)
def test_nonlinear_pipeline_equality(pipeline_class):
    if pipeline_class in [
        BinaryClassificationPipeline,
        MulticlassClassificationPipeline,
    ]:
        final_estimator = "Random Forest Classifier"
    else:
        final_estimator = "Random Forest Regressor"

    parameters = {
        "Imputer": {
            "categorical_impute_strategy": "most_frequent",
            "numeric_impute_strategy": "mean",
        },
        "OHE_1": {"top_n": 5},
    }

    different_parameters = {
        "Imputer": {
            "categorical_impute_strategy": "constant",
            "numeric_impute_strategy": "mean",
        },
        "OHE_2": {
            "top_n": 7,
        },
    }

    class MockPipeline(pipeline_class):
        custom_name = "Mock Pipeline"
        component_graph = {
            "Imputer": ["Imputer", "X", "y"],
            "OHE_1": ["One Hot Encoder", "Imputer.x", "y"],
            "OHE_2": ["One Hot Encoder", "Imputer.x", "y"],
            "Estimator": [final_estimator, "OHE_1.x", "OHE_2.x", "y"],
        }

        def __init__(self, parameters, random_seed=0):
            super().__init__(
                self.component_graph,
                parameters=parameters,
                custom_name=self.custom_name,
                random_seed=random_seed,
            )

        def fit(self, X, y=None):
            return self

    # Test self-equality
    mock_pipeline = MockPipeline(parameters={})
    assert mock_pipeline == mock_pipeline

    # Test defaults
    assert MockPipeline(parameters={}) == MockPipeline(parameters={})

    # Test random_seed
    assert MockPipeline(parameters={}, random_seed=10) == MockPipeline(
        parameters={}, random_seed=10
    )
    assert MockPipeline(parameters={}, random_seed=10) != MockPipeline(
        parameters={}, random_seed=0
    )

    # Test parameters
    assert MockPipeline(parameters=parameters) != MockPipeline(
        parameters=different_parameters
    )

    # Test fitted equality
    X = pd.DataFrame({})
    mock_pipeline.fit(X)
    assert mock_pipeline != MockPipeline(parameters={})

    mock_pipeline_equal = MockPipeline(parameters={})
    mock_pipeline_equal.fit(X)
    assert mock_pipeline == mock_pipeline_equal


@pytest.mark.parametrize(
    "problem_type",
    [ProblemTypes.BINARY, ProblemTypes.MULTICLASS, ProblemTypes.REGRESSION],
)
def test_pipeline_equality_different_fitted_data(
    problem_type,
    X_y_binary,
    X_y_multi,
    X_y_regression,
    linear_regression_pipeline_class,
    logistic_regression_binary_pipeline_class,
    logistic_regression_multiclass_pipeline_class,
):
    # Test fitted on different data
    if problem_type == ProblemTypes.BINARY:
        pipeline = logistic_regression_binary_pipeline_class(
            parameters={"Logistic Regression Classifier": {"n_jobs": 1}}
        )
        X, y = X_y_binary
    elif problem_type == ProblemTypes.MULTICLASS:
        pipeline = logistic_regression_multiclass_pipeline_class(
            parameters={"Logistic Regression Classifier": {"n_jobs": 1}}
        )
        X, y = X_y_multi
    elif problem_type == ProblemTypes.REGRESSION:
        pipeline = linear_regression_pipeline_class(
            parameters={"Linear Regressor": {"n_jobs": 1}}
        )
        X, y = X_y_regression

    pipeline_diff_data = pipeline.clone()
    assert pipeline == pipeline_diff_data

    pipeline.fit(X, y)
    # Add new column to data to make it different
    X = np.append(X, np.zeros((len(X), 1)), axis=1)
    pipeline_diff_data.fit(X, y)

    assert pipeline != pipeline_diff_data


def test_pipeline_str():
    class MockBinaryPipeline(BinaryClassificationPipeline):
        custom_name = "Mock Binary Pipeline"
        component_graph = ["Imputer", "Random Forest Classifier"]

        def __init__(self, parameters, random_seed=0):
            super().__init__(
                self.component_graph,
                parameters=parameters,
                custom_name=self.custom_name,
                random_seed=random_seed,
            )

    class MockMulticlassPipeline(MulticlassClassificationPipeline):
        custom_name = "Mock Multiclass Pipeline"
        component_graph = ["Imputer", "Random Forest Classifier"]

        def __init__(self, parameters, random_seed=0):
            super().__init__(
                self.component_graph,
                parameters=parameters,
                custom_name=self.custom_name,
                random_seed=random_seed,
            )

    class MockRegressionPipeline(RegressionPipeline):
        custom_name = "Mock Regression Pipeline"
        component_graph = ["Imputer", "Random Forest Regressor"]

        def __init__(self, parameters, random_seed=0):
            super().__init__(
                self.component_graph,
                parameters=parameters,
                custom_name=self.custom_name,
                random_seed=random_seed,
            )

    binary_pipeline = MockBinaryPipeline(parameters={})
    multiclass_pipeline = MockMulticlassPipeline(parameters={})
    regression_pipeline = MockRegressionPipeline(parameters={})

    assert str(binary_pipeline) == "Mock Binary Pipeline"
    assert str(multiclass_pipeline) == "Mock Multiclass Pipeline"
    assert str(regression_pipeline) == "Mock Regression Pipeline"


@pytest.mark.parametrize(
    "pipeline_class",
    [
        BinaryClassificationPipeline,
        MulticlassClassificationPipeline,
        RegressionPipeline,
    ],
)
def test_pipeline_repr(pipeline_class):
    if pipeline_class in [
        BinaryClassificationPipeline,
        MulticlassClassificationPipeline,
    ]:
        final_estimator = "Random Forest Classifier"
    else:
        final_estimator = "Random Forest Regressor"

    custom_name = "Mock Pipeline"
    component_graph = ["Imputer", final_estimator]
    component_graph_str = f"{{'Imputer': ['Imputer', 'X', 'y'], '{final_estimator}': ['{final_estimator}', 'Imputer.x', 'y']}}"

    pipeline = pipeline_class(component_graph=component_graph, custom_name=custom_name)
    expected_repr = (
        f"pipeline = {pipeline_class.__name__}(component_graph={component_graph_str}, "
        f"parameters={{'Imputer':{{'categorical_impute_strategy': 'most_frequent', 'numeric_impute_strategy': 'mean', 'categorical_fill_value': None, 'numeric_fill_value': None}}, '{final_estimator}':{{'n_estimators': 100, 'max_depth': 6, 'n_jobs': -1}}}}, "
        "custom_name='Mock Pipeline', random_seed=0)"
    )
    assert repr(pipeline) == expected_repr

    pipeline_with_parameters = pipeline_class(
        component_graph=component_graph,
        parameters={"Imputer": {"numeric_fill_value": 42}},
        custom_name=custom_name,
    )
    expected_repr = (
        f"pipeline = {pipeline_class.__name__}(component_graph={component_graph_str}, "
        f"parameters={{'Imputer':{{'categorical_impute_strategy': 'most_frequent', 'numeric_impute_strategy': 'mean', 'categorical_fill_value': None, 'numeric_fill_value': 42}}, '{final_estimator}':{{'n_estimators': 100, 'max_depth': 6, 'n_jobs': -1}}}}, "
        "custom_name='Mock Pipeline', random_seed=0)"
    )
    assert repr(pipeline_with_parameters) == expected_repr

    pipeline_with_inf_parameters = pipeline_class(
        component_graph=component_graph,
        parameters={
            "Imputer": {
                "numeric_fill_value": float("inf"),
                "categorical_fill_value": np.inf,
            }
        },
    )
    expected_repr = (
        f"pipeline = {pipeline_class.__name__}(component_graph={component_graph_str}, "
        f"parameters={{'Imputer':{{'categorical_impute_strategy': 'most_frequent', 'numeric_impute_strategy': 'mean', 'categorical_fill_value': float('inf'), 'numeric_fill_value': float('inf')}}, '{final_estimator}':{{'n_estimators': 100, 'max_depth': 6, 'n_jobs': -1}}}}, random_seed=0)"
    )
    assert repr(pipeline_with_inf_parameters) == expected_repr

    pipeline_with_nan_parameters = pipeline_class(
        component_graph=component_graph,
        parameters={
            "Imputer": {
                "numeric_fill_value": float("nan"),
                "categorical_fill_value": np.nan,
            }
        },
    )
    expected_repr = (
        f"pipeline = {pipeline_class.__name__}(component_graph={component_graph_str}, "
        f"parameters={{'Imputer':{{'categorical_impute_strategy': 'most_frequent', 'numeric_impute_strategy': 'mean', 'categorical_fill_value': np.nan, 'numeric_fill_value': np.nan}}, '{final_estimator}':{{'n_estimators': 100, 'max_depth': 6, 'n_jobs': -1}}}}, random_seed=0)"
    )
    assert repr(pipeline_with_nan_parameters) == expected_repr


@pytest.mark.parametrize(
    "pipeline_class",
    [
        BinaryClassificationPipeline,
        MulticlassClassificationPipeline,
        RegressionPipeline,
    ],
)
def test_nonlinear_pipeline_repr(pipeline_class):
    if pipeline_class in [
        BinaryClassificationPipeline,
        MulticlassClassificationPipeline,
    ]:
        final_estimator = "Random Forest Classifier"
    else:
        final_estimator = "Random Forest Regressor"

    custom_name = "Mock Pipeline"
    component_graph = {
        "Imputer": ["Imputer", "X", "y"],
        "OHE_1": ["One Hot Encoder", "Imputer.x", "y"],
        "OHE_2": ["One Hot Encoder", "Imputer.x", "y"],
        "Estimator": [final_estimator, "OHE_1.x", "OHE_2.x", "y"],
    }

    pipeline = pipeline_class(component_graph=component_graph, custom_name=custom_name)
    component_graph_str = f"{{'Imputer': ['Imputer', 'X', 'y'], 'OHE_1': ['One Hot Encoder', 'Imputer.x', 'y'], 'OHE_2': ['One Hot Encoder', 'Imputer.x', 'y'], 'Estimator': ['{final_estimator}', 'OHE_1.x', 'OHE_2.x', 'y']}}"
    expected_repr = (
        f"pipeline = {pipeline_class.__name__}(component_graph={component_graph_str}, "
        "parameters={'Imputer':{'categorical_impute_strategy': 'most_frequent', 'numeric_impute_strategy': 'mean', 'categorical_fill_value': None, 'numeric_fill_value': None}, "
        "'OHE_1':{'top_n': 10, 'features_to_encode': None, 'categories': None, 'drop': 'if_binary', 'handle_unknown': 'ignore', 'handle_missing': 'error'}, "
        "'OHE_2':{'top_n': 10, 'features_to_encode': None, 'categories': None, 'drop': 'if_binary', 'handle_unknown': 'ignore', 'handle_missing': 'error'}, "
        "'Estimator':{'n_estimators': 100, 'max_depth': 6, 'n_jobs': -1}}, custom_name='Mock Pipeline', random_seed=0)"
    )
    assert repr(pipeline) == expected_repr

    pipeline_with_parameters = pipeline_class(
        component_graph=component_graph,
        custom_name=custom_name,
        parameters={"Imputer": {"numeric_fill_value": 42}},
    )
    expected_repr = (
        f"pipeline = {pipeline_class.__name__}(component_graph={component_graph_str}, "
        "parameters={'Imputer':{'categorical_impute_strategy': 'most_frequent', 'numeric_impute_strategy': 'mean', 'categorical_fill_value': None, 'numeric_fill_value': 42}, "
        "'OHE_1':{'top_n': 10, 'features_to_encode': None, 'categories': None, 'drop': 'if_binary', 'handle_unknown': 'ignore', 'handle_missing': 'error'}, "
        "'OHE_2':{'top_n': 10, 'features_to_encode': None, 'categories': None, 'drop': 'if_binary', 'handle_unknown': 'ignore', 'handle_missing': 'error'}, "
        "'Estimator':{'n_estimators': 100, 'max_depth': 6, 'n_jobs': -1}}, custom_name='Mock Pipeline', random_seed=0)"
    )
    assert repr(pipeline_with_parameters) == expected_repr

    pipeline_with_inf_parameters = pipeline_class(
        component_graph=component_graph,
        custom_name=custom_name,
        parameters={
            "Imputer": {
                "numeric_fill_value": float("inf"),
                "categorical_fill_value": np.inf,
            }
        },
    )
    expected_repr = (
        f"pipeline = {pipeline_class.__name__}(component_graph={component_graph_str}, "
        "parameters={'Imputer':{'categorical_impute_strategy': 'most_frequent', 'numeric_impute_strategy': 'mean', 'categorical_fill_value': float('inf'), 'numeric_fill_value': float('inf')}, "
        "'OHE_1':{'top_n': 10, 'features_to_encode': None, 'categories': None, 'drop': 'if_binary', 'handle_unknown': 'ignore', 'handle_missing': 'error'}, "
        "'OHE_2':{'top_n': 10, 'features_to_encode': None, 'categories': None, 'drop': 'if_binary', 'handle_unknown': 'ignore', 'handle_missing': 'error'}, "
        "'Estimator':{'n_estimators': 100, 'max_depth': 6, 'n_jobs': -1}}, custom_name='Mock Pipeline', random_seed=0)"
    )
    assert repr(pipeline_with_inf_parameters) == expected_repr

    pipeline_with_nan_parameters = pipeline_class(
        component_graph=component_graph,
        custom_name=custom_name,
        parameters={
            "Imputer": {
                "numeric_fill_value": float("nan"),
                "categorical_fill_value": np.nan,
            }
        },
    )
    expected_repr = (
        f"pipeline = {pipeline_class.__name__}(component_graph={component_graph_str}, "
        "parameters={'Imputer':{'categorical_impute_strategy': 'most_frequent', 'numeric_impute_strategy': 'mean', 'categorical_fill_value': np.nan, 'numeric_fill_value': np.nan}, "
        "'OHE_1':{'top_n': 10, 'features_to_encode': None, 'categories': None, 'drop': 'if_binary', 'handle_unknown': 'ignore', 'handle_missing': 'error'}, "
        "'OHE_2':{'top_n': 10, 'features_to_encode': None, 'categories': None, 'drop': 'if_binary', 'handle_unknown': 'ignore', 'handle_missing': 'error'}, "
        "'Estimator':{'n_estimators': 100, 'max_depth': 6, 'n_jobs': -1}}, custom_name='Mock Pipeline', random_seed=0)"
    )
    assert repr(pipeline_with_nan_parameters) == expected_repr


@pytest.mark.parametrize("problem_type", ProblemTypes.all_problem_types)
def test_predict_has_input_target_name(
    problem_type,
    X_y_binary,
    X_y_multi,
    X_y_regression,
    ts_data,
    logistic_regression_binary_pipeline_class,
    logistic_regression_multiclass_pipeline_class,
    linear_regression_pipeline_class,
    time_series_regression_pipeline_class,
    time_series_binary_classification_pipeline_class,
    time_series_multiclass_classification_pipeline_class,
):
    if problem_type == ProblemTypes.BINARY:
        X, y = X_y_binary
        clf = logistic_regression_binary_pipeline_class(
            parameters={"Logistic Regression Classifier": {"n_jobs": 1}}
        )

    elif problem_type == ProblemTypes.MULTICLASS:
        X, y = X_y_multi
        clf = logistic_regression_multiclass_pipeline_class(
            parameters={"Logistic Regression Classifier": {"n_jobs": 1}}
        )

    elif problem_type == ProblemTypes.REGRESSION:
        X, y = X_y_regression
        clf = linear_regression_pipeline_class(
            parameters={"Linear Regressor": {"n_jobs": 1}}
        )

    elif problem_type == ProblemTypes.TIME_SERIES_REGRESSION:
        X, y = ts_data
        X_validation = X[29:31]
        X, y = X[:29], y[:29]
        clf = time_series_regression_pipeline_class(
            parameters={
                "pipeline": {
                    "gap": 0,
                    "max_delay": 0,
                    "date_index": None,
                    "forecast_horizon": 2,
                },
                "Delayed Feature Transformer": {
                    "gap": 0,
                    "max_delay": 0,
                    "forecast_horizon": 2,
                },
            }
        )
    elif problem_type == ProblemTypes.TIME_SERIES_BINARY:
        X, y = X_y_binary
        X, y = pd.DataFrame(X), pd.Series(y)
        X_validation = X[50:52]
        X, y = X[:50], y[:50]
        clf = time_series_binary_classification_pipeline_class(
            parameters={
                "Logistic Regression Classifier": {"n_jobs": 1},
                "Delayed Feature Transformer": {
                    "gap": 0,
                    "max_delay": 0,
                    "forecast_horizon": 2,
                },
                "pipeline": {
                    "gap": 0,
                    "max_delay": 0,
                    "date_index": None,
                    "forecast_horizon": 2,
                },
            }
        )
    elif problem_type == ProblemTypes.TIME_SERIES_MULTICLASS:
        X, y = X_y_multi
        X, y = pd.DataFrame(X), pd.Series(y)
        X_validation = X[50:52]
        X, y = X[:50], y[:50]
        clf = time_series_multiclass_classification_pipeline_class(
            parameters={
                "Logistic Regression Classifier": {"n_jobs": 1},
                "Delayed Feature Transformer": {
                    "gap": 0,
                    "max_delay": 0,
                    "forecast_horizon": 2,
                },
                "pipeline": {
                    "gap": 0,
                    "max_delay": 0,
                    "date_index": None,
                    "forecast_horizon": 2,
                },
            }
        )
    y = pd.Series(y, name="test target name")
    clf.fit(X, y)
    if is_time_series(problem_type):
        predictions = clf.predict(X_validation, None, X, y)
    else:
        predictions = clf.predict(X)
    assert predictions.name == "test target name"


def test_linear_pipeline_iteration(logistic_regression_binary_pipeline_class):
    expected_order = [
        Imputer(),
        OneHotEncoder(),
        StandardScaler(),
        LogisticRegressionClassifier(),
    ]

    pipeline = logistic_regression_binary_pipeline_class({})
    order = [c for c in pipeline]
    order_again = [c for c in pipeline]

    assert order == expected_order
    assert order_again == expected_order

    expected_order_params = [
        Imputer(numeric_impute_strategy="median"),
        OneHotEncoder(top_n=2),
        StandardScaler(),
        LogisticRegressionClassifier(),
    ]

    pipeline = logistic_regression_binary_pipeline_class(
        {
            "One Hot Encoder": {"top_n": 2},
            "Imputer": {"numeric_impute_strategy": "median"},
        }
    )
    order_params = [c for c in pipeline]
    order_again_params = [c for c in pipeline]

    assert order_params == expected_order_params
    assert order_again_params == expected_order_params


def test_nonlinear_pipeline_iteration(nonlinear_binary_pipeline_class):
    expected_order = [
        Imputer(),
        OneHotEncoder(),
        ElasticNetClassifier(),
        OneHotEncoder(),
        RandomForestClassifier(),
        LogisticRegressionClassifier(),
    ]

    pipeline = nonlinear_binary_pipeline_class({})
    order = [c for c in pipeline]
    order_again = [c for c in pipeline]

    assert order == expected_order
    assert order_again == expected_order

    expected_order_params = [
        Imputer(),
        OneHotEncoder(top_n=2),
        ElasticNetClassifier(),
        OneHotEncoder(top_n=5),
        RandomForestClassifier(),
        LogisticRegressionClassifier(),
    ]

    pipeline = nonlinear_binary_pipeline_class(
        {"OneHot_ElasticNet": {"top_n": 2}, "OneHot_RandomForest": {"top_n": 5}}
    )
    order_params = [c for c in pipeline]
    order_again_params = [c for c in pipeline]

    assert order_params == expected_order_params
    assert order_again_params == expected_order_params


def test_linear_getitem(logistic_regression_binary_pipeline_class):
    pipeline = logistic_regression_binary_pipeline_class(
        {"One Hot Encoder": {"top_n": 4}}
    )

    assert pipeline[0] == Imputer()
    assert pipeline[1] == OneHotEncoder(top_n=4)
    assert pipeline[2] == StandardScaler()
    assert pipeline[3] == LogisticRegressionClassifier()

    assert pipeline["Imputer"] == Imputer()
    assert pipeline["One Hot Encoder"] == OneHotEncoder(top_n=4)
    assert pipeline["Standard Scaler"] == StandardScaler()
    assert pipeline["Logistic Regression Classifier"] == LogisticRegressionClassifier()


def test_nonlinear_getitem(nonlinear_binary_pipeline_class):
    pipeline = nonlinear_binary_pipeline_class({"OneHot_RandomForest": {"top_n": 4}})

    assert pipeline[0] == Imputer()
    assert pipeline[1] == OneHotEncoder()
    assert pipeline[2] == ElasticNetClassifier()
    assert pipeline[3] == OneHotEncoder(top_n=4)
    assert pipeline[4] == RandomForestClassifier()
    assert pipeline[5] == LogisticRegressionClassifier()

    assert pipeline["Imputer"] == Imputer()
    assert pipeline["OneHot_ElasticNet"] == OneHotEncoder()
    assert pipeline["Elastic Net"] == ElasticNetClassifier()
    assert pipeline["OneHot_RandomForest"] == OneHotEncoder(top_n=4)
    assert pipeline["Random Forest"] == RandomForestClassifier()
    assert pipeline["Logistic Regression"] == LogisticRegressionClassifier()


def test_get_component(
    logistic_regression_binary_pipeline_class, nonlinear_binary_pipeline_class
):
    pipeline = logistic_regression_binary_pipeline_class(
        {"One Hot Encoder": {"top_n": 4}}
    )

    assert pipeline.get_component("Imputer") == Imputer()
    assert pipeline.get_component("One Hot Encoder") == OneHotEncoder(top_n=4)
    assert pipeline.get_component("Standard Scaler") == StandardScaler()
    assert (
        pipeline.get_component("Logistic Regression Classifier")
        == LogisticRegressionClassifier()
    )

    pipeline = nonlinear_binary_pipeline_class({"OneHot_RandomForest": {"top_n": 4}})

    assert pipeline.get_component("Imputer") == Imputer()
    assert pipeline.get_component("OneHot_ElasticNet") == OneHotEncoder()
    assert pipeline.get_component("Elastic Net") == ElasticNetClassifier()
    assert pipeline.get_component("OneHot_RandomForest") == OneHotEncoder(top_n=4)
    assert pipeline.get_component("Random Forest") == RandomForestClassifier()
    assert (
        pipeline.get_component("Logistic Regression") == LogisticRegressionClassifier()
    )


@pytest.mark.parametrize("problem_type", ProblemTypes.all_problem_types)
def test_score_error_when_custom_objective_not_instantiated(
    problem_type,
    logistic_regression_binary_pipeline_class,
    dummy_multiclass_pipeline_class,
    dummy_regression_pipeline_class,
    X_y_binary,
):
    pipeline = dummy_regression_pipeline_class({})
    if is_binary(problem_type):
        pipeline = logistic_regression_binary_pipeline_class({})
    elif is_multiclass(problem_type):
        pipeline = dummy_multiclass_pipeline_class({})

    X, y = X_y_binary
    pipeline.fit(X, y)
    msg = "Cannot pass cost benefit matrix as a string in pipeline.score. Instantiate first and then add it to the list of objectives."
    with pytest.raises(ObjectiveCreationError, match=msg):
        pipeline.score(X, y, objectives=["cost benefit matrix", "F1"])

    # Verify ObjectiveCreationError only raised when string matches an existing objective
    with pytest.raises(
        ObjectiveNotFoundError, match="cost benefit is not a valid Objective!"
    ):
        pipeline.score(X, y, objectives=["cost benefit", "F1"])

    # Verify no exception when objective properly specified
    if is_binary(problem_type):
        pipeline.score(X, y, objectives=[CostBenefitMatrix(1, 1, -1, -1), "F1"])


@pytest.mark.parametrize("is_time_series", [True, False])
def test_binary_pipeline_string_target_thresholding(
    is_time_series,
    make_data_type,
    time_series_binary_classification_pipeline_class,
    logistic_regression_binary_pipeline_class,
    X_y_binary,
):
    X, y = X_y_binary
    X = make_data_type("ww", X)
    y = ww.init_series(pd.Series([f"String value {i}" for i in y]), "Categorical")
    X_train, y_train = X.ww.iloc[:80], y.ww.iloc[:80]
    X_validation, y_validation = X.ww.iloc[80:83], y.ww.iloc[80:83]
    objective = get_objective("F1", return_instance=True)
    pipeline_class = (
        time_series_binary_classification_pipeline_class
        if is_time_series
        else logistic_regression_binary_pipeline_class
    )

    pipeline = pipeline_class(
        parameters={
            "Logistic Regression Classifier": {"n_jobs": 1},
            "pipeline": {
                "gap": 0,
                "max_delay": 1,
                "date_index": None,
                "forecast_horizon": 3,
            },
        }
    )
    pipeline.fit(X_train, y_train)
    assert pipeline.threshold is None
    pred_proba = (
        pipeline.predict_proba(X_validation, X_train, y_train).iloc[:, 1]
        if is_time_series
        else pipeline.predict_proba(X_validation).iloc[:, 1]
    )
    pipeline.optimize_threshold(X_validation, y_validation, pred_proba, objective)
    assert pipeline.threshold is not None


@patch("evalml.pipelines.components.LogisticRegressionClassifier.fit")
def test_undersampler_component_in_pipeline_fit(mock_fit):
    X = pd.DataFrame({"a": [i for i in range(1000)], "b": [i % 3 for i in range(1000)]})
    y = pd.Series([0] * 100 + [1] * 900)
    pipeline = BinaryClassificationPipeline(
        {
            "Imputer": ["Imputer", "X", "y"],
            "Undersampler": ["Undersampler", "Imputer.x", "y"],
            "Logistic Regression Classifier": [
                "Logistic Regression Classifier",
                "Undersampler.x",
                "Undersampler.y",
            ],
        }
    )
    pipeline.fit(X, y)
    # make sure we undersample to 500 values in the X and y
    assert len(mock_fit.call_args[0][0]) == 500
    assert all(mock_fit.call_args[0][1].value_counts().values == [400, 100])

    # balance the data
    y_balanced = pd.Series([0] * 400 + [1] * 600)
    pipeline.fit(X, y_balanced)
    assert len(mock_fit.call_args[0][0]) == 1000


def test_undersampler_component_in_pipeline_predict():
    X = pd.DataFrame({"a": [i for i in range(1000)], "b": [i % 3 for i in range(1000)]})
    y = pd.Series([0] * 100 + [1] * 900)
    pipeline = BinaryClassificationPipeline(
        {
            "Imputer": ["Imputer", "X", "y"],
            "Undersampler": ["Undersampler", "Imputer.x", "y"],
            "Logistic Regression Classifier": [
                "Logistic Regression Classifier",
                "Undersampler.x",
                "Undersampler.y",
            ],
        }
    )
    pipeline.fit(X, y)
    preds = pipeline.predict(X)
    assert len(preds) == 1000
    preds = pipeline.predict_proba(X)
    assert len(preds) == 1000


@patch("evalml.pipelines.components.LogisticRegressionClassifier.fit")
def test_oversampler_component_in_pipeline_fit(mock_fit):
    pytest.importorskip(
        "imblearn.over_sampling",
        reason="Skipping test because imbalanced-learn not installed",
    )

    X = pd.DataFrame(
        {
            "a": [i for i in range(1000)],
            "b": [i % 3 for i in range(1000)],
            "c": [i % 7 for i in range(1, 1001)],
        }
    )
    X.ww.init(logical_types={"c": "Categorical"})
    y = pd.Series([0] * 100 + [1] * 900)
    pipeline = BinaryClassificationPipeline(
        {
            "Imputer": ["Imputer", "X", "y"],
            "Oversampler": ["Oversampler", "Imputer.x", "y"],
            "Logistic Regression Classifier": [
                "Logistic Regression Classifier",
                "Oversampler.x",
                "Oversampler.y",
            ],
        }
    )
    pipeline.fit(X, y)
    # make sure we oversample 0 to 225 values values in the X and y
    assert len(mock_fit.call_args[0][0]) == 1125
    assert all(mock_fit.call_args[0][1].value_counts().values == [900, 225])

    # balance the data
    y_balanced = pd.Series([0] * 400 + [1] * 600)
    pipeline.fit(X, y_balanced)
    assert len(mock_fit.call_args[0][0]) == 1000


def test_oversampler_component_in_pipeline_predict():
    pytest.importorskip(
        "imblearn.over_sampling",
        reason="Skipping test because imbalanced-learn not installed",
    )
    X = pd.DataFrame(
        {
            "a": [i for i in range(1000)],
            "b": [i % 3 for i in range(1000)],
            "c": [i % 7 for i in range(1, 1001)],
        }
    )
    X.ww.init(logical_types={"c": "Categorical"})
    y = pd.Series([0] * 100 + [1] * 900)
    pipeline = BinaryClassificationPipeline(
        {
            "Imputer": ["Imputer", "X", "y"],
            "Oversampler": ["Oversampler", "Imputer.x", "y"],
            "Logistic Regression Classifier": [
                "Logistic Regression Classifier",
                "Oversampler.x",
                "Oversampler.y",
            ],
        }
    )
    pipeline.fit(X, y)
    preds = pipeline.predict(X)
    assert len(preds) == 1000
    preds = pipeline.predict_proba(X)
    assert len(preds) == 1000


@pytest.mark.parametrize(
    "pipeline_class",
    [
        BinaryClassificationPipeline,
        MulticlassClassificationPipeline,
        RegressionPipeline,
    ],
)
def test_pipeline_init_from_component_list(pipeline_class):
    if pipeline_class in [
        BinaryClassificationPipeline,
        MulticlassClassificationPipeline,
    ]:
        estimator = "Random Forest Classifier"
        estimator_class = RandomForestClassifier
    else:
        estimator = "Random Forest Regressor"
        estimator_class = RandomForestRegressor

    assert pipeline_class([estimator]).component_graph == ComponentGraph(
        {estimator: [estimator_class, "X", "y"]}
    )
    assert pipeline_class([Imputer]).component_graph == ComponentGraph(
        {"Imputer": [Imputer, "X", "y"]}
    )
    assert pipeline_class(
        [Imputer, OneHotEncoder, DropNullColumns]
    ).component_graph == ComponentGraph(
        {
            "Imputer": [Imputer, "X", "y"],
            "One Hot Encoder": [OneHotEncoder, "Imputer.x", "y"],
            "Drop Null Columns Transformer": [
                DropNullColumns,
                "One Hot Encoder.x",
                "y",
            ],
        }
    )

    # Test with component after estimator
    assert pipeline_class(
        [Imputer, estimator, Imputer]
    ).component_graph == ComponentGraph(
        {
            "Imputer": [Imputer, "X", "y"],
            estimator: [
                estimator_class,
                "Imputer.x",
                "y",
            ],
            "Imputer_2": [Imputer, f"{estimator}.x", "y"],
        }
    )


@pytest.mark.parametrize(
    "pipeline_class",
    [
        BinaryClassificationPipeline,
        MulticlassClassificationPipeline,
        RegressionPipeline,
    ],
)
def test_pipeline_init_from_component_list_with_duplicate_components(pipeline_class):
    if pipeline_class in [
        BinaryClassificationPipeline,
        MulticlassClassificationPipeline,
    ]:
        estimator = "Random Forest Classifier"
        estimator_class = RandomForestClassifier
    else:
        estimator = "Random Forest Regressor"
        estimator_class = RandomForestRegressor

    assert pipeline_class([estimator, estimator]).component_graph == ComponentGraph(
        {
            estimator: [estimator_class, "X", "y"],
            f"{estimator}_1": [
                estimator_class,
                f"{estimator}.x",
                "y",
            ],
        }
    )
    assert pipeline_class(
        [Imputer, Imputer, estimator_class]
    ).component_graph == ComponentGraph(
        {
            "Imputer": [Imputer, "X", "y"],
            "Imputer_1": [Imputer, "Imputer.x", "y"],
            estimator: [
                estimator_class,
                "Imputer_1.x",
                "y",
            ],
        }
    )


def test_make_component_dict_from_component_list():
    assert PipelineBase._make_component_dict_from_component_list(
        [RandomForestClassifier]
    ) == {"Random Forest Classifier": [RandomForestClassifier, "X", "y"]}
    assert PipelineBase._make_component_dict_from_component_list([Imputer]) == {
        "Imputer": [Imputer, "X", "y"]
    }
    assert PipelineBase._make_component_dict_from_component_list(
        [Imputer, OneHotEncoder, DropNullColumns]
    ) == {
        "Imputer": [Imputer, "X", "y"],
        "One Hot Encoder": [OneHotEncoder, "Imputer.x", "y"],
        "Drop Null Columns Transformer": [DropNullColumns, "One Hot Encoder.x", "y"],
    }

    # Test with component that modifies y (Target Imputer)
    assert PipelineBase._make_component_dict_from_component_list(
        [Imputer, OneHotEncoder, TargetImputer, RandomForestClassifier]
    ) == {
        "Imputer": [Imputer, "X", "y"],
        "One Hot Encoder": [OneHotEncoder, "Imputer.x", "y"],
        "Target Imputer": [TargetImputer, "One Hot Encoder.x", "y"],
        "Random Forest Classifier": [
            RandomForestClassifier,
            "One Hot Encoder.x",
            "Target Imputer.y",
        ],
    }

    # Test with component that modifies X and y (Undersampler)
    assert PipelineBase._make_component_dict_from_component_list(
        [
            Imputer,
            OneHotEncoder,
            TargetImputer,
            DropNullColumns,
            Undersampler,
            RandomForestClassifier,
        ]
    ) == {
        "Imputer": [Imputer, "X", "y"],
        "One Hot Encoder": [OneHotEncoder, "Imputer.x", "y"],
        "Target Imputer": [TargetImputer, "One Hot Encoder.x", "y"],
        "Drop Null Columns Transformer": [
            DropNullColumns,
            "One Hot Encoder.x",
            "Target Imputer.y",
        ],
        "Undersampler": [
            Undersampler,
            "Drop Null Columns Transformer.x",
            "Target Imputer.y",
        ],
        "Random Forest Classifier": [
            RandomForestClassifier,
            "Undersampler.x",
            "Undersampler.y",
        ],
    }

    # Test with component after estimator
    assert PipelineBase._make_component_dict_from_component_list(
        [Imputer, RandomForestClassifier, Imputer]
    ) == {
        "Imputer": [Imputer, "X", "y"],
        "Random Forest Classifier": [
            RandomForestClassifier,
            "Imputer.x",
            "y",
        ],
        "Imputer_2": [Imputer, "Random Forest Classifier.x", "y"],
    }


def test_make_component_dict_from_component_list_with_duplicate_names():
    assert PipelineBase._make_component_dict_from_component_list(
        [RandomForestClassifier, RandomForestClassifier]
    ) == {
        "Random Forest Classifier": [RandomForestClassifier, "X", "y"],
        "Random Forest Classifier_1": [
            RandomForestClassifier,
            "Random Forest Classifier.x",
            "y",
        ],
    }
    assert PipelineBase._make_component_dict_from_component_list(
        [Imputer, Imputer, RandomForestClassifier]
    ) == {
        "Imputer": [Imputer, "X", "y"],
        "Imputer_1": [Imputer, "Imputer.x", "y"],
        "Random Forest Classifier": [
            RandomForestClassifier,
            "Imputer_1.x",
            "y",
        ],
    }
    assert PipelineBase._make_component_dict_from_component_list(
        [TargetImputer, TargetImputer, RandomForestClassifier]
    ) == {
        "Target Imputer": [TargetImputer, "X", "y"],
        "Target Imputer_1": [TargetImputer, "X", "Target Imputer.y"],
        "Random Forest Classifier": [
            RandomForestClassifier,
            "X",
            "Target Imputer_1.y",
        ],
    }
    assert PipelineBase._make_component_dict_from_component_list(
        [Undersampler, Undersampler, RandomForestClassifier]
    ) == {
        "Undersampler": [Undersampler, "X", "y"],
        "Undersampler_1": [Undersampler, "Undersampler.x", "Undersampler.y"],
        "Random Forest Classifier": [
            RandomForestClassifier,
            "Undersampler_1.x",
            "Undersampler_1.y",
        ],
    }


def test_get_hyperparameter_ranges():
    pipeline = BinaryClassificationPipeline(
        component_graph=["Imputer", "Random Forest Classifier"]
    )
    custom_hyperparameters = {
        "One Hot Encoder": {"top_n": 3},
        "Imputer": {"numeric_impute_strategy": Categorical(["most_frequent", "mean"])},
        "Random Forest Classifier": {"n_estimators": Integer(150, 160)},
    }

    expected_ranges = {
        "Imputer": {
            "categorical_impute_strategy": ["most_frequent"],
            "numeric_impute_strategy": Categorical(
                categories=("most_frequent", "mean"), prior=None
            ),
        },
        "Random Forest Classifier": {
            "n_estimators": Integer(
                low=150, high=160, prior="uniform", transform="identity"
            ),
            "max_depth": Integer(low=1, high=10, prior="uniform", transform="identity"),
        },
    }
    hyperparameter_ranges = pipeline.get_hyperparameter_ranges(custom_hyperparameters)
    assert expected_ranges == hyperparameter_ranges


@pytest.mark.parametrize(
    "problem_type",
    [
        ProblemTypes.BINARY,
        ProblemTypes.MULTICLASS,
        ProblemTypes.REGRESSION,
    ],
)
def test_pipeline_predict_without_final_estimator(
    problem_type, make_data_type, X_y_binary
):
    X, y = X_y_binary
    X = make_data_type("ww", X)
    y = make_data_type("ww", y)
    pipeline_class = _get_pipeline_base_class(problem_type)
    pipeline = pipeline_class(
        component_graph={
            "Imputer": ["Imputer", "X", "y"],
            "OHE": ["One Hot Encoder", "Imputer.x", "y"],
        },
    )
    pipeline.fit(X, y)
    if is_classification(problem_type):
        with pytest.raises(
            ValueError,
            match=re.escape(
                "Cannot call predict_proba() on a component graph because the final component is not an Estimator."
            ),
        ):
            pipeline.predict_proba(X)

    with pytest.raises(
        ValueError,
        match=re.escape(
            "Cannot call predict() on a component graph because the final component is not an Estimator."
        ),
    ):
        pipeline.predict(X)


@patch("evalml.pipelines.components.Imputer.transform")
@patch("evalml.pipelines.components.OneHotEncoder.transform")
@pytest.mark.parametrize(
    "problem_type",
    [
        ProblemTypes.BINARY,
        ProblemTypes.MULTICLASS,
        ProblemTypes.REGRESSION,
    ],
)
def test_pipeline_transform(
    mock_ohe_transform, mock_imputer_transform, problem_type, X_y_binary, make_data_type
):
    component_graph = {
        "Imputer": ["Imputer", "X", "y"],
        "OHE": ["One Hot Encoder", "Imputer.x", "y"],
    }
    X, y = X_y_binary
    X = make_data_type("ww", X)
    y = make_data_type("ww", y)
    mock_imputer_transform.return_value = X
    mock_ohe_transform.return_value = X
    pipeline_class = _get_pipeline_base_class(problem_type)

    pipeline = pipeline_class(component_graph=component_graph)
    pipeline.fit(X, y)
    transformed_X = pipeline.transform(X, y)
    assert_frame_equal(X, transformed_X)


@pytest.mark.parametrize(
    "problem_type",
    [
        ProblemTypes.BINARY,
        ProblemTypes.MULTICLASS,
        ProblemTypes.REGRESSION,
    ],
)
def test_pipeline_transform_with_final_estimator(
    problem_type, X_y_binary, X_y_multi, X_y_regression
):
    X, y = X_y_binary
    if problem_type == ProblemTypes.BINARY:
        pipeline = BinaryClassificationPipeline(
            component_graph=["Logistic Regression Classifier"],
            parameters={
                "Logistic Regression Classifier": {"n_jobs": 1},
            },
        )

    elif problem_type == ProblemTypes.MULTICLASS:
        X, y = X_y_multi
        pipeline = MulticlassClassificationPipeline(
            component_graph=["Logistic Regression Classifier"],
            parameters={
                "Logistic Regression Classifier": {"n_jobs": 1},
            },
        )
    elif problem_type == ProblemTypes.REGRESSION:
        X, y = X_y_regression
        pipeline = RegressionPipeline(
            component_graph=["Random Forest Regressor"],
            parameters={
                "Random Forest Regressor": {"n_jobs": 1},
            },
        )

    pipeline.fit(X, y)
    with pytest.raises(
        ValueError,
        match=re.escape(
            "Cannot call transform() on a component graph because the final component is not a Transformer."
        ),
    ):
        pipeline.transform(X, y)


@patch("evalml.pipelines.components.LogisticRegressionClassifier.fit")
def test_training_only_component_in_pipeline_fit(mock_fit, X_y_binary):
    X, y = X_y_binary
    pipeline = BinaryClassificationPipeline(
        {
            "Imputer": ["Imputer", "X", "y"],
            "Drop Rows Transformer": [DropRowsTransformer, "Imputer.x", "y"],
            "Logistic Regression Classifier": [
                "Logistic Regression Classifier",
                "Drop Rows Transformer.x",
                "Drop Rows Transformer.y",
            ],
        },
        parameters={"Drop Rows Transformer": {"indices_to_drop": [0, 9]}},
    )
    pipeline.fit(X, y)
    assert len(mock_fit.call_args[0][0]) == len(X) - 2


def test_training_only_component_in_pipeline_predict_and_compute_estimator_features(
    X_y_binary,
):
    # Test that calling predict() and `compute_estimator_features` will not evaluate any training-only transformations
    X, y = X_y_binary
    pipeline = BinaryClassificationPipeline(
        {
            "Imputer": ["Imputer", "X", "y"],
            "Drop Rows Transformer": [DropRowsTransformer, "Imputer.x", "y"],
            "Logistic Regression Classifier": [
                "Logistic Regression Classifier",
                "Drop Rows Transformer.x",
                "Drop Rows Transformer.y",
            ],
        },
        parameters={"Drop Rows Transformer": {"indices_to_drop": [9]}},
    )
    pipeline.fit(X, y)
    preds = pipeline.predict(X)
    assert len(preds) == len(X)
    preds = pipeline.predict_proba(X)
    assert len(preds) == len(X)
    estimator_features = pipeline.compute_estimator_features(X, y)
    assert len(estimator_features) == len(X)


def test_training_only_component_in_pipeline_transform(X_y_binary):
    # Test that calling transform() will evaluate all training-only transformations
    X, y = X_y_binary
    pipeline = BinaryClassificationPipeline(
        {
            "Imputer": ["Imputer", "X", "y"],
            "Drop Rows Transformer": [DropRowsTransformer, "Imputer.x", "y"],
        },
        parameters={"Drop Rows Transformer": {"indices_to_drop": [0, 9]}},
    )
    pipeline.fit(X, y)
    transformed = pipeline.transform(X)
    assert len(transformed) == len(X) - 2


def test_component_graph_pipeline():
    classification_cg = ComponentGraph(
        {
            "Imputer": ["Imputer", "X", "y"],
            "Undersampler": ["Undersampler", "Imputer.x", "y"],
            "Logistic Regression Classifier": [
                "Logistic Regression Classifier",
                "Undersampler.x",
                "Undersampler.y",
            ],
        }
    )

    regression_cg = ComponentGraph(
        {
            "Imputer": ["Imputer", "X", "y"],
            "Linear Regressor": [
                "Linear Regressor",
                "Imputer.x",
                "y",
            ],
        }
    )

    no_estimator_cg = ComponentGraph(
        {
            "Imputer": ["Imputer", "X", "y"],
            "Undersampler": ["Undersampler", "Imputer.x", "y"],
        }
    )

    assert (
        BinaryClassificationPipeline(classification_cg).component_graph
        == classification_cg
    )
    assert RegressionPipeline(regression_cg).component_graph == regression_cg
    assert (
        BinaryClassificationPipeline(no_estimator_cg).component_graph == no_estimator_cg
    )
    with pytest.raises(
        ValueError, match="Problem type regression not valid for this component graph"
    ):
        RegressionPipeline(classification_cg)


def test_component_graph_pipeline_initialized():
    component_graph1 = ComponentGraph(
        {
            "Imputer": ["Imputer", "X", "y"],
            "Undersampler": ["Undersampler", "Imputer.x", "y"],
            "Logistic Regression Classifier": [
                "Logistic Regression Classifier",
                "Undersampler.x",
                "Undersampler.y",
            ],
        }
    )
    component_graph1.instantiate({"Imputer": {"numeric_impute_strategy": "mean"}})
    assert (
        component_graph1.component_instances["Imputer"].parameters[
            "numeric_impute_strategy"
        ]
        == "mean"
    )

    # make sure the value gets overwritten when reinitialized
    bcp = BinaryClassificationPipeline(
        component_graph1, parameters={"Imputer": {"numeric_impute_strategy": "median"}}
    )
    assert bcp.parameters["Imputer"]["numeric_impute_strategy"] == "median"
    assert (
        bcp.component_graph.component_instances["Imputer"].parameters[
            "numeric_impute_strategy"
        ]
        == "median"
    )
