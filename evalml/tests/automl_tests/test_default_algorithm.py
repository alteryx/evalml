from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from skopt.space import Categorical, Integer

from evalml.automl.automl_algorithm import DefaultAlgorithm
from evalml.model_family import ModelFamily
from evalml.pipelines.components import (
    ElasticNetClassifier,
    ElasticNetRegressor,
    LogisticRegressionClassifier,
    RandomForestClassifier,
    StackedEnsembleClassifier,
    StackedEnsembleRegressor,
)
from evalml.problem_types import ProblemTypes


def test_default_algorithm_init(X_y_binary):
    X, y = X_y_binary
    problem_type = ProblemTypes.BINARY
    sampler_name = "Undersampler"

    algo = DefaultAlgorithm(X, y, problem_type, sampler_name, verbose=True)

    assert algo.problem_type == problem_type
    assert algo.sampler_name == sampler_name
    assert algo.pipeline_number == 0
    assert algo.batch_number == 0
    assert algo.allowed_pipelines == []
    assert algo.verbose is True


def test_default_algorithm_init_clustering(X_y_binary):
    X, _ = X_y_binary

    algo = DefaultAlgorithm(X, None, "clustering", "Undersampler")
    assert algo.y is None
    assert algo.problem_type == "clustering"

    with pytest.raises(
        ValueError, match="y cannot be None for supervised learning problems"
    ):
        DefaultAlgorithm(X, None, "binary", "Undersampler")


def test_default_algorithm_custom_hyperparameters_error(X_y_binary):
    X, y = X_y_binary
    problem_type = ProblemTypes.BINARY
    sampler_name = "Undersampler"

    custom_hyperparameters = [
        {"Imputer": {"numeric_impute_strategy": ["median"]}},
        {"One Hot Encoder": {"value1": ["value2"]}},
    ]

    with pytest.raises(
        ValueError, match="If custom_hyperparameters provided, must be of type dict"
    ):
        DefaultAlgorithm(
            X,
            y,
            problem_type,
            sampler_name,
            custom_hyperparameters=custom_hyperparameters,
        )

    with pytest.raises(
        ValueError, match="Custom hyperparameters should only contain skopt"
    ):
        DefaultAlgorithm(
            X,
            y,
            problem_type,
            sampler_name,
            random_seed=0,
            custom_hyperparameters={"Imputer": {"impute_strategy": (1, 2)}},
        )

    with pytest.raises(
        ValueError, match="Pipeline parameters should not contain skopt.Space variables"
    ):
        DefaultAlgorithm(
            X,
            y,
            problem_type,
            sampler_name,
            random_seed=0,
            pipeline_params={"Imputer": {"impute_strategy": Categorical([1, 3, 4])}},
        )


def add_result(algo, batch):
    scores = np.arange(0, len(batch))
    for score, pipeline in zip(scores, batch):
        algo.add_result(score, pipeline, {"id": algo.pipeline_number})


@patch("evalml.pipelines.components.FeatureSelector.get_names")
@pytest.mark.parametrize(
    "automl_type",
    [ProblemTypes.BINARY, ProblemTypes.MULTICLASS, ProblemTypes.REGRESSION],
)
def test_default_algorithm(
    mock_get_names,
    automl_type,
    X_y_categorical_classification,
    X_y_multi,
    X_y_regression,
):
    pipeline_names = [
        "Numeric Pipeline - Select Columns Transformer",
        "Categorical Pipeline - Select Columns Transformer",
    ]
    if automl_type == ProblemTypes.BINARY:
        X, y = X_y_categorical_classification
        fs = "RF Classifier Select From Model"
    elif automl_type == ProblemTypes.MULTICLASS:
        X, y = X_y_multi
        fs = "RF Classifier Select From Model"
    elif automl_type == ProblemTypes.REGRESSION:
        X, y = X_y_regression
        fs = "RF Regressor Select From Model"

    mock_get_names.return_value = ["0", "1", "2"]
    problem_type = automl_type
    sampler_name = None
    algo = DefaultAlgorithm(X, y, problem_type, sampler_name)
    naive_model_families = set([ModelFamily.LINEAR_MODEL, ModelFamily.RANDOM_FOREST])

    first_batch = algo.next_batch()
    assert len(first_batch) == 2
    assert {p.model_family for p in first_batch} == naive_model_families
    add_result(algo, first_batch)

    second_batch = algo.next_batch()
    assert len(second_batch) == 2
    assert {p.model_family for p in second_batch} == naive_model_families
    for pipeline in second_batch:
        assert pipeline.get_component(fs)
    add_result(algo, second_batch)
    algo._selected_cat_cols = ["A", "B", "C"]

    assert algo._selected_cols == ["0", "1", "2"]
    assert algo._selected_cat_cols == ["A", "B", "C"]
    final_batch = algo.next_batch()
    for pipeline in final_batch:
        if not isinstance(
            pipeline.estimator, (ElasticNetClassifier, ElasticNetRegressor)
        ):
            assert pipeline.model_family not in naive_model_families
        assert pipeline.parameters[pipeline_names[0]]["columns"] == [
            "0",
            "1",
            "2",
        ]
        assert pipeline.parameters[pipeline_names[1]]["columns"] == [
            "A",
            "B",
            "C",
        ]
        assert algo._tuners[pipeline.name]
    add_result(algo, final_batch)

    final_ensemble = algo.next_batch()
    assert isinstance(
        final_ensemble[0].estimator,
        (StackedEnsembleClassifier, StackedEnsembleRegressor),
    )
    add_result(algo, final_ensemble)

    long_explore = algo.next_batch()

    long_estimators = set([pipeline.estimator.name for pipeline in long_explore])
    assert len(long_explore) == 150
    assert len(long_estimators) == 3

    long_first_ensemble = algo.next_batch()
    assert isinstance(
        long_first_ensemble[0].estimator,
        (StackedEnsembleClassifier, StackedEnsembleRegressor),
    )

    long = algo.next_batch()
    long_estimators = set([pipeline.estimator.name for pipeline in long])
    assert len(long) == 30
    assert len(long_estimators) == 3

    long_second_ensemble = algo.next_batch()
    assert isinstance(
        long_second_ensemble[0].estimator,
        (StackedEnsembleClassifier, StackedEnsembleRegressor),
    )

    long_2 = algo.next_batch()
    long_estimators = set([pipeline.estimator.name for pipeline in long_2])
    assert len(long_2) == 30
    assert len(long_estimators) == 3


@patch("evalml.pipelines.components.FeatureSelector.get_names")
def test_evalml_algo_pipeline_params(mock_get_names, X_y_binary):
    X, y = X_y_binary
    mock_get_names.return_value = ["0", "1", "2"]

    problem_type = ProblemTypes.BINARY
    sampler_name = None
    pipeline_params = {
        "pipeline": {"gap": 2, "max_delay": 10},
        "Logistic Regression Classifier": {"C": 5},
    }
    algo = DefaultAlgorithm(
        X,
        y,
        problem_type,
        sampler_name,
        pipeline_params=pipeline_params,
        num_long_explore_pipelines=1,
        num_long_pipelines_per_batch=1,
    )

    for _ in range(6):
        batch = algo.next_batch()
        add_result(algo, batch)
        for pipeline in batch:
            if not isinstance(pipeline.estimator, StackedEnsembleClassifier):
                assert pipeline.parameters["pipeline"] == {"gap": 2, "max_delay": 10}
            if isinstance(pipeline.estimator, LogisticRegressionClassifier):
                assert pipeline.parameters["Logistic Regression Classifier"]["C"] == 5


@patch("evalml.pipelines.components.FeatureSelector.get_names")
@patch("evalml.pipelines.components.OneHotEncoder._get_feature_provenance")
def test_evalml_algo_custom_hyperparameters(
    mock_get_feature_provenance, mock_get_names, X_y_categorical_classification
):
    X, y = X_y_categorical_classification
    X.ww.init()
    cat_cols = list(X.ww.select("categorical").columns)
    mock_get_names.return_value = ["0", "1", "2", "Sex_male", "Embarked_S"]
    mock_get_feature_provenance.return_value = {
        "Sex": ["Sex_male"],
        "Embarked": ["Embarked_S"],
    }

    problem_type = ProblemTypes.BINARY
    sampler_name = None
    impute_strategy = Categorical(["mean", "median"])
    custom_hyperparameters = {
        "Random Forest Classifier": {
            "n_estimators": Integer(5, 7),
            "max_depth": Categorical([5, 6, 7]),
        },
        "Imputer": {"numeric_impute_strategy": impute_strategy},
    }

    algo = DefaultAlgorithm(
        X,
        y,
        problem_type,
        sampler_name,
        custom_hyperparameters=custom_hyperparameters,
        num_long_explore_pipelines=3,
        num_long_pipelines_per_batch=3,
    )

    for _ in range(2):
        batch = algo.next_batch()
        add_result(algo, batch)
        for pipeline in batch:
            if isinstance(pipeline.estimator, RandomForestClassifier):
                assert pipeline.parameters["Random Forest Classifier"][
                    "n_estimators"
                ] in Integer(5, 7)
                assert pipeline.parameters["Random Forest Classifier"][
                    "max_depth"
                ] in Categorical([5, 6, 7])

    assert algo._selected_cols == ["0", "1", "2"]
    assert algo._selected_cat_cols == cat_cols

    batch = algo.next_batch()
    add_result(algo, batch)
    for pipeline in batch:
        assert (
            pipeline.parameters["Numeric Pipeline - Imputer"]["numeric_impute_strategy"]
            in impute_strategy
        )
        assert (
            pipeline.parameters["Categorical Pipeline - Imputer"][
                "numeric_impute_strategy"
            ]
            in impute_strategy
        )


@patch("evalml.pipelines.components.FeatureSelector.get_names")
@pytest.mark.parametrize("columns", [["unknown_col"], ["unknown1, unknown2"]])
def test_default_algo_drop_columns(mock_get_names, columns, X_y_binary):
    X, y = X_y_binary
    mock_get_names.return_value = ["0", "1", "2"]

    X = pd.DataFrame(X)
    for col in columns:
        X[col] = pd.Series(range(len(X)))
    X.ww.init()
    X.ww.set_types({col: "Unknown" for col in columns})

    algo = DefaultAlgorithm(X, y, ProblemTypes.BINARY, sampler_name=None)

    assert algo._pipeline_params["Drop Columns Transformer"]["columns"] == columns

    for _ in range(2):
        batch = algo.next_batch()
        add_result(algo, batch)
        for pipeline in batch:
            if not isinstance(pipeline.estimator, StackedEnsembleClassifier):
                assert (
                    pipeline.parameters["Drop Columns Transformer"]["columns"]
                    == columns
                )

    batch = algo.next_batch()
    add_result(algo, batch)
    for pipeline in batch:
        for component_name in pipeline.component_graph.compute_order:
            split = component_name.split(" - ")
            if "Drop Columns Transformer" in split:
                assert algo._pipeline_params[component_name]["columns"] == columns
                assert pipeline.parameters[component_name]["columns"] == columns


def test_make_split_pipeline(X_y_binary):
    X, y = X_y_binary
    algo = DefaultAlgorithm(X, y, ProblemTypes.BINARY, sampler_name=None)
    algo._selected_cols = ["1", "2", "3"]
    algo._selected_cat_cols = ["A", "B", "C"]
    pipeline = algo._make_split_pipeline(RandomForestClassifier, "test_pipeline")
    compute_order = [
        "Label Encoder",
        "Categorical Pipeline - Select Columns Transformer",
        "Categorical Pipeline - Label Encoder",
        "Categorical Pipeline - Imputer",
        "Numeric Pipeline - Label Encoder",
        "Numeric Pipeline - Imputer",
        "Numeric Pipeline - Select Columns Transformer",
        "Random Forest Classifier",
    ]
    assert pipeline.component_graph.compute_order == compute_order
    assert pipeline.name == "test_pipeline"
    assert pipeline.parameters["Numeric Pipeline - Select Columns Transformer"][
        "columns"
    ] == ["1", "2", "3"]
    assert pipeline.parameters["Categorical Pipeline - Select Columns Transformer"][
        "columns"
    ] == ["A", "B", "C"]
    assert isinstance(pipeline.estimator, RandomForestClassifier)


@patch("evalml.pipelines.components.FeatureSelector.get_names")
@patch("evalml.pipelines.components.OneHotEncoder._get_feature_provenance")
def test_select_cat_cols(
    mock_get_feature_provenance, mock_get_names, X_y_categorical_classification
):
    X, y = X_y_categorical_classification
    X.ww.init()
    cat_cols = list(X.ww.select("categorical").columns)
    mock_get_names.return_value = ["0", "1", "2", "Sex_male", "Embarked_S"]
    mock_get_feature_provenance.return_value = {
        "Sex": ["Sex_male"],
        "Embarked": ["Embarked_S"],
    }

    algo = DefaultAlgorithm(X, y, ProblemTypes.BINARY, None)

    batch = algo.next_batch()
    add_result(algo, batch)

    batch = algo.next_batch()
    add_result(algo, batch)

    assert algo._selected_cols == ["0", "1", "2"]
    assert algo._selected_cat_cols == cat_cols

    batch = algo.next_batch()
    add_result(algo, batch)

    batch = algo.next_batch()
    add_result(algo, batch)
    for component, value in batch[0].parameters.items():
        if "Numeric Pipeline - Select Columns Transformer" in component:
            assert value["columns"] == algo._selected_cols
        elif "Categorical Pipeline - Select Columns Transformer" in component:
            assert value["columns"] == algo._selected_cat_cols
