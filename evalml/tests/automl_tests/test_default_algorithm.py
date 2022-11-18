from unittest.mock import patch

import featuretools as ft
import numpy as np
import pandas as pd
import pytest
from skopt.space import Categorical, Integer

from evalml.automl.automl_algorithm import DefaultAlgorithm
from evalml.model_family import ModelFamily
from evalml.pipelines.components import (
    ARIMARegressor,
    DateTimeFeaturizer,
    ElasticNetClassifier,
    ElasticNetRegressor,
    EmailFeaturizer,
    LogisticRegressionClassifier,
    NaturalLanguageFeaturizer,
    ProphetRegressor,
    RandomForestClassifier,
    StackedEnsembleClassifier,
    StackedEnsembleRegressor,
    TimeSeriesFeaturizer,
    URLFeaturizer,
)
from evalml.problem_types import ProblemTypes, is_time_series


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
    assert algo.ensembling is False
    assert algo.default_max_batches == 3

    algo = DefaultAlgorithm(
        X,
        y,
        ProblemTypes.TIME_SERIES_BINARY,
        sampler_name,
        verbose=True,
    )
    assert algo.default_max_batches == 3

    algo = DefaultAlgorithm(
        X,
        y,
        ProblemTypes.BINARY,
        sampler_name,
        verbose=True,
        ensembling=True,
    )
    assert algo.default_max_batches == 4


def test_default_algorithm_search_parameters_error(X_y_binary):
    X, y = X_y_binary
    problem_type = ProblemTypes.BINARY
    sampler_name = "Undersampler"

    search_parameters = [
        {"Imputer": {"numeric_impute_strategy": ["median"]}},
        {"One Hot Encoder": {"value1": ["value2"]}},
    ]

    with pytest.raises(
        ValueError,
        match="If search_parameters provided, must be of type dict",
    ):
        DefaultAlgorithm(
            X,
            y,
            problem_type,
            sampler_name,
            search_parameters=search_parameters,
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
@pytest.mark.parametrize("split", ["split", "numeric-only", "categorical-only"])
def test_default_algorithm(
    mock_get_names,
    automl_type,
    split,
    X_y_categorical_classification,
    X_y_multi,
    X_y_regression,
):
    if split == "split":
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

    X = pd.DataFrame(X)
    X["1"] = 0
    X["2"] = 0
    X["3"] = 0
    X["A"] = "a"
    X["B"] = "b"
    X["C"] = "c"

    non_categorical_columns = ["0", "1", "2"]
    categorical_columns = ["A", "B", "C"]

    if split == "split" or split == "numeric-only":
        mock_get_names.return_value = non_categorical_columns
    else:
        mock_get_names.return_value = None

    problem_type = automl_type
    sampler_name = None
    algo = DefaultAlgorithm(X, y, problem_type, sampler_name, ensembling=True)
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

    if split == "split" or split == "numeric-only":
        assert algo._selected_cols == non_categorical_columns
    if split == "split" or split == "categorical-only":
        algo._selected_cat_cols = categorical_columns
        assert algo._selected_cat_cols == categorical_columns

    final_batch = algo.next_batch()
    for pipeline in final_batch:
        if not isinstance(
            pipeline.estimator,
            (ElasticNetClassifier, ElasticNetRegressor),
        ):
            assert pipeline.model_family not in naive_model_families
        if split == "split":
            assert (
                pipeline.parameters[pipeline_names[0]]["columns"]
                == non_categorical_columns
            )
            assert (
                pipeline.parameters[pipeline_names[1]]["columns"] == categorical_columns
            )
        elif split == "numeric-only":
            assert (
                pipeline.parameters["Select Columns Transformer"]["columns"]
                == non_categorical_columns
            )
        elif split == "categorical-only":
            assert (
                pipeline.parameters["Select Columns Transformer"]["columns"]
                == categorical_columns
            )
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
def test_evalml_algo_search_params(mock_get_names, X_y_binary):
    X, y = X_y_binary
    mock_get_names.return_value = ["0", "1", "2"]

    problem_type = ProblemTypes.BINARY
    sampler_name = None
    search_params = {
        "pipeline": {"gap": 2, "max_delay": 10},
        "Logistic Regression Classifier": {"C": 5},
    }
    algo = DefaultAlgorithm(
        X,
        y,
        problem_type,
        sampler_name,
        search_parameters=search_params,
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
def test_evalml_algo_search_hyperparameters(
    mock_get_feature_provenance,
    mock_get_names,
    X_y_categorical_classification,
):
    X, y = X_y_categorical_classification
    cat_cols = list(X.ww.select("categorical").columns)
    mock_get_names.return_value = [
        "0",
        "1",
        "2",
        "Sex_male",
        "Ticket_A/5 21171",
        "C85",
        "Embarked_S",
    ]
    mock_get_feature_provenance.return_value = {
        "Sex": ["Sex_male"],
        "Ticket": ["Ticket_A/5 21171"],
        "Cabin": ["C85"],
        "Embarked": ["Embarked_S"],
    }

    problem_type = ProblemTypes.BINARY
    sampler_name = None
    impute_strategy = Categorical(["mean", "median"])
    search_parameters = {
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
        search_parameters=search_parameters,
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

    for col in columns:
        X.ww[col] = pd.Series(range(len(X)))
    X.ww.set_types({col: "Unknown" for col in columns})

    algo = DefaultAlgorithm(X, y, ProblemTypes.BINARY, sampler_name=None)

    assert algo.search_parameters["Drop Columns Transformer"]["columns"] == columns

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
                assert algo.search_parameters[component_name]["columns"] == columns
                assert pipeline.parameters[component_name]["columns"] == columns


@pytest.mark.parametrize("sampler", ["Oversampler", None])
@pytest.mark.parametrize("features", [True, False])
def test_make_split_pipeline(sampler, features, X_y_binary):
    X, y = X_y_binary

    X = pd.DataFrame(X)
    X["1"] = 0
    X["2"] = 0
    X["3"] = 0
    X["A"] = "a"
    X["B"] = "b"
    X["C"] = "c"

    algo = DefaultAlgorithm(
        X,
        y,
        ProblemTypes.BINARY,
        sampler_name=sampler,
        features=features,
    )
    algo._selected_cols = ["1", "2", "3"]
    algo._selected_cat_cols = ["A", "B", "C"]
    pipeline = algo._make_split_pipeline(RandomForestClassifier, "test_pipeline")
    compute_order = ["Label Encoder"]
    if features:
        compute_order.extend(["DFS Transformer"])
    compute_order.extend(
        [
            "Categorical Pipeline - Select Columns Transformer",
            "Categorical Pipeline - Label Encoder",
            "Categorical Pipeline - Replace Nullable Types Transformer",
            "Categorical Pipeline - Imputer",
            "Categorical Pipeline - One Hot Encoder",
            "Numeric Pipeline - Select Columns By Type Transformer",
            "Numeric Pipeline - Label Encoder",
            "Numeric Pipeline - Replace Nullable Types Transformer",
            "Numeric Pipeline - Imputer",
            "Numeric Pipeline - Select Columns Transformer",
        ],
    )
    if sampler:
        compute_order.extend([sampler])
    compute_order.append("Random Forest Classifier")

    assert pipeline.component_graph.compute_order == compute_order
    assert pipeline.name == "test_pipeline"
    assert pipeline.parameters["Numeric Pipeline - Select Columns By Type Transformer"][
        "column_types"
    ] == ["Categorical", "EmailAddress", "URL"]
    assert pipeline.parameters["Numeric Pipeline - Select Columns By Type Transformer"][
        "exclude"
    ]
    assert pipeline.parameters["Numeric Pipeline - Select Columns Transformer"][
        "columns"
    ] == ["1", "2", "3"]
    assert pipeline.parameters["Categorical Pipeline - Select Columns Transformer"][
        "columns"
    ] == ["A", "B", "C"]
    assert isinstance(pipeline.estimator, RandomForestClassifier)


def test_make_split_pipeline_categorical_only(X_y_binary):
    X, y = X_y_binary

    X = pd.DataFrame(X)
    X["A"] = "a"
    X["B"] = "b"
    X["C"] = "c"

    algo = DefaultAlgorithm(X, y, ProblemTypes.BINARY, sampler_name=None)
    algo._selected_cat_cols = ["A", "B", "C"]
    pipeline = algo._make_split_pipeline(RandomForestClassifier)
    compute_order = [
        "Select Columns Transformer",
        "Label Encoder",
        "Replace Nullable Types Transformer",
        "Imputer",
        "One Hot Encoder",
        "Random Forest Classifier",
    ]
    assert pipeline.component_graph.compute_order == compute_order
    assert pipeline.parameters["Select Columns Transformer"]["columns"] == [
        "A",
        "B",
        "C",
    ]
    assert isinstance(pipeline.estimator, RandomForestClassifier)


@patch("evalml.pipelines.components.FeatureSelector.get_names")
@patch("evalml.pipelines.components.OneHotEncoder._get_feature_provenance")
def test_select_cat_cols(
    mock_get_feature_provenance,
    mock_get_names,
    X_y_categorical_classification,
):
    X, y = X_y_categorical_classification
    X.ww.init()
    cat_cols = list(X.ww.select("categorical").columns)
    mock_get_names.return_value = [
        "0",
        "1",
        "2",
        "Sex_male",
        "Ticket_A/5 21171",
        "C85",
        "Embarked_S",
    ]
    mock_get_feature_provenance.return_value = {
        "Sex": ["Sex_male"],
        "Ticket": ["Ticket_A/5 21171"],
        "Cabin": ["C85"],
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
    for component, value in batch[0].parameters.items():
        if "Numeric Pipeline - Select Columns Transformer" in component:
            assert value["columns"] == algo._selected_cols
        elif "Numeric Pipeline - Select Columns By Type Transformer" in component:
            assert value["column_types"] == ["Categorical", "EmailAddress", "URL"]
            assert value["exclude"]
        elif "Categorical Pipeline - Select Columns Transformer" in component:
            assert value["columns"] == algo._selected_cat_cols

    batch = algo.next_batch()
    add_result(algo, batch)
    for component, value in batch[0].parameters.items():
        if "Numeric Pipeline - Select Columns Transformer" in component:
            assert value["columns"] == algo._selected_cols
        elif "Numeric Pipeline - Select Columns By Type Transformer" in component:
            assert value["column_types"] == ["Categorical", "EmailAddress", "URL"]
            assert value["exclude"]
        elif "Categorical Pipeline - Select Columns Transformer" in component:
            assert value["columns"] == algo._selected_cat_cols


@pytest.mark.parametrize(
    "problem_type,n_unique",
    [
        ("binary", 2),
        ("multiclass", 10),
        ("multiclass", 200),
        ("regression", 80),
        ("regression", 200),
    ],
)
@pytest.mark.parametrize("allow_long_running_models", [True, False])
def test_default_algorithm_allow_long_running_models_next_batch(
    allow_long_running_models,
    problem_type,
    n_unique,
):
    if allow_long_running_models and problem_type != "multiclass":
        pytest.skip("Skipping to shorten tests")

    models_to_check = [
        "Elastic Net",
        "XGBoost",
        "CatBoost",
    ]

    X = pd.DataFrame()
    y = pd.Series([i for i in range(n_unique)] * 5)

    algo = DefaultAlgorithm(
        X=X,
        y=y,
        sampler_name=None,
        problem_type=problem_type,
        random_seed=0,
        allow_long_running_models=allow_long_running_models,
    )
    algo._selected_cols = []
    next_batch = algo.next_batch()
    found_models = False
    for pipeline in next_batch:
        found_models |= any([m in pipeline.name for m in models_to_check])

    # the "best" score will be the 1st dummy pipeline
    add_result(algo, next_batch)

    for i in range(1, 6):
        next_batch = algo.next_batch()
        for pipeline in next_batch:
            found_models |= any([m in pipeline.name for m in models_to_check])
        # if found_models becomes true, we already have our needed results
        if found_models:
            break
        scores = -np.arange(0, len(next_batch))
        for score, pipeline in zip(scores, next_batch):
            algo.add_result(score, pipeline, {"id": algo.pipeline_number})

    if (
        problem_type == "multiclass"
        and not allow_long_running_models
        and n_unique == 200
    ):
        assert not found_models
    else:
        assert found_models


@pytest.mark.parametrize(
    "problem_type",
    [
        "time series binary",
        "time series multiclass",
        "time series regression",
    ],
)
@patch("evalml.pipelines.components.FeatureSelector.get_names")
def test_default_algorithm_time_series(
    mock_get_names,
    problem_type,
    ts_data,
):
    X, _, y = ts_data(problem_type=problem_type)

    mock_get_names.return_value = ["0", "1", "2"]
    sampler_name = None
    search_parameters = {
        "pipeline": {
            "time_index": "date",
            "gap": 1,
            "max_delay": 3,
            "delay_features": False,
            "forecast_horizon": 10,
        },
    }

    algo = DefaultAlgorithm(
        X,
        y,
        problem_type,
        sampler_name,
        search_parameters=search_parameters,
    )
    naive_model_families = set([ModelFamily.LINEAR_MODEL, ModelFamily.RANDOM_FOREST])

    first_batch = algo.next_batch()
    assert len(first_batch) == 2 if problem_type != "time series regression" else 4
    assert {p.model_family for p in first_batch} == naive_model_families
    for pipeline in first_batch:
        assert pipeline.parameters["pipeline"] == search_parameters["pipeline"]
        assert pipeline.parameters["DateTime Featurizer"]["time_index"]
    add_result(algo, first_batch)

    second_batch = algo.next_batch()
    assert len(second_batch) == 2 if problem_type != "time series regression" else 4
    assert {p.model_family for p in second_batch} == naive_model_families
    for pipeline in second_batch:
        assert pipeline.parameters["pipeline"] == search_parameters["pipeline"]
        assert pipeline.parameters["DateTime Featurizer"]["time_index"]
    add_result(algo, second_batch)

    final_batch = algo.next_batch()
    for pipeline in final_batch:
        if not isinstance(
            pipeline.estimator,
            (ElasticNetClassifier, ElasticNetRegressor),
        ):
            assert pipeline.model_family not in naive_model_families
        assert algo._tuners[pipeline.name]
        assert pipeline.parameters["pipeline"] == search_parameters["pipeline"]
        if not isinstance(pipeline.estimator, (ARIMARegressor, ProphetRegressor)):
            assert pipeline.parameters["DateTime Featurizer"]["time_index"]
    add_result(algo, final_batch)

    long_explore = algo.next_batch()
    long_estimators = set([pipeline.estimator.name for pipeline in long_explore])
    assert len(long_explore) == 150 if problem_type != "time series regression" else 300
    assert len(long_estimators) == 3

    long = algo.next_batch()
    long_estimators = set([pipeline.estimator.name for pipeline in long])
    assert len(long) == 30 if problem_type != "time series regression" else 60
    assert len(long_estimators) == 3

    long_2 = algo.next_batch()
    long_estimators = set([pipeline.estimator.name for pipeline in long_2])
    assert len(long_2) == 30 if problem_type != "time series regression" else 60
    assert len(long_estimators) == 3


@pytest.mark.parametrize(
    "problem_type",
    [
        "time series binary",
        "time series multiclass",
        "time series regression",
    ],
)
@patch("evalml.pipelines.components.FeatureSelector.get_names")
def test_default_algorithm_time_series_known_in_advance(
    mock_get_names,
    problem_type,
    ts_data,
):
    X, _, y = ts_data(problem_type=problem_type)

    X.ww["email"] = pd.Series(["foo@foo.com"] * X.shape[0], index=X.index)
    X.ww["category"] = pd.Series(["a"] * X.shape[0], index=X.index)
    X.ww.set_types({"email": "EmailAddress", "category": "Categorical"})
    known_in_advance = ["email", "category"]

    mock_get_names.return_value = ["0", "1", "2"]
    sampler_name = None
    search_parameters = {
        "pipeline": {
            "time_index": "date",
            "gap": 1,
            "max_delay": 3,
            "delay_features": False,
            "forecast_horizon": 10,
            "known_in_advance": known_in_advance,
        },
    }

    algo = DefaultAlgorithm(
        X,
        y,
        problem_type,
        sampler_name,
        search_parameters=search_parameters,
    )
    naive_model_families = set([ModelFamily.LINEAR_MODEL, ModelFamily.RANDOM_FOREST])

    first_batch = algo.next_batch()
    assert len(first_batch) == 2
    assert {p.model_family for p in first_batch} == naive_model_families
    for pipeline in first_batch:
        assert (
            pipeline.parameters[
                "Known In Advance Pipeline - Select Columns Transformer"
            ]["columns"]
            == known_in_advance
        )
        assert pipeline.parameters[
            "Not Known In Advance Pipeline - Select Columns Transformer"
        ]["columns"] == ["feature", "date"]
    add_result(algo, first_batch)

    second_batch = algo.next_batch()
    assert len(second_batch) == 2
    assert {p.model_family for p in second_batch} == naive_model_families
    for pipeline in second_batch:
        assert (
            pipeline.parameters[
                "Known In Advance Pipeline - Select Columns Transformer"
            ]["columns"]
            == known_in_advance
        )
        assert pipeline.parameters[
            "Not Known In Advance Pipeline - Select Columns Transformer"
        ]["columns"] == ["feature", "date"]
    add_result(algo, second_batch)

    final_batch = algo.next_batch()
    for pipeline in final_batch:
        if not isinstance(
            pipeline.estimator,
            (ElasticNetClassifier, ElasticNetRegressor),
        ):
            assert pipeline.model_family not in naive_model_families
        assert algo._tuners[pipeline.name]
        assert pipeline.parameters["pipeline"] == search_parameters["pipeline"]
        assert (
            pipeline.parameters[
                "Known In Advance Pipeline - Select Columns Transformer"
            ]["columns"]
            == known_in_advance
        )
        assert pipeline.parameters[
            "Not Known In Advance Pipeline - Select Columns Transformer"
        ]["columns"] == ["feature", "date"]
    add_result(algo, final_batch)

    long_explore = algo.next_batch()
    long_estimators = set([pipeline.estimator.name for pipeline in long_explore])
    assert len(long_explore) == 150 if problem_type != "time series regression" else 300
    assert len(long_estimators) == 3

    long = algo.next_batch()
    long_estimators = set([pipeline.estimator.name for pipeline in long])
    assert len(long) == 30 if problem_type != "time series regression" else 60
    assert len(long_estimators) == 3

    long_2 = algo.next_batch()
    long_estimators = set([pipeline.estimator.name for pipeline in long_2])
    assert len(long_2) == 30 if problem_type != "time series regression" else 60
    assert len(long_estimators) == 3


@pytest.mark.parametrize(
    "automl_type",
    [
        "time series binary",
        "time series multiclass",
        "time series regression",
    ],
)
@patch("evalml.pipelines.components.FeatureSelector.get_names")
def test_default_algorithm_time_series_ensembling(
    mock_get_names,
    automl_type,
    ts_data,
):
    X, _, y = ts_data()
    with pytest.raises(
        ValueError,
        match="Ensembling is not available for time series problems in DefaultAlgorithm.",
    ):
        DefaultAlgorithm(X, y, automl_type, None, ensembling=True)


@pytest.mark.parametrize("split", ["split", "numeric-only", "categorical-only"])
@pytest.mark.parametrize("problem_type", ProblemTypes.all_problem_types)
@patch("evalml.pipelines.components.FeatureSelector.get_names")
def test_default_algorithm_accept_features(
    mock_get_names,
    ts_data,
    X_y_binary,
    problem_type,
    split,
):
    if is_time_series(problem_type):
        X, _, y = ts_data()
    else:
        X, y = X_y_binary

    X = pd.DataFrame(X)
    X["A"] = "a"
    X["B"] = "b"
    X["C"] = "c"

    non_categorical_columns = ["0", "1", "2"]
    categorical_columns = ["A", "B", "C"]

    if split == "split" or split == "numeric-only":
        mock_get_names.return_value = non_categorical_columns
    else:
        mock_get_names.return_value = None

    X_pd = pd.DataFrame(X)
    X_pd.columns = X_pd.columns.astype(str)
    X_transform = X_pd.iloc[len(X) // 3 :]

    es = ft.EntitySet()
    es = es.add_dataframe(
        dataframe_name="X",
        dataframe=X_transform,
        index="index",
        make_index=True,
    )
    _, features = ft.dfs(
        entityset=es,
        target_dataframe_name="X",
        trans_primitives=["absolute"],
    )

    sampler_name = None
    search_parameters = {"DFS Transformer": {"features": features}}
    if is_time_series(problem_type):
        search_parameters["pipeline"] = {
            "time_index": "date",
            "gap": 1,
            "max_delay": 3,
            "delay_features": False,
            "forecast_horizon": 10,
        }

    algo = DefaultAlgorithm(
        X,
        y,
        problem_type,
        sampler_name,
        search_parameters=search_parameters,
        num_long_explore_pipelines=1,
        num_long_pipelines_per_batch=1,
        features=features,
    )

    for _ in range(2):
        batch = algo.next_batch()
        add_result(algo, batch)
        for pipeline in batch:
            if not isinstance(pipeline.estimator, StackedEnsembleClassifier):
                assert "DFS Transformer" in pipeline.component_graph.compute_order
                assert pipeline.parameters["DFS Transformer"]["features"] == features

    if not is_time_series(
        problem_type,
    ):  # DefaultAlgorithm doesn't generate split pipelines for time series problems
        if split == "split" or split == "numeric-only":
            assert algo._selected_cols == non_categorical_columns
        if split == "split" or split == "categorical-only":
            algo._selected_cat_cols = categorical_columns
            assert algo._selected_cat_cols == categorical_columns

    for _ in range(2):
        batch = algo.next_batch()
        add_result(algo, batch)
        for pipeline in batch:
            if not isinstance(pipeline.estimator, StackedEnsembleClassifier):
                assert "DFS Transformer" in pipeline.component_graph.compute_order
                assert pipeline.parameters["DFS Transformer"]["features"] == features


def test_default_algorithm_add_result_cache(X_y_binary):
    X, y = X_y_binary
    algo = DefaultAlgorithm(
        X=X,
        y=y,
        problem_type="binary",
        sampler_name=None,
    )

    cache = {"some_cache_key": "some_cache_value"}
    # initial batch contains one of each pipeline, with default parameters
    next_batch = algo.next_batch()
    scores = np.arange(0, len(next_batch))
    for pipeline_num, (score, pipeline) in enumerate(zip(scores, next_batch)):
        algo.add_result(
            score,
            pipeline,
            {"id": algo.pipeline_number + pipeline_num},
            cached_data=cache,
        )

    for values in algo._best_pipeline_info.values():
        assert values["cached_data"] == cache


def test_default_algorithm_ensembling_off(X_y_binary):
    X, y = X_y_binary
    algo = DefaultAlgorithm(
        X=X,
        y=y,
        problem_type="binary",
        sampler_name=None,
        ensembling=False,
    )

    for i in range(10):
        next_batch = algo.next_batch()
        for pipeline in next_batch:
            assert (
                "Stacked Ensemble Classifier"
                not in pipeline.component_graph.compute_order
            )


@patch("evalml.pipelines.components.URLFeaturizer._get_feature_provenance")
@patch("evalml.pipelines.components.EmailFeaturizer._get_feature_provenance")
@patch("evalml.pipelines.components.OneHotEncoder._get_feature_provenance")
@patch("evalml.pipelines.components.FeatureSelector.get_names")
def test_default_algorithm_accepts_URL_email_features(
    mock_get_names,
    mock_ohe_get_feature_provenance,
    mock_email_get_feature_provenance,
    mock_url_get_feature_provenance,
    df_with_url_and_email,
):
    X = df_with_url_and_email
    y = pd.Series(range(len(X)))

    mock_ohe_get_feature_provenance.return_value = {
        "URL_TO_DOMAIN(url)": ["URL_TO_DOMAIN(url)_alteryx.com"],
        "EMAIL_ADDRESS_TO_DOMAIN(email)": ["EMAIL_ADDRESS_TO_DOMAIN(email)_gmail.com"],
    }
    mock_url_get_feature_provenance.return_value = {"url": ["URL_TO_DOMAIN(url)"]}
    mock_email_get_feature_provenance.return_value = {
        "email": ["EMAIL_ADDRESS_TO_DOMAIN(email)"],
    }

    mock_get_names.return_value = [
        "numeric",
        "EMAIL_ADDRESS_TO_DOMAIN(email)_gmail.com",
        "URL_TO_DOMAIN(url)_alteryx.com",
    ]
    algo = DefaultAlgorithm(X, y, ProblemTypes.REGRESSION, sampler_name=None)
    for _ in range(2):
        batch = algo.next_batch()
        add_result(algo, batch)

    assert algo._selected_cat_cols == ["url", "email"]

    batch = algo.next_batch()
    for pipeline in batch:
        pipeline.fit(X, y)
        assert pipeline.parameters["Categorical Pipeline - Select Columns Transformer"][
            "columns"
        ] == ["url", "email"]


@pytest.mark.parametrize(
    "automl_type",
    [ProblemTypes.BINARY, ProblemTypes.MULTICLASS, ProblemTypes.REGRESSION],
)
@pytest.mark.parametrize(
    "ensembling",
    [True, False],
)
@patch("evalml.pipelines.components.FeatureSelector.get_names")
def test_default_algorithm_num_pipelines_per_batch(
    mock_get_names,
    automl_type,
    ensembling,
    X_y_categorical_classification,
    X_y_multi,
    X_y_regression,
):
    mock_get_names.return_value = None
    if automl_type == ProblemTypes.BINARY:
        X, y = X_y_categorical_classification
    elif automl_type == ProblemTypes.MULTICLASS:
        X, y = X_y_multi
    elif automl_type == ProblemTypes.REGRESSION:
        X, y = X_y_regression
    algo = DefaultAlgorithm(
        X=X,
        y=y,
        problem_type=automl_type,
        sampler_name=None,
        ensembling=ensembling,
        top_n=2,
        num_long_explore_pipelines=10,
        num_long_pipelines_per_batch=5,
    )
    for i in range(7):
        batch = algo.next_batch()
        add_result(algo, batch)
        assert len(batch) == algo.num_pipelines_per_batch(i)


@pytest.mark.parametrize("input_type", ["pd", "ww"])
@pytest.mark.parametrize("problem_type", ProblemTypes.all_problem_types)
def test_exclude_featurizers_default_algorithm(
    problem_type,
    input_type,
    get_test_data_from_configuration,
):
    parameters = {}
    if is_time_series(problem_type):
        parameters = {
            "time_index": "dates",
            "gap": 1,
            "max_delay": 1,
            "forecast_horizon": 3,
        }

    X, y = get_test_data_from_configuration(
        input_type,
        problem_type,
        column_names=["dates", "text", "email", "url"],
    )

    algo = DefaultAlgorithm(
        X,
        y,
        problem_type,
        sampler_name=None,
        search_parameters={"pipeline": parameters},
        exclude_featurizers=[
            "DatetimeFeaturizer",
            "EmailFeaturizer",
            "URLFeaturizer",
            "NaturalLanguageFeaturizer",
            "TimeSeriesFeaturizer",
        ],
    )

    pipelines = []
    for _ in range(4):
        pipelines.extend([pl for pl in algo.next_batch()])

    # A check to make sure we actually retrieve constructed pipelines from the algo.
    assert len(pipelines) > 0

    assert not any(
        [
            DateTimeFeaturizer.name in pl.component_graph.compute_order
            for pl in pipelines
        ],
    )
    assert not any(
        [EmailFeaturizer.name in pl.component_graph.compute_order for pl in pipelines],
    )
    assert not any(
        [URLFeaturizer.name in pl.component_graph.compute_order for pl in pipelines],
    )
    assert not any(
        [
            NaturalLanguageFeaturizer.name in pl.component_graph.compute_order
            for pl in pipelines
        ],
    )
    assert not any(
        [
            TimeSeriesFeaturizer.name in pl.component_graph.compute_order
            for pl in pipelines
        ],
    )
