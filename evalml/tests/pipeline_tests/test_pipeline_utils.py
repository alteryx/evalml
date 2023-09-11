import os
from unittest.mock import patch

import black
import numpy as np
import pandas as pd
import pytest

from evalml.data_checks import DataCheckAction, DataCheckActionCode
from evalml.model_family import ModelFamily
from evalml.pipelines import (
    BinaryClassificationPipeline,
    MulticlassClassificationPipeline,
    RegressionPipeline,
)
from evalml.pipelines.components import (
    DateTimeFeaturizer,
    DFSTransformer,
    DropColumns,
    DropNaNRowsTransformer,
    DropRowsTransformer,
    EmailFeaturizer,
    Estimator,
    Imputer,
    LinearRegressor,
    LogisticRegressionClassifier,
    NaturalLanguageFeaturizer,
    OneHotEncoder,
    StackedEnsembleBase,
    StandardScaler,
    STLDecomposer,
    TargetImputer,
    TimeSeriesFeaturizer,
    Transformer,
    URLFeaturizer,
)
from evalml.pipelines.components.transformers.encoders.label_encoder import LabelEncoder
from evalml.pipelines.components.transformers.imputers.per_column_imputer import (
    PerColumnImputer,
)
from evalml.pipelines.components.utils import (
    estimator_unable_to_handle_nans,
    handle_component_class,
)
from evalml.pipelines.utils import (
    _get_pipeline_base_class,
    _get_preprocessing_components,
    _make_pipeline_from_multiple_graphs,
    generate_pipeline_code,
    generate_pipeline_example,
    get_estimators,
    is_classification,
    is_regression,
    make_pipeline,
    make_pipeline_from_actions,
    rows_of_interest,
    stack_data,
    stack_X,
    unstack_multiseries,
)
from evalml.problem_types import ProblemTypes, is_multiseries, is_time_series


@pytest.mark.parametrize("input_type", ["pd", "ww"])
@pytest.mark.parametrize("problem_type", ProblemTypes.all_problem_types)
@pytest.mark.parametrize("features", [True, False])
@pytest.mark.parametrize(
    "test_description, column_names",
    [
        ("all nan is not categorical", ["all_null", "numerical"]),
        ("mixed types", ["all_null", "categorical", "dates", "numerical"]),
        ("no all_null columns", ["numerical", "categorical", "dates"]),
        ("date, numerical", ["dates", "numerical"]),
        ("only text", ["text"]),
        ("only dates", ["dates"]),
        ("only numerical", ["numerical"]),
        ("only ip", ["ip"]),
        ("only all_null", ["all_null"]),
        ("only categorical", ["categorical"]),
        ("text with other features", ["text", "numerical", "categorical"]),
        ("url with other features", ["url", "numerical", "categorical"]),
        ("ip with other features", ["ip", "numerical", "categorical"]),
        ("email with other features", ["email", "numerical", "categorical"]),
        ("only null int", ["int_null"]),
        ("only null bool", ["bool_null"]),
        ("only null age", ["age_null"]),
        ("nullable_types", ["numerical", "int_null", "bool_null", "age_null"]),
    ],
)
def test_make_pipeline(
    problem_type,
    input_type,
    features,
    test_description,
    column_names,
    get_test_data_from_configuration,
    multiseries_ts_data_stacked,
):
    X, y = get_test_data_from_configuration(
        input_type,
        problem_type,
        column_names=column_names,
    )
    estimators = get_estimators(problem_type=problem_type)
    pipeline_class = _get_pipeline_base_class(problem_type)
    for estimator_class in estimators:
        if problem_type in estimator_class.supported_problem_types:
            parameters = {}
            if is_time_series(problem_type):
                parameters = {
                    "pipeline": {
                        "time_index": "date",
                        "gap": 1,
                        "max_delay": 1,
                        "forecast_horizon": 3,
                        "series_id": "series_id"
                        if is_multiseries(problem_type)
                        else None,
                    },
                }

            pipeline = make_pipeline(
                X,
                y,
                estimator_class,
                problem_type,
                parameters,
                features=features,
            )
            assert isinstance(pipeline, pipeline_class)
            label_encoder = [LabelEncoder] if is_classification(problem_type) else []
            delayed_features = (
                [TimeSeriesFeaturizer] if is_time_series(problem_type) else []
            )

            if estimator_class.model_family != ModelFamily.CATBOOST and any(
                column_name in ["url", "email", "categorical"]
                for column_name in column_names
            ):
                ohe = [OneHotEncoder]
            else:
                ohe = []
            dfs = [DFSTransformer] if features else []
            decomposer = [STLDecomposer] if is_regression(problem_type) else []
            datetime = (
                [DateTimeFeaturizer]
                if estimator_class.model_family
                not in [ModelFamily.ARIMA, ModelFamily.PROPHET]
                and "dates" in column_names
                else []
            )
            standard_scaler = (
                [StandardScaler]
                if estimator_class.model_family == ModelFamily.LINEAR_MODEL
                else []
            )
            drop_null = [DropColumns] if "all_null" in column_names else []
            natural_language_featurizer = (
                [NaturalLanguageFeaturizer] if "text" in column_names else []
            )
            email_featurizer = [EmailFeaturizer] if "email" in column_names else []
            url_featurizer = [URLFeaturizer] if "url" in column_names else []
            imputer = [] if (column_names in [["ip"], ["all_null"]]) else [Imputer]
            drop_nan_rows_transformer = (
                [DropNaNRowsTransformer]
                if is_time_series(problem_type)
                and estimator_unable_to_handle_nans(estimator_class)
                else []
            )

            if is_time_series(problem_type):
                if is_multiseries(problem_type):
                    expected_components = dfs + decomposer + [estimator_class]
                else:
                    expected_components = (
                        dfs
                        + label_encoder
                        + email_featurizer
                        + url_featurizer
                        + drop_null
                        + natural_language_featurizer
                        + imputer
                        + delayed_features
                        + decomposer
                        + datetime
                        + ohe
                        + drop_nan_rows_transformer
                        + standard_scaler
                        + [estimator_class]
                    )
            else:
                expected_components = (
                    dfs
                    + label_encoder
                    + email_featurizer
                    + url_featurizer
                    + drop_null
                    + delayed_features
                    + datetime
                    + natural_language_featurizer
                    + imputer
                    + ohe
                    + standard_scaler
                    + [estimator_class]
                )
            assert pipeline.component_graph.compute_order == [
                component.name for component in expected_components
            ], test_description


@pytest.mark.parametrize(
    "problem_type",
    [
        ProblemTypes.TIME_SERIES_BINARY,
        ProblemTypes.TIME_SERIES_MULTICLASS,
        ProblemTypes.TIME_SERIES_REGRESSION,
    ],
)
@pytest.mark.parametrize(
    "frequency, period, should_decomp",
    [
        ("D", 7, True),
        ("MS", 12, True),
        ("A", None, False),
        ("T", 24, True),
        ("10T", None, False),
        ("AS-JAN", None, False),
        ("YS", 6, True),
        ("Q", 4, True),
        ("Q", 2, False),
        ("D", 1500, False),
        (None, None, False),
    ],
)
def test_make_pipeline_controls_decomposer_time_series(
    problem_type,
    frequency,
    period,
    should_decomp,
    generate_seasonal_data,
):
    X, y = generate_seasonal_data(real_or_synthetic="synthetic")(
        period,
        freq_str=frequency,
    )
    X.ww.init()
    if frequency is None:
        X.ww["dates"] = pd.Series(
            pd.date_range("2000-02-03", periods=10, freq=frequency).append(
                pd.date_range("2000-02-15", periods=10, freq=frequency),
            ),
        )
    else:
        X.ww["dates"] = pd.Series(X.index)
    parameters = {
        "pipeline": {
            "time_index": "date",
            "gap": 1,
            "max_delay": 1,
            "forecast_horizon": 3,
        },
    }
    estimators = get_estimators(problem_type=problem_type)
    pipeline_class = _get_pipeline_base_class(problem_type)
    for estimator_class in estimators:
        if problem_type in estimator_class.supported_problem_types:
            pipeline = make_pipeline(X, y, estimator_class, problem_type, parameters)
            assert isinstance(pipeline, pipeline_class)

            if is_regression(problem_type) and should_decomp:
                assert "STL Decomposer" in pipeline.component_graph.compute_order
            else:
                assert "STL Decomposer" not in pipeline.component_graph.compute_order

            pipeline = make_pipeline(
                X,
                y,
                estimator_class,
                problem_type,
                parameters,
                include_decomposer=False,
            )
            assert isinstance(pipeline, pipeline_class)
            assert "STL Decomposer" not in pipeline.component_graph.compute_order


@pytest.mark.parametrize(
    "sampler",
    ["Oversampler", "Undersampler"],
)
@pytest.mark.parametrize(
    "problem_type",
    [
        ProblemTypes.TIME_SERIES_MULTICLASS,
        ProblemTypes.TIME_SERIES_REGRESSION,
        ProblemTypes.TIME_SERIES_BINARY,
    ],
)
@pytest.mark.parametrize(
    "test_description, known_in_advance",
    [
        ("categorical", ["categorical"]),
        ("email", ["email"]),
        ("url", ["url"]),
        ("text", ["text"]),
        ("nullable", ["int_null", "bool_null", "age_null"]),
        (
            "all",
            [
                "int_null",
                "bool_null",
                "age_null",
                "categorical",
                "email",
                "url",
                "text",
            ],
        ),
        ("other numerical", []),
    ],
)
def test_make_pipeline_known_in_advance(
    test_description,
    known_in_advance,
    problem_type,
    sampler,
    get_test_data_from_configuration,
):
    X, y = get_test_data_from_configuration(
        "ww",
        problem_type,
        column_names=["numerical"] + known_in_advance,
    )
    if test_description == "other numerical":
        X.ww["other numerical"] = pd.Series(range(X.shape[0]))
        known_in_advance = ["other numerical"]

    estimators = get_estimators(problem_type=problem_type)
    for estimator_class in estimators:
        parameters = {
            "pipeline": {
                "time_index": "date",
                "gap": 1,
                "max_delay": 1,
                "forecast_horizon": 3,
            },
            "Known In Advance Pipeline - Select Columns Transformer": {
                "columns": known_in_advance,
            },
            "Not Known In Advance Pipeline - Select Columns Transformer": {
                "columns": ["numerical"],
            },
        }

        pipeline = make_pipeline(
            X,
            y,
            estimator_class,
            problem_type,
            parameters,
            known_in_advance=known_in_advance,
            sampler_name=sampler if is_classification(problem_type) else None,
        )
        expected_known_in_advance_components = _get_preprocessing_components(
            X.ww[known_in_advance],
            y,
            "regression",
            estimator_class,
            sampler_name=sampler if is_classification(problem_type) else None,
        )
        expected_known_in_advance_components = [
            c.name
            for c in expected_known_in_advance_components
            if c.name != "Label Encoder"
        ]
        expected_known_in_advance_components = [
            "Select Columns Transformer",
        ] + expected_known_in_advance_components
        known_in_advance_components = [
            c.split("-")[1].strip()
            for c in pipeline.component_graph.compute_order
            if c.startswith("Known In Advance")
        ]

        assert expected_known_in_advance_components == known_in_advance_components
        for k in [
            "pipeline",
            "Known In Advance Pipeline - Select Columns Transformer",
            "Not Known In Advance Pipeline - Select Columns Transformer",
        ]:
            assert pipeline.parameters[k] == parameters[k]
        if is_classification(problem_type):
            assert (
                len([c for c in pipeline.component_graph if "Label Encoder" in c.name])
                == 2
            )
            assert len([c for c in pipeline.component_graph if sampler in c.name]) == 2


def test_make_pipeline_problem_type_mismatch():
    with pytest.raises(
        ValueError,
        match=f"{LogisticRegressionClassifier.name} is not a valid estimator for problem type",
    ):
        make_pipeline(
            pd.DataFrame(),
            pd.Series(),
            LogisticRegressionClassifier,
            ProblemTypes.REGRESSION,
        )
    with pytest.raises(
        ValueError,
        match=f"{LinearRegressor.name} is not a valid estimator for problem type",
    ):
        make_pipeline(
            pd.DataFrame(),
            pd.Series(),
            LinearRegressor,
            ProblemTypes.MULTICLASS,
        )
    with pytest.raises(
        ValueError,
        match=f"{Transformer.name} is not a valid estimator for problem type",
    ):
        make_pipeline(pd.DataFrame(), pd.Series(), Transformer, ProblemTypes.MULTICLASS)


@pytest.mark.parametrize("problem_type", ["binary", "multiclass", "regression"])
def test_make_pipeline_from_actions(problem_type):
    pipeline_class = _get_pipeline_base_class(problem_type)

    assert make_pipeline_from_actions(problem_type, []) == pipeline_class(
        component_graph={},
    )

    actions = [
        DataCheckAction(DataCheckActionCode.DROP_COL, None, {"columns": ["some col"]}),
    ]
    assert make_pipeline_from_actions(problem_type, actions) == pipeline_class(
        component_graph={"Drop Columns Transformer": [DropColumns, "X", "y"]},
        parameters={"Drop Columns Transformer": {"columns": ["some col"]}},
        random_seed=0,
    )

    actions = [
        DataCheckAction(
            DataCheckActionCode.DROP_COL,
            None,
            metadata={"columns": ["some col"]},
        ),
        DataCheckAction(
            DataCheckActionCode.IMPUTE_COL,
            None,
            metadata={
                "columns": None,
                "is_target": True,
                "parameters": {"impute_strategy": "most_frequent"},
            },
        ),
        DataCheckAction(DataCheckActionCode.DROP_ROWS, None, metadata={"rows": [1, 2]}),
        DataCheckAction(
            DataCheckActionCode.IMPUTE_COL,
            None,
            metadata={
                "columns": None,
                "is_target": False,
                "parameters": {
                    "impute_strategies": {
                        "some_column": {
                            "impute_strategy": "most_frequent",
                            "fill_value": 0.0,
                        },
                        "some_other_column": {
                            "impute_strategy": "mean",
                            "fill_value": 1.0,
                        },
                    },
                },
            },
        ),
    ]

    assert make_pipeline_from_actions(problem_type, actions) == pipeline_class(
        component_graph={
            "Target Imputer": [TargetImputer, "X", "y"],
            "Per Column Imputer": [PerColumnImputer, "X", "Target Imputer.y"],
            "Drop Columns Transformer": [
                DropColumns,
                "Per Column Imputer.x",
                "Target Imputer.y",
            ],
            "Drop Rows Transformer": [
                DropRowsTransformer,
                "Drop Columns Transformer.x",
                "Target Imputer.y",
            ],
        },
        parameters={
            "Target Imputer": {"impute_strategy": "most_frequent", "fill_value": None},
            "Drop Columns Transformer": {"columns": ["some col"]},
            "Drop Rows Transformer": {"indices_to_drop": [1, 2]},
            "Per Column Imputer": {
                "impute_strategies": {
                    "some_column": {
                        "impute_strategy": "most_frequent",
                        "fill_value": 0.0,
                    },
                    "some_other_column": {
                        "impute_strategy": "mean",
                        "fill_value": 1.0,
                    },
                },
                "default_impute_strategy": "most_frequent",
            },
        },
        random_seed=0,
    )


@pytest.mark.parametrize("problem_type", ["binary", "multiclass", "regression"])
@pytest.mark.parametrize("different_names", [True, False])
def test_make_pipeline_from_actions_with_duplicate_actions(
    problem_type,
    different_names,
):
    pipeline_class = _get_pipeline_base_class(problem_type)

    actions = [
        DataCheckAction(DataCheckActionCode.DROP_COL, None, {"columns": ["some col"]}),
        DataCheckAction(
            DataCheckActionCode.DROP_COL,
            None if different_names else "Data check name",
            {"columns": ["some other col"]},
        ),
    ]
    assert make_pipeline_from_actions(problem_type, actions) == pipeline_class(
        component_graph={"Drop Columns Transformer": [DropColumns, "X", "y"]},
        parameters={
            "Drop Columns Transformer": {"columns": ["some col", "some other col"]},
        },
        random_seed=0,
    )
    actions = [
        DataCheckAction(
            DataCheckActionCode.DROP_ROWS,
            None,
            metadata={"rows": [0, 1, 3]},
        ),
        DataCheckAction(
            DataCheckActionCode.DROP_ROWS,
            None if different_names else "Data check name",
            metadata={"rows": [1, 2]},
        ),
    ]
    assert make_pipeline_from_actions(problem_type, actions) == pipeline_class(
        component_graph={"Drop Rows Transformer": [DropRowsTransformer, "X", "y"]},
        parameters={"Drop Rows Transformer": {"indices_to_drop": [0, 1, 2, 3]}},
        random_seed=0,
    )


@pytest.mark.parametrize(
    "sampler",
    [
        None,
        "Undersampler",
        "Oversampler",
    ],
)
@pytest.mark.parametrize("problem_type", ["binary", "multiclass", "regression"])
def test_make_pipeline_samplers(
    problem_type,
    sampler,
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
    estimators = get_estimators(problem_type=problem_type)

    for estimator in estimators:
        if problem_type == "regression" and sampler is not None:
            with pytest.raises(ValueError, match="Sampling is unsupported for"):
                make_pipeline(X, y, estimator, problem_type, sampler_name=sampler)
        else:
            pipeline = make_pipeline(
                X,
                y,
                estimator,
                problem_type,
                sampler_name=sampler,
            )
            # check that we do add the sampler properly
            if sampler is not None and problem_type != "regression":
                assert any("sampler" in comp.name for comp in pipeline.component_graph)
            else:
                assert not any(
                    "sampler" in comp.name for comp in pipeline.component_graph
                )


def test_get_estimators():
    assert len(get_estimators(problem_type=ProblemTypes.BINARY)) == 6
    assert (
        len(
            get_estimators(
                problem_type=ProblemTypes.BINARY,
                model_families=[ModelFamily.LINEAR_MODEL],
            ),
        )
        == 2
    )
    assert len(get_estimators(problem_type=ProblemTypes.MULTICLASS)) == 6
    assert len(get_estimators(problem_type=ProblemTypes.REGRESSION)) == 5
    assert (
        len(
            get_estimators(
                problem_type=ProblemTypes.MULTISERIES_TIME_SERIES_REGRESSION,
            ),
        )
        == 1
    )

    assert len(get_estimators(problem_type=ProblemTypes.BINARY, model_families=[])) == 0
    assert (
        len(get_estimators(problem_type=ProblemTypes.MULTICLASS, model_families=[]))
        == 0
    )
    assert (
        len(get_estimators(problem_type=ProblemTypes.REGRESSION, model_families=[]))
        == 0
    )

    with pytest.raises(RuntimeError, match="Unrecognized model type for problem type"):
        get_estimators(
            problem_type=ProblemTypes.REGRESSION,
            model_families=["random_forest", "none"],
        )
    with pytest.raises(TypeError, match="model_families parameter is not a list."):
        get_estimators(
            problem_type=ProblemTypes.REGRESSION,
            model_families="random_forest",
        )
    with pytest.raises(KeyError):
        get_estimators(problem_type="Not A Valid Problem Type")

    match_text = "Both `model_families` and `excluded_model_families` cannot be set."
    with pytest.raises(ValueError, match=match_text):
        get_estimators(
            ProblemTypes.REGRESSION,
            model_families=[ModelFamily.RANDOM_FOREST],
            excluded_model_families=[ModelFamily.XGBOOST],
        )


def test_generate_code_pipeline_errors():
    with pytest.raises(ValueError, match="Element must be a pipeline instance"):
        generate_pipeline_code(BinaryClassificationPipeline)

    with pytest.raises(ValueError, match="Element must be a pipeline instance"):
        generate_pipeline_code(RegressionPipeline)

    with pytest.raises(ValueError, match="Element must be a pipeline instance"):
        generate_pipeline_code(MulticlassClassificationPipeline)

    with pytest.raises(ValueError, match="Element must be a pipeline instance"):
        generate_pipeline_code([Imputer])

    with pytest.raises(ValueError, match="Element must be a pipeline instance"):
        generate_pipeline_code([Imputer, LogisticRegressionClassifier])

    with pytest.raises(ValueError, match="Element must be a pipeline instance"):
        generate_pipeline_code([Imputer(), LogisticRegressionClassifier()])


def test_generate_code_pipeline_json_with_objects(get_black_config):
    class CustomEstimator(Estimator):
        name = "My Custom Estimator"
        hyperparameter_ranges = {}
        supported_problem_types = [ProblemTypes.BINARY, ProblemTypes.MULTICLASS]
        model_family = ModelFamily.NONE

        def __init__(self, random_arg=False, numpy_arg=[], random_seed=0):
            parameters = {"random_arg": random_arg, "numpy_arg": numpy_arg}

            super().__init__(
                parameters=parameters,
                component_obj=None,
                random_seed=random_seed,
            )

    component_graph = ["Imputer", CustomEstimator]
    pipeline = BinaryClassificationPipeline(
        component_graph,
        custom_name="Mock Binary Pipeline with Transformer",
        parameters={"My Custom Estimator": {"numpy_arg": np.array([0])}},
    )
    generated_pipeline_code = generate_pipeline_code(pipeline)
    assert generated_pipeline_code == black.format_str(
        "from evalml.pipelines.binary_classification_pipeline import BinaryClassificationPipeline\n"
        "pipeline = BinaryClassificationPipeline(component_graph={'Imputer': ['Imputer', 'X', 'y'], 'My Custom Estimator': [CustomEstimator, 'Imputer.x', 'y']}, "
        "parameters={'Imputer':{'categorical_impute_strategy': 'most_frequent', 'numeric_impute_strategy': 'mean', 'boolean_impute_strategy': 'most_frequent', 'categorical_fill_value': None, 'numeric_fill_value': None, 'boolean_fill_value': None}, "
        "'My Custom Estimator':{'random_arg': False, 'numpy_arg': array([0])}}, custom_name='Mock Binary Pipeline with Transformer', random_seed=0)",
        mode=black.Mode(**get_black_config),
    )

    pipeline = BinaryClassificationPipeline(
        component_graph,
        custom_name="Mock Binary Pipeline with Transformer",
        parameters={"My Custom Estimator": {"random_arg": Imputer()}},
    )
    generated_pipeline_code = generate_pipeline_code(pipeline)
    assert generated_pipeline_code == black.format_str(
        "from evalml.pipelines.binary_classification_pipeline import BinaryClassificationPipeline\n"
        "pipeline = BinaryClassificationPipeline(component_graph={'Imputer': ['Imputer', 'X', 'y'], 'My Custom Estimator': [CustomEstimator, 'Imputer.x', 'y']}, "
        "parameters={'Imputer':{'categorical_impute_strategy': 'most_frequent', 'numeric_impute_strategy': 'mean', 'boolean_impute_strategy': 'most_frequent', 'categorical_fill_value': None, 'numeric_fill_value': None, 'boolean_fill_value': None}, "
        "'My Custom Estimator':{'random_arg': Imputer(categorical_impute_strategy='most_frequent', numeric_impute_strategy='mean', boolean_impute_strategy='most_frequent', categorical_fill_value=None, numeric_fill_value=None, boolean_fill_value=None), 'numpy_arg': []}}, "
        "custom_name='Mock Binary Pipeline with Transformer', random_seed=0)",
        mode=black.Mode(**get_black_config),
    )


def test_generate_code_pipeline(get_black_config):
    binary_pipeline = BinaryClassificationPipeline(
        ["Imputer", "Random Forest Classifier"],
    )
    expected_code = black.format_str(
        "from evalml.pipelines.binary_classification_pipeline import BinaryClassificationPipeline\n"
        "pipeline = BinaryClassificationPipeline(component_graph={'Imputer': ['Imputer', 'X', 'y'], 'Random Forest Classifier': ['Random Forest Classifier', 'Imputer.x', 'y']}, "
        "parameters={'Imputer':{'categorical_impute_strategy': 'most_frequent', 'numeric_impute_strategy': 'mean', 'boolean_impute_strategy': 'most_frequent', 'categorical_fill_value': None, 'numeric_fill_value': None, 'boolean_fill_value': None}, "
        "'Random Forest Classifier':{'n_estimators': 100, 'max_depth': 6, 'n_jobs': -1}}, random_seed=0)",
        mode=black.Mode(**get_black_config),
    )

    pipeline = generate_pipeline_code(binary_pipeline)
    assert expected_code == pipeline

    regression_pipeline = RegressionPipeline(
        ["Imputer", "Random Forest Regressor"],
        custom_name="Mock Regression Pipeline",
    )
    expected_code = black.format_str(
        "from evalml.pipelines.regression_pipeline import RegressionPipeline\n"
        "pipeline = RegressionPipeline(component_graph={'Imputer': ['Imputer', 'X', 'y'], 'Random Forest Regressor': ['Random Forest Regressor', 'Imputer.x', 'y']}, parameters={'Imputer':{'categorical_impute_strategy': 'most_frequent', 'numeric_impute_strategy': 'mean', 'boolean_impute_strategy': 'most_frequent', 'categorical_fill_value': None, 'numeric_fill_value': None, 'boolean_fill_value': None}, "
        "'Random Forest Regressor':{'n_estimators': 100, 'max_depth': 6, 'n_jobs': -1}}, custom_name='Mock Regression Pipeline', random_seed=0)",
        mode=black.Mode(**get_black_config),
    )
    pipeline = generate_pipeline_code(regression_pipeline)
    assert pipeline == expected_code

    regression_pipeline_with_params = RegressionPipeline(
        ["DFS Transformer", "Imputer", "Random Forest Regressor"],
        custom_name="Mock Regression Pipeline",
        parameters={
            "DFS Transformer": {"features": None},
            "Imputer": {"numeric_impute_strategy": "most_frequent"},
            "Random Forest Regressor": {"n_estimators": 50},
        },
    )
    expected_code_params = black.format_str(
        "from evalml.pipelines.regression_pipeline import RegressionPipeline\n"
        "pipeline = RegressionPipeline(component_graph={'DFS Transformer': ['DFS Transformer', 'X', 'y'],'Imputer': ['Imputer', 'DFS Transformer.x', 'y'], 'Random Forest Regressor': ['Random Forest Regressor', 'Imputer.x', 'y']}, "
        "parameters={'DFS Transformer':{}, 'Imputer':{'categorical_impute_strategy': 'most_frequent', 'numeric_impute_strategy': 'most_frequent', 'boolean_impute_strategy': 'most_frequent', 'categorical_fill_value': None, 'numeric_fill_value': None, 'boolean_fill_value': None}, "
        "'Random Forest Regressor':{'n_estimators': 50, 'max_depth': 6, 'n_jobs': -1}}, custom_name='Mock Regression Pipeline', random_seed=0)",
        mode=black.Mode(**get_black_config),
    )
    pipeline = generate_pipeline_code(regression_pipeline_with_params)
    assert pipeline == expected_code_params


def test_generate_code_nonlinear_pipeline(get_black_config):
    custom_name = "Non Linear Binary Pipeline"
    component_graph = {
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
    pipeline = BinaryClassificationPipeline(
        component_graph=component_graph,
        custom_name=custom_name,
    )
    expected = black.format_str(
        "from evalml.pipelines.binary_classification_pipeline import BinaryClassificationPipeline\n"
        "pipeline = BinaryClassificationPipeline("
        "component_graph={'Imputer': ['Imputer', 'X', 'y'], "
        "'OneHot_RandomForest': ['One Hot Encoder', 'Imputer.x', 'y'], "
        "'OneHot_ElasticNet': ['One Hot Encoder', 'Imputer.x', 'y'], "
        "'Random Forest': ['Random Forest Classifier', 'OneHot_RandomForest.x', 'y'], "
        "'Elastic Net': ['Elastic Net Classifier', 'OneHot_ElasticNet.x', 'y'], "
        "'Logistic Regression Classifier': ['Logistic Regression Classifier', 'Random Forest.x', 'Elastic Net.x', 'y']}, "
        "parameters={'Imputer':{'categorical_impute_strategy': 'most_frequent', 'numeric_impute_strategy': 'mean', 'boolean_impute_strategy': 'most_frequent', 'categorical_fill_value': None, 'numeric_fill_value': None, 'boolean_fill_value': None}, "
        "'OneHot_RandomForest':{'top_n': 10, 'features_to_encode': None, 'categories': None, 'drop': 'if_binary', 'handle_unknown': 'ignore', 'handle_missing': 'error'}, "
        "'OneHot_ElasticNet':{'top_n': 10, 'features_to_encode': None, 'categories': None, 'drop': 'if_binary', 'handle_unknown': 'ignore', 'handle_missing': 'error'}, "
        "'Random Forest':{'n_estimators': 100, 'max_depth': 6, 'n_jobs': -1}, "
        "'Elastic Net':{'penalty': 'elasticnet', 'C': 1.0, 'l1_ratio': 0.15, 'n_jobs': -1, 'multi_class': 'auto', 'solver': 'saga'}, "
        "'Logistic Regression Classifier':{'penalty': 'l2', 'C': 1.0, 'n_jobs': -1, 'multi_class': 'auto', 'solver': 'lbfgs'}}, "
        "custom_name='Non Linear Binary Pipeline', random_seed=0)",
        mode=black.Mode(**get_black_config),
    )
    pipeline_code = generate_pipeline_code(pipeline)
    assert pipeline_code == expected


def test_generate_code_pipeline_with_custom_components(get_black_config):
    class CustomTransformer(Transformer):
        name = "My Custom Transformer"
        hyperparameter_ranges = {}

        def __init__(self, random_seed=0):
            parameters = {}

            super().__init__(
                parameters=parameters,
                component_obj=None,
                random_seed=random_seed,
            )

        def transform(self, X, y=None):
            return X

    class CustomEstimator(Estimator):
        name = "My Custom Estimator"
        hyperparameter_ranges = {}
        supported_problem_types = [ProblemTypes.BINARY, ProblemTypes.MULTICLASS]
        model_family = ModelFamily.NONE

        def __init__(self, random_arg=False, random_seed=0):
            parameters = {"random_arg": random_arg}

            super().__init__(
                parameters=parameters,
                component_obj=None,
                random_seed=random_seed,
            )

    mock_pipeline_with_custom_components = BinaryClassificationPipeline(
        [CustomTransformer, CustomEstimator],
    )
    expected_code = black.format_str(
        "from evalml.pipelines.binary_classification_pipeline import BinaryClassificationPipeline\n"
        "pipeline = BinaryClassificationPipeline(component_graph={'My Custom Transformer': [CustomTransformer, 'X', 'y'], 'My Custom Estimator': [CustomEstimator, 'My Custom Transformer.x', 'y']}, "
        "parameters={'My Custom Estimator':{'random_arg': False}}, random_seed=0)",
        mode=black.Mode(**get_black_config),
    )
    pipeline = generate_pipeline_code(mock_pipeline_with_custom_components)
    assert pipeline == expected_code


@pytest.mark.parametrize(
    "automl_type",
    [
        ProblemTypes.BINARY,
        ProblemTypes.MULTICLASS,
        ProblemTypes.REGRESSION,
        ProblemTypes.TIME_SERIES_REGRESSION,
        ProblemTypes.TIME_SERIES_MULTICLASS,
        ProblemTypes.TIME_SERIES_BINARY,
    ],
)
@pytest.mark.parametrize(
    "ensembling",
    [True, False],
)
def test_generate_pipeline_example(
    automl_type,
    ensembling,
    tmpdir,
    AutoMLTestEnv,
    X_y_binary,
    X_y_multi,
    X_y_regression,
    ts_data,
    is_using_windows,
):
    if is_using_windows:
        pytest.xfail(
            "Pipeline code generation is not expected to work on Windows machines",
        )
    if automl_type == ProblemTypes.BINARY:
        X, y = X_y_binary
    elif automl_type == ProblemTypes.MULTICLASS:
        X, y = X_y_multi
    elif automl_type == ProblemTypes.REGRESSION:
        X, y = X_y_regression
    elif is_time_series(automl_type):
        X, _, y = ts_data(problem_type=automl_type)

    problem_configuration = (
        {
            "time_index": "date",
            "gap": 0,
            "max_delay": 2,
            "forecast_horizon": 1,
        }
        if is_time_series(automl_type)
        else None
    )

    import featuretools as ft

    from evalml import AutoMLSearch
    from evalml.preprocessing import split_data

    X_train, X_test, y_train, y_test = split_data(
        X,
        y,
        problem_type=automl_type,
        test_size=0.2,
        problem_configuration=problem_configuration,
    )

    X_train = pd.DataFrame(X_train)
    X_train.columns = X_train.columns.astype(str)
    es = ft.EntitySet()
    es = es.add_dataframe(
        dataframe_name="X",
        dataframe=X_train,
        index="index",
        make_index=False,
    )
    X_train_t, features = ft.dfs(
        entityset=es,
        target_dataframe_name="X",
        trans_primitives=["absolute"],
        return_types="all",
    )

    X_train_t.index = y_train.index
    features_path = os.path.join(str(tmpdir), "features.json")
    ft.save_features(features, features_path)
    ensembling = ensembling if not is_time_series(automl_type) else False
    aml = AutoMLSearch(
        X_train=X_train_t,
        y_train=y_train,
        problem_type=automl_type,
        optimize_thresholds=False,
        problem_configuration=problem_configuration,
        features=features,
        ensembling=ensembling,
    )

    if ensembling:
        aml.search()
    else:
        env = AutoMLTestEnv(automl_type)
        with env.test_context(score_return_value={aml.objective.name: 1.0}):
            aml.search()

    if ensembling:
        for i in range(len(aml.full_rankings)):
            pipeline = aml.get_pipeline(i)
            if isinstance(
                pipeline.estimator,
                StackedEnsembleBase,
            ):
                break
        assert isinstance(
            pipeline.estimator,
            StackedEnsembleBase,
        )
    else:
        pipeline = aml.get_pipeline(1)

    y_train.index = X_train_t.index
    y_test.index = X_test.index
    X_train_t.ww["target"] = y_train
    X_test.ww["target"] = y_test.reindex(X_test.index)

    train_path = os.path.join(str(tmpdir), "train")
    holdout_path = os.path.join(str(tmpdir), "holdout")

    X_train_t.ww.to_disk(train_path)
    X_test.ww.to_disk(holdout_path)
    output_path = os.path.join(str(tmpdir), "example.py")

    # extra features provided to example
    with pytest.raises(
        ValueError,
        match="Provided features in `features_path` do not match pipeline features. There is a different amount of features in the loaded features.",
    ):
        _, false_features = ft.dfs(
            entityset=es,
            target_dataframe_name="X",
            trans_primitives=["absolute", "is_null"],
            return_types="all",
        )
        false_features_path = os.path.join(str(tmpdir), "false_features.json")
        ft.save_features(false_features, false_features_path)
        _ = generate_pipeline_example(
            pipeline=pipeline,
            path_to_train=train_path,
            path_to_holdout=holdout_path,
            path_to_features=false_features_path,
            target="target",
            output_file_path=output_path,
        )

    # different features provided to example
    with pytest.raises(
        ValueError,
        match="Provided features in `features_path` do not match pipeline features.",
    ):
        _, false_features = ft.dfs(
            entityset=es,
            target_dataframe_name="X",
            trans_primitives=["sine"],
            return_types="all",
        )
        false_features_path = os.path.join(str(tmpdir), "false_features.json")
        ft.save_features(false_features, false_features_path)
        _ = generate_pipeline_example(
            pipeline=pipeline,
            path_to_train=train_path,
            path_to_holdout=holdout_path,
            path_to_features=false_features_path,
            target="target",
            output_file_path=output_path,
        )

    pipeline_example = generate_pipeline_example(
        pipeline=pipeline,
        path_to_train=train_path,
        path_to_holdout=holdout_path,
        path_to_features=features_path,
        target="target",
        output_file_path=output_path,
    )
    assert f'PATH_TO_TRAIN = "{train_path}"' in pipeline_example
    assert f'PATH_TO_HOLDOUT = "{holdout_path}"' in pipeline_example
    assert 'TARGET = "target"' in pipeline_example
    assert 'column_mapping = ""' in pipeline_example
    assert generate_pipeline_code(pipeline, features_path) in pipeline_example

    if is_time_series(automl_type):
        assert "predict(X_test, X_train=X_train, y_train=y_train)" in pipeline_example
    else:
        assert "predict(X_test)" in pipeline_example
    exec(pipeline_example)
    assert os.path.exists(output_path)


def test_rows_of_interest_errors(X_y_binary):
    pipeline = BinaryClassificationPipeline(
        component_graph=["Logistic Regression Classifier"],
    )
    pipeline_mc = MulticlassClassificationPipeline(
        component_graph=["Logistic Regression Classifier"],
    )
    X, y = X_y_binary

    with pytest.raises(ValueError, match="Invalid arg for"):
        rows_of_interest(pipeline, X, y, types="ball")

    with pytest.raises(ValueError, match="Need an input y in order to"):
        rows_of_interest(pipeline, X, types="correct")

    with pytest.raises(ValueError, match="Pipeline provided must be a fitted"):
        rows_of_interest(pipeline, X, y, types="all")

    with pytest.raises(ValueError, match="Pipeline provided must be a fitted"):
        rows_of_interest(pipeline_mc, X, y, types="all")

    with pytest.raises(ValueError, match="Pipeline provided must be a fitted"):
        rows_of_interest(pipeline_mc, X, y, types="all")

    pipeline._is_fitted = True
    with pytest.raises(ValueError, match="Provided threshold 1.1 must be between"):
        rows_of_interest(pipeline, X, y, threshold=1.1)

    with pytest.raises(ValueError, match="Provided threshold -0.1 must be between"):
        rows_of_interest(pipeline, X, y, threshold=-0.1)


@patch("evalml.pipelines.BinaryClassificationPipeline.predict_proba")
@patch("evalml.pipelines.BinaryClassificationPipeline.fit")
@pytest.mark.parametrize("threshold", [0.3, None, 0.7])
@pytest.mark.parametrize("y", [pd.Series([i % 2 for i in range(100)]), None])
def test_rows_of_interest_threshold(mock_fit, mock_pred_proba, threshold, y):
    pipeline = BinaryClassificationPipeline(
        component_graph=[
            "Imputer",
            "Standard Scaler",
            "Logistic Regression Classifier",
        ],
    )
    X = pd.DataFrame([i for i in range(100)])
    y = y
    pipeline._is_fitted = True

    vals = [0.2] * 25 + [0.5] * 50 + [0.8] * 25
    predicted_proba_values = pd.DataFrame({0: [1 - v for v in vals], 1: vals})
    mock_pred_proba.return_value = predicted_proba_values
    vals = rows_of_interest(
        pipeline,
        X,
        y,
        threshold=threshold,
        epsilon=0.5,
        sort_values=True,
    )
    if threshold == 0.3:
        assert vals == list(range(100))
    elif threshold == 0.7:
        assert vals == list(range(75, 100)) + list(range(25, 75)) + list(range(25))
    else:
        assert vals == list(range(25, 75)) + list(range(25)) + list(range(75, 100))

    pipeline._threshold = 0.9
    vals = rows_of_interest(
        pipeline,
        X,
        y,
        threshold=None,
        epsilon=0.5,
        sort_values=True,
    )
    assert vals == list(range(75, 100)) + list(range(25, 75))


@patch("evalml.pipelines.BinaryClassificationPipeline.predict_proba")
@patch("evalml.pipelines.BinaryClassificationPipeline.fit")
@pytest.mark.parametrize(
    "types,expected_val",
    [
        ("incorrect", list(range(75, 100))),
        ("correct", list(range(75))),
        ("true_positive", list(range(25, 75))),
        ("true_negative", list(range(25)) + list(range(75, 100))),
        ("all", list(range(100))),
    ],
)
def test_rows_of_interest_types(mock_fit, mock_pred_proba, types, expected_val):
    pipeline = BinaryClassificationPipeline(
        component_graph=["Logistic Regression Classifier"],
    )
    X = pd.DataFrame([i for i in range(100)])
    y = pd.Series([0] * 25 + [1] * 50 + [0] * 25)
    pipeline._is_fitted = True

    vals = [0.2] * 25 + [0.5] * 50 + [0.8] * 25
    predicted_proba_values = pd.DataFrame({0: [1 - v for v in vals], 1: vals})
    mock_pred_proba.return_value = predicted_proba_values
    vals = rows_of_interest(pipeline, X, y, types=types, epsilon=0.5, sort_values=False)
    assert vals == expected_val


@patch("evalml.pipelines.BinaryClassificationPipeline.predict_proba")
@patch("evalml.pipelines.BinaryClassificationPipeline.fit")
@pytest.mark.parametrize("epsilon,expected_len", [(0.01, 50), (0.3, 75), (0.5, 100)])
def test_rows_of_interest_epsilon(mock_fit, mock_pred_proba, epsilon, expected_len):
    pipeline = BinaryClassificationPipeline(
        component_graph=["Logistic Regression Classifier"],
    )
    X = pd.DataFrame([i for i in range(100)])
    y = pd.Series([0] * 25 + [1] * 50 + [0] * 25)
    pipeline._is_fitted = True

    vals = [0.2] * 25 + [0.5] * 50 + [0.85] * 25
    predicted_proba_values = pd.DataFrame({0: [1 - v for v in vals], 1: vals})
    mock_pred_proba.return_value = predicted_proba_values
    vals = rows_of_interest(pipeline, X, y, epsilon=epsilon)
    assert len(vals) == expected_len

    if epsilon == 0.01:
        vals = [0.2] * 25 + [0.65] * 50 + [0.85] * 25
        predicted_proba_values = pd.DataFrame({0: [1 - v for v in vals], 1: vals})
        mock_pred_proba.return_value = predicted_proba_values
        vals = rows_of_interest(pipeline, X, y, epsilon=epsilon)
        assert len(vals) == 0


@patch("evalml.pipelines.BinaryClassificationPipeline.predict_proba")
@patch("evalml.pipelines.BinaryClassificationPipeline.fit")
@pytest.mark.parametrize(
    "sorts,expected_val",
    [
        (True, list(range(75, 100)) + list(range(25, 75)) + list(range(25))),
        (False, list(range(100))),
    ],
)
def test_rows_of_interest_sorted(mock_fit, mock_pred_proba, sorts, expected_val):
    pipeline = BinaryClassificationPipeline(
        component_graph=["Logistic Regression Classifier"],
    )
    X = pd.DataFrame([i for i in range(100)])
    y = pd.Series([0] * 25 + [1] * 50 + [0] * 25)
    pipeline._is_fitted = True

    vals = [0.2] * 25 + [0.5] * 50 + [0.8] * 25
    predicted_proba_values = pd.DataFrame({0: [1 - v for v in vals], 1: vals})
    mock_pred_proba.return_value = predicted_proba_values
    vals = rows_of_interest(
        pipeline,
        X,
        y,
        threshold=0.9,
        epsilon=0.9,
        sort_values=sorts,
    )
    assert vals == expected_val


@patch("evalml.pipelines.BinaryClassificationPipeline.predict_proba")
@patch("evalml.pipelines.BinaryClassificationPipeline.fit")
def test_rows_of_interest_index(mock_fit, mock_pred_proba):
    pipeline = BinaryClassificationPipeline(
        component_graph=["Logistic Regression Classifier"],
    )
    X = pd.DataFrame(
        [i for i in range(100)],
        index=["index_{}".format(i) for i in range(100)],
    )
    pipeline._is_fitted = True

    vals = [0.2] * 25 + [0.5] * 50 + [0.8] * 25
    predicted_proba_values = pd.DataFrame({0: [1 - v for v in vals], 1: vals})
    mock_pred_proba.return_value = predicted_proba_values
    vals = rows_of_interest(pipeline, X, epsilon=0.5)
    assert vals == list(range(25, 75)) + list(range(25)) + list(range(75, 100))


@patch("evalml.pipelines.BinaryClassificationPipeline.predict_proba")
@patch("evalml.pipelines.BinaryClassificationPipeline.fit")
@pytest.mark.parametrize(
    "types,sorts,epsilon,expected_vals",
    [
        ("correct", True, 0.01, list(range(25, 75))),
        ("true_negative", True, 0.3, list(range(25))),
        ("all", False, 0.3, list(range(75))),
    ],
)
def test_rows_of_interest(
    mock_fit,
    mock_pred_proba,
    types,
    sorts,
    epsilon,
    expected_vals,
):
    pipeline = BinaryClassificationPipeline(
        component_graph=["Logistic Regression Classifier"],
    )
    X = pd.DataFrame([i for i in range(100)])
    y = pd.Series([0] * 25 + [1] * 50 + [0] * 25)
    pipeline._is_fitted = True

    vals = [0.2] * 25 + [0.5] * 50 + [0.85] * 25
    predicted_proba_values = pd.DataFrame({0: [1 - v for v in vals], 1: vals})
    mock_pred_proba.return_value = predicted_proba_values
    vals = rows_of_interest(
        pipeline,
        X,
        y,
        types=types,
        sort_values=sorts,
        epsilon=epsilon,
    )
    assert vals == expected_vals

    if types == "all":
        vals = rows_of_interest(
            pipeline,
            X,
            types=types,
            sort_values=sorts,
            epsilon=epsilon,
        )
        assert vals == expected_vals


@patch("evalml.pipelines.BinaryClassificationPipeline.predict_proba")
@patch("evalml.pipelines.BinaryClassificationPipeline.fit")
def test_rows_of_interest_empty(mock_fit, mock_pred_proba):
    pipeline = BinaryClassificationPipeline(
        component_graph=["Logistic Regression Classifier"],
    )
    X = pd.DataFrame([i for i in range(100)])
    y = pd.Series([0] * 25 + [1] * 50 + [0] * 25)
    pipeline._is_fitted = True

    vals = [1] * 25 + [0] * 50 + [1] * 25
    predicted_proba_values = pd.DataFrame({0: [1 - v for v in vals], 1: vals})
    mock_pred_proba.return_value = predicted_proba_values
    vals = rows_of_interest(pipeline, X, y, epsilon=0.5, types="correct")
    assert len(vals) == 0


def test_make_pipeline_from_multiple_graphs_with_sampler(X_y_binary):
    X, y = X_y_binary
    estimator = handle_component_class("Random Forest Classifier")
    pipeline_1 = make_pipeline(
        X,
        y,
        estimator,
        ProblemTypes.BINARY,
        sampler_name="Undersampler",
        use_estimator=False,
    )
    pipeline_2 = make_pipeline(
        X,
        y,
        estimator,
        ProblemTypes.BINARY,
        sampler_name="Undersampler",
        use_estimator=False,
    )
    input_pipelines = [pipeline_1, pipeline_2]

    combined_pipeline = _make_pipeline_from_multiple_graphs(
        input_pipelines=input_pipelines,
        estimator=estimator,
        problem_type=ProblemTypes.BINARY,
    )
    second_pipeline_sampler = (
        "Pipeline w/ Label Encoder + Imputer + Undersampler Pipeline 2 - Undersampler.y"
    )
    assert (
        combined_pipeline.component_graph.get_inputs("Random Forest Classifier")[2]
        == second_pipeline_sampler
    )


def test_make_pipeline_from_multiple_graphs_pre_pipeline_components(X_y_binary):
    X, y = X_y_binary
    estimator = handle_component_class("Random Forest Classifier")
    pre_pipeline_components = {"DFS Transformer": ["DFS Transformer", "X", "y"]}
    pipeline_1 = make_pipeline(
        X,
        y,
        estimator,
        ProblemTypes.BINARY,
        sampler_name="Undersampler",
        use_estimator=False,
    )
    pipeline_2 = make_pipeline(
        X,
        y,
        estimator,
        ProblemTypes.BINARY,
        sampler_name="Undersampler",
        use_estimator=False,
    )

    input_pipelines = [pipeline_1, pipeline_2]
    pipeline_1._custom_name = "First"
    pipeline_2._custom_name = "Second"
    sub_pipeline_names = {
        pipeline_1.name: "First",
        pipeline_2.name: "Second",
    }
    combined_pipeline = _make_pipeline_from_multiple_graphs(
        input_pipelines=input_pipelines,
        estimator=estimator,
        problem_type=ProblemTypes.BINARY,
        pre_pipeline_components=pre_pipeline_components,
        sub_pipeline_names=sub_pipeline_names,
    )
    assert (
        combined_pipeline.component_graph.get_inputs("First Pipeline - Imputer")[0]
        == "DFS Transformer.x"
    )
    assert (
        combined_pipeline.component_graph.get_inputs("Second Pipeline - Imputer")[0]
        == "DFS Transformer.x"
    )


def test_make_pipeline_features_and_dfs(X_y_binary):
    X, y = X_y_binary
    estimator = handle_component_class("Random Forest Classifier")
    features = True
    pipeline = make_pipeline(
        X,
        y,
        estimator,
        ProblemTypes.BINARY,
        sampler_name="Undersampler",
        features=features,
    )

    assert "DFS Transformer" == pipeline.component_graph.compute_order[0]


@pytest.mark.parametrize("target_name", ["target", "Target_Data"])
def test_unstack_multiseries(
    target_name,
    multiseries_ts_data_stacked,
    multiseries_ts_data_unstacked,
):
    X, y = multiseries_ts_data_stacked
    X_unstacked, y_unstacked = multiseries_ts_data_unstacked
    y.name = target_name
    y_unstacked.columns = [
        f"{target_name}_{i}" for i in range(len(y_unstacked.columns))
    ]

    X_unstacked_transformed, y_unstacked_transformed = unstack_multiseries(
        X,
        y,
        "series_id",
        "date",
        target_name=target_name,
    )
    pd.testing.assert_frame_equal(
        X_unstacked.sort_index(axis=1),
        X_unstacked_transformed.sort_index(axis=1),
        check_freq=False,
    )
    pd.testing.assert_frame_equal(
        y_unstacked,
        y_unstacked_transformed,
        check_freq=False,
    )


@pytest.mark.parametrize("include_series_id", [True, False])
@pytest.mark.parametrize("series_id_name", [None, "SERIES"])
@pytest.mark.parametrize("starting_index", [None, 1, 132])
def test_stack_data(
    include_series_id,
    series_id_name,
    starting_index,
    multiseries_ts_data_stacked,
    multiseries_ts_data_unstacked,
):
    _, y = multiseries_ts_data_unstacked
    _, y_stacked = multiseries_ts_data_stacked

    y_stacked.name = "target"
    y_stacked_transformed = stack_data(
        y,
        include_series_id=include_series_id,
        series_id_name=series_id_name,
        starting_index=starting_index,
    )

    if starting_index is not None:
        y_stacked.index = y_stacked.index + starting_index

    if include_series_id:
        series_id_name = series_id_name or "series_id"
        series_id_col = pd.Series(
            list(range(5)) * 20,
            dtype="str",
            index=y_stacked.index,
        )
        y_stacked = pd.DataFrame({series_id_name: series_id_col, "target": y_stacked})
        pd.testing.assert_frame_equal(y_stacked, y_stacked_transformed)

    else:
        pd.testing.assert_series_equal(y_stacked, y_stacked_transformed)


def test_stack_data_noop():
    none_y = None
    series_y = pd.Series(range(100))

    assert stack_data(none_y) is None
    pd.testing.assert_series_equal(stack_data(series_y), series_y)


@pytest.mark.parametrize("series_id_values_type", [set, list])
@pytest.mark.parametrize("no_features", [True, False])
@pytest.mark.parametrize("starting_index", [None, 1, 132])
def test_stack_X(
    starting_index,
    no_features,
    series_id_values_type,
    multiseries_ts_data_stacked,
    multiseries_ts_data_unstacked,
):
    X, _ = multiseries_ts_data_unstacked
    X_expected, _ = multiseries_ts_data_stacked

    if starting_index is not None:
        X_expected.index = X_expected.index + starting_index

    if no_features:
        series_id_values = series_id_values_type(str(i) for i in range(0, 5))
        X = pd.DataFrame(X["date"])
        X_expected = X_expected[["date", "series_id"]]

        with pytest.raises(
            ValueError,
            match="Series ID values need to be passed in X column values or as a set with the `series_id_values` parameter.",
        ):
            stack_X(X, "series_id", "date", starting_index=starting_index)

        X_transformed = stack_X(
            X,
            "series_id",
            "date",
            starting_index=starting_index,
            series_id_values=series_id_values,
        )

    else:
        X_transformed = stack_X(X, "series_id", "date", starting_index=starting_index)

    pd.testing.assert_frame_equal(
        X_expected.sort_index(axis=1),
        X_transformed.sort_index(axis=1),
    )
