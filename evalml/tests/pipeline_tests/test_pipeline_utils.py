from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import woodwork as ww

from evalml.data_checks import DataCheckAction, DataCheckActionCode
from evalml.model_family import ModelFamily
from evalml.pipelines import (
    BinaryClassificationPipeline,
    MulticlassClassificationPipeline,
    RegressionPipeline,
)
from evalml.pipelines.components import (
    DateTimeFeaturizer,
    DropColumns,
    DropNullColumns,
    DropRowsTransformer,
    EmailFeaturizer,
    Estimator,
    Imputer,
    LinearRegressor,
    LogisticRegressionClassifier,
    NaturalLanguageFeaturizer,
    OneHotEncoder,
    StandardScaler,
    TargetImputer,
    TimeSeriesFeaturizer,
    Transformer,
    URLFeaturizer,
)
from evalml.pipelines.components.transformers.encoders.label_encoder import (
    LabelEncoder,
)
from evalml.pipelines.components.utils import handle_component_class
from evalml.pipelines.utils import (
    _get_pipeline_base_class,
    _make_pipeline_from_multiple_graphs,
    generate_pipeline_code,
    get_estimators,
    is_classification,
    make_pipeline,
    make_pipeline_from_actions,
    rows_of_interest,
)
from evalml.problem_types import ProblemTypes, is_regression, is_time_series


@pytest.fixture
def get_test_data_from_configuration():
    def _get_test_data_from_configuration(input_type, problem_type, column_names=None):
        X_all = pd.DataFrame(
            {
                "all_null": [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]
                * 2,
                "numerical": range(14),
                "categorical": ["a", "b", "a", "b", "b", "a", "b"] * 2,
                "dates": pd.date_range("2000-02-03", periods=14, freq="W"),
                "text": [
                    "this is a string",
                    "this is another string",
                    "this is just another string",
                    "evalml should handle string input",
                    "cats are gr8",
                    "hello world",
                    "evalml is gr8",
                ]
                * 2,
                "email": [
                    "abalone_0@gmail.com",
                    "AbaloneRings@yahoo.com",
                    "abalone_2@abalone.com",
                    "titanic_data@hotmail.com",
                    "fooEMAIL@email.org",
                    "evalml@evalml.org",
                    "evalml@alteryx.org",
                ]
                * 2,
                "url": [
                    "https://evalml.alteryx.com/en/stable/",
                    "https://woodwork.alteryx.com/en/stable/guides/statistical_insights.html",
                    "https://twitter.com/AlteryxOSS",
                    "https://www.twitter.com/AlteryxOSS",
                    "https://www.evalml.alteryx.com/en/stable/demos/text_input.html",
                    "https://github.com/alteryx/evalml",
                    "https://github.com/alteryx/featuretools",
                ]
                * 2,
                "ip": [
                    "0.0.0.0",
                    "1.1.1.101",
                    "1.1.101.1",
                    "1.101.1.1",
                    "101.1.1.1",
                    "192.168.1.1",
                    "255.255.255.255",
                ]
                * 2,
            }
        )
        y = pd.Series([0, 0, 1, 0, 0, 1, 1] * 2)
        if problem_type == ProblemTypes.MULTICLASS:
            y = pd.Series([0, 2, 1, 2, 0, 2, 1] * 2)
        elif is_regression(problem_type):
            y = pd.Series([1, 2, 3, 3, 3, 4, 5] * 2)
        X = X_all[column_names]

        if input_type == "ww":
            logical_types = {}
            if "text" in column_names:
                logical_types.update({"text": "NaturalLanguage"})
            if "categorical" in column_names:
                logical_types.update({"categorical": "Categorical"})
            if "url" in column_names:
                logical_types.update({"url": "URL"})
            if "email" in column_names:
                logical_types.update({"email": "EmailAddress"})

            X.ww.init(logical_types=logical_types)

            y = ww.init_series(y)

        return X, y

    return _get_test_data_from_configuration


@pytest.mark.parametrize("input_type", ["pd", "ww"])
@pytest.mark.parametrize("problem_type", ProblemTypes.all_problem_types)
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
    ],
)
def test_make_pipeline(
    problem_type,
    input_type,
    test_description,
    column_names,
    get_test_data_from_configuration,
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
                        "date_index": "date",
                        "gap": 1,
                        "max_delay": 1,
                        "forecast_horizon": 3,
                    },
                }

            pipeline = make_pipeline(X, y, estimator_class, problem_type, parameters)
            assert isinstance(pipeline, pipeline_class)
            label_encoder = [LabelEncoder] if is_classification(problem_type) else []
            delayed_features = (
                [TimeSeriesFeaturizer]
                if is_time_series(problem_type)
                and estimator_class.model_family != ModelFamily.ARIMA
                else []
            )
            ohe = (
                [OneHotEncoder]
                if estimator_class.model_family != ModelFamily.CATBOOST
                and (
                    any(
                        ltype in column_names
                        for ltype in ["url", "email", "categorical"]
                    )
                )
                else []
            )
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
            drop_null = [DropNullColumns] if "all_null" in column_names else []
            natural_language_featurizer = (
                [NaturalLanguageFeaturizer]
                if "text" in column_names and input_type == "ww"
                else []
            )
            email_featurizer = [EmailFeaturizer] if "email" in column_names else []
            url_featurizer = [URLFeaturizer] if "url" in column_names else []
            imputer = (
                []
                if ((column_names in [["ip"]]) and input_type == "ww")
                or ((column_names in [["ip"], ["text"]]) and input_type == "pd")
                else [Imputer]
            )
            drop_col = (
                [DropColumns]
                if any(ltype in column_names for ltype in ["text"])
                and input_type == "pd"
                else []
            )
            if is_time_series(problem_type):
                expected_components = (
                    label_encoder
                    + email_featurizer
                    + url_featurizer
                    + drop_null
                    + drop_col
                    + natural_language_featurizer
                    + imputer
                    + delayed_features
                    + datetime
                    + ohe
                    + standard_scaler
                    + [estimator_class]
                )
            else:
                expected_components = (
                    label_encoder
                    + email_featurizer
                    + url_featurizer
                    + drop_null
                    + drop_col
                    + delayed_features
                    + natural_language_featurizer
                    + datetime
                    + imputer
                    + ohe
                    + standard_scaler
                    + [estimator_class]
                )
            assert pipeline.component_graph.compute_order == [
                component.name for component in expected_components
            ], test_description


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
            pd.DataFrame(), pd.Series(), LinearRegressor, ProblemTypes.MULTICLASS
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
        component_graph={}
    )

    actions = [
        DataCheckAction(DataCheckActionCode.DROP_COL, None, {"columns": ["some col"]})
    ]
    assert make_pipeline_from_actions(problem_type, actions) == pipeline_class(
        component_graph={"Drop Columns Transformer": [DropColumns, "X", "y"]},
        parameters={"Drop Columns Transformer": {"columns": ["some col"]}},
        random_seed=0,
    )

    actions = [
        DataCheckAction(
            DataCheckActionCode.DROP_COL, None, metadata={"columns": ["some col"]}
        ),
        DataCheckAction(
            DataCheckActionCode.IMPUTE_COL,
            None,
            metadata={
                "columns": None,
                "is_target": True,
                "impute_strategy": "most_frequent",
            },
        ),
        DataCheckAction(DataCheckActionCode.DROP_ROWS, None, metadata={"rows": [1, 2]}),
    ]

    assert make_pipeline_from_actions(problem_type, actions) == pipeline_class(
        component_graph={
            "Target Imputer": [TargetImputer, "X", "y"],
            "Drop Columns Transformer": [DropColumns, "X", "Target Imputer.y"],
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
        },
        random_seed=0,
    )


@pytest.mark.parametrize("problem_type", ["binary", "multiclass", "regression"])
@pytest.mark.parametrize("different_names", [True, False])
def test_make_pipeline_from_actions_with_duplicate_actions(
    problem_type, different_names
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
            "Drop Columns Transformer": {"columns": ["some col", "some other col"]}
        },
        random_seed=0,
    )
    actions = [
        DataCheckAction(
            DataCheckActionCode.DROP_ROWS, None, metadata={"rows": [0, 1, 3]}
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
    "samplers",
    [
        None,
        "Undersampler",
        "Oversampler",
    ],
)
@pytest.mark.parametrize("problem_type", ["binary", "multiclass", "regression"])
def test_make_pipeline_samplers(
    problem_type,
    samplers,
    X_y_binary,
    X_y_multi,
    X_y_regression,
    has_minimal_dependencies,
):
    if problem_type == "binary":
        X, y = X_y_binary
    elif problem_type == "multiclass":
        X, y = X_y_multi
    else:
        X, y = X_y_regression
    estimators = get_estimators(problem_type=problem_type)

    for estimator in estimators:
        if problem_type == "regression" and samplers is not None:
            with pytest.raises(ValueError, match="Sampling is unsupported for"):
                make_pipeline(X, y, estimator, problem_type, sampler_name=samplers)
        else:
            pipeline = make_pipeline(
                X, y, estimator, problem_type, sampler_name=samplers
            )
            if has_minimal_dependencies and samplers is not None:
                samplers = "Undersampler"
            # check that we do add the sampler properly
            if samplers is not None and problem_type != "regression":
                assert any("sampler" in comp.name for comp in pipeline.component_graph)
            else:
                assert not any(
                    "sampler" in comp.name for comp in pipeline.component_graph
                )


def test_get_estimators(has_minimal_dependencies):
    if has_minimal_dependencies:
        assert len(get_estimators(problem_type=ProblemTypes.BINARY)) == 5
        assert (
            len(
                get_estimators(
                    problem_type=ProblemTypes.BINARY,
                    model_families=[ModelFamily.LINEAR_MODEL],
                )
            )
            == 2
        )
        assert len(get_estimators(problem_type=ProblemTypes.MULTICLASS)) == 5
        assert len(get_estimators(problem_type=ProblemTypes.REGRESSION)) == 4
    else:
        assert len(get_estimators(problem_type=ProblemTypes.BINARY)) == 8
        assert (
            len(
                get_estimators(
                    problem_type=ProblemTypes.BINARY,
                    model_families=[ModelFamily.LINEAR_MODEL],
                )
            )
            == 2
        )
        assert len(get_estimators(problem_type=ProblemTypes.MULTICLASS)) == 8
        assert len(get_estimators(problem_type=ProblemTypes.REGRESSION)) == 7

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
            problem_type=ProblemTypes.REGRESSION, model_families="random_forest"
        )
    with pytest.raises(KeyError):
        get_estimators(problem_type="Not A Valid Problem Type")


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


def test_generate_code_pipeline_json_with_objects():
    class CustomEstimator(Estimator):
        name = "My Custom Estimator"
        hyperparameter_ranges = {}
        supported_problem_types = [ProblemTypes.BINARY, ProblemTypes.MULTICLASS]
        model_family = ModelFamily.NONE

        def __init__(self, random_arg=False, numpy_arg=[], random_seed=0):
            parameters = {"random_arg": random_arg, "numpy_arg": numpy_arg}

            super().__init__(
                parameters=parameters, component_obj=None, random_seed=random_seed
            )

    component_graph = ["Imputer", CustomEstimator]
    pipeline = BinaryClassificationPipeline(
        component_graph,
        custom_name="Mock Binary Pipeline with Transformer",
        parameters={"My Custom Estimator": {"numpy_arg": np.array([0])}},
    )
    generated_pipeline_code = generate_pipeline_code(pipeline)
    assert (
        generated_pipeline_code
        == "from evalml.pipelines.binary_classification_pipeline import BinaryClassificationPipeline\n"
        "pipeline = BinaryClassificationPipeline(component_graph={'Imputer': ['Imputer', 'X', 'y'], 'My Custom Estimator': [CustomEstimator, 'Imputer.x', 'y']}, "
        "parameters={'Imputer':{'categorical_impute_strategy': 'most_frequent', 'numeric_impute_strategy': 'mean', 'categorical_fill_value': None, 'numeric_fill_value': None}, "
        "'My Custom Estimator':{'random_arg': False, 'numpy_arg': array([0])}}, custom_name='Mock Binary Pipeline with Transformer', random_seed=0)"
    )

    pipeline = BinaryClassificationPipeline(
        component_graph,
        custom_name="Mock Binary Pipeline with Transformer",
        parameters={"My Custom Estimator": {"random_arg": Imputer()}},
    )
    generated_pipeline_code = generate_pipeline_code(pipeline)
    assert (
        generated_pipeline_code
        == "from evalml.pipelines.binary_classification_pipeline import BinaryClassificationPipeline\n"
        "pipeline = BinaryClassificationPipeline(component_graph={'Imputer': ['Imputer', 'X', 'y'], 'My Custom Estimator': [CustomEstimator, 'Imputer.x', 'y']}, "
        "parameters={'Imputer':{'categorical_impute_strategy': 'most_frequent', 'numeric_impute_strategy': 'mean', 'categorical_fill_value': None, 'numeric_fill_value': None}, "
        "'My Custom Estimator':{'random_arg': Imputer(categorical_impute_strategy='most_frequent', numeric_impute_strategy='mean', categorical_fill_value=None, numeric_fill_value=None), 'numpy_arg': []}}, "
        "custom_name='Mock Binary Pipeline with Transformer', random_seed=0)"
    )


def test_generate_code_pipeline():

    binary_pipeline = BinaryClassificationPipeline(
        ["Imputer", "Random Forest Classifier"]
    )
    expected_code = (
        "from evalml.pipelines.binary_classification_pipeline import BinaryClassificationPipeline\n"
        "pipeline = BinaryClassificationPipeline(component_graph={'Imputer': ['Imputer', 'X', 'y'], 'Random Forest Classifier': ['Random Forest Classifier', 'Imputer.x', 'y']}, "
        "parameters={'Imputer':{'categorical_impute_strategy': 'most_frequent', 'numeric_impute_strategy': 'mean', 'categorical_fill_value': None, 'numeric_fill_value': None}, "
        "'Random Forest Classifier':{'n_estimators': 100, 'max_depth': 6, 'n_jobs': -1}}, random_seed=0)"
    )
    pipeline = generate_pipeline_code(binary_pipeline)
    assert expected_code == pipeline

    regression_pipeline = RegressionPipeline(
        ["Imputer", "Random Forest Regressor"], custom_name="Mock Regression Pipeline"
    )
    expected_code = (
        "from evalml.pipelines.regression_pipeline import RegressionPipeline\n"
        "pipeline = RegressionPipeline(component_graph={'Imputer': ['Imputer', 'X', 'y'], 'Random Forest Regressor': ['Random Forest Regressor', 'Imputer.x', 'y']}, parameters={'Imputer':{'categorical_impute_strategy': 'most_frequent', 'numeric_impute_strategy': 'mean', 'categorical_fill_value': None, 'numeric_fill_value': None}, "
        "'Random Forest Regressor':{'n_estimators': 100, 'max_depth': 6, 'n_jobs': -1}}, custom_name='Mock Regression Pipeline', random_seed=0)"
    )
    pipeline = generate_pipeline_code(regression_pipeline)
    assert pipeline == expected_code

    regression_pipeline_with_params = RegressionPipeline(
        ["Imputer", "Random Forest Regressor"],
        custom_name="Mock Regression Pipeline",
        parameters={
            "Imputer": {"numeric_impute_strategy": "most_frequent"},
            "Random Forest Regressor": {"n_estimators": 50},
        },
    )
    expected_code_params = (
        "from evalml.pipelines.regression_pipeline import RegressionPipeline\n"
        "pipeline = RegressionPipeline(component_graph={'Imputer': ['Imputer', 'X', 'y'], 'Random Forest Regressor': ['Random Forest Regressor', 'Imputer.x', 'y']}, "
        "parameters={'Imputer':{'categorical_impute_strategy': 'most_frequent', 'numeric_impute_strategy': 'most_frequent', 'categorical_fill_value': None, 'numeric_fill_value': None}, "
        "'Random Forest Regressor':{'n_estimators': 50, 'max_depth': 6, 'n_jobs': -1}}, custom_name='Mock Regression Pipeline', random_seed=0)"
    )
    pipeline = generate_pipeline_code(regression_pipeline_with_params)
    assert pipeline == expected_code_params


def test_generate_code_nonlinear_pipeline():
    custom_name = "Non Linear Binary Pipeline"
    component_graph = {
        "Imputer": ["Imputer", "X", "y"],
        "OneHot_RandomForest": ["One Hot Encoder", "Imputer.x", "y"],
        "OneHot_ElasticNet": ["One Hot Encoder", "Imputer.x", "y"],
        "Random Forest": ["Random Forest Classifier", "OneHot_RandomForest.x", "y"],
        "Elastic Net": ["Elastic Net Classifier", "OneHot_ElasticNet.x", "y"],
        "Logistic Regression": [
            "Logistic Regression Classifier",
            "Random Forest.x",
            "Elastic Net.x",
            "y",
        ],
    }
    pipeline = BinaryClassificationPipeline(
        component_graph=component_graph, custom_name=custom_name
    )
    expected = (
        "from evalml.pipelines.binary_classification_pipeline import BinaryClassificationPipeline\n"
        "pipeline = BinaryClassificationPipeline("
        "component_graph={'Imputer': ['Imputer', 'X', 'y'], "
        "'OneHot_RandomForest': ['One Hot Encoder', 'Imputer.x', 'y'], "
        "'OneHot_ElasticNet': ['One Hot Encoder', 'Imputer.x', 'y'], "
        "'Random Forest': ['Random Forest Classifier', 'OneHot_RandomForest.x', 'y'], "
        "'Elastic Net': ['Elastic Net Classifier', 'OneHot_ElasticNet.x', 'y'], "
        "'Logistic Regression': ['Logistic Regression Classifier', 'Random Forest.x', 'Elastic Net.x', 'y']}, "
        "parameters={'Imputer':{'categorical_impute_strategy': 'most_frequent', 'numeric_impute_strategy': 'mean', 'categorical_fill_value': None, 'numeric_fill_value': None}, "
        "'OneHot_RandomForest':{'top_n': 10, 'features_to_encode': None, 'categories': None, 'drop': 'if_binary', 'handle_unknown': 'ignore', 'handle_missing': 'error'}, "
        "'OneHot_ElasticNet':{'top_n': 10, 'features_to_encode': None, 'categories': None, 'drop': 'if_binary', 'handle_unknown': 'ignore', 'handle_missing': 'error'}, "
        "'Random Forest':{'n_estimators': 100, 'max_depth': 6, 'n_jobs': -1}, "
        "'Elastic Net':{'penalty': 'elasticnet', 'C': 1.0, 'l1_ratio': 0.15, 'n_jobs': -1, 'multi_class': 'auto', 'solver': 'saga'}, "
        "'Logistic Regression':{'penalty': 'l2', 'C': 1.0, 'n_jobs': -1, 'multi_class': 'auto', 'solver': 'lbfgs'}}, "
        "custom_name='Non Linear Binary Pipeline', random_seed=0)"
    )
    pipeline_code = generate_pipeline_code(pipeline)
    assert pipeline_code == expected


def test_generate_code_pipeline_with_custom_components():
    class CustomTransformer(Transformer):
        name = "My Custom Transformer"
        hyperparameter_ranges = {}

        def __init__(self, random_seed=0):
            parameters = {}

            super().__init__(
                parameters=parameters, component_obj=None, random_seed=random_seed
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
                parameters=parameters, component_obj=None, random_seed=random_seed
            )

    mock_pipeline_with_custom_components = BinaryClassificationPipeline(
        [CustomTransformer, CustomEstimator]
    )
    expected_code = (
        "from evalml.pipelines.binary_classification_pipeline import BinaryClassificationPipeline\n"
        "pipeline = BinaryClassificationPipeline(component_graph={'My Custom Transformer': [CustomTransformer, 'X', 'y'], 'My Custom Estimator': [CustomEstimator, 'My Custom Transformer.x', 'y']}, "
        "parameters={'My Custom Estimator':{'random_arg': False}}, random_seed=0)"
    )
    pipeline = generate_pipeline_code(mock_pipeline_with_custom_components)
    assert pipeline == expected_code


def test_rows_of_interest_errors(X_y_binary):
    pipeline = BinaryClassificationPipeline(
        component_graph=["Logistic Regression Classifier"]
    )
    pipeline_mc = MulticlassClassificationPipeline(
        component_graph=["Logistic Regression Classifier"]
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
        component_graph=["Imputer", "Standard Scaler", "Logistic Regression Classifier"]
    )
    X = pd.DataFrame([i for i in range(100)])
    y = y
    pipeline._is_fitted = True

    vals = [0.2] * 25 + [0.5] * 50 + [0.8] * 25
    predicted_proba_values = pd.DataFrame({0: [1 - v for v in vals], 1: vals})
    mock_pred_proba.return_value = predicted_proba_values
    vals = rows_of_interest(
        pipeline, X, y, threshold=threshold, epsilon=0.5, sort_values=True
    )
    if threshold == 0.3:
        assert vals == list(range(100))
    elif threshold == 0.7:
        assert vals == list(range(75, 100)) + list(range(25, 75)) + list(range(25))
    else:
        assert vals == list(range(25, 75)) + list(range(25)) + list(range(75, 100))

    pipeline._threshold = 0.9
    vals = rows_of_interest(
        pipeline, X, y, threshold=None, epsilon=0.5, sort_values=True
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
        component_graph=["Logistic Regression Classifier"]
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
        component_graph=["Logistic Regression Classifier"]
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
        component_graph=["Logistic Regression Classifier"]
    )
    X = pd.DataFrame([i for i in range(100)])
    y = pd.Series([0] * 25 + [1] * 50 + [0] * 25)
    pipeline._is_fitted = True

    vals = [0.2] * 25 + [0.5] * 50 + [0.8] * 25
    predicted_proba_values = pd.DataFrame({0: [1 - v for v in vals], 1: vals})
    mock_pred_proba.return_value = predicted_proba_values
    vals = rows_of_interest(
        pipeline, X, y, threshold=0.9, epsilon=0.9, sort_values=sorts
    )
    assert vals == expected_val


@patch("evalml.pipelines.BinaryClassificationPipeline.predict_proba")
@patch("evalml.pipelines.BinaryClassificationPipeline.fit")
def test_rows_of_interest_index(mock_fit, mock_pred_proba):
    pipeline = BinaryClassificationPipeline(
        component_graph=["Logistic Regression Classifier"]
    )
    X = pd.DataFrame(
        [i for i in range(100)], index=["index_{}".format(i) for i in range(100)]
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
    mock_fit, mock_pred_proba, types, sorts, epsilon, expected_vals
):
    pipeline = BinaryClassificationPipeline(
        component_graph=["Logistic Regression Classifier"]
    )
    X = pd.DataFrame([i for i in range(100)])
    y = pd.Series([0] * 25 + [1] * 50 + [0] * 25)
    pipeline._is_fitted = True

    vals = [0.2] * 25 + [0.5] * 50 + [0.85] * 25
    predicted_proba_values = pd.DataFrame({0: [1 - v for v in vals], 1: vals})
    mock_pred_proba.return_value = predicted_proba_values
    vals = rows_of_interest(
        pipeline, X, y, types=types, sort_values=sorts, epsilon=epsilon
    )
    assert vals == expected_vals

    if types == "all":
        vals = rows_of_interest(
            pipeline, X, types=types, sort_values=sorts, epsilon=epsilon
        )
        assert vals == expected_vals


@patch("evalml.pipelines.BinaryClassificationPipeline.predict_proba")
@patch("evalml.pipelines.BinaryClassificationPipeline.fit")
def test_rows_of_interest_empty(mock_fit, mock_pred_proba):
    pipeline = BinaryClassificationPipeline(
        component_graph=["Logistic Regression Classifier"]
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
