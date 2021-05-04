
import numpy as np
import pandas as pd
import pytest
import woodwork as ww

from evalml.data_checks import DataCheckAction, DataCheckActionCode
from evalml.model_family import ModelFamily
from evalml.pipelines import (
    BinaryClassificationPipeline,
    MulticlassClassificationPipeline,
    RegressionPipeline
)
from evalml.pipelines.components import (
    DateTimeFeaturizer,
    DelayedFeatureTransformer,
    DropColumns,
    DropNullColumns,
    Estimator,
    Imputer,
    LinearRegressor,
    LogisticRegressionClassifier,
    OneHotEncoder,
    StackedEnsembleClassifier,
    StackedEnsembleRegressor,
    StandardScaler,
    TargetImputer,
    TextFeaturizer,
    Transformer
)
from evalml.pipelines.utils import (
    _get_pipeline_base_class,
    _make_component_list_from_actions,
    generate_pipeline_code,
    get_estimators,
    make_pipeline
)
from evalml.problem_types import ProblemTypes, is_time_series


def test_make_pipeline_error():
    X = pd.DataFrame([[0, 1], [1, 0]])
    y = pd.Series([1, 0])
    estimators = get_estimators(problem_type="binary")
    custom_hyperparameters = [{"Imputer": {"numeric_imput_strategy": ["median"]}}, {"One Hot Encoder": {"value1": ["value2"]}}]

    for estimator in estimators:
        with pytest.raises(ValueError, match="if custom_hyperparameters provided, must be dictionary"):
            make_pipeline(X, y, estimator, "binary", {}, custom_hyperparameters)


@pytest.mark.parametrize("problem_type", [ProblemTypes.BINARY, ProblemTypes.MULTICLASS, ProblemTypes.REGRESSION,
                                          ProblemTypes.TIME_SERIES_REGRESSION])
def test_make_pipeline_custom_hyperparameters(problem_type):
    X = pd.DataFrame({"all_null": [np.nan, np.nan, np.nan, np.nan, np.nan],
                      "categorical": ["a", "b", "a", "c", "c"],
                      "some dates": pd.date_range('2000-02-03', periods=5, freq='W')})
    custom_hyperparameters = {'Imputer': {
        'numeric_impute_strategy': ['median']
    }}

    y = pd.Series([0, 0, 1, 0, 0])
    estimators = get_estimators(problem_type=problem_type)

    for estimator_class in estimators:
        for problem_type in estimator_class.supported_problem_types:
            parameters = {}
            if is_time_series(problem_type):
                parameters = {"pipeline": {"date_index": "some dates", "gap": 1, "max_delay": 1},
                              "Time Series Baseline Estimator": {"date_index": "some dates", "gap": 1, "max_delay": 1}}

            pipeline = make_pipeline(X, y, estimator_class, problem_type, parameters, custom_hyperparameters)
            assert pipeline.custom_hyperparameters == custom_hyperparameters

            pipeline2 = make_pipeline(X, y, estimator_class, problem_type, parameters)
            assert not pipeline2.custom_hyperparameters


@pytest.mark.parametrize("input_type", ["pd", "ww"])
@pytest.mark.parametrize("problem_type", ProblemTypes.all_problem_types)
def test_make_pipeline_all_nan_no_categoricals(input_type, problem_type):
    # testing that all_null column is not considered categorical
    X = pd.DataFrame({"all_null": [np.nan, np.nan, np.nan, np.nan, np.nan],
                      "num": [1, 2, 3, 4, 5]})
    y = pd.Series([0, 0, 1, 1, 0])
    if input_type == 'ww':
        X = ww.DataTable(X)
        y = ww.DataColumn(y)

    estimators = get_estimators(problem_type=problem_type)
    pipeline_class = _get_pipeline_base_class(problem_type)
    if problem_type == ProblemTypes.MULTICLASS:
        y = pd.Series([0, 2, 1, 2])

    for estimator_class in estimators:
        if problem_type in estimator_class.supported_problem_types:
            parameters = {}
            if is_time_series(problem_type):
                parameters = {"pipeline": {"date_index": None, "gap": 1, "max_delay": 1},
                              "Time Series Baseline Estimator": {"date_index": None, "gap": 1, "max_delay": 1}}

            pipeline = make_pipeline(X, y, estimator_class, problem_type, parameters)
            assert isinstance(pipeline, pipeline_class)
            assert pipeline.custom_hyperparameters is None
            delayed_features = []
            if is_time_series(problem_type):
                delayed_features = [DelayedFeatureTransformer]
            if estimator_class.model_family == ModelFamily.LINEAR_MODEL:
                estimator_components = [StandardScaler, estimator_class]
            else:
                estimator_components = [estimator_class]
            assert pipeline.component_graph == [DropNullColumns, Imputer] + delayed_features + estimator_components


@pytest.mark.parametrize("input_type", ["pd", "ww"])
@pytest.mark.parametrize("problem_type", ProblemTypes.all_problem_types)
def test_make_pipeline(input_type, problem_type):
    X = pd.DataFrame({"all_null": [np.nan, np.nan, np.nan, np.nan, np.nan],
                      "categorical": ["a", "b", "a", "c", "c"],
                      "some dates": pd.date_range('2000-02-03', periods=5, freq='W')})
    y = pd.Series([0, 0, 1, 0, 0])
    if input_type == 'ww':
        X = ww.DataTable(X)
        y = ww.DataColumn(y)

    estimators = get_estimators(problem_type=problem_type)
    pipeline_class = _get_pipeline_base_class(problem_type)
    if problem_type == ProblemTypes.MULTICLASS:
        y = pd.Series([0, 2, 1, 2])

    for estimator_class in estimators:
        if problem_type in estimator_class.supported_problem_types:
            parameters = {}
            if is_time_series(problem_type):
                parameters = {"pipeline": {"date_index": "some dates", "gap": 1, "max_delay": 1},
                              "Time Series Baseline Estimator": {"date_index": "some dates", "gap": 1, "max_delay": 1}}

            pipeline = make_pipeline(X, y, estimator_class, problem_type, parameters)
            assert isinstance(pipeline, pipeline_class)
            assert pipeline.custom_hyperparameters is None
            delayed_features = []
            if is_time_series(problem_type):
                delayed_features = [DelayedFeatureTransformer]
            if estimator_class.model_family == ModelFamily.LINEAR_MODEL:
                estimator_components = [OneHotEncoder, StandardScaler, estimator_class]
            elif estimator_class.model_family == ModelFamily.CATBOOST:
                estimator_components = [estimator_class]
            else:
                estimator_components = [OneHotEncoder, estimator_class]
            assert pipeline.component_graph == [DropNullColumns, Imputer, DateTimeFeaturizer] + delayed_features + estimator_components


@pytest.mark.parametrize("input_type", ["pd", "ww"])
@pytest.mark.parametrize("problem_type", ProblemTypes.all_problem_types)
def test_make_pipeline_no_nulls(input_type, problem_type):
    X = pd.DataFrame({"numerical": [1, 2, 3, 1, 2],
                      "categorical": ["a", "b", "a", "c", "c"],
                      "some dates": pd.date_range('2000-02-03', periods=5, freq='W')})
    y = pd.Series([0, 1, 1, 0, 0])
    if input_type == 'ww':
        X = ww.DataTable(X)
        y = ww.DataColumn(y)

    estimators = get_estimators(problem_type=problem_type)
    pipeline_class = _get_pipeline_base_class(problem_type)
    if problem_type == ProblemTypes.MULTICLASS:
        y = pd.Series([0, 2, 1, 2])

    for estimator_class in estimators:
        if problem_type in estimator_class.supported_problem_types:
            parameters = {}
            if is_time_series(problem_type):
                parameters = {"pipeline": {"date_index": "some dates", "gap": 1, "max_delay": 1},
                              "Time Series Baseline Estimator": {"date_index": "some dates", "gap": 1, "max_delay": 1}}

            pipeline = make_pipeline(X, y, estimator_class, problem_type, parameters)
            assert isinstance(pipeline, pipeline_class)
            assert pipeline.custom_hyperparameters is None
            delayed_features = []
            if is_time_series(problem_type):
                delayed_features = [DelayedFeatureTransformer]
            if estimator_class.model_family == ModelFamily.LINEAR_MODEL:
                estimator_components = [OneHotEncoder, StandardScaler, estimator_class]
            elif estimator_class.model_family == ModelFamily.CATBOOST:
                estimator_components = [estimator_class]
            else:
                estimator_components = [OneHotEncoder, estimator_class]
            assert pipeline.component_graph == [Imputer, DateTimeFeaturizer] + delayed_features + estimator_components


@pytest.mark.parametrize("input_type", ["pd", "ww"])
@pytest.mark.parametrize("problem_type", ProblemTypes.all_problem_types)
def test_make_pipeline_no_datetimes(input_type, problem_type):
    X = pd.DataFrame({"numerical": [1, 2, 3, 1, 2],
                      "categorical": ["a", "b", "a", "c", "c"],
                      "all_null": [np.nan, np.nan, np.nan, np.nan, np.nan]})
    y = pd.Series([0, 1, 1, 0, 0])
    if input_type == 'ww':
        X = ww.DataTable(X)
        y = ww.DataColumn(y)

    estimators = get_estimators(problem_type=problem_type)
    pipeline_class = _get_pipeline_base_class(problem_type)
    if problem_type == ProblemTypes.MULTICLASS:
        y = pd.Series([0, 2, 1, 2])

    for estimator_class in estimators:
        if problem_type in estimator_class.supported_problem_types:
            parameters = {}
            if is_time_series(problem_type):
                parameters = {"pipeline": {"date_index": None, "gap": 1, "max_delay": 1},
                              "Time Series Baseline Estimator": {"date_index": None, "gap": 1, "max_delay": 1}}

            pipeline = make_pipeline(X, y, estimator_class, problem_type, parameters)
            assert isinstance(pipeline, pipeline_class)
            assert pipeline.custom_hyperparameters is None
            delayed_features = []
            if is_time_series(problem_type):
                delayed_features = [DelayedFeatureTransformer]
            if estimator_class.model_family == ModelFamily.LINEAR_MODEL:
                estimator_components = [OneHotEncoder, StandardScaler, estimator_class]
            elif estimator_class.model_family == ModelFamily.CATBOOST:
                estimator_components = [estimator_class]
            else:
                estimator_components = [OneHotEncoder, estimator_class]
            assert pipeline.component_graph == [DropNullColumns, Imputer] + delayed_features + estimator_components


@pytest.mark.parametrize("input_type", ["pd", "ww"])
@pytest.mark.parametrize("problem_type", ProblemTypes.all_problem_types)
def test_make_pipeline_no_column_names(input_type, problem_type):
    X = pd.DataFrame([[1, "a", np.nan], [2, "b", np.nan], [5, "b", np.nan]])
    y = pd.Series([0, 0, 1])
    if input_type == 'ww':
        X = ww.DataTable(X)
        y = ww.DataColumn(y)
    estimators = get_estimators(problem_type=problem_type)
    pipeline_class = _get_pipeline_base_class(problem_type)
    if problem_type == ProblemTypes.MULTICLASS:
        y = pd.Series([0, 2, 1, 2])

    for estimator_class in estimators:
        if problem_type in estimator_class.supported_problem_types:
            parameters = {}
            if is_time_series(problem_type):
                parameters = {"pipeline": {"date_index": None, "gap": 1, "max_delay": 1},
                              "Time Series Baseline Estimator": {"date_index": None, "gap": 1, "max_delay": 1}}

            pipeline = make_pipeline(X, y, estimator_class, problem_type, parameters)
            assert isinstance(pipeline, pipeline_class)
            assert pipeline.custom_hyperparameters is None
            delayed_features = []
            if is_time_series(problem_type):
                delayed_features = [DelayedFeatureTransformer]
            if estimator_class.model_family == ModelFamily.LINEAR_MODEL:
                estimator_components = [OneHotEncoder, StandardScaler, estimator_class]
            elif estimator_class.model_family == ModelFamily.CATBOOST:
                estimator_components = [estimator_class]
            else:
                estimator_components = [OneHotEncoder, estimator_class]
            assert pipeline.component_graph == [DropNullColumns, Imputer] + delayed_features + estimator_components


@pytest.mark.parametrize("input_type", ["pd", "ww"])
@pytest.mark.parametrize("problem_type", ProblemTypes.all_problem_types)
def test_make_pipeline_text_columns(input_type, problem_type):
    X = pd.DataFrame({"numerical": [1, 2, 3, 1, 2],
                      "categorical": ["a", "b", "a", "c", "c"],
                      "text": ["string one", "another", "text for a column, this should be a text column!!", "text string", "hello world"]})
    y = pd.Series([0, 0, 1, 1, 0])
    if input_type == 'ww':
        X = ww.DataTable(X)
        y = ww.DataColumn(y)
    estimators = get_estimators(problem_type=problem_type)

    pipeline_class = _get_pipeline_base_class(problem_type)
    if problem_type == ProblemTypes.MULTICLASS:
        y = pd.Series([0, 2, 1, 2])

    for estimator_class in estimators:
        if problem_type in estimator_class.supported_problem_types:
            parameters = {}
            if is_time_series(problem_type):
                parameters = {"pipeline": {"date_index": None, "gap": 1, "max_delay": 1},
                              "Time Series Baseline Estimator": {"date_index": None, "gap": 1, "max_delay": 1}}

            pipeline = make_pipeline(X, y, estimator_class, problem_type, parameters)
            assert isinstance(pipeline, pipeline_class)
            assert pipeline.custom_hyperparameters is None
            delayed_features = []
            if is_time_series(problem_type):
                delayed_features = [DelayedFeatureTransformer]
            if estimator_class.model_family == ModelFamily.LINEAR_MODEL:
                estimator_components = [OneHotEncoder, StandardScaler, estimator_class]
            elif estimator_class.model_family == ModelFamily.CATBOOST:
                estimator_components = [estimator_class]
            else:
                estimator_components = [OneHotEncoder, estimator_class]
            assert pipeline.component_graph == [Imputer, TextFeaturizer] + delayed_features + estimator_components


@pytest.mark.parametrize("input_type", ["pd", "ww"])
@pytest.mark.parametrize("problem_type", ProblemTypes.all_problem_types)
def test_make_pipeline_only_text_columns(input_type, problem_type):
    X = pd.DataFrame({"text": ["string one", "the evalml team is full of wonderful people", "text for a column, this should be a text column!!", "text string", "hello world"],
                      "another text": ["ladidididididida", "cats are great", "text for a column, this should be a text column!!", "text string", "goodbye world"]})
    y = pd.Series([0, 0, 1, 1, 0])
    if input_type == 'ww':
        X = ww.DataTable(X)
        y = ww.DataColumn(y)
    estimators = get_estimators(problem_type=problem_type)

    pipeline_class = _get_pipeline_base_class(problem_type)
    if problem_type == ProblemTypes.MULTICLASS:
        y = pd.Series([0, 2, 1, 2])

    for estimator_class in estimators:
        if problem_type in estimator_class.supported_problem_types:
            parameters = {}
            if is_time_series(problem_type):
                parameters = {"pipeline": {"date_index": None, "gap": 1, "max_delay": 1},
                              "Time Series Baseline Estimator": {"date_index": None, "gap": 1, "max_delay": 1}}

            pipeline = make_pipeline(X, y, estimator_class, problem_type, parameters)
            assert isinstance(pipeline, pipeline_class)
            assert pipeline.custom_hyperparameters is None
            delayed_features = []
            if is_time_series(problem_type):
                delayed_features = [DelayedFeatureTransformer]
            standard_scaler = []
            if estimator_class.model_family == ModelFamily.LINEAR_MODEL:
                standard_scaler = [StandardScaler]
            assert pipeline.component_graph == [TextFeaturizer] + delayed_features + standard_scaler + [estimator_class]


@pytest.mark.parametrize("input_type", ["pd", "ww"])
@pytest.mark.parametrize("problem_type", ProblemTypes.all_problem_types)
def test_make_pipeline_only_datetime_columns(input_type, problem_type):
    X = pd.DataFrame({"some dates": pd.date_range('2000-02-03', periods=5, freq='W'),
                      "some other dates": pd.date_range('2000-05-19', periods=5, freq='W')})
    y = pd.Series([0, 0, 1, 1, 0])
    if input_type == 'ww':
        X = ww.DataTable(X)
        y = ww.DataColumn(y)
    estimators = get_estimators(problem_type=problem_type)

    pipeline_class = _get_pipeline_base_class(problem_type)
    if problem_type == ProblemTypes.MULTICLASS:
        y = pd.Series([0, 2, 1, 2])

    for estimator_class in estimators:
        if problem_type in estimator_class.supported_problem_types:
            parameters = {}
            if is_time_series(problem_type):
                parameters = {"pipeline": {"date_index": "some dates", "gap": 1, "max_delay": 1},
                              "Time Series Baseline Estimator": {"date_index": "some dates", "gap": 1, "max_delay": 1}}

            pipeline = make_pipeline(X, y, estimator_class, problem_type, parameters)
            assert isinstance(pipeline, pipeline_class)
            assert pipeline.custom_hyperparameters is None
            delayed_features = []
            if is_time_series(problem_type):
                delayed_features = [DelayedFeatureTransformer]
            standard_scaler = []
            if estimator_class.model_family == ModelFamily.LINEAR_MODEL:
                standard_scaler = [StandardScaler]
            assert pipeline.component_graph == [DateTimeFeaturizer] + delayed_features + standard_scaler + [estimator_class]


@pytest.mark.parametrize("problem_type", ProblemTypes.all_problem_types)
def test_make_pipeline_numpy_input(problem_type):
    X = np.array([[1, 2, 0, np.nan], [2, 2, 1, np.nan], [5, 1, np.nan, np.nan]])
    y = np.array([0, 0, 1, 0])

    estimators = get_estimators(problem_type=problem_type)
    pipeline_class = _get_pipeline_base_class(problem_type)
    if problem_type == ProblemTypes.MULTICLASS:
        y = pd.Series([0, 2, 1, 2])

    for estimator_class in estimators:
        if problem_type in estimator_class.supported_problem_types:
            parameters = {}
            if is_time_series(problem_type):
                parameters = {"pipeline": {"date_index": None, "gap": 1, "max_delay": 1},
                              "Time Series Baseline Estimator": {"date_index": None, "gap": 1, "max_delay": 1}}

            pipeline = make_pipeline(X, y, estimator_class, problem_type, parameters)
            assert isinstance(pipeline, pipeline_class)
            delayed_features = []
            if is_time_series(problem_type):
                delayed_features = [DelayedFeatureTransformer]
            if estimator_class.model_family == ModelFamily.LINEAR_MODEL:
                estimator_components = [StandardScaler, estimator_class]
            else:
                estimator_components = [estimator_class]
            assert pipeline.component_graph == [DropNullColumns, Imputer] + delayed_features + estimator_components


@pytest.mark.parametrize("input_type", ["pd", "ww"])
@pytest.mark.parametrize("problem_type", ProblemTypes.all_problem_types)
def test_make_pipeline_datetime_no_categorical(input_type, problem_type):
    X = pd.DataFrame({"numerical": [1, 2, 3, 1, 2],
                      "some dates": pd.date_range('2000-02-03', periods=5, freq='W')})
    y = pd.Series([0, 1, 1, 0, 0])
    if input_type == 'ww':
        X = ww.DataTable(X)
        y = ww.DataColumn(y)

    estimators = get_estimators(problem_type=problem_type)
    pipeline_class = _get_pipeline_base_class(problem_type)
    if problem_type == ProblemTypes.MULTICLASS:
        y = pd.Series([0, 2, 1, 2])

    for estimator_class in estimators:
        if problem_type in estimator_class.supported_problem_types:
            parameters = {}
            if is_time_series(problem_type):
                parameters = {"pipeline": {"date_index": "soem dates", "gap": 1, "max_delay": 1},
                              "Time Series Baseline Estimator": {"date_index": "some dates", "gap": 1, "max_delay": 1}}

            pipeline = make_pipeline(X, y, estimator_class, problem_type, parameters)
            assert isinstance(pipeline, pipeline_class)
            assert pipeline.custom_hyperparameters is None
            delayed_features = []
            if is_time_series(problem_type):
                delayed_features = [DelayedFeatureTransformer]
            if estimator_class.model_family == ModelFamily.LINEAR_MODEL:
                estimator_components = [StandardScaler, estimator_class]
            elif estimator_class.model_family == ModelFamily.CATBOOST:
                estimator_components = [estimator_class]
            else:
                estimator_components = [estimator_class]
            assert pipeline.component_graph == [Imputer, DateTimeFeaturizer] + delayed_features + estimator_components


def test_make_pipeline_problem_type_mismatch():
    with pytest.raises(ValueError, match=f"{LogisticRegressionClassifier.name} is not a valid estimator for problem type"):
        make_pipeline(pd.DataFrame(), pd.Series(), LogisticRegressionClassifier, ProblemTypes.REGRESSION)
    with pytest.raises(ValueError, match=f"{LinearRegressor.name} is not a valid estimator for problem type"):
        make_pipeline(pd.DataFrame(), pd.Series(), LinearRegressor, ProblemTypes.MULTICLASS)
    with pytest.raises(ValueError, match=f"{Transformer.name} is not a valid estimator for problem type"):
        make_pipeline(pd.DataFrame(), pd.Series(), Transformer, ProblemTypes.MULTICLASS)


@pytest.mark.parametrize("problem_type", [ProblemTypes.BINARY, ProblemTypes.MULTICLASS, ProblemTypes.REGRESSION])
def test_stacked_estimator_in_pipeline(problem_type, X_y_binary, X_y_multi, X_y_regression,
                                       stackable_classifiers,
                                       stackable_regressors,
                                       logistic_regression_binary_pipeline_class,
                                       logistic_regression_multiclass_pipeline_class,
                                       linear_regression_pipeline_class):
    if problem_type == ProblemTypes.BINARY:
        X, y = X_y_binary
        base_pipeline_class = BinaryClassificationPipeline
        stacking_component_name = StackedEnsembleClassifier.name
        input_pipelines = [BinaryClassificationPipeline([classifier]) for classifier in stackable_classifiers]
        comparison_pipeline = logistic_regression_binary_pipeline_class(parameters={"Logistic Regression Classifier": {"n_jobs": 1}})
        objective = 'Log Loss Binary'
    elif problem_type == ProblemTypes.MULTICLASS:
        X, y = X_y_multi
        base_pipeline_class = MulticlassClassificationPipeline
        stacking_component_name = StackedEnsembleClassifier.name
        input_pipelines = [MulticlassClassificationPipeline([classifier]) for classifier in stackable_classifiers]
        comparison_pipeline = logistic_regression_multiclass_pipeline_class(parameters={"Logistic Regression Classifier": {"n_jobs": 1}})
        objective = 'Log Loss Multiclass'
    elif problem_type == ProblemTypes.REGRESSION:
        X, y = X_y_regression
        base_pipeline_class = RegressionPipeline
        stacking_component_name = StackedEnsembleRegressor.name
        input_pipelines = [RegressionPipeline([regressor]) for regressor in stackable_regressors]
        comparison_pipeline = linear_regression_pipeline_class(parameters={"Linear Regressor": {"n_jobs": 1}})
        objective = 'R2'
    parameters = {
        stacking_component_name: {
            "input_pipelines": input_pipelines,
            "n_jobs": 1
        }
    }
    graph = ['Simple Imputer', stacking_component_name]

    pipeline = base_pipeline_class(component_graph=graph, parameters=parameters)
    pipeline.fit(X, y)
    comparison_pipeline.fit(X, y)
    assert not np.isnan(pipeline.predict(X).to_series()).values.any()

    pipeline_score = pipeline.score(X, y, [objective])[objective]
    comparison_pipeline_score = comparison_pipeline.score(X, y, [objective])[objective]

    if problem_type == ProblemTypes.BINARY or problem_type == ProblemTypes.MULTICLASS:
        assert not np.isnan(pipeline.predict_proba(X).to_dataframe()).values.any()
        assert (pipeline_score <= comparison_pipeline_score)
    else:
        assert (pipeline_score >= comparison_pipeline_score)


def test_make_component_list_from_actions():
    assert _make_component_list_from_actions([]) == []

    actions = [DataCheckAction(DataCheckActionCode.DROP_COL, {"columns": ['some col']})]
    assert _make_component_list_from_actions(actions) == [DropColumns(columns=['some col'])]

    actions = [DataCheckAction(DataCheckActionCode.DROP_COL, metadata={"columns": ['some col']}),
               DataCheckAction(DataCheckActionCode.IMPUTE_COL, metadata={"column": None, "is_target": True, "impute_strategy": "most_frequent"})]
    assert _make_component_list_from_actions(actions) == [DropColumns(columns=['some col']),
                                                          TargetImputer(impute_strategy="most_frequent")]


@pytest.mark.parametrize("samplers", [None, "Undersampler", "SMOTE Oversampler", "SMOTENC Oversampler", "SMOTEN Oversampler"])
@pytest.mark.parametrize("problem_type", ['binary', 'multiclass', 'regression'])
def test_make_pipeline_samplers(problem_type, samplers, X_y_binary, X_y_multi, X_y_regression, has_minimal_dependencies):
    if problem_type == 'binary':
        X, y = X_y_binary
    elif problem_type == 'multiclass':
        X, y = X_y_multi
    else:
        X, y = X_y_regression
    estimators = get_estimators(problem_type=problem_type)

    for estimator in estimators:
        if problem_type == 'regression' and samplers is not None:
            with pytest.raises(ValueError, match='Sampling is unsupported for'):
                make_pipeline(X, y, estimator, problem_type, sampler_name=samplers)
        else:
            pipeline = make_pipeline(X, y, estimator, problem_type, sampler_name=samplers)
            if has_minimal_dependencies and samplers is not None:
                samplers = 'Undersampler'
            # check that we do add the sampler properly
            if samplers is not None and problem_type != 'regression':
                # we add the sampler before the scaler if it exists
                if pipeline.component_graph[-2].name == 'Standard Scaler':
                    assert pipeline.component_graph[-3].name == samplers
                else:
                    assert pipeline.component_graph[-2].name == samplers
            else:
                assert not any('sampler' in comp.name for comp in pipeline.component_graph)


def test_get_estimators(has_minimal_dependencies):
    if has_minimal_dependencies:
        assert len(get_estimators(problem_type=ProblemTypes.BINARY)) == 5
        assert len(get_estimators(problem_type=ProblemTypes.BINARY, model_families=[ModelFamily.LINEAR_MODEL])) == 2
        assert len(get_estimators(problem_type=ProblemTypes.MULTICLASS)) == 5
        assert len(get_estimators(problem_type=ProblemTypes.REGRESSION)) == 5
    else:
        assert len(get_estimators(problem_type=ProblemTypes.BINARY)) == 8
        assert len(get_estimators(problem_type=ProblemTypes.BINARY, model_families=[ModelFamily.LINEAR_MODEL])) == 2
        assert len(get_estimators(problem_type=ProblemTypes.MULTICLASS)) == 8
        assert len(get_estimators(problem_type=ProblemTypes.REGRESSION)) == 8

    assert len(get_estimators(problem_type=ProblemTypes.BINARY, model_families=[])) == 0
    assert len(get_estimators(problem_type=ProblemTypes.MULTICLASS, model_families=[])) == 0
    assert len(get_estimators(problem_type=ProblemTypes.REGRESSION, model_families=[])) == 0

    with pytest.raises(RuntimeError, match="Unrecognized model type for problem type"):
        get_estimators(problem_type=ProblemTypes.REGRESSION, model_families=["random_forest", "none"])
    with pytest.raises(TypeError, match="model_families parameter is not a list."):
        get_estimators(problem_type=ProblemTypes.REGRESSION, model_families='random_forest')
    with pytest.raises(KeyError):
        get_estimators(problem_type="Not A Valid Problem Type")


def test_generate_code_pipeline_errors():
    class MockBinaryPipeline(BinaryClassificationPipeline):
        name = "Mock Binary Pipeline"
        component_graph = ['Imputer', 'Random Forest Classifier']

    class MockMulticlassPipeline(MulticlassClassificationPipeline):
        name = "Mock Multiclass Pipeline"
        component_graph = ['Imputer', 'Random Forest Classifier']

    class MockRegressionPipeline(RegressionPipeline):
        name = "Mock Regression Pipeline"
        component_graph = ['Imputer', 'Random Forest Regressor']

    with pytest.raises(ValueError, match="Element must be a pipeline instance"):
        generate_pipeline_code(MockBinaryPipeline)

    with pytest.raises(ValueError, match="Element must be a pipeline instance"):
        generate_pipeline_code(MockMulticlassPipeline)

    with pytest.raises(ValueError, match="Element must be a pipeline instance"):
        generate_pipeline_code(MockRegressionPipeline)

    with pytest.raises(ValueError, match="Element must be a pipeline instance"):
        generate_pipeline_code([Imputer])

    with pytest.raises(ValueError, match="Element must be a pipeline instance"):
        generate_pipeline_code([Imputer, LogisticRegressionClassifier])

    with pytest.raises(ValueError, match="Element must be a pipeline instance"):
        generate_pipeline_code([Imputer(), LogisticRegressionClassifier()])


def test_generate_code_pipeline_json_errors():
    class CustomEstimator(Estimator):
        name = "My Custom Estimator"
        hyperparameter_ranges = {}
        supported_problem_types = [ProblemTypes.BINARY, ProblemTypes.MULTICLASS]
        model_family = ModelFamily.NONE

        def __init__(self, random_arg=False, numpy_arg=[], random_seed=0):
            parameters = {'random_arg': random_arg,
                          'numpy_arg': numpy_arg}

            super().__init__(parameters=parameters,
                             component_obj=None,
                             random_seed=random_seed)

    class MockBinaryPipelineTransformer(BinaryClassificationPipeline):
        custom_name = "Mock Binary Pipeline with Transformer"
        component_graph = ['Imputer', CustomEstimator]

        def __init__(self, parameters, random_seed=0):
            super().__init__(self.component_graph, parameters=parameters, custom_name=self.custom_name, custom_hyperparameters=None, random_seed=random_seed)

    pipeline = MockBinaryPipelineTransformer({})
    generate_pipeline_code(pipeline)

    pipeline = MockBinaryPipelineTransformer({'My Custom Estimator': {'numpy_arg': np.array([0])}})
    with pytest.raises(TypeError, match="cannot be JSON-serialized"):
        generate_pipeline_code(pipeline)

    pipeline = MockBinaryPipelineTransformer({'My Custom Estimator': {'random_arg': pd.DataFrame()}})
    with pytest.raises(TypeError, match="cannot be JSON-serialized"):
        generate_pipeline_code(pipeline)

    pipeline = MockBinaryPipelineTransformer({'My Custom Estimator': {'random_arg': ProblemTypes.BINARY}})
    with pytest.raises(TypeError, match="cannot be JSON-serialized"):
        generate_pipeline_code(pipeline)

    pipeline = MockBinaryPipelineTransformer({'My Custom Estimator': {'random_arg': BinaryClassificationPipeline}})
    with pytest.raises(TypeError, match="cannot be JSON-serialized"):
        generate_pipeline_code(pipeline)

    pipeline = MockBinaryPipelineTransformer({'My Custom Estimator': {'random_arg': Estimator}})
    with pytest.raises(TypeError, match="cannot be JSON-serialized"):
        generate_pipeline_code(pipeline)

    pipeline = MockBinaryPipelineTransformer({'My Custom Estimator': {'random_arg': Imputer()}})
    with pytest.raises(TypeError, match="cannot be JSON-serialized"):
        generate_pipeline_code(pipeline)


def test_generate_code_pipeline():
    class MockBinaryPipeline(BinaryClassificationPipeline):
        component_graph = ['Imputer', 'Random Forest Classifier']
        custom_hyperparameters = {
            "Imputer": {
                "numeric_impute_strategy": 'most_frequent'
            }
        }

        def __init__(self, parameters, random_seed=0):
            super().__init__(self.component_graph, parameters=parameters, custom_hyperparameters=self.custom_hyperparameters, random_seed=random_seed)

    class MockRegressionPipeline(RegressionPipeline):
        name = "Mock Regression Pipeline"
        component_graph = ['Imputer', 'Random Forest Regressor']

        def __init__(self, parameters, random_seed=0):
            super().__init__(self.component_graph, parameters=parameters, custom_name=self.name, custom_hyperparameters=None, random_seed=random_seed)

    mock_binary_pipeline = MockBinaryPipeline({})
    expected_code = 'import json\n' \
                    'from evalml.pipelines.binary_classification_pipeline import BinaryClassificationPipeline' \
                    '\n\nclass MockBinaryPipeline(BinaryClassificationPipeline):' \
                    '\n\tcomponent_graph = [\n\t\t\'Imputer\',\n\t\t\'Random Forest Classifier\'\n\t]' \
                    '\n\tcustom_hyperparameters = {\'Imputer\': {\'numeric_impute_strategy\': \'most_frequent\'}}\n' \
                    '\n\tdef __init__(self, parameters, random_seed=0):'\
                    '\n\t\tsuper().__init__(self.component_graph, custom_name=self.custom_name, parameters=parameters, custom_hyperparameters=custom_hyperparameters, random_seed=random_seed)'\
                    '\n\nparameters = json.loads("""{\n\t"Imputer": {\n\t\t"categorical_impute_strategy": "most_frequent",\n\t\t"numeric_impute_strategy": "mean",\n\t\t"categorical_fill_value": null,\n\t\t"numeric_fill_value": null\n\t},' \
                    '\n\t"Random Forest Classifier": {\n\t\t"n_estimators": 100,\n\t\t"max_depth": 6,\n\t\t"n_jobs": -1\n\t}\n}""")\n' \
                    'pipeline = MockBinaryPipeline(parameters)'
    pipeline = generate_pipeline_code(mock_binary_pipeline)

    assert expected_code == pipeline

    mock_regression_pipeline = MockRegressionPipeline({})
    expected_code = 'import json\n' \
                    'from evalml.pipelines.regression_pipeline import RegressionPipeline' \
                    '\n\nclass MockRegressionPipeline(RegressionPipeline):' \
                    '\n\tcomponent_graph = [\n\t\t\'Imputer\',\n\t\t\'Random Forest Regressor\'\n\t]\n\t' \
                    'name = \'Mock Regression Pipeline\'\n\n' \
                    '\tdef __init__(self, parameters, random_seed=0):'\
                    '\n\t\tsuper().__init__(self.component_graph, custom_name=self.custom_name, parameters=parameters, custom_hyperparameters=custom_hyperparameters, random_seed=random_seed)'\
                    '\n\nparameters = json.loads("""{\n\t"Imputer": {\n\t\t"categorical_impute_strategy": "most_frequent",\n\t\t"numeric_impute_strategy": "mean",\n\t\t"categorical_fill_value": null,\n\t\t"numeric_fill_value": null\n\t},' \
                    '\n\t"Random Forest Regressor": {\n\t\t"n_estimators": 100,\n\t\t"max_depth": 6,\n\t\t"n_jobs": -1\n\t}\n}""")' \
                    '\npipeline = MockRegressionPipeline(parameters)'
    pipeline = generate_pipeline_code(mock_regression_pipeline)
    assert pipeline == expected_code

    mock_regression_pipeline_params = MockRegressionPipeline({"Imputer": {"numeric_impute_strategy": "most_frequent"}, "Random Forest Regressor": {"n_estimators": 50}})
    expected_code_params = 'import json\n' \
                           'from evalml.pipelines.regression_pipeline import RegressionPipeline' \
                           '\n\nclass MockRegressionPipeline(RegressionPipeline):' \
                           '\n\tcomponent_graph = [\n\t\t\'Imputer\',\n\t\t\'Random Forest Regressor\'\n\t]' \
                           '\n\tname = \'Mock Regression Pipeline\'' \
                           '\n\n\tdef __init__(self, parameters, random_seed=0):'\
                           '\n\t\tsuper().__init__(self.component_graph, custom_name=self.custom_name, parameters=parameters, custom_hyperparameters=custom_hyperparameters, random_seed=random_seed)'\
                           '\n\nparameters = json.loads("""{\n\t"Imputer": {\n\t\t"categorical_impute_strategy": "most_frequent",\n\t\t"numeric_impute_strategy": "most_frequent",\n\t\t"categorical_fill_value": null,\n\t\t"numeric_fill_value": null\n\t},' \
                           '\n\t"Random Forest Regressor": {\n\t\t"n_estimators": 50,\n\t\t"max_depth": 6,\n\t\t"n_jobs": -1\n\t}\n}""")' \
                           '\npipeline = MockRegressionPipeline(parameters)'
    pipeline = generate_pipeline_code(mock_regression_pipeline_params)
    assert pipeline == expected_code_params


def test_generate_code_nonlinear_pipeline_error(nonlinear_binary_pipeline_class):
    pipeline = nonlinear_binary_pipeline_class({})
    with pytest.raises(ValueError, match="Code generation for nonlinear pipelines is not supported yet"):
        generate_pipeline_code(pipeline)


def test_generate_code_pipeline_custom():
    class CustomTransformer(Transformer):
        name = "My Custom Transformer"
        hyperparameter_ranges = {}

        def __init__(self, random_seed=0):
            parameters = {}

            super().__init__(parameters=parameters,
                             component_obj=None,
                             random_seed=random_seed)

    class CustomEstimator(Estimator):
        name = "My Custom Estimator"
        hyperparameter_ranges = {}
        supported_problem_types = [ProblemTypes.BINARY, ProblemTypes.MULTICLASS]
        model_family = ModelFamily.NONE

        def __init__(self, random_arg=False, random_seed=0):
            parameters = {'random_arg': random_arg}

            super().__init__(parameters=parameters,
                             component_obj=None,
                             random_seed=random_seed)

    class MockBinaryPipelineTransformer(BinaryClassificationPipeline):
        name = "Mock Binary Pipeline with Transformer"
        component_graph = [CustomTransformer, 'Random Forest Classifier']

        def __init__(self, parameters, random_seed=0):
            super().__init__(self.component_graph, parameters=parameters, custom_name=self.name, custom_hyperparameters=None, random_seed=random_seed)

    class MockBinaryPipelineEstimator(BinaryClassificationPipeline):
        name = "Mock Binary Pipeline with Estimator"
        component_graph = ['Imputer', CustomEstimator]
        custom_hyperparameters = {
            'Imputer': {
                'numeric_impute_strategy': 'most_frequent'
            }
        }

        def __init__(self, parameters, random_seed=0):
            super().__init__(self.component_graph, parameters=parameters, custom_name=self.name, custom_hyperparameters=None, random_seed=random_seed)

    class MockAllCustom(BinaryClassificationPipeline):
        name = "Mock All Custom Pipeline"
        component_graph = [CustomTransformer, CustomEstimator]

        def __init__(self, parameters, random_seed=0):
            super().__init__(self.component_graph, parameters=parameters, custom_name=self.name, custom_hyperparameters=None, random_seed=random_seed)

    mockBinaryTransformer = MockBinaryPipelineTransformer({})
    expected_code = 'import json\n' \
                    'from evalml.pipelines.binary_classification_pipeline import BinaryClassificationPipeline' \
                    '\n\nclass MockBinaryPipelineTransformer(BinaryClassificationPipeline):' \
                    '\n\tcomponent_graph = [\n\t\tCustomTransformer,\n\t\t\'Random Forest Classifier\'\n\t]' \
                    '\n\tname = \'Mock Binary Pipeline with Transformer\'' \
                    '\n\n\tdef __init__(self, parameters, random_seed=0):'\
                    '\n\t\tsuper().__init__(self.component_graph, custom_name=self.custom_name, parameters=parameters, custom_hyperparameters=custom_hyperparameters, random_seed=random_seed)'\
                    '\n\nparameters = json.loads("""{\n\t"Random Forest Classifier": {\n\t\t"n_estimators": 100,\n\t\t"max_depth": 6,\n\t\t"n_jobs": -1\n\t}\n}""")' \
                    '\npipeline = MockBinaryPipelineTransformer(parameters)'
    pipeline = generate_pipeline_code(mockBinaryTransformer)
    assert pipeline == expected_code

    mockBinaryPipeline = MockBinaryPipelineEstimator({})
    expected_code = 'import json\n' \
                    'from evalml.pipelines.binary_classification_pipeline import BinaryClassificationPipeline' \
                    '\n\nclass MockBinaryPipelineEstimator(BinaryClassificationPipeline):' \
                    '\n\tcomponent_graph = [\n\t\t\'Imputer\',\n\t\tCustomEstimator\n\t]' \
                    '\n\tcustom_hyperparameters = {\'Imputer\': {\'numeric_impute_strategy\': \'most_frequent\'}}' \
                    '\n\tname = \'Mock Binary Pipeline with Estimator\'' \
                    '\n\n\tdef __init__(self, parameters, random_seed=0):'\
                    '\n\t\tsuper().__init__(self.component_graph, custom_name=self.custom_name, parameters=parameters, custom_hyperparameters=custom_hyperparameters, random_seed=random_seed)'\
                    '\n\nparameters = json.loads("""{\n\t"Imputer": {\n\t\t"categorical_impute_strategy": "most_frequent",\n\t\t"numeric_impute_strategy": "mean",\n\t\t"categorical_fill_value": null,\n\t\t"numeric_fill_value": null\n\t},' \
                    '\n\t"My Custom Estimator": {\n\t\t"random_arg": false\n\t}\n}""")' \
                    '\npipeline = MockBinaryPipelineEstimator(parameters)'
    pipeline = generate_pipeline_code(mockBinaryPipeline)
    assert pipeline == expected_code

    mockAllCustom = MockAllCustom({})
    expected_code = 'import json\n' \
                    'from evalml.pipelines.binary_classification_pipeline import BinaryClassificationPipeline' \
                    '\n\nclass MockAllCustom(BinaryClassificationPipeline):' \
                    '\n\tcomponent_graph = [\n\t\tCustomTransformer,\n\t\tCustomEstimator\n\t]' \
                    '\n\tname = \'Mock All Custom Pipeline\''\
                    '\n\n\tdef __init__(self, parameters, random_seed=0):'\
                    '\n\t\tsuper().__init__(self.component_graph, custom_name=self.custom_name, parameters=parameters, custom_hyperparameters=custom_hyperparameters, random_seed=random_seed)'\
                    '\n\nparameters = json.loads("""{\n\t"My Custom Estimator": {\n\t\t"random_arg": false\n\t}\n}""")' \
                    '\npipeline = MockAllCustom(parameters)'
    pipeline = generate_pipeline_code(mockAllCustom)
    assert pipeline == expected_code
