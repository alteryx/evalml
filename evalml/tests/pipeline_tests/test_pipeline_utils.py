
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
    RandomForestClassifier,
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
    get_estimators,
    make_pipeline,
    make_pipeline_from_components
)
from evalml.problem_types import ProblemTypes, is_time_series


def test_make_pipeline_error():
    X = pd.DataFrame([[0, 1], [1, 0]])
    y = pd.Series([1, 0])
    estimators = get_estimators(problem_type="binary")
    custom_hyperparameters = [{"Imputer": {"numeric_imput_strategy": ["median"]}}, {"One Hot Encoder": {"value1": ["value2"]}}]

    for estimator in estimators:
        with pytest.raises(ValueError, match="if custom_hyperparameters provided, must be dictionary"):
            make_pipeline(X, y, estimator, "binary", custom_hyperparameters)


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
            pipeline = make_pipeline(X, y, estimator_class, problem_type, custom_hyperparameters)
            assert pipeline.custom_hyperparameters == custom_hyperparameters

            pipeline2 = make_pipeline(X, y, estimator_class, problem_type)
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
            pipeline = make_pipeline(X, y, estimator_class, problem_type)
            assert isinstance(pipeline, type(pipeline_class))
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
            pipeline = make_pipeline(X, y, estimator_class, problem_type)
            assert isinstance(pipeline, type(pipeline_class))
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
            pipeline = make_pipeline(X, y, estimator_class, problem_type)
            assert isinstance(pipeline, type(pipeline_class))
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
            pipeline = make_pipeline(X, y, estimator_class, problem_type)
            assert isinstance(pipeline, type(pipeline_class))
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
            pipeline = make_pipeline(X, y, estimator_class, problem_type)
            assert isinstance(pipeline, type(pipeline_class))
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
            pipeline = make_pipeline(X, y, estimator_class, problem_type)
            assert isinstance(pipeline, type(pipeline_class))
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
            pipeline = make_pipeline(X, y, estimator_class, problem_type)
            assert isinstance(pipeline, type(pipeline_class))
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
            pipeline = make_pipeline(X, y, estimator_class, problem_type)
            assert isinstance(pipeline, type(pipeline_class))
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
            pipeline = make_pipeline(X, y, estimator_class, problem_type)
            assert isinstance(pipeline, type(pipeline_class))
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
            pipeline = make_pipeline(X, y, estimator_class, problem_type)
            assert isinstance(pipeline, type(pipeline_class))
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


def test_make_pipeline_from_components(X_y_binary, logistic_regression_binary_pipeline_class):
    with pytest.raises(ValueError, match="Pipeline needs to have an estimator at the last position of the component list"):
        make_pipeline_from_components([Imputer()], problem_type='binary')

    with pytest.raises(KeyError, match="Problem type 'invalid_type' does not exist"):
        make_pipeline_from_components([RandomForestClassifier()], problem_type='invalid_type')

    with pytest.raises(TypeError, match="Custom pipeline name must be a string"):
        make_pipeline_from_components([RandomForestClassifier()], problem_type='binary', custom_name=True)

    with pytest.raises(TypeError, match="Every element of `component_instances` must be an instance of ComponentBase"):
        make_pipeline_from_components([RandomForestClassifier], problem_type='binary')

    with pytest.raises(TypeError, match="Every element of `component_instances` must be an instance of ComponentBase"):
        make_pipeline_from_components(['RandomForestClassifier'], problem_type='binary')

    imp = Imputer(numeric_impute_strategy='median', random_seed=5)
    est = RandomForestClassifier(random_seed=7)
    pipeline = make_pipeline_from_components([imp, est], ProblemTypes.BINARY, custom_name='My Pipeline',
                                             random_seed=15)
    assert [c.__class__ for c in pipeline] == [Imputer, RandomForestClassifier]
    assert [(c.random_seed == 15) for c in pipeline]
    assert pipeline.problem_type == ProblemTypes.BINARY
    assert pipeline.custom_name == 'My Pipeline'
    expected_parameters = {
        'Imputer': {
            'categorical_impute_strategy': 'most_frequent',
            'numeric_impute_strategy': 'median',
            'categorical_fill_value': None,
            'numeric_fill_value': None},
        'Random Forest Classifier': {
            'n_estimators': 100,
            'max_depth': 6,
            'n_jobs': -1}
    }
    assert pipeline.parameters == expected_parameters
    assert pipeline.random_seed == 15

    class DummyEstimator(Estimator):
        name = "Dummy!"
        model_family = "foo"
        supported_problem_types = [ProblemTypes.BINARY]
        parameters = {'bar': 'baz'}
    random_seed = 42
    pipeline = make_pipeline_from_components([DummyEstimator(random_seed=3)], ProblemTypes.BINARY,
                                             random_seed=random_seed)
    components_list = [c for c in pipeline]
    assert len(components_list) == 1
    assert isinstance(components_list[0], DummyEstimator)
    assert components_list[0].random_seed == random_seed
    expected_parameters = {'Dummy!': {'bar': 'baz'}}
    assert pipeline.parameters == expected_parameters
    assert pipeline.random_seed == random_seed

    X, y = X_y_binary
    pipeline = logistic_regression_binary_pipeline_class(parameters={"Logistic Regression Classifier": {"n_jobs": 1}},
                                                         random_seed=42)
    component_instances = [c for c in pipeline]
    new_pipeline = make_pipeline_from_components(component_instances, ProblemTypes.BINARY)
    pipeline.fit(X, y)
    predictions = pipeline.predict(X)
    new_pipeline.fit(X, y)
    new_predictions = new_pipeline.predict(X)
    assert np.array_equal(predictions, new_predictions)
    assert np.array_equal(pipeline.feature_importance, new_pipeline.feature_importance)
    assert new_pipeline.name == 'Templated Pipeline'
    assert pipeline.parameters == new_pipeline.parameters
    for component, new_component in zip(pipeline._component_graph, new_pipeline._component_graph):
        assert isinstance(new_component, type(component))


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
        input_pipelines = [make_pipeline_from_components([classifier], problem_type) for classifier in stackable_classifiers]
        comparison_pipeline = logistic_regression_binary_pipeline_class(parameters={"Logistic Regression Classifier": {"n_jobs": 1}})
        objective = 'Log Loss Binary'
    elif problem_type == ProblemTypes.MULTICLASS:
        X, y = X_y_multi
        base_pipeline_class = MulticlassClassificationPipeline
        stacking_component_name = StackedEnsembleClassifier.name
        input_pipelines = [make_pipeline_from_components([classifier], problem_type) for classifier in stackable_classifiers]
        comparison_pipeline = logistic_regression_multiclass_pipeline_class(parameters={"Logistic Regression Classifier": {"n_jobs": 1}})
        objective = 'Log Loss Multiclass'
    elif problem_type == ProblemTypes.REGRESSION:
        X, y = X_y_regression
        base_pipeline_class = RegressionPipeline
        stacking_component_name = StackedEnsembleRegressor.name
        input_pipelines = [make_pipeline_from_components([regressor], problem_type) for regressor in stackable_regressors]
        comparison_pipeline = linear_regression_pipeline_class(parameters={"Linear Regressor": {"n_jobs": 1}})
        objective = 'R2'
    parameters = {
        stacking_component_name: {
            "input_pipelines": input_pipelines,
            "n_jobs": 1
        }
    }
    graph = ['Simple Imputer', stacking_component_name]

    class StackedPipeline(base_pipeline_class):
        component_graph = graph
        model_family = ModelFamily.ENSEMBLE

    pipeline = StackedPipeline(parameters=parameters)
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
