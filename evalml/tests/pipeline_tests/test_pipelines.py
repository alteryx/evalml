import os
from unittest.mock import patch

import cloudpickle
import numpy as np
import pandas as pd
import pytest
from skopt.space import Integer, Real

from evalml.demos import load_breast_cancer, load_wine
from evalml.exceptions import (
    ComponentNotYetFittedError,
    IllFormattedClassNameError,
    MissingComponentError,
    PipelineScoreError
)
from evalml.model_family import ModelFamily
from evalml.objectives import FraudCost, Precision
from evalml.pipelines import (
    BinaryClassificationPipeline,
    MulticlassClassificationPipeline,
    PipelineBase,
    RegressionPipeline
)
from evalml.pipelines.components import (
    DateTimeFeaturizer,
    DropNullColumns,
    Imputer,
    LinearRegressor,
    LogisticRegressionClassifier,
    OneHotEncoder,
    RandomForestClassifier,
    RandomForestRegressor,
    RFClassifierSelectFromModel,
    StandardScaler,
    Transformer
)
from evalml.pipelines.components.utils import (
    _all_estimators_used_in_search,
    allowed_model_families
)
from evalml.pipelines.utils import get_estimators, make_pipeline
from evalml.problem_types import ProblemTypes
from evalml.utils.gen_utils import (
    categorical_dtypes,
    numeric_and_boolean_dtypes
)


def test_allowed_model_families(has_minimal_dependencies):
    families = [ModelFamily.RANDOM_FOREST, ModelFamily.LINEAR_MODEL, ModelFamily.EXTRA_TREES]
    expected_model_families_binary = set(families)
    expected_model_families_regression = set(families)
    if not has_minimal_dependencies:
        expected_model_families_binary.add(ModelFamily.XGBOOST)
        expected_model_families_binary.add(ModelFamily.CATBOOST)
        expected_model_families_regression.add(ModelFamily.CATBOOST)
        expected_model_families_regression.add(ModelFamily.XGBOOST)
    assert set(allowed_model_families(ProblemTypes.BINARY)) == expected_model_families_binary
    assert set(allowed_model_families(ProblemTypes.REGRESSION)) == expected_model_families_regression


def test_all_estimators(has_minimal_dependencies):
    if has_minimal_dependencies:
        assert len((_all_estimators_used_in_search())) == 8
    else:
        assert len(_all_estimators_used_in_search()) == 12


def test_get_estimators(has_minimal_dependencies):
    if has_minimal_dependencies:
        assert len(get_estimators(problem_type=ProblemTypes.BINARY)) == 4
        assert len(get_estimators(problem_type=ProblemTypes.BINARY, model_families=[ModelFamily.LINEAR_MODEL])) == 2
        assert len(get_estimators(problem_type=ProblemTypes.MULTICLASS)) == 4
        assert len(get_estimators(problem_type=ProblemTypes.REGRESSION)) == 4
    else:
        assert len(get_estimators(problem_type=ProblemTypes.BINARY)) == 6
        assert len(get_estimators(problem_type=ProblemTypes.BINARY, model_families=[ModelFamily.LINEAR_MODEL])) == 2
        assert len(get_estimators(problem_type=ProblemTypes.MULTICLASS)) == 6
        assert len(get_estimators(problem_type=ProblemTypes.REGRESSION)) == 6

    assert len(get_estimators(problem_type=ProblemTypes.BINARY, model_families=[])) == 0
    assert len(get_estimators(problem_type=ProblemTypes.MULTICLASS, model_families=[])) == 0
    assert len(get_estimators(problem_type=ProblemTypes.REGRESSION, model_families=[])) == 0

    with pytest.raises(RuntimeError, match="Unrecognized model type for problem type"):
        get_estimators(problem_type=ProblemTypes.REGRESSION, model_families=["random_forest", "none"])
    with pytest.raises(TypeError, match="model_families parameter is not a list."):
        get_estimators(problem_type=ProblemTypes.REGRESSION, model_families='random_forest')
    with pytest.raises(KeyError):
        get_estimators(problem_type="Not A Valid Problem Type")


def test_make_pipeline_all_nan_no_categoricals():
    # testing that all_null column is not considered categorical
    X = pd.DataFrame({"all_null": [np.nan, np.nan, np.nan, np.nan, np.nan],
                      "num": [1, 2, 3, 4, 5]})
    y = pd.Series([0, 0, 1, 1, 0])
    binary_pipeline = make_pipeline(X, y, LogisticRegressionClassifier, ProblemTypes.BINARY)
    assert isinstance(binary_pipeline, type(BinaryClassificationPipeline))
    assert binary_pipeline.component_graph == [DropNullColumns, Imputer, StandardScaler, LogisticRegressionClassifier]

    binary_pipeline = make_pipeline(X, y, RandomForestClassifier, ProblemTypes.BINARY)
    assert isinstance(binary_pipeline, type(BinaryClassificationPipeline))
    assert binary_pipeline.component_graph == [DropNullColumns, Imputer, RandomForestClassifier]


def test_make_pipeline():
    X = pd.DataFrame({"all_null": [np.nan, np.nan, np.nan, np.nan, np.nan],
                      "categorical": ["a", "b", "a", "c", "c"],
                      "some dates": pd.date_range('2000-02-03', periods=5, freq='W')})
    y = pd.Series([0, 0, 1, 0, 0])
    binary_pipeline = make_pipeline(X, y, LogisticRegressionClassifier, ProblemTypes.BINARY)
    assert isinstance(binary_pipeline, type(BinaryClassificationPipeline))
    assert binary_pipeline.component_graph == [DropNullColumns, Imputer, DateTimeFeaturizer, OneHotEncoder, StandardScaler, LogisticRegressionClassifier]

    binary_pipeline = make_pipeline(X, y, RandomForestClassifier, ProblemTypes.BINARY)
    assert isinstance(binary_pipeline, type(BinaryClassificationPipeline))
    assert binary_pipeline.component_graph == [DropNullColumns, Imputer, DateTimeFeaturizer, OneHotEncoder, RandomForestClassifier]

    y = pd.Series([0, 2, 1, 2, 0])
    multiclass_pipeline = make_pipeline(X, y, LogisticRegressionClassifier, ProblemTypes.MULTICLASS)
    assert isinstance(multiclass_pipeline, type(MulticlassClassificationPipeline))
    assert multiclass_pipeline.component_graph == [DropNullColumns, Imputer, DateTimeFeaturizer, OneHotEncoder, StandardScaler, LogisticRegressionClassifier]

    regression_pipeline = make_pipeline(X, y, RandomForestRegressor, ProblemTypes.REGRESSION)
    assert isinstance(regression_pipeline, type(RegressionPipeline))
    assert regression_pipeline.component_graph == [DropNullColumns, Imputer, DateTimeFeaturizer, OneHotEncoder, RandomForestRegressor]

    regression_pipeline = make_pipeline(X, y, LinearRegressor, ProblemTypes.REGRESSION)
    assert isinstance(regression_pipeline, type(RegressionPipeline))
    assert regression_pipeline.component_graph == [DropNullColumns, Imputer, DateTimeFeaturizer, OneHotEncoder, StandardScaler, LinearRegressor]


def test_make_pipeline_no_nulls():
    X = pd.DataFrame({"numerical": [1, 2, 3, 1, 2],
                      "categorical": ["a", "b", "a", "c", "c"],
                      "some dates": pd.date_range('2000-02-03', periods=5, freq='W')})
    y = pd.Series([0, 1, 1, 0, 0])
    binary_pipeline = make_pipeline(X, y, LogisticRegressionClassifier, ProblemTypes.BINARY)
    assert isinstance(binary_pipeline, type(BinaryClassificationPipeline))
    assert binary_pipeline.component_graph == [Imputer, DateTimeFeaturizer, OneHotEncoder, StandardScaler, LogisticRegressionClassifier]
    assert binary_pipeline.custom_hyperparameters is None

    y = pd.Series([0, 2, 1, 2, 0])
    multiclass_pipeline = make_pipeline(X, y, LogisticRegressionClassifier, ProblemTypes.MULTICLASS)
    assert isinstance(multiclass_pipeline, type(MulticlassClassificationPipeline))
    assert multiclass_pipeline.component_graph == [Imputer, DateTimeFeaturizer, OneHotEncoder, StandardScaler, LogisticRegressionClassifier]
    assert multiclass_pipeline.custom_hyperparameters is None

    regression_pipeline = make_pipeline(X, y, RandomForestRegressor, ProblemTypes.REGRESSION)
    assert isinstance(regression_pipeline, type(RegressionPipeline))
    assert regression_pipeline.component_graph == [Imputer, DateTimeFeaturizer, OneHotEncoder, RandomForestRegressor]
    assert regression_pipeline.custom_hyperparameters is None


def test_make_pipeline_no_datetimes():
    X = pd.DataFrame({"numerical": [1, 2, 3, 1, 2],
                      "categorical": ["a", "b", "a", "c", "c"],
                      "all_null": [np.nan, np.nan, np.nan, np.nan, np.nan]})
    y = pd.Series([0, 0, 1, 2, 0])
    binary_pipeline = make_pipeline(X, y, LogisticRegressionClassifier, ProblemTypes.BINARY)
    assert isinstance(binary_pipeline, type(BinaryClassificationPipeline))
    assert binary_pipeline.component_graph == [DropNullColumns, Imputer, OneHotEncoder, StandardScaler, LogisticRegressionClassifier]
    assert binary_pipeline.custom_hyperparameters is None

    multiclass_pipeline = make_pipeline(X, y, LogisticRegressionClassifier, ProblemTypes.MULTICLASS)
    assert isinstance(multiclass_pipeline, type(MulticlassClassificationPipeline))
    assert multiclass_pipeline.component_graph == [DropNullColumns, Imputer, OneHotEncoder, StandardScaler, LogisticRegressionClassifier]
    assert multiclass_pipeline.custom_hyperparameters is None

    regression_pipeline = make_pipeline(X, y, RandomForestRegressor, ProblemTypes.REGRESSION)
    assert isinstance(regression_pipeline, type(RegressionPipeline))
    assert regression_pipeline.component_graph == [DropNullColumns, Imputer, OneHotEncoder, RandomForestRegressor]
    assert regression_pipeline.custom_hyperparameters is None


def test_make_pipeline_no_column_names():
    X = pd.DataFrame([[1, "a", np.nan], [2, "b", np.nan], [5, "b", np.nan]])
    y = pd.Series([0, 0, 1])
    binary_pipeline = make_pipeline(X, y, LogisticRegressionClassifier, ProblemTypes.BINARY)
    assert isinstance(binary_pipeline, type(BinaryClassificationPipeline))
    assert binary_pipeline.component_graph == [DropNullColumns, Imputer, OneHotEncoder, StandardScaler, LogisticRegressionClassifier]
    assert binary_pipeline.custom_hyperparameters is None

    multiclass_pipeline = make_pipeline(X, y, LogisticRegressionClassifier, ProblemTypes.MULTICLASS)
    assert isinstance(multiclass_pipeline, type(MulticlassClassificationPipeline))
    assert multiclass_pipeline.component_graph == [DropNullColumns, Imputer, OneHotEncoder, StandardScaler, LogisticRegressionClassifier]
    assert multiclass_pipeline.custom_hyperparameters is None

    regression_pipeline = make_pipeline(X, y, RandomForestRegressor, ProblemTypes.REGRESSION)
    assert isinstance(regression_pipeline, type(RegressionPipeline))
    assert regression_pipeline.component_graph == [DropNullColumns, Imputer, OneHotEncoder, RandomForestRegressor]
    assert regression_pipeline.custom_hyperparameters is None


def test_make_pipeline_numpy_input():
    X = np.array([[1, 2, 0, np.nan], [2, 2, 1, np.nan], [5, 1, np.nan, np.nan]])
    y = np.array([0, 0, 1, 0])
    binary_pipeline = make_pipeline(X, y, LogisticRegressionClassifier, ProblemTypes.BINARY)
    assert isinstance(binary_pipeline, type(BinaryClassificationPipeline))
    assert binary_pipeline.component_graph == [DropNullColumns, Imputer, StandardScaler, LogisticRegressionClassifier]
    assert binary_pipeline.custom_hyperparameters is None

    multiclass_pipeline = make_pipeline(X, y, LogisticRegressionClassifier, ProblemTypes.MULTICLASS)
    assert isinstance(multiclass_pipeline, type(MulticlassClassificationPipeline))
    assert multiclass_pipeline.component_graph == [DropNullColumns, Imputer, StandardScaler, LogisticRegressionClassifier]
    assert multiclass_pipeline.custom_hyperparameters is None

    regression_pipeline = make_pipeline(X, y, RandomForestRegressor, ProblemTypes.REGRESSION)
    assert isinstance(regression_pipeline, type(RegressionPipeline))
    assert regression_pipeline.component_graph == [DropNullColumns, Imputer, RandomForestRegressor]
    assert regression_pipeline.custom_hyperparameters is None


def test_make_pipeline_problem_type_mismatch():
    with pytest.raises(ValueError, match=f"{LogisticRegressionClassifier.name} is not a valid estimator for problem type"):
        make_pipeline(pd.DataFrame(), pd.Series(), LogisticRegressionClassifier, ProblemTypes.REGRESSION)
    with pytest.raises(ValueError, match=f"{LinearRegressor.name} is not a valid estimator for problem type"):
        make_pipeline(pd.DataFrame(), pd.Series(), LinearRegressor, ProblemTypes.MULTICLASS)
    with pytest.raises(ValueError, match=f"{Transformer.name} is not a valid estimator for problem type"):
        make_pipeline(pd.DataFrame(), pd.Series(), Transformer, ProblemTypes.MULTICLASS)


def test_required_fields():
    class TestPipelineWithoutComponentGraph(PipelineBase):
        pass

    with pytest.raises(TypeError):
        TestPipelineWithoutComponentGraph(parameters={})


def test_serialization(X_y_binary, tmpdir, logistic_regression_binary_pipeline_class):
    X, y = X_y_binary
    path = os.path.join(str(tmpdir), 'pipe.pkl')
    pipeline = logistic_regression_binary_pipeline_class(parameters={})
    pipeline.fit(X, y)
    pipeline.save(path)
    assert pipeline.score(X, y, ['precision']) == PipelineBase.load(path).score(X, y, ['precision'])


@patch('cloudpickle.dump')
def test_serialization_protocol(mock_cloudpickle_dump, tmpdir, logistic_regression_binary_pipeline_class):
    path = os.path.join(str(tmpdir), 'pipe.pkl')
    pipeline = logistic_regression_binary_pipeline_class(parameters={})

    pipeline.save(path)
    assert len(mock_cloudpickle_dump.call_args_list) == 1
    assert mock_cloudpickle_dump.call_args_list[0][1]['protocol'] == cloudpickle.DEFAULT_PROTOCOL

    mock_cloudpickle_dump.reset_mock()

    pipeline.save(path, pickle_protocol=42)
    assert len(mock_cloudpickle_dump.call_args_list) == 1
    assert mock_cloudpickle_dump.call_args_list[0][1]['protocol'] == 42


@pytest.fixture
def pickled_pipeline_path(X_y_binary, tmpdir, logistic_regression_binary_pipeline_class):
    X, y = X_y_binary
    path = os.path.join(str(tmpdir), 'pickled_pipe.pkl')
    pipeline = logistic_regression_binary_pipeline_class(parameters={})
    pipeline.fit(X, y)
    pipeline.save(path)
    return path


def test_load_pickled_pipeline_with_custom_objective(X_y_binary, pickled_pipeline_path, logistic_regression_binary_pipeline_class):
    X, y = X_y_binary
    # checks that class is not defined before loading in pipeline
    with pytest.raises(NameError):
        MockPrecision()  # noqa: F821: ignore flake8's "undefined name" error
    objective = Precision()
    pipeline = logistic_regression_binary_pipeline_class(parameters={})
    pipeline.fit(X, y)
    assert PipelineBase.load(pickled_pipeline_path).score(X, y, [objective]) == pipeline.score(X, y, [objective])


def test_reproducibility(X_y_binary, logistic_regression_binary_pipeline_class):
    X, y = X_y_binary
    objective = FraudCost(
        retry_percentage=.5,
        interchange_fee=.02,
        fraud_payout_percentage=.75,
        amount_col=10
    )

    parameters = {
        'Imputer': {
            "categorical_impute_strategy": "most_frequent",
            "numeric_impute_strategy": "mean",
        },
        'Logistic Regression Classifier': {
            'penalty': 'l2',
            'C': 1.0,
        }
    }

    clf = logistic_regression_binary_pipeline_class(parameters=parameters)
    clf.fit(X, y)

    clf_1 = logistic_regression_binary_pipeline_class(parameters=parameters)
    clf_1.fit(X, y)

    assert clf_1.score(X, y, [objective]) == clf.score(X, y, [objective])


def test_indexing(X_y_binary, logistic_regression_binary_pipeline_class):
    X, y = X_y_binary
    clf = logistic_regression_binary_pipeline_class(parameters={})
    clf.fit(X, y)

    assert isinstance(clf[1], OneHotEncoder)
    assert isinstance(clf['Imputer'], Imputer)

    setting_err_msg = 'Setting pipeline components is not supported.'
    with pytest.raises(NotImplementedError, match=setting_err_msg):
        clf[1] = OneHotEncoder()

    slicing_err_msg = 'Slicing pipelines is currently not supported.'
    with pytest.raises(NotImplementedError, match=slicing_err_msg):
        clf[:1]


def test_describe(caplog, logistic_regression_binary_pipeline_class):
    lrp = logistic_regression_binary_pipeline_class(parameters={})
    lrp.describe()
    out = caplog.text
    assert "Logistic Regression Binary Pipeline" in out
    assert "Problem Type: Binary Classification" in out
    assert "Model Family: Linear" in out
    assert "Number of features: " not in out

    for component in lrp.component_graph:
        if component.hyperparameter_ranges:
            for parameter in component.hyperparameter_ranges:
                assert parameter in out
        assert component.name in out


def test_describe_fitted(X_y_binary, caplog, logistic_regression_binary_pipeline_class):
    X, y = X_y_binary
    lrp = logistic_regression_binary_pipeline_class(parameters={})
    lrp.fit(X, y)
    lrp.describe()
    out = caplog.text
    assert "Logistic Regression Binary Pipeline" in out
    assert "Problem Type: Binary Classification" in out
    assert "Model Family: Linear" in out
    assert "Number of features: {}".format(X.shape[1]) in out

    for component in lrp.component_graph:
        if component.hyperparameter_ranges:
            for parameter in component.hyperparameter_ranges:
                assert parameter in out
        assert component.name in out


def test_parameters(logistic_regression_binary_pipeline_class):
    parameters = {
        'Imputer': {
            "categorical_impute_strategy": "most_frequent",
            "numeric_impute_strategy": "median"
        },
        'Logistic Regression Classifier': {
            'penalty': 'l2',
            'C': 3.0,
        }
    }
    lrp = logistic_regression_binary_pipeline_class(parameters=parameters)
    expected_parameters = {
        'Imputer': {
            "categorical_impute_strategy": "most_frequent",
            "numeric_impute_strategy": "median",
            'categorical_fill_value': None,
            'numeric_fill_value': None
        },
        'One Hot Encoder': {
            'top_n': 10,
            'categories': None,
            'drop': None,
            'handle_unknown': 'ignore',
            'handle_missing': 'error'
        },
        'Logistic Regression Classifier': {
            'penalty': 'l2',
            'C': 3.0,
            'n_jobs': -1
        }
    }
    assert lrp.parameters == expected_parameters


def test_name():
    class TestNamePipeline(BinaryClassificationPipeline):
        component_graph = ['Logistic Regression Classifier']

    class TestDefinedNamePipeline(BinaryClassificationPipeline):
        custom_name = "Cool Logistic Regression"
        component_graph = ['Logistic Regression Classifier']

    class testillformattednamepipeline(BinaryClassificationPipeline):
        component_graph = ['Logistic Regression Classifier']

    assert TestNamePipeline.name == "Test Name Pipeline"
    assert TestNamePipeline.custom_name is None
    assert TestDefinedNamePipeline.name == "Cool Logistic Regression"
    assert TestDefinedNamePipeline.custom_name == "Cool Logistic Regression"
    assert TestDefinedNamePipeline(parameters={}).name == "Cool Logistic Regression"
    with pytest.raises(IllFormattedClassNameError):
        testillformattednamepipeline.name == "Test Illformatted Name Pipeline"


def test_estimator_not_last():
    class MockBinaryClassificationPipelineWithoutEstimator(BinaryClassificationPipeline):
        name = "Mock Binary Classification Pipeline Without Estimator"
        component_graph = ['One Hot Encoder', 'Imputer', 'Logistic Regression Classifier', 'Standard Scaler']

    err_msg = "A pipeline must have an Estimator as the last component in component_graph."
    with pytest.raises(ValueError, match=err_msg):
        MockBinaryClassificationPipelineWithoutEstimator(parameters={})


def test_multi_format_creation(X_y_binary):
    X, y = X_y_binary

    class TestPipeline(BinaryClassificationPipeline):
        component_graph = component_graph = ['Imputer', 'One Hot Encoder', StandardScaler, 'Logistic Regression Classifier']

        hyperparameters = {
            'Imputer': {
                "categorical_impute_strategy": ["most_frequent"],
                "numeric_impute_strategy": ["mean", "median", "most_frequent"]
            },
            'Logistic Regression Classifier': {
                "penalty": ["l2"],
                "C": Real(.01, 10)
            }
        }

    parameters = {
        'Imputer': {
            "categorical_impute_strategy": "most_frequent",
            "numeric_impute_strategy": "mean",
        },
        'Logistic Regression Classifier': {
            'penalty': 'l2',
            'C': 1.0,
        }
    }

    clf = TestPipeline(parameters=parameters)
    correct_components = [Imputer, OneHotEncoder, StandardScaler, LogisticRegressionClassifier]
    for component, correct_components in zip(clf.component_graph, correct_components):
        assert isinstance(component, correct_components)
    assert clf.model_family == ModelFamily.LINEAR_MODEL

    clf.fit(X, y)
    clf.score(X, y, ['precision'])
    assert not clf.feature_importance.isnull().all().all()


def test_multiple_feature_selectors(X_y_binary):
    X, y = X_y_binary

    class TestPipeline(BinaryClassificationPipeline):
        component_graph = ['Imputer', 'One Hot Encoder', 'RF Classifier Select From Model', StandardScaler, 'RF Classifier Select From Model', 'Logistic Regression Classifier']

        hyperparameters = {
            'Imputer': {
                "categorical_impute_strategy": ["most_frequent"],
                "numeric_impute_strategy": ["mean", "median", "most_frequent"]
            },
            'Logistic Regression Classifier': {
                "penalty": ["l2"],
                "C": Real(.01, 10)
            }
        }

    clf = TestPipeline(parameters={})
    correct_components = [Imputer, OneHotEncoder, RFClassifierSelectFromModel, StandardScaler, RFClassifierSelectFromModel, LogisticRegressionClassifier]
    for component, correct_components in zip(clf.component_graph, correct_components):
        assert isinstance(component, correct_components)
    assert clf.model_family == ModelFamily.LINEAR_MODEL

    clf.fit(X, y)
    clf.score(X, y, ['precision'])
    assert not clf.feature_importance.isnull().all().all()


def test_problem_types():
    class TestPipeline(BinaryClassificationPipeline):
        component_graph = ['Random Forest Regressor']

    with pytest.raises(ValueError, match="not valid for this component graph. Valid problem types include *."):
        TestPipeline(parameters={})


def make_mock_regression_pipeline():
    class MockRegressionPipeline(RegressionPipeline):
        component_graph = ['Random Forest Regressor']

    return MockRegressionPipeline({})


def make_mock_binary_pipeline():
    class MockBinaryClassificationPipeline(BinaryClassificationPipeline):
        component_graph = ['Random Forest Classifier']

    return MockBinaryClassificationPipeline({})


def make_mock_multiclass_pipeline():
    class MockMulticlassClassificationPipeline(MulticlassClassificationPipeline):
        component_graph = ['Random Forest Classifier']

    return MockMulticlassClassificationPipeline({})


@patch('evalml.pipelines.RegressionPipeline.fit')
@patch('evalml.pipelines.RegressionPipeline.predict')
def test_score_regression_single(mock_predict, mock_fit, X_y_binary):
    X, y = X_y_binary
    mock_predict.return_value = y
    clf = make_mock_regression_pipeline()
    clf.fit(X, y)
    objective_names = ['r2']
    scores = clf.score(X, y, objective_names)
    mock_predict.assert_called()
    assert scores == {'R2': 1.0}


@patch('evalml.pipelines.BinaryClassificationPipeline._encode_targets')
@patch('evalml.pipelines.BinaryClassificationPipeline.fit')
@patch('evalml.pipelines.components.Estimator.predict')
def test_score_binary_single(mock_predict, mock_fit, mock_encode, X_y_binary):
    X, y = X_y_binary
    mock_predict.return_value = y
    mock_encode.return_value = y
    clf = make_mock_binary_pipeline()
    clf.fit(X, y)
    objective_names = ['f1']
    scores = clf.score(X, y, objective_names)
    mock_encode.assert_called()
    mock_fit.assert_called()
    mock_predict.assert_called()
    assert scores == {'F1': 1.0}


@patch('evalml.pipelines.MulticlassClassificationPipeline._encode_targets')
@patch('evalml.pipelines.MulticlassClassificationPipeline.fit')
@patch('evalml.pipelines.components.Estimator.predict')
def test_score_multiclass_single(mock_predict, mock_fit, mock_encode, X_y_binary):
    X, y = X_y_binary
    mock_predict.return_value = y
    mock_encode.return_value = y
    clf = make_mock_multiclass_pipeline()
    clf.fit(X, y)
    objective_names = ['f1_micro']
    scores = clf.score(X, y, objective_names)
    mock_encode.assert_called()
    mock_fit.assert_called()
    mock_predict.assert_called()
    assert scores == {'F1 Micro': 1.0}


@patch('evalml.pipelines.RegressionPipeline.fit')
@patch('evalml.pipelines.RegressionPipeline.predict')
def test_score_regression_list(mock_predict, mock_fit, X_y_binary):
    X, y = X_y_binary
    mock_predict.return_value = y
    clf = make_mock_regression_pipeline()
    clf.fit(X, y)
    objective_names = ['r2', 'mse']
    scores = clf.score(X, y, objective_names)
    mock_predict.assert_called()
    assert scores == {'R2': 1.0, 'MSE': 0.0}


@patch('evalml.pipelines.BinaryClassificationPipeline._encode_targets')
@patch('evalml.pipelines.BinaryClassificationPipeline.fit')
@patch('evalml.pipelines.components.Estimator.predict')
def test_score_binary_list(mock_predict, mock_fit, mock_encode, X_y_binary):
    X, y = X_y_binary
    mock_predict.return_value = y
    mock_encode.return_value = y
    clf = make_mock_binary_pipeline()
    clf.fit(X, y)
    objective_names = ['f1', 'precision']
    scores = clf.score(X, y, objective_names)
    mock_fit.assert_called()
    mock_encode.assert_called()
    mock_predict.assert_called()
    assert scores == {'F1': 1.0, 'Precision': 1.0}


@patch('evalml.pipelines.MulticlassClassificationPipeline._encode_targets')
@patch('evalml.pipelines.MulticlassClassificationPipeline.fit')
@patch('evalml.pipelines.components.Estimator.predict')
def test_score_multi_list(mock_predict, mock_fit, mock_encode, X_y_binary):
    X, y = X_y_binary
    mock_predict.return_value = y
    mock_encode.return_value = y
    clf = make_mock_multiclass_pipeline()
    clf.fit(X, y)
    objective_names = ['f1_micro', 'precision_micro']
    scores = clf.score(X, y, objective_names)
    mock_predict.assert_called()
    assert scores == {'F1 Micro': 1.0, 'Precision Micro': 1.0}


@patch('evalml.objectives.R2.score')
@patch('evalml.pipelines.RegressionPipeline.fit')
@patch('evalml.pipelines.RegressionPipeline.predict')
def test_score_regression_objective_error(mock_predict, mock_fit, mock_objective_score, X_y_binary):
    mock_objective_score.side_effect = Exception('finna kabooom 💣')
    X, y = X_y_binary
    mock_predict.return_value = y
    clf = make_mock_regression_pipeline()
    clf.fit(X, y)
    objective_names = ['r2', 'mse']
    # Using pytest.raises to make sure we error if an error is not thrown.
    with pytest.raises(PipelineScoreError):
        _ = clf.score(X, y, objective_names)
    try:
        _ = clf.score(X, y, objective_names)
    except PipelineScoreError as e:
        assert e.scored_successfully == {"MSE": 0.0}
        assert 'finna kabooom 💣' in e.message
        assert "R2" in e.exceptions


@patch('evalml.pipelines.BinaryClassificationPipeline._encode_targets')
@patch('evalml.objectives.F1.score')
@patch('evalml.pipelines.BinaryClassificationPipeline.fit')
@patch('evalml.pipelines.components.Estimator.predict')
def test_score_binary_objective_error(mock_predict, mock_fit, mock_objective_score, mock_encode, X_y_binary):
    mock_objective_score.side_effect = Exception('finna kabooom 💣')
    X, y = X_y_binary
    mock_predict.return_value = y
    mock_encode.return_value = y
    clf = make_mock_binary_pipeline()
    clf.fit(X, y)
    objective_names = ['f1', 'precision']
    # Using pytest.raises to make sure we error if an error is not thrown.
    with pytest.raises(PipelineScoreError):
        _ = clf.score(X, y, objective_names)
    try:
        _ = clf.score(X, y, objective_names)
    except PipelineScoreError as e:
        assert e.scored_successfully == {"Precision": 1.0}
        assert 'finna kabooom 💣' in e.message


@patch('evalml.pipelines.MulticlassClassificationPipeline._encode_targets')
@patch('evalml.objectives.F1Micro.score')
@patch('evalml.pipelines.MulticlassClassificationPipeline.fit')
@patch('evalml.pipelines.components.Estimator.predict')
def test_score_multiclass_objective_error(mock_predict, mock_fit, mock_objective_score, mock_encode, X_y_binary):
    mock_objective_score.side_effect = Exception('finna kabooom 💣')
    X, y = X_y_binary
    mock_predict.return_value = y
    mock_encode.return_value = y
    clf = make_mock_multiclass_pipeline()
    clf.fit(X, y)
    objective_names = ['f1_micro', 'precision_micro']
    # Using pytest.raises to make sure we error if an error is not thrown.
    with pytest.raises(PipelineScoreError):
        _ = clf.score(X, y, objective_names)
    try:
        _ = clf.score(X, y, objective_names)
    except PipelineScoreError as e:
        assert e.scored_successfully == {"Precision Micro": 1.0}
        assert 'finna kabooom 💣' in e.message
        assert "F1 Micro" in e.exceptions


def test_no_default_parameters():
    class MockComponent(Transformer):
        name = "Mock Component"
        hyperparameter_ranges = {
            'a': [0, 1, 2]
        }

        def __init__(self, a, b=1, c='2', random_state=0):
            self.a = a
            self.b = b
            self.c = c

    class TestPipeline(BinaryClassificationPipeline):
        component_graph = [MockComponent, 'Logistic Regression Classifier']

    with pytest.raises(ValueError, match="Error received when instantiating component *."):
        TestPipeline(parameters={})

    assert TestPipeline(parameters={'Mock Component': {'a': 42}})


def test_init_components_invalid_parameters():
    class TestPipeline(BinaryClassificationPipeline):
        component_graph = ['RF Classifier Select From Model', 'Logistic Regression Classifier']

    parameters = {
        'Logistic Regression Classifier': {
            "cool_parameter": "yes"
        }
    }

    with pytest.raises(ValueError, match="Error received when instantiating component"):
        TestPipeline(parameters=parameters)


def test_correct_parameters(logistic_regression_binary_pipeline_class):
    parameters = {
        'Imputer': {
            'categorical_impute_strategy': 'most_frequent',
            'numeric_impute_strategy': 'mean'
        },
        'Logistic Regression Classifier': {
            'penalty': 'l2',
            'C': 3.0,
        }
    }
    lr_pipeline = logistic_regression_binary_pipeline_class(parameters=parameters)
    assert lr_pipeline.estimator.random_state.get_state()[0] == np.random.RandomState(1).get_state()[0]
    assert lr_pipeline.estimator.parameters['C'] == 3.0
    assert lr_pipeline['Imputer'].parameters['categorical_impute_strategy'] == 'most_frequent'
    assert lr_pipeline['Imputer'].parameters['numeric_impute_strategy'] == 'mean'


def test_hyperparameters():
    class MockPipeline(BinaryClassificationPipeline):
        component_graph = ['Imputer', 'Random Forest Classifier']

    hyperparameters = {
        'Imputer': {
            "categorical_impute_strategy": ["most_frequent"],
            "numeric_impute_strategy": ["mean", "median", "most_frequent"]
        },
        'Random Forest Classifier': {
            "n_estimators": Integer(10, 1000),
            "max_depth": Integer(1, 10)
        }
    }

    assert MockPipeline.hyperparameters == hyperparameters
    assert MockPipeline(parameters={}).hyperparameters == hyperparameters


def test_hyperparameters_override():
    class MockPipelineOverRide(BinaryClassificationPipeline):
        component_graph = ['Imputer', 'Random Forest Classifier']

        custom_hyperparameters = {
            'Imputer': {
                "categorical_impute_strategy": ["most_frequent"],
                "numeric_impute_strategy": ["median", "most_frequent"]
            },
            'Random Forest Classifier': {
                "n_estimators": [1, 100, 200],
                "max_depth": [5]
            }
        }

    hyperparameters = {
        'Imputer': {
            "categorical_impute_strategy": ["most_frequent"],
            "numeric_impute_strategy": ["median", "most_frequent"]
        },
        'Random Forest Classifier': {
            "n_estimators": [1, 100, 200],
            "max_depth": [5]
        }
    }

    assert MockPipelineOverRide.hyperparameters == hyperparameters
    assert MockPipelineOverRide(parameters={}).hyperparameters == hyperparameters


def test_hyperparameters_none(dummy_classifier_estimator_class):
    MockEstimator = dummy_classifier_estimator_class

    class MockPipelineNone(BinaryClassificationPipeline):
        component_graph = [MockEstimator]

    assert MockPipelineNone.hyperparameters == {'Mock Classifier': {}}
    assert MockPipelineNone(parameters={}).hyperparameters == {'Mock Classifier': {}}


@patch('evalml.pipelines.components.Estimator.predict')
def test_score_with_objective_that_requires_predict_proba(mock_predict, dummy_regression_pipeline_class, X_y_binary):
    X, y = X_y_binary
    mock_predict.return_value = np.array([1] * 100)
    # Using pytest.raises to make sure we error if an error is not thrown.
    with pytest.raises(PipelineScoreError):
        dummy_regression_pipeline_class(parameters={}).score(X, y, ['precision', 'auc'])
    try:
        dummy_regression_pipeline_class(parameters={}).score(X, y, ['precision', 'auc'])
    except PipelineScoreError as e:
        assert "Invalid objective AUC specified for problem type Regression" in e.message
        assert "Invalid objective Precision specified for problem type Regression" in e.message
    mock_predict.assert_called()


def test_score_auc(X_y_binary, logistic_regression_binary_pipeline_class):
    X, y = X_y_binary
    lr_pipeline = logistic_regression_binary_pipeline_class(parameters={})
    lr_pipeline.fit(X, y)
    lr_pipeline.score(X, y, ['auc'])


def test_pipeline_summary():
    class MockPipelineWithoutEstimator(PipelineBase):
        component_graph = ["Imputer", "One Hot Encoder"]
    assert MockPipelineWithoutEstimator.summary == "Pipeline w/ Imputer + One Hot Encoder"

    class MockPipelineWithSingleComponent(PipelineBase):
        component_graph = ["Imputer"]
    assert MockPipelineWithSingleComponent.summary == "Pipeline w/ Imputer"

    class MockPipelineWithOnlyAnEstimator(PipelineBase):
        component_graph = ["Random Forest Classifier"]
    assert MockPipelineWithOnlyAnEstimator.summary == "Random Forest Classifier"

    class MockPipelineWithNoComponents(PipelineBase):
        component_graph = []
    assert MockPipelineWithNoComponents.summary == "Empty Pipeline"

    class MockPipeline(PipelineBase):
        component_graph = ["Imputer", "One Hot Encoder", "Random Forest Classifier"]
    assert MockPipeline.summary == "Random Forest Classifier w/ Imputer + One Hot Encoder"


def test_drop_columns_in_pipeline():
    class PipelineWithDropCol(BinaryClassificationPipeline):
        component_graph = ['Drop Columns Transformer', 'Imputer', 'Logistic Regression Classifier']

    parameters = {
        'Drop Columns Transformer': {
            'columns': ["column to drop"]
        },
        'Imputer': {
            "categorical_impute_strategy": "most_frequent",
            "numeric_impute_strategy": "mean"
        },
        'Logistic Regression Classifier': {
            'penalty': 'l2',
            'C': 3.0,
        }
    }
    pipeline_with_drop_col = PipelineWithDropCol(parameters=parameters)
    X = pd.DataFrame({"column to drop": [1, 0, 1, 3], "other col": [1, 2, 4, 1]})
    y = pd.Series([1, 0, 1, 0])
    pipeline_with_drop_col.fit(X, y)
    pipeline_with_drop_col.score(X, y, ['auc'])
    assert list(pipeline_with_drop_col.feature_importance["feature"]) == ['other col']


def test_clone_init(linear_regression_pipeline_class):
    parameters = {
        'Imputer': {
            "categorical_impute_strategy": "most_frequent",
            "numeric_impute_strategy": "mean",
        },
        'Linear Regressor': {
            'fit_intercept': True,
            'normalize': True,
        }
    }
    pipeline = linear_regression_pipeline_class(parameters=parameters)
    pipeline_clone = pipeline.clone()
    assert pipeline.parameters == pipeline_clone.parameters


def test_clone_random_state(linear_regression_pipeline_class):
    parameters = {
        'Imputer': {
            "categorical_impute_strategy": "most_frequent",
            "numeric_impute_strategy": "mean"
        },
        'Linear Regressor': {
            'fit_intercept': True,
            'normalize': True,
        }
    }
    pipeline = linear_regression_pipeline_class(parameters=parameters, random_state=np.random.RandomState(42))
    pipeline_clone = pipeline.clone(random_state=np.random.RandomState(42))
    assert pipeline_clone.random_state.randint(2**30) == pipeline.random_state.randint(2**30)

    pipeline = linear_regression_pipeline_class(parameters=parameters, random_state=2)
    pipeline_clone = pipeline.clone(random_state=2)
    assert pipeline_clone.random_state.randint(2**30) == pipeline.random_state.randint(2**30)


def test_clone_fitted(X_y_binary, logistic_regression_binary_pipeline_class):
    X, y = X_y_binary
    pipeline = logistic_regression_binary_pipeline_class(parameters={}, random_state=42)
    random_state_first_val = pipeline.random_state.randint(2**30)
    pipeline.fit(X, y)
    X_t = pipeline.predict_proba(X)

    pipeline_clone = pipeline.clone(random_state=42)
    assert pipeline_clone.random_state.randint(2**30) == random_state_first_val
    assert pipeline.parameters == pipeline_clone.parameters
    with pytest.raises(ComponentNotYetFittedError):
        pipeline_clone.predict(X)
    pipeline_clone.fit(X, y)
    X_t_clone = pipeline_clone.predict_proba(X)
    pd.testing.assert_frame_equal(X_t, X_t_clone)


def test_feature_importance_has_feature_names(X_y_binary, logistic_regression_binary_pipeline_class):
    X, y = X_y_binary
    col_names = ["col_{}".format(i) for i in range(len(X[0]))]
    X = pd.DataFrame(X, columns=col_names)
    parameters = {
        'Imputer': {
            "categorical_impute_strategy": "most_frequent",
            "numeric_impute_strategy": "mean"
        },
        'RF Classifier Select From Model': {
            "percent_features": 1.0,
            "number_features": len(X.columns),
            "n_estimators": 20
        },
        'Logistic Regression Classifier': {
            'penalty': 'l2',
            'C': 1.0,
        }
    }

    clf = logistic_regression_binary_pipeline_class(parameters=parameters)
    clf.fit(X, y)
    assert len(clf.feature_importance) == len(X.columns)
    assert not clf.feature_importance.isnull().all().all()
    assert sorted(clf.feature_importance["feature"]) == sorted(col_names)


def test_component_not_found(X_y_binary, logistic_regression_binary_pipeline_class):
    class FakePipeline(BinaryClassificationPipeline):
        component_graph = ['Imputer', 'One Hot Encoder', 'This Component Does Not Exist', 'Standard Scaler', 'Logistic Regression Classifier']
    with pytest.raises(MissingComponentError, match="Error recieved when retrieving class for component 'This Component Does Not Exist'"):
        FakePipeline(parameters={})


def test_get_default_parameters(logistic_regression_binary_pipeline_class):
    expected_defaults = {
        'Imputer': {
            'categorical_impute_strategy': 'most_frequent',
            'numeric_impute_strategy': 'mean',
            'categorical_fill_value': None,
            'numeric_fill_value': None
        },
        'One Hot Encoder': {
            'top_n': 10,
            'categories': None,
            'drop': None,
            'handle_unknown': 'ignore',
            'handle_missing': 'error'
        },
        'Logistic Regression Classifier': {
            'penalty': 'l2',
            'C': 1.0,
            'n_jobs': -1
        }
    }
    assert logistic_regression_binary_pipeline_class.default_parameters == expected_defaults


@pytest.mark.parametrize("problem_type", [ProblemTypes.BINARY, ProblemTypes.MULTICLASS])
@pytest.mark.parametrize("target_type", numeric_and_boolean_dtypes + categorical_dtypes)
def test_targets_data_types_classification_pipelines(problem_type, target_type, all_binary_pipeline_classes, all_multiclass_pipeline_classes):
    if problem_type == ProblemTypes.BINARY:
        objective = "log_loss_binary"
        pipeline_classes = all_binary_pipeline_classes
        X, y = load_breast_cancer()
        if target_type == "bool":
            y = y.map({"malignant": False, "benign": True})
    elif problem_type == ProblemTypes.MULTICLASS:
        if target_type == "bool":
            pytest.skip("Skipping test where problem type is multiclass but target type is boolean")
        objective = "log_loss_multi"
        pipeline_classes = all_multiclass_pipeline_classes
        X, y = load_wine()

    if target_type == "category":
        y = pd.Categorical(y)
    elif "int" in target_type:
        unique_vals = y.unique()
        y = y.map({unique_vals[i]: int(i) for i in range(len(unique_vals))})
    elif "float" in target_type:
        unique_vals = y.unique()
        y = y.map({unique_vals[i]: float(i) for i in range(len(unique_vals))})

    unique_vals = y.unique()
    for pipeline_class in pipeline_classes:
        pipeline = pipeline_class(parameters={})
        pipeline.fit(X, y)
        predictions = pipeline.predict(X, objective)
        assert set(predictions.unique()).issubset(unique_vals)
        predict_proba = pipeline.predict_proba(X)
        assert set(predict_proba.columns) == set(unique_vals)
