import os
from importlib import import_module
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from skopt.space import Integer, Real

from evalml.exceptions import IllFormattedClassNameError
from evalml.model_family import ModelFamily
from evalml.objectives import FraudCost, Precision
from evalml.pipelines import (
    BinaryClassificationPipeline,
    LinearRegressionPipeline,
    LogisticRegressionBinaryPipeline,
    MulticlassClassificationPipeline,
    PipelineBase,
    RegressionPipeline
)
from evalml.pipelines.components import (
    CatBoostClassifier,
    CatBoostRegressor,
    DateTimeFeaturization,
    DropNullColumns,
    LinearRegressor,
    LogisticRegressionClassifier,
    OneHotEncoder,
    RandomForestClassifier,
    RandomForestRegressor,
    RFClassifierSelectFromModel,
    SimpleImputer,
    StandardScaler,
    Transformer
)
from evalml.pipelines.utils import (
    all_estimators,
    all_pipelines,
    get_estimators,
    get_pipelines,
    list_model_families,
    make_pipeline
)
from evalml.problem_types import ProblemTypes


def test_list_model_families(has_minimal_dependencies):
    expected_model_families_binary = set([ModelFamily.RANDOM_FOREST, ModelFamily.LINEAR_MODEL])
    expected_model_families_regression = set([ModelFamily.RANDOM_FOREST, ModelFamily.LINEAR_MODEL])
    if not has_minimal_dependencies:
        expected_model_families_binary.add(ModelFamily.XGBOOST)
        expected_model_families_binary.add(ModelFamily.CATBOOST)
        expected_model_families_regression.add(ModelFamily.CATBOOST)
        expected_model_families_regression.add(ModelFamily.XGBOOST)
    assert set(list_model_families(ProblemTypes.BINARY)) == expected_model_families_binary
    assert set(list_model_families(ProblemTypes.REGRESSION)) == expected_model_families_regression


def test_all_pipelines(has_minimal_dependencies):
    if has_minimal_dependencies:
        assert len(all_pipelines()) == 6
    else:
        assert len(all_pipelines()) == 12


def test_all_estimators(has_minimal_dependencies):
    if has_minimal_dependencies:
        assert len(all_estimators()) == 4
    else:
        assert len(all_estimators()) == 8


def make_mock_import_module(libs_to_exclude):
    def _import_module(library):
        if library in libs_to_exclude:
            raise ImportError("Cannot import {}; excluded by mock muahahaha".format(library))
        return import_module(library)
    return _import_module


@patch('importlib.import_module', make_mock_import_module({'xgboost', 'catboost'}))
def test_all_pipelines_core_dependencies_mock():
    assert len(all_pipelines()) == 6


@patch('importlib.import_module', make_mock_import_module({'xgboost', 'catboost'}))
def test_all_estimators_core_dependencies_mock():
    assert len(all_estimators()) == 4


def test_get_pipelines(has_minimal_dependencies):
    if has_minimal_dependencies:
        assert len(get_pipelines(problem_type=ProblemTypes.BINARY)) == 2
        assert len(get_pipelines(problem_type=ProblemTypes.BINARY, model_families=[ModelFamily.LINEAR_MODEL])) == 1
        assert len(get_pipelines(problem_type=ProblemTypes.MULTICLASS)) == 2
        assert len(get_pipelines(problem_type=ProblemTypes.REGRESSION)) == 2
    else:
        assert len(get_pipelines(problem_type=ProblemTypes.BINARY)) == 4
        assert len(get_pipelines(problem_type=ProblemTypes.BINARY, model_families=[ModelFamily.LINEAR_MODEL])) == 1
        assert len(get_pipelines(problem_type=ProblemTypes.MULTICLASS)) == 4
        assert len(get_pipelines(problem_type=ProblemTypes.REGRESSION)) == 4

    with pytest.raises(RuntimeError, match="Unrecognized model type for problem type"):
        get_pipelines(problem_type=ProblemTypes.REGRESSION, model_families=["random_forest", "none"])
    with pytest.raises(TypeError, match="model_families parameter is not a list."):
        get_pipelines(problem_type=ProblemTypes.REGRESSION, model_families='random_forest')
    with pytest.raises(KeyError):
        get_pipelines(problem_type="Not A Valid Problem Type")


def test_get_estimators(has_minimal_dependencies):
    if has_minimal_dependencies:
        assert len(get_estimators(problem_type=ProblemTypes.BINARY)) == 2
        assert len(get_estimators(problem_type=ProblemTypes.BINARY, model_families=[ModelFamily.LINEAR_MODEL])) == 1
        assert len(get_estimators(problem_type=ProblemTypes.MULTICLASS)) == 2
        assert len(get_estimators(problem_type=ProblemTypes.REGRESSION)) == 2
    else:
        assert len(get_estimators(problem_type=ProblemTypes.BINARY)) == 4
        assert len(get_estimators(problem_type=ProblemTypes.BINARY, model_families=[ModelFamily.LINEAR_MODEL])) == 1
        assert len(get_estimators(problem_type=ProblemTypes.MULTICLASS)) == 4
        assert len(get_estimators(problem_type=ProblemTypes.REGRESSION)) == 4

    assert len(get_estimators(problem_type=ProblemTypes.BINARY, model_families=[])) == 0
    assert len(get_estimators(problem_type=ProblemTypes.MULTICLASS, model_families=[])) == 0
    assert len(get_estimators(problem_type=ProblemTypes.REGRESSION, model_families=[])) == 0

    with pytest.raises(RuntimeError, match="Unrecognized model type for problem type"):
        get_estimators(problem_type=ProblemTypes.REGRESSION, model_families=["random_forest", "none"])
    with pytest.raises(TypeError, match="model_families parameter is not a list."):
        get_estimators(problem_type=ProblemTypes.REGRESSION, model_families='random_forest')
    with pytest.raises(KeyError):
        get_estimators(problem_type="Not A Valid Problem Type")


@patch('importlib.import_module', make_mock_import_module({'xgboost', 'catboost'}))
def test_get_pipelines_core_dependencies_mock():
    assert len(get_pipelines(problem_type=ProblemTypes.BINARY)) == 2
    assert len(get_pipelines(problem_type=ProblemTypes.BINARY, model_families=[ModelFamily.LINEAR_MODEL])) == 1
    assert len(get_pipelines(problem_type=ProblemTypes.MULTICLASS)) == 2
    assert len(get_pipelines(problem_type=ProblemTypes.REGRESSION)) == 2
    with pytest.raises(RuntimeError, match="Unrecognized model type for problem type"):
        get_pipelines(problem_type=ProblemTypes.REGRESSION, model_families=["random_forest", "none"])
    with pytest.raises(KeyError):
        get_pipelines(problem_type="Not A Valid Problem Type")


def test_make_pipeline():
    X = pd.DataFrame({"all_null": [np.nan, np.nan, np.nan, np.nan, np.nan],
                      "categorical": ["a", "b", "a", "c", "c"],
                      "some dates": pd.date_range('2000-02-03', periods=5, freq='W')})
    y = pd.Series([0, 0, 1, 2, 0])
    binary_pipeline = make_pipeline(X, y, LogisticRegressionClassifier, ProblemTypes.BINARY)
    assert isinstance(binary_pipeline, type(BinaryClassificationPipeline))
    assert binary_pipeline.component_graph == [DropNullColumns, SimpleImputer, DateTimeFeaturization, OneHotEncoder, StandardScaler, LogisticRegressionClassifier]

    binary_pipeline = make_pipeline(X, y, RandomForestClassifier, ProblemTypes.BINARY)
    assert isinstance(binary_pipeline, type(BinaryClassificationPipeline))
    assert binary_pipeline.component_graph == [DropNullColumns, SimpleImputer, DateTimeFeaturization, OneHotEncoder, RandomForestClassifier]

    multiclass_pipeline = make_pipeline(X, y, LogisticRegressionClassifier, ProblemTypes.MULTICLASS)
    assert isinstance(multiclass_pipeline, type(MulticlassClassificationPipeline))
    assert multiclass_pipeline.component_graph == [DropNullColumns, SimpleImputer, DateTimeFeaturization, OneHotEncoder, StandardScaler, LogisticRegressionClassifier]

    regression_pipeline = make_pipeline(X, y, RandomForestRegressor, ProblemTypes.REGRESSION)
    assert isinstance(regression_pipeline, type(RegressionPipeline))
    assert regression_pipeline.component_graph == [DropNullColumns, SimpleImputer, DateTimeFeaturization, OneHotEncoder, RandomForestRegressor]

    regression_pipeline = make_pipeline(X, y, LinearRegressor, ProblemTypes.REGRESSION)
    assert isinstance(regression_pipeline, type(RegressionPipeline))
    assert regression_pipeline.component_graph == [DropNullColumns, SimpleImputer, DateTimeFeaturization, OneHotEncoder, StandardScaler, LinearRegressor]


def test_make_pipelines_catboost(has_minimal_dependencies):
    if not has_minimal_dependencies:
        X = pd.DataFrame({"all_null": [np.nan, np.nan, np.nan, np.nan, np.nan],
                          "categorical": ["a", "b", "a", "c", "c"],
                          "some dates": pd.date_range('2000-02-03', periods=5, freq='W')})
        y = pd.Series([0, 0, 1, 2, 0])
        catboost_pipeline = make_pipeline(X, y, CatBoostClassifier, ProblemTypes.MULTICLASS)
        assert isinstance(catboost_pipeline, type(MulticlassClassificationPipeline))
        assert catboost_pipeline.component_graph == [DropNullColumns, SimpleImputer, DateTimeFeaturization, CatBoostClassifier]

        catboost_pipeline = make_pipeline(X, y, CatBoostRegressor, ProblemTypes.REGRESSION)
        assert isinstance(catboost_pipeline, type(RegressionPipeline))
        assert catboost_pipeline.component_graph == [DropNullColumns, SimpleImputer, DateTimeFeaturization, CatBoostRegressor]


def test_make_pipeline_no_nulls():
    X = pd.DataFrame({"numerical": [1, 2, 3, 1, 2],
                      "categorical": ["a", "b", "a", "c", "c"],
                      "some dates": pd.date_range('2000-02-03', periods=5, freq='W')})
    y = pd.Series([0, 0, 1, 2, 0])
    binary_pipeline = make_pipeline(X, y, LogisticRegressionClassifier, ProblemTypes.BINARY)
    assert isinstance(binary_pipeline, type(BinaryClassificationPipeline))
    assert binary_pipeline.component_graph == [SimpleImputer, DateTimeFeaturization, OneHotEncoder, StandardScaler, LogisticRegressionClassifier]

    multiclass_pipeline = make_pipeline(X, y, LogisticRegressionClassifier, ProblemTypes.MULTICLASS)
    assert isinstance(multiclass_pipeline, type(MulticlassClassificationPipeline))
    assert multiclass_pipeline.component_graph == [SimpleImputer, DateTimeFeaturization, OneHotEncoder, StandardScaler, LogisticRegressionClassifier]

    regression_pipeline = make_pipeline(X, y, RandomForestRegressor, ProblemTypes.REGRESSION)
    assert isinstance(regression_pipeline, type(RegressionPipeline))
    assert regression_pipeline.component_graph == [SimpleImputer, DateTimeFeaturization, OneHotEncoder, RandomForestRegressor]


def test_make_pipeline_no_datetimes():
    X = pd.DataFrame({"numerical": [1, 2, 3, 1, 2],
                      "categorical": ["a", "b", "a", "c", "c"],
                      "all_null": [np.nan, np.nan, np.nan, np.nan, np.nan]})
    y = pd.Series([0, 0, 1, 2, 0])
    binary_pipeline = make_pipeline(X, y, LogisticRegressionClassifier, ProblemTypes.BINARY)
    assert isinstance(binary_pipeline, type(BinaryClassificationPipeline))
    assert binary_pipeline.component_graph == [DropNullColumns, SimpleImputer, OneHotEncoder, StandardScaler, LogisticRegressionClassifier]

    multiclass_pipeline = make_pipeline(X, y, LogisticRegressionClassifier, ProblemTypes.MULTICLASS)
    assert isinstance(multiclass_pipeline, type(MulticlassClassificationPipeline))
    assert multiclass_pipeline.component_graph == [DropNullColumns, SimpleImputer, OneHotEncoder, StandardScaler, LogisticRegressionClassifier]

    regression_pipeline = make_pipeline(X, y, RandomForestRegressor, ProblemTypes.REGRESSION)
    assert isinstance(regression_pipeline, type(RegressionPipeline))
    assert regression_pipeline.component_graph == [DropNullColumns, SimpleImputer, OneHotEncoder, RandomForestRegressor]


def test_make_pipeline_no_column_names():
    X = pd.DataFrame([[1, "a", np.nan], [2, "b", np.nan], [5, "b", np.nan]])
    y = pd.Series([0, 0, 1])
    binary_pipeline = make_pipeline(X, y, LogisticRegressionClassifier, ProblemTypes.BINARY)
    assert isinstance(binary_pipeline, type(BinaryClassificationPipeline))
    assert binary_pipeline.component_graph == [DropNullColumns, SimpleImputer, OneHotEncoder, StandardScaler, LogisticRegressionClassifier]

    multiclass_pipeline = make_pipeline(X, y, LogisticRegressionClassifier, ProblemTypes.MULTICLASS)
    assert isinstance(multiclass_pipeline, type(MulticlassClassificationPipeline))
    assert multiclass_pipeline.component_graph == [DropNullColumns, SimpleImputer, OneHotEncoder, StandardScaler, LogisticRegressionClassifier]

    regression_pipeline = make_pipeline(X, y, RandomForestRegressor, ProblemTypes.REGRESSION)
    assert isinstance(regression_pipeline, type(RegressionPipeline))
    assert regression_pipeline.component_graph == [DropNullColumns, SimpleImputer, OneHotEncoder, RandomForestRegressor]


def test_make_pipeline_numpy_input():
    X = np.array([[1, 2, 0, np.nan], [2, 2, 1, np.nan], [5, 1, np.nan, np.nan]])
    y = np.array([0, 0, 1, 0])
    binary_pipeline = make_pipeline(X, y, LogisticRegressionClassifier, ProblemTypes.BINARY)
    assert isinstance(binary_pipeline, type(BinaryClassificationPipeline))
    assert binary_pipeline.component_graph == [DropNullColumns, SimpleImputer, StandardScaler, LogisticRegressionClassifier]

    multiclass_pipeline = make_pipeline(X, y, LogisticRegressionClassifier, ProblemTypes.MULTICLASS)
    assert isinstance(multiclass_pipeline, type(MulticlassClassificationPipeline))
    assert multiclass_pipeline.component_graph == [DropNullColumns, SimpleImputer, StandardScaler, LogisticRegressionClassifier]

    regression_pipeline = make_pipeline(X, y, RandomForestRegressor, ProblemTypes.REGRESSION)
    assert isinstance(regression_pipeline, type(RegressionPipeline))
    assert regression_pipeline.component_graph == [DropNullColumns, SimpleImputer, RandomForestRegressor]


def test_make_pipeline_problem_type_mismatch():
    with pytest.raises(ValueError, match=f"{LogisticRegressionClassifier.name} is not a valid estimator for problem type"):
        make_pipeline(pd.DataFrame(), pd.Series(), LogisticRegressionClassifier, ProblemTypes.REGRESSION)
    with pytest.raises(ValueError, match=f"{LinearRegressor.name} is not a valid estimator for problem type"):
        make_pipeline(pd.DataFrame(), pd.Series(), LinearRegressor, ProblemTypes.MULTICLASS)
    with pytest.raises(ValueError, match=f"{Transformer.name} is not a valid estimator for problem type"):
        make_pipeline(pd.DataFrame(), pd.Series(), Transformer, ProblemTypes.MULTICLASS)


@pytest.fixture
def lr_pipeline():
    parameters = {
        'Simple Imputer': {
            'impute_strategy': 'median'
        },
        'Logistic Regression Classifier': {
            'penalty': 'l2',
            'C': 3.0,
        }
    }
    return LogisticRegressionBinaryPipeline(parameters=parameters, random_state=42)


def test_required_fields():
    class TestPipelineWithoutComponentGraph(PipelineBase):
        pass

    with pytest.raises(TypeError):
        TestPipelineWithoutComponentGraph(parameters={})


def test_serialization(X_y, tmpdir, lr_pipeline):
    X, y = X_y
    path = os.path.join(str(tmpdir), 'pipe.pkl')
    pipeline = lr_pipeline
    pipeline.fit(X, y)
    pipeline.save(path)
    assert pipeline.score(X, y, ['precision']) == PipelineBase.load(path).score(X, y, ['precision'])


@pytest.fixture
def pickled_pipeline_path(X_y, tmpdir, lr_pipeline):
    X, y = X_y
    path = os.path.join(str(tmpdir), 'pickled_pipe.pkl')
    pipeline = LogisticRegressionBinaryPipeline(parameters=lr_pipeline.parameters)
    pipeline.fit(X, y)
    pipeline.save(path)
    return path


def test_load_pickled_pipeline_with_custom_objective(X_y, pickled_pipeline_path, lr_pipeline):
    X, y = X_y
    # checks that class is not defined before loading in pipeline
    with pytest.raises(NameError):
        MockPrecision()  # noqa: F821: ignore flake8's "undefined name" error
    objective = Precision()
    pipeline = LogisticRegressionBinaryPipeline(parameters=lr_pipeline.parameters)
    pipeline.fit(X, y)
    assert PipelineBase.load(pickled_pipeline_path).score(X, y, [objective]) == pipeline.score(X, y, [objective])


def test_reproducibility(X_y):
    X, y = X_y
    objective = FraudCost(
        retry_percentage=.5,
        interchange_fee=.02,
        fraud_payout_percentage=.75,
        amount_col=10
    )

    parameters = {
        'Simple Imputer': {
            'impute_strategy': 'mean'
        },
        'Logistic Regression Classifier': {
            'penalty': 'l2',
            'C': 1.0,
        }
    }

    clf = LogisticRegressionBinaryPipeline(parameters=parameters)
    clf.fit(X, y)

    clf_1 = LogisticRegressionBinaryPipeline(parameters=parameters)
    clf_1.fit(X, y)

    assert clf_1.score(X, y, [objective]) == clf.score(X, y, [objective])


def test_indexing(X_y, lr_pipeline):
    X, y = X_y
    clf = lr_pipeline
    clf.fit(X, y)

    assert isinstance(clf[1], OneHotEncoder)
    assert isinstance(clf['Simple Imputer'], SimpleImputer)

    setting_err_msg = 'Setting pipeline components is not supported.'
    with pytest.raises(NotImplementedError, match=setting_err_msg):
        clf[1] = OneHotEncoder()

    slicing_err_msg = 'Slicing pipelines is currently not supported.'
    with pytest.raises(NotImplementedError, match=slicing_err_msg):
        clf[:1]


def test_describe(X_y, caplog, lr_pipeline):
    X, y = X_y
    lrp = lr_pipeline
    lrp.describe()
    out = caplog.text
    assert "Logistic Regression Binary Pipeline" in out
    assert "Problem Type: Binary Classification" in out
    assert "Model Family: Linear" in out

    for component in lrp.component_graph:
        if component.hyperparameter_ranges:
            for parameter in component.hyperparameter_ranges:
                assert parameter in out
        assert component.name in out


def test_parameters(X_y, lr_pipeline):
    X, y = X_y
    lrp = lr_pipeline
    params = {
        'Simple Imputer': {
            'impute_strategy': 'median',
            'fill_value': None
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
    assert params == lrp.parameters


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


def test_estimator_not_last(X_y):
    X, y = X_y

    parameters = {
        'Simple Imputer': {
            'impute_strategy': 'mean'
        },
        'Logistic Regression Classifier': {
            'penalty': 'l2',
            'C': 1.0,
        }
    }

    class MockLogisticRegressionBinaryPipeline(BinaryClassificationPipeline):
        name = "Mock Logistic Regression Pipeline"
        component_graph = ['One Hot Encoder', 'Simple Imputer', 'Logistic Regression Classifier', 'Standard Scaler']

    err_msg = "A pipeline must have an Estimator as the last component in component_graph."
    with pytest.raises(ValueError, match=err_msg):
        MockLogisticRegressionBinaryPipeline(parameters=parameters)


def test_multi_format_creation(X_y):
    X, y = X_y

    class TestPipeline(BinaryClassificationPipeline):
        component_graph = component_graph = ['Simple Imputer', 'One Hot Encoder', StandardScaler, 'Logistic Regression Classifier']

        hyperparameters = {
            'Simple Imputer': {
                "impute_strategy": ["mean", "median", "most_frequent"]
            },
            'Logistic Regression Classifier': {
                "penalty": ["l2"],
                "C": Real(.01, 10)
            }
        }

    parameters = {
        'Simple Imputer': {
            'impute_strategy': 'mean'
        },
        'Logistic Regression Classifier': {
            'penalty': 'l2',
            'C': 1.0,
        }
    }

    clf = TestPipeline(parameters=parameters)
    correct_components = [SimpleImputer, OneHotEncoder, StandardScaler, LogisticRegressionClassifier]
    for component, correct_components in zip(clf.component_graph, correct_components):
        assert isinstance(component, correct_components)
    assert clf.model_family == ModelFamily.LINEAR_MODEL

    clf.fit(X, y)
    clf.score(X, y, ['precision'])
    assert not clf.feature_importance.isnull().all().all()


def test_multiple_feature_selectors(X_y):
    X, y = X_y

    class TestPipeline(BinaryClassificationPipeline):
        component_graph = ['Simple Imputer', 'One Hot Encoder', 'RF Classifier Select From Model', StandardScaler, 'RF Classifier Select From Model', 'Logistic Regression Classifier']

        hyperparameters = {
            'Simple Imputer': {
                "impute_strategy": ["mean", "median", "most_frequent"]
            },
            'Logistic Regression Classifier': {
                "penalty": ["l2"],
                "C": Real(.01, 10)
            }
        }

    clf = TestPipeline(parameters={})
    correct_components = [SimpleImputer, OneHotEncoder, RFClassifierSelectFromModel, StandardScaler, RFClassifierSelectFromModel, LogisticRegressionClassifier]
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
def test_score_regression_single(mock_predict, mock_fit, X_y):
    X, y = X_y
    mock_predict.return_value = y
    clf = make_mock_regression_pipeline()
    clf.fit(X, y)
    objective_names = ['r2']
    scores = clf.score(X, y, objective_names)
    mock_predict.assert_called()
    assert scores == {'R2': 1.0}


@patch('evalml.pipelines.BinaryClassificationPipeline.fit')
@patch('evalml.pipelines.BinaryClassificationPipeline.predict')
def test_score_binary_single(mock_predict, mock_fit, X_y):
    X, y = X_y
    mock_predict.return_value = y
    clf = make_mock_binary_pipeline()
    clf.fit(X, y)
    objective_names = ['f1']
    scores = clf.score(X, y, objective_names)
    mock_predict.assert_called()
    assert scores == {'F1': 1.0}


@patch('evalml.pipelines.MulticlassClassificationPipeline.fit')
@patch('evalml.pipelines.MulticlassClassificationPipeline.predict')
def test_score_multiclass_single(mock_predict, mock_fit, X_y):
    X, y = X_y
    mock_predict.return_value = y
    clf = make_mock_multiclass_pipeline()
    clf.fit(X, y)
    objective_names = ['f1_micro']
    scores = clf.score(X, y, objective_names)
    mock_predict.assert_called()
    assert scores == {'F1 Micro': 1.0}


@patch('evalml.pipelines.RegressionPipeline.fit')
@patch('evalml.pipelines.RegressionPipeline.predict')
def test_score_regression_list(mock_predict, mock_fit, X_y):
    X, y = X_y
    mock_predict.return_value = y
    clf = make_mock_regression_pipeline()
    clf.fit(X, y)
    objective_names = ['r2', 'mse']
    scores = clf.score(X, y, objective_names)
    mock_predict.assert_called()
    assert scores == {'R2': 1.0, 'MSE': 0.0}


@patch('evalml.pipelines.BinaryClassificationPipeline.fit')
@patch('evalml.pipelines.BinaryClassificationPipeline.predict')
def test_score_binary_list(mock_predict, mock_fit, X_y):
    X, y = X_y
    mock_predict.return_value = y
    clf = make_mock_binary_pipeline()
    clf.fit(X, y)
    objective_names = ['f1', 'precision']
    scores = clf.score(X, y, objective_names)
    mock_predict.assert_called()
    assert scores == {'F1': 1.0, 'Precision': 1.0}


@patch('evalml.pipelines.MulticlassClassificationPipeline.fit')
@patch('evalml.pipelines.MulticlassClassificationPipeline.predict')
def test_score_multi_list(mock_predict, mock_fit, X_y):
    X, y = X_y
    mock_predict.return_value = y
    clf = make_mock_multiclass_pipeline()
    clf.fit(X, y)
    objective_names = ['f1_micro', 'precision_micro']
    scores = clf.score(X, y, objective_names)
    mock_predict.assert_called()
    assert scores == {'F1 Micro': 1.0, 'Precision Micro': 1.0}


@patch('evalml.objectives.R2.score')
@patch('evalml.pipelines.RegressionPipeline.fit')
@patch('evalml.pipelines.RegressionPipeline.predict')
def test_score_regression_objective_error(mock_predict, mock_fit, mock_objective_score, X_y):
    mock_objective_score.side_effect = Exception('finna kabooom 💣')
    X, y = X_y
    mock_predict.return_value = y
    clf = make_mock_regression_pipeline()
    clf.fit(X, y)
    objective_names = ['r2', 'mse']
    scores = clf.score(X, y, objective_names)
    mock_predict.assert_called()
    assert scores == {'R2': np.nan, 'MSE': 0.0}


@patch('evalml.objectives.F1.score')
@patch('evalml.pipelines.BinaryClassificationPipeline.fit')
@patch('evalml.pipelines.BinaryClassificationPipeline.predict')
def test_score_binary_objective_error(mock_predict, mock_fit, mock_objective_score, X_y):
    mock_objective_score.side_effect = Exception('finna kabooom 💣')
    X, y = X_y
    mock_predict.return_value = y
    clf = make_mock_binary_pipeline()
    clf.fit(X, y)
    objective_names = ['f1', 'precision']
    scores = clf.score(X, y, objective_names)
    mock_predict.assert_called()
    assert scores == {'F1': np.nan, 'Precision': 1.0}


@patch('evalml.objectives.F1Micro.score')
@patch('evalml.pipelines.MulticlassClassificationPipeline.fit')
@patch('evalml.pipelines.MulticlassClassificationPipeline.predict')
def test_score_multiclass_objective_error(mock_predict, mock_fit, mock_objective_score, X_y):
    mock_objective_score.side_effect = Exception('finna kabooom 💣')
    X, y = X_y
    mock_predict.return_value = y
    clf = make_mock_multiclass_pipeline()
    clf.fit(X, y)
    objective_names = ['f1_micro', 'precision_micro']
    scores = clf.score(X, y, objective_names)
    mock_predict.assert_called()
    assert scores == {'F1 Micro': np.nan, 'Precision Micro': 1.0}


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


def test_correct_parameters(lr_pipeline):
    lr_pipeline = lr_pipeline

    assert lr_pipeline.estimator.random_state.get_state()[0] == np.random.RandomState(1).get_state()[0]
    assert lr_pipeline.estimator.parameters['C'] == 3.0
    assert lr_pipeline['Simple Imputer'].parameters['impute_strategy'] == 'median'


def test_hyperparameters():
    class MockPipeline(BinaryClassificationPipeline):
        component_graph = ['Simple Imputer', 'Random Forest Classifier']

    hyperparameters = {
        'Simple Imputer': {
            "impute_strategy": ['mean', 'median', 'most_frequent'],
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
        component_graph = ['Simple Imputer', 'Random Forest Classifier']

        custom_hyperparameters = {
            'Simple Imputer': {
                "impute_strategy": ['median']
            },
            'Random Forest Classifier': {
                "n_estimators": [1, 100, 200],
                "max_depth": [5]
            }
        }

    hyperparameters = {
        'Simple Imputer': {
            "impute_strategy": ['median']
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
def test_score_with_objective_that_requires_predict_proba(mock_predict, dummy_regression_pipeline, X_y):
    X, y = X_y
    mock_predict.return_value = np.array([1] * 100)
    with pytest.raises(ValueError, match="Objective `AUC` does not support score_needs_proba"):
        dummy_regression_pipeline.score(X, y, ['precision', 'auc'])
    mock_predict.assert_called()


def test_score_auc(X_y, lr_pipeline):
    X, y = X_y
    lr_pipeline.fit(X, y)
    lr_pipeline.score(X, y, ['auc'])


def test_pipeline_summary():
    class MockPipelineWithoutEstimator(PipelineBase):
        component_graph = ["Simple Imputer", "One Hot Encoder"]
    assert MockPipelineWithoutEstimator.summary == "Pipeline w/ Simple Imputer + One Hot Encoder"

    class MockPipelineWithSingleComponent(PipelineBase):
        component_graph = ["Simple Imputer"]
    assert MockPipelineWithSingleComponent.summary == "Pipeline w/ Simple Imputer"

    class MockPipelineWithOnlyAnEstimator(PipelineBase):
        component_graph = ["Random Forest Classifier"]
    assert MockPipelineWithOnlyAnEstimator.summary == "Random Forest Classifier"

    class MockPipelineWithNoComponents(PipelineBase):
        component_graph = []
    assert MockPipelineWithNoComponents.summary == "Empty Pipeline"

    class MockPipeline(PipelineBase):
        component_graph = ["Simple Imputer", "One Hot Encoder", "Random Forest Classifier"]
    assert MockPipeline.summary == "Random Forest Classifier w/ Simple Imputer + One Hot Encoder"


def test_drop_columns_in_pipeline():
    class PipelineWithDropCol(BinaryClassificationPipeline):
        component_graph = ['Drop Columns Transformer', 'Simple Imputer', 'Logistic Regression Classifier']

    parameters = {
        'Drop Columns Transformer': {
            'columns': ["column to drop"]
        },
        'Simple Imputer': {
            'impute_strategy': 'median'
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


def test_clone_init():
    parameters = {
        'Simple Imputer': {
            'impute_strategy': 'most_frequent'
        },
        'Linear Regressor': {
            'fit_intercept': True,
            'normalize': True,
        }
    }
    pipeline = LinearRegressionPipeline(parameters=parameters)
    pipeline_clone = pipeline.clone()
    assert pipeline.parameters == pipeline_clone.parameters


def test_clone_random_state():
    parameters = {
        'Simple Imputer': {
            'impute_strategy': 'most_frequent'
        },
        'Linear Regressor': {
            'fit_intercept': True,
            'normalize': True,
        }
    }
    pipeline = LinearRegressionPipeline(parameters=parameters, random_state=np.random.RandomState(42))
    pipeline_clone = pipeline.clone(random_state=np.random.RandomState(42))
    assert pipeline_clone.random_state.randint(2**30) == pipeline.random_state.randint(2**30)

    pipeline = LinearRegressionPipeline(parameters=parameters, random_state=2)
    pipeline_clone = pipeline.clone(random_state=2)
    assert pipeline_clone.random_state.randint(2**30) == pipeline.random_state.randint(2**30)


def test_clone_fitted(X_y, lr_pipeline):
    X, y = X_y
    pipeline = lr_pipeline
    random_state_first_val = pipeline.random_state.randint(2**30)
    pipeline.fit(X, y)
    X_t = pipeline.predict_proba(X)

    pipeline_clone = pipeline.clone(random_state=42)
    assert pipeline_clone.random_state.randint(2**30) == random_state_first_val
    assert pipeline.parameters == pipeline_clone.parameters
    with pytest.raises(RuntimeError):
        pipeline_clone.predict(X)
    pipeline_clone.fit(X, y)
    X_t_clone = pipeline_clone.predict_proba(X)
    pd.testing.assert_frame_equal(X_t, X_t_clone)


@pytest.mark.parametrize("cls", all_pipelines())
def test_pipeline_default_parameters(cls):

    assert cls.default_parameters == cls({}).parameters, f"{cls.__name__}'s default parameters don't match __init__."
