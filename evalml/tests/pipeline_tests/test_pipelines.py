import os
from importlib import import_module
from unittest.mock import patch

import numpy as np
import pytest
from skopt.space import Integer, Real

from evalml.exceptions import IllFormattedClassNameError
from evalml.model_family import ModelFamily
from evalml.objectives import FraudCost, Precision, Recall
from evalml.pipelines import (
    BinaryClassificationPipeline,
    LogisticRegressionBinaryPipeline,
    PipelineBase
)
from evalml.pipelines.components import (
    LogisticRegressionClassifier,
    OneHotEncoder,
    RFClassifierSelectFromModel,
    SimpleImputer,
    StandardScaler,
    Transformer
)
from evalml.pipelines.utils import (
    all_pipelines,
    get_pipelines,
    list_model_families
)
from evalml.problem_types import ProblemTypes


def test_list_model_families(has_minimal_dependencies):
    expected_model_families_binary = set([ModelFamily.RANDOM_FOREST, ModelFamily.LINEAR_MODEL, ModelFamily.EXTRA_TREES])
    expected_model_families_regression = set([ModelFamily.RANDOM_FOREST, ModelFamily.LINEAR_MODEL, ModelFamily.EXTRA_TREES])
    if not has_minimal_dependencies:
        expected_model_families_binary.add(ModelFamily.XGBOOST)
        expected_model_families_binary.add(ModelFamily.CATBOOST)
        expected_model_families_regression.add(ModelFamily.CATBOOST)
        expected_model_families_regression.add(ModelFamily.XGBOOST)
    assert set(list_model_families(ProblemTypes.BINARY)) == expected_model_families_binary
    assert set(list_model_families(ProblemTypes.REGRESSION)) == expected_model_families_regression


def test_all_pipelines(has_minimal_dependencies):
    if has_minimal_dependencies:
        assert len(all_pipelines()) == 9
    else:
        assert len(all_pipelines()) == 15


def make_mock_import_module(libs_to_blacklist):
    def _import_module(library):
        if library in libs_to_blacklist:
            raise ImportError("Cannot import {}; blacklisted by mock muahahaha".format(library))
        return import_module(library)
    return _import_module


@patch('importlib.import_module', make_mock_import_module({'xgboost', 'catboost'}))
def test_all_pipelines_core_dependencies_mock():
    assert len(all_pipelines()) == 9


def test_get_pipelines(has_minimal_dependencies):
    if has_minimal_dependencies:
        assert len(get_pipelines(problem_type=ProblemTypes.BINARY)) == 3
        assert len(get_pipelines(problem_type=ProblemTypes.BINARY, model_families=[ModelFamily.LINEAR_MODEL])) == 1
        assert len(get_pipelines(problem_type=ProblemTypes.MULTICLASS)) == 3
        assert len(get_pipelines(problem_type=ProblemTypes.REGRESSION)) == 3
    else:
        assert len(get_pipelines(problem_type=ProblemTypes.BINARY)) == 5
        assert len(get_pipelines(problem_type=ProblemTypes.BINARY, model_families=[ModelFamily.LINEAR_MODEL])) == 1
        assert len(get_pipelines(problem_type=ProblemTypes.MULTICLASS)) == 5
        assert len(get_pipelines(problem_type=ProblemTypes.REGRESSION)) == 5

    with pytest.raises(RuntimeError, match="Unrecognized model type for problem type"):
        get_pipelines(problem_type=ProblemTypes.REGRESSION, model_families=["random_forest", "none"])
    with pytest.raises(KeyError):
        get_pipelines(problem_type="Not A Valid Problem Type")


@patch('importlib.import_module', make_mock_import_module({'xgboost', 'catboost'}))
def test_get_pipelines_core_dependencies_mock():
    assert len(get_pipelines(problem_type=ProblemTypes.BINARY)) == 3
    assert len(get_pipelines(problem_type=ProblemTypes.BINARY, model_families=[ModelFamily.LINEAR_MODEL])) == 1
    assert len(get_pipelines(problem_type=ProblemTypes.MULTICLASS)) == 3
    assert len(get_pipelines(problem_type=ProblemTypes.REGRESSION)) == 3
    with pytest.raises(RuntimeError, match="Unrecognized model type for problem type"):
        get_pipelines(problem_type=ProblemTypes.REGRESSION, model_families=["random_forest", "none"])
    with pytest.raises(KeyError):
        get_pipelines(problem_type="Not A Valid Problem Type")


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

    assert isinstance(clf[0], OneHotEncoder)
    assert isinstance(clf['Simple Imputer'], SimpleImputer)

    setting_err_msg = 'Setting pipeline components is not supported.'
    with pytest.raises(NotImplementedError, match=setting_err_msg):
        clf[1] = OneHotEncoder()

    slicing_err_msg = 'Slicing pipelines is currently not supported.'
    with pytest.raises(NotImplementedError, match=slicing_err_msg):
        clf[:1]


def test_describe(X_y, capsys, lr_pipeline):
    X, y = X_y
    lrp = lr_pipeline
    lrp.describe()
    out, err = capsys.readouterr()
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
        'One Hot Encoder': {'top_n': 10},
        'Logistic Regression Classifier': {
            'penalty': 'l2',
            'C': 3.0
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
        component_graph = component_graph = ['Simple Imputer', 'One Hot Encoder', StandardScaler(), 'Logistic Regression Classifier']

        hyperparameters = {
            "penalty": ["l2"],
            "C": Real(.01, 10),
            "impute_strategy": ["mean", "median", "most_frequent"],
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
    assert not clf.feature_importances.isnull().all().all()


def test_multiple_feature_selectors(X_y):
    X, y = X_y

    class TestPipeline(BinaryClassificationPipeline):
        component_graph = ['Simple Imputer', 'One Hot Encoder', 'RF Classifier Select From Model', StandardScaler(), 'RF Classifier Select From Model', 'Logistic Regression Classifier']

        hyperparameters = {
            "penalty": ["l2"],
            "C": Real(.01, 10),
            "impute_strategy": ["mean", "median", "most_frequent"],
        }

    clf = TestPipeline(parameters={})
    correct_components = [SimpleImputer, OneHotEncoder, RFClassifierSelectFromModel, StandardScaler, RFClassifierSelectFromModel, LogisticRegressionClassifier]
    for component, correct_components in zip(clf.component_graph, correct_components):
        assert isinstance(component, correct_components)
    assert clf.model_family == ModelFamily.LINEAR_MODEL

    clf.fit(X, y)
    clf.score(X, y, ['precision'])
    assert not clf.feature_importances.isnull().all().all()


def test_problem_types():
    class TestPipeline(BinaryClassificationPipeline):
        component_graph = ['Random Forest Regressor']

    with pytest.raises(ValueError, match="not valid for this component graph. Valid problem types include *."):
        TestPipeline(parameters={})


def test_score_with_list_of_multiple_objectives(X_y):
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

    clf = LogisticRegressionBinaryPipeline(parameters=parameters)
    clf.fit(X, y)
    recall_name = Recall.name
    precision_name = Precision.name
    objective_names = [recall_name, precision_name]
    scores = clf.score(X, y, objective_names)
    assert len(scores.values()) == 2
    assert all(name in scores.keys() for name in objective_names)
    assert not any(np.isnan(val) for val in scores.values())


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
        component_graph = [MockComponent(a=0), 'Logistic Regression Classifier']

    with pytest.raises(ValueError, match="Error received when instantiating component *."):
        TestPipeline(parameters={})

    assert TestPipeline(parameters={'Mock Component': {'a': 42}})


def test_no_random_state_argument_in_component():
    class MockComponent(Transformer):
        name = "Mock Component"
        hyperparameter_ranges = {
            'a': [0, 1, 2]
        }

        def __init__(self, a, b=1, c='2'):
            self.a = a
            self.b = b
            self.c = c

    class TestPipeline(BinaryClassificationPipeline):
        component_graph = [MockComponent(a=0), 'Logistic Regression Classifier']

    with pytest.raises(ValueError, match="Error received when instantiating component *."):
        TestPipeline(parameters={'Mock Component': {'a': 42}}, random_state=0)


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
        "impute_strategy": ['mean', 'median', 'most_frequent'],
        "n_estimators": Integer(10, 1000),
        "max_depth": Integer(1, 32),
    }

    assert MockPipeline.hyperparameters == hyperparameters
    assert MockPipeline(parameters={}).hyperparameters == hyperparameters


def test_hyperparameters_override():
    class MockPipelineOverRide(BinaryClassificationPipeline):
        component_graph = ['Simple Imputer', 'Random Forest Classifier']

        custom_hyperparameters = {
            "impute_strategy": ['median'],
            "n_estimators": [1, 100, 200],
            "max_depth": [5]
        }

    hyperparameters = {
        "impute_strategy": ['median'],
        "n_estimators": [1, 100, 200],
        "max_depth": [5]
    }

    assert MockPipelineOverRide.hyperparameters == hyperparameters
    assert MockPipelineOverRide(parameters={}).hyperparameters == hyperparameters


def test_hyperparameters_none(dummy_estimator):
    class MockPipelineNone(BinaryClassificationPipeline):
        component_graph = [dummy_estimator()]

    assert MockPipelineNone.hyperparameters == {}
    assert MockPipelineNone(parameters={}).hyperparameters == {}


@patch('evalml.pipelines.components.Estimator.predict')
def test_score_with_objective_that_requires_predict_proba(mock_predict, dummy_regression_pipeline, X_y):
    X, y = X_y
    mock_predict.return_value = np.array([1] * 100)
    with pytest.raises(ValueError, match="Objective `AUC` does not support score_needs_proba"):
        dummy_regression_pipeline.score(X, y, ['recall', 'auc'])
    mock_predict.assert_called()


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
