import os

import pytest
from skopt.space import Real

from evalml.model_types import ModelTypes
from evalml.objectives import FraudCost, Precision
from evalml.pipelines import LogisticRegressionPipeline, PipelineBase
from evalml.pipelines.components import (
    LogisticRegressionClassifier,
    OneHotEncoder,
    RFClassifierSelectFromModel,
    SimpleImputer,
    StandardScaler,
    Transformer
)
from evalml.pipelines.utils import (
    get_pipelines,
    list_model_types,
    load_pipeline,
    save_pipeline
)
from evalml.problem_types import ProblemTypes


def test_list_model_types():
    assert set(list_model_types(ProblemTypes.BINARY)) == set([ModelTypes.RANDOM_FOREST, ModelTypes.XGBOOST, ModelTypes.LINEAR_MODEL, ModelTypes.CATBOOST])
    assert set(list_model_types(ProblemTypes.REGRESSION)) == set([ModelTypes.RANDOM_FOREST, ModelTypes.LINEAR_MODEL, ModelTypes.CATBOOST])


def test_get_pipelines():
    assert len(get_pipelines(problem_type=ProblemTypes.BINARY)) == 4
    assert len(get_pipelines(problem_type=ProblemTypes.BINARY, model_types=[ModelTypes.LINEAR_MODEL])) == 1
    assert len(get_pipelines(problem_type=ProblemTypes.MULTICLASS)) == 4
    assert len(get_pipelines(problem_type=ProblemTypes.REGRESSION)) == 3
    with pytest.raises(RuntimeError, match="Unrecognized model type for problem type"):
        get_pipelines(problem_type=ProblemTypes.REGRESSION, model_types=["random_forest", "xgboost"])
    with pytest.raises(KeyError):
        get_pipelines(problem_type="Not A Valid Problem Type")


@pytest.fixture
def lr_pipeline():
    objective = Precision()
    parameters = {
        'Simple Imputer': {
            'impute_strategy': 'median'
        },
        'Logistic Regression Classifier': {
            'penalty': 'l2',
            'C': 3.0,
            'random_state': 1
        }
    }

    return LogisticRegressionPipeline(objective=objective, parameters=parameters)


def test_required_fields():
    class TestPipelineWithComponentGraph(PipelineBase):
        component_graph = ['Logistic Regression']

    with pytest.raises(TypeError):
        TestPipelineWithComponentGraph(parameters={}, objective='precision')

    class TestPipelineWithProblemTypes(PipelineBase):
        component_graph = ['Logistic Regression']

    with pytest.raises(TypeError):
        TestPipelineWithProblemTypes(parameters={}, objective='precision')


def test_serialization(X_y, tmpdir, lr_pipeline):
    X, y = X_y
    path = os.path.join(str(tmpdir), 'pipe.pkl')
    pipeline = lr_pipeline
    pipeline.fit(X, y)
    save_pipeline(pipeline, path)
    assert pipeline.score(X, y) == load_pipeline(path).score(X, y)


@pytest.fixture
def pickled_pipeline_path(X_y, tmpdir, lr_pipeline):
    X, y = X_y
    path = os.path.join(str(tmpdir), 'pickled_pipe.pkl')
    MockPrecision = type('MockPrecision', (Precision,), {})
    pipeline = LogisticRegressionPipeline(objective=MockPrecision(), parameters=lr_pipeline.parameters)
    pipeline.fit(X, y)
    save_pipeline(pipeline, path)
    return path


def test_load_pickled_pipeline_with_custom_objective(X_y, pickled_pipeline_path, lr_pipeline):
    X, y = X_y
    # checks that class is not defined before loading in pipeline
    with pytest.raises(NameError):
        MockPrecision()  # noqa: F821: ignore flake8's "undefined name" error
    objective = Precision()
    pipeline = LogisticRegressionPipeline(objective=objective, parameters=lr_pipeline.parameters)
    pipeline.fit(X, y)
    assert load_pipeline(pickled_pipeline_path).score(X, y) == pipeline.score(X, y)


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
            'random_state': 1
        }
    }

    clf = LogisticRegressionPipeline(objective=objective, parameters=parameters)
    clf.fit(X, y)

    clf_1 = LogisticRegressionPipeline(objective=objective, parameters=parameters)
    clf_1.fit(X, y)

    assert clf_1.score(X, y) == clf.score(X, y)


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


def test_describe(X_y, lr_pipeline):
    X, y = X_y
    lrp = lr_pipeline
    assert lrp.describe(True) == {
        'Simple Imputer': {
            'impute_strategy': 'median'
        },
        'Logistic Regression Classifier': {
            'penalty': 'l2',
            'C': 3.0,
            'random_state': 1
        }
    }


def test_name():
    class TestNamePipeline(PipelineBase):
        component_graph = ['Logistic Regression Classifier']
        problem_types = ['binary']

    class TestDefinedNamePipeline(PipelineBase):
        _name = "Cool Logistic Regression"
        component_graph = ['Logistic Regression Classifier']
        problem_types = ['binary']

    assert TestNamePipeline.name == "Test Name Pipeline"
    assert TestDefinedNamePipeline.name == "Cool Logistic Regression"
    assert TestDefinedNamePipeline(parameters={}, objective='precision').name == "Cool Logistic Regression"


def test_summary(X_y, lr_pipeline):
    X, y = X_y
    clf = lr_pipeline
    assert clf.summary == 'Logistic Regression Classifier w/ One Hot Encoder + Simple Imputer + Standard Scaler'


def test_estimator_not_last(X_y):
    X, y = X_y

    parameters = {
        'Simple Imputer': {
            'impute_strategy': 'mean'
        },
        'Logistic Regression Classifier': {
            'penalty': 'l2',
            'C': 1.0,
            'random_state': 1
        }
    }

    class MockLogisticRegressionPipeline(PipelineBase):
        name = "Mock Logistic Regression Pipeline"
        problem_types = ['binary', 'multiclass']
        component_graph = ['One Hot Encoder', 'Simple Imputer', 'Logistic Regression Classifier', 'Standard Scaler']

        def __init__(self, objective, parameters):
            super().__init__(objective=objective,
                             parameters=parameters)

    err_msg = "A pipeline must have an Estimator as the last component in component_graph."
    with pytest.raises(ValueError, match=err_msg):
        MockLogisticRegressionPipeline(objective='recall', parameters=parameters)


def test_multi_format_creation(X_y):
    X, y = X_y

    class TestPipeline(PipelineBase):
        model_type = ModelTypes.LINEAR_MODEL
        component_graph = component_graph = ['Simple Imputer', 'One Hot Encoder', StandardScaler(), 'Logistic Regression Classifier']
        problem_types = ['binary', 'multiclass']

        hyperparameters = {
            "penalty": ["l2"],
            "C": Real(.01, 10),
            "impute_strategy": ["mean", "median", "most_frequent"],
        }

        def __init__(self, objective, parameters):
            super().__init__(objective=objective,
                             parameters=parameters)

    parameters = {
        'Simple Imputer': {
            'impute_strategy': 'mean'
        },
        'Logistic Regression Classifier': {
            'penalty': 'l2',
            'C': 1.0,
            'random_state': 1
        }
    }

    clf = TestPipeline(parameters=parameters, objective='precision')
    correct_components = [SimpleImputer, OneHotEncoder, StandardScaler, LogisticRegressionClassifier]
    for component, correct_components in zip(clf.component_graph, correct_components):
        assert isinstance(component, correct_components)
    assert clf.model_type == ModelTypes.LINEAR_MODEL
    assert clf.problem_types == [ProblemTypes.BINARY, ProblemTypes.MULTICLASS]

    clf.fit(X, y)
    clf.score(X, y)
    assert not clf.feature_importances.isnull().all().all()


def test_multiple_feature_selectors(X_y):
    X, y = X_y

    class TestPipeline(PipelineBase):
        model_type = ModelTypes.LINEAR_MODEL
        component_graph = ['Simple Imputer', 'One Hot Encoder', 'RF Classifier Select From Model', StandardScaler(), 'RF Classifier Select From Model', 'Logistic Regression Classifier']
        problem_types = ['binary', 'multiclass']

        hyperparameters = {
            "penalty": ["l2"],
            "C": Real(.01, 10),
            "impute_strategy": ["mean", "median", "most_frequent"],
        }

        def __init__(self, objective, parameters):
            super().__init__(objective=objective,
                             parameters=parameters)

    clf = TestPipeline(parameters={}, objective='precision')
    correct_components = [SimpleImputer, OneHotEncoder, RFClassifierSelectFromModel, StandardScaler, RFClassifierSelectFromModel, LogisticRegressionClassifier]
    for component, correct_components in zip(clf.component_graph, correct_components):
        assert isinstance(component, correct_components)
    assert clf.model_type == ModelTypes.LINEAR_MODEL
    assert clf.problem_types == [ProblemTypes.BINARY, ProblemTypes.MULTICLASS]

    clf.fit(X, y)
    clf.score(X, y)
    assert not clf.feature_importances.isnull().all().all()


def test_problem_types():
    class TestPipeline(PipelineBase):
        model_type = ModelTypes.LINEAR_MODEL
        component_graph = ['Logistic Regression Classifier']
        problem_types = ['binary', 'regression']

        def __init__(self, objective, parameters):
            super().__init__(objective=objective,
                             parameters=parameters)

    with pytest.raises(ValueError, match="not valid for this component graph. Valid problem types include *."):
        TestPipeline(parameters={}, objective='precision')


def test_no_default_parameters():
    class MockComponent(Transformer):
        name = "Mock Component"
        hyperparameter_ranges = {
            'a': [0, 1, 2]
        }

        def __init__(self, a, b=1, c='2',):
            self.a = a
            self.b = b
            self.c = c

    class TestPipeline(PipelineBase):
        component_graph = [MockComponent(a=0), 'Logistic Regression Classifier']
        problem_types = ['binary']

        def __init__(self, objective, parameters):
            super().__init__(objective=objective,
                             parameters=parameters)

    with pytest.raises(ValueError, match="Error received when instantiating component *."):
        TestPipeline(parameters={}, objective='precision')

    assert TestPipeline(parameters={'Mock Component': {'a': 42}}, objective='precision')


def test_init_components_invalid_parameters():
    class TestPipeline(PipelineBase):
        component_graph = ['RF Classifier Select From Model', 'Logistic Regression Classifier']
        problem_types = ['binary']

        def __init__(self, objective, parameters):
            super().__init__(objective=objective,
                             parameters=parameters)

    parameters = {
        'Logistic Regression Classifier': {
            "cool_parameter": "yes"
        }
    }

    with pytest.raises(ValueError, match="Error received when instantiating component"):
        TestPipeline(parameters=parameters, objective='precision')


def test_correct_parameters(lr_pipeline):
    lr_pipeline = lr_pipeline

    assert lr_pipeline.estimator.random_state == 1
    assert lr_pipeline.estimator.parameters['C'] == 3.0
    assert lr_pipeline['Simple Imputer'].parameters['impute_strategy'] == 'median'
