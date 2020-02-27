import os

import pytest

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
            'impute_strategy': 'mean'
        },
        'Logistic Regression Classifier': {
            'penalty': 'l2',
            'C': 1.0,
            'random_state': 1
        }
    }

    return LogisticRegressionPipeline(objective=objective, parameters=parameters)


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

    assert isinstance(clf[0], SimpleImputer)
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
            'impute_strategy': 'mean'
        },
        'Logistic Regression Classifier': {
            'penalty': 'l2',
            'C': 1.0,
            'random_state': 1
        }
    }


def test_name(X_y, lr_pipeline):
    X, y = X_y
    clf = lr_pipeline
    assert clf.name == 'Logistic Regression Classifier w/ Simple Imputer + One Hot Encoder + Standard Scaler'


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
                             parameters=parameters,
                             component_graph=self.__class__.component_graph,
                             problem_types=self.__class__.problem_types)

    err_msg = "A pipeline must have an Estimator as the last component in component_graph."
    with pytest.raises(ValueError, match=err_msg):
        MockLogisticRegressionPipeline(objective='recall', parameters=parameters)


def test_multi_format_creation(X_y):
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

    clf = PipelineBase(objective='precision', component_graph=['Simple Imputer', 'One Hot Encoder', StandardScaler(), 'Logistic Regression Classifier'],
                       parameters=parameters, problem_types=['binary', 'multiclass'])
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
    clf = PipelineBase(objective='precision', component_graph=['Simple Imputer', 'One Hot Encoder', 'RF Classifier Select From Model', StandardScaler(), 'RF Classifier Select From Model', 'Logistic Regression Classifier'],
                       parameters={}, problem_types=['binary', 'multiclass'])
    correct_components = [SimpleImputer, OneHotEncoder, RFClassifierSelectFromModel, StandardScaler, RFClassifierSelectFromModel, LogisticRegressionClassifier]
    for component, correct_components in zip(clf.component_graph, correct_components):
        assert isinstance(component, correct_components)
    assert clf.model_type == ModelTypes.LINEAR_MODEL
    assert clf.problem_types == [ProblemTypes.BINARY, ProblemTypes.MULTICLASS]

    clf.fit(X, y)
    clf.score(X, y)
    assert not clf.feature_importances.isnull().all().all()


def test_n_jobs(X_y):
    with pytest.raises(ValueError, match='n_jobs must be an non-zero integer*.'):
        PipelineBase(objective='precision', component_graph=['Simple Imputer', 'One Hot Encoder', StandardScaler(), 'Logistic Regression Classifier'],
                     n_jobs='5', random_state=0, parameters={}, problem_types=['binary', 'multiclass'])

    with pytest.raises(ValueError, match='n_jobs must be an non-zero integer*.'):
        PipelineBase(objective='precision', component_graph=['Simple Imputer', 'One Hot Encoder', StandardScaler(), 'Logistic Regression Classifier'],
                     n_jobs=0, random_state=0, parameters={}, problem_types=['binary', 'multiclass'])

    assert PipelineBase(objective='precision', component_graph=['Simple Imputer', 'One Hot Encoder', StandardScaler(), 'Logistic Regression Classifier'],
                        n_jobs=-4, random_state=0, parameters={}, problem_types=['binary', 'multiclass'])

    assert PipelineBase(objective='precision', component_graph=['Simple Imputer', 'One Hot Encoder', StandardScaler(), 'Logistic Regression Classifier'],
                        n_jobs=4, random_state=0, parameters={}, problem_types=['binary', 'multiclass'])

    assert PipelineBase(objective='precision', component_graph=['Simple Imputer', 'One Hot Encoder', StandardScaler(), 'Logistic Regression Classifier'],
                        n_jobs=None, random_state=0, parameters={}, problem_types=['binary', 'multiclass'])


def test_problem_types():
    component_graph = ['Simple Imputer', 'Logistic Regression Classifier']

    with pytest.raises(ValueError, match="not valid for this component graph. Valid problem types include *."):
        PipelineBase(component_graph=component_graph, parameters={}, objective='precision', problem_types=['regression'])


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

    component_graph = [MockComponent(a=0)]
    with pytest.raises(ValueError, match="Please provide the required parameters for *."):
        PipelineBase(component_graph=component_graph, parameters={}, objective='precision', problem_types=['binary'])


def test_num_features():
    component_graph = ['RF Classifier Select From Model', 'Logistic Regression Classifier']
    pipeline = PipelineBase(component_graph=component_graph, parameters={}, objective="precision", problem_types=['binary'], number_features=100)
    assert pipeline.number_features == 100
    assert pipeline.component_graph[0]._component_obj.get_params()['max_features'] == 50  # default percent_features=0.5 so 100 * 0.5 == 50


def test_initiate_components():
    component_graph = ['RF Classifier Select From Model', 'Logistic Regression Classifier']
    parameters = {
        'Logistic Regression Classifier': {
            "cool_parameter": "yes"
        }
    }

    with pytest.raises(ValueError, match="Error received when instantiating component"):
        PipelineBase(component_graph=component_graph, parameters=parameters, objective='precision', problem_types=['binary'])

    component_graph = ['RF Classifier Select From Model', 'Logistic Regression Classifier']
    parameters = {
        'Logistic Regression Classifier': {
            "C": 100
        }
    }

    with pytest.raises(ValueError, match="Error received when instantiating component"):
        PipelineBase(component_graph=component_graph, parameters=parameters, objective='precision', problem_types=['binary'])
