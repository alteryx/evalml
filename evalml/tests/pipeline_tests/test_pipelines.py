import os

import pytest

from evalml.model_types import ModelTypes
from evalml.objectives import FraudCost, Precision
from evalml.pipelines import LogisticRegressionPipeline, PipelineBase
from evalml.pipelines.components import (
    ComponentTypes,
    Estimator,
    LogisticRegressionClassifier,
    OneHotEncoder,
    RFClassifierSelectFromModel,
    SimpleImputer,
    StandardScaler
)
from evalml.pipelines.utils import (
    get_component_lists,
    list_model_types,
    load_pipeline,
    register_pipeline,
    register_pipeline_yaml,
    register_pipelines,
    save_pipeline
)
from evalml.problem_types import ProblemTypes


def test_list_model_types():
    assert set(list_model_types(ProblemTypes.BINARY)) == set([ModelTypes.RANDOM_FOREST, ModelTypes.XGBOOST, ModelTypes.LINEAR_MODEL])
    assert set(list_model_types(ProblemTypes.REGRESSION)) == set([ModelTypes.RANDOM_FOREST, ModelTypes.LINEAR_MODEL])


def test_get_pipelines():
    assert len(get_component_lists(problem_type=ProblemTypes.BINARY)) == 3
    assert len(get_component_lists(problem_type=ProblemTypes.BINARY, model_types=[ModelTypes.LINEAR_MODEL])) == 1
    assert len(get_component_lists(problem_type=ProblemTypes.REGRESSION)) == 2
    with pytest.raises(RuntimeError, match="Unrecognized model type for problem type"):
        get_component_lists(problem_type=ProblemTypes.REGRESSION, model_types=["random_forest", "xgboost"])


def test_serialization(X_y, tmpdir):
    X, y = X_y
    path = os.path.join(str(tmpdir), 'pipe.pkl')
    objective = Precision()

    pipeline = LogisticRegressionPipeline(objective=objective, penalty='l2', C=1.0, impute_strategy='mean', number_features=len(X[0]))
    pipeline.fit(X, y)
    save_pipeline(pipeline, path)
    assert pipeline.score(X, y) == load_pipeline(path).score(X, y)


@pytest.fixture
def pickled_pipeline_path(X_y, tmpdir):
    X, y = X_y
    path = os.path.join(str(tmpdir), 'pickled_pipe.pkl')
    MockPrecision = type('MockPrecision', (Precision,), {})
    objective = MockPrecision()
    pipeline = LogisticRegressionPipeline(objective=objective, penalty='l2', C=1.0, impute_strategy='mean', number_features=len(X[0]))
    pipeline.fit(X, y)
    save_pipeline(pipeline, path)
    return path


def test_load_pickled_pipeline_with_custom_objective(X_y, pickled_pipeline_path):
    X, y = X_y
    # checks that class is not defined before loading in pipeline
    with pytest.raises(NameError):
        MockPrecision()  # noqa: F821: ignore flake8's "undefined name" error
    objective = Precision()
    pipeline = LogisticRegressionPipeline(objective=objective, penalty='l2', C=1.0, impute_strategy='mean', number_features=len(X[0]))
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

    clf = LogisticRegressionPipeline(objective=objective, penalty='l2', C=1.0, impute_strategy='mean', number_features=len(X[0]), random_state=0)
    clf.fit(X, y)

    clf_1 = LogisticRegressionPipeline(objective=objective, penalty='l2', C=1.0, impute_strategy='mean', number_features=len(X[0]), random_state=0)
    clf_1.fit(X, y)

    assert clf_1.score(X, y) == clf.score(X, y)


def test_indexing(X_y):
    X, y = X_y
    clf = LogisticRegressionPipeline(objective='recall', penalty='l2', C=1.0, impute_strategy='mean', number_features=len(X[0]), random_state=0)
    clf.fit(X, y)

    assert isinstance(clf[0], OneHotEncoder)
    assert isinstance(clf['One Hot Encoder'], OneHotEncoder)

    setting_err_msg = 'Setting pipeline components is not supported.'
    with pytest.raises(NotImplementedError, match=setting_err_msg):
        clf[1] = OneHotEncoder()

    slicing_err_msg = 'Slicing pipelines is currently not supported.'
    with pytest.raises(NotImplementedError, match=slicing_err_msg):
        clf[:1]


def test_describe(X_y):
    X, y = X_y
    lrp = LogisticRegressionPipeline(objective='recall', penalty='l2', C=1.0, impute_strategy='mean', number_features=len(X[0]), random_state=0)
    assert lrp.describe(True) == {'C': 1.0, 'impute_strategy': 'mean', 'penalty': 'l2'}


def test_name(X_y):
    X, y = X_y
    clf = LogisticRegressionPipeline(objective='recall', penalty='l2', C=1.0, impute_strategy='mean', number_features=len(X[0]), random_state=0)
    assert clf.name == 'Logistic Regression Classifier w/ One Hot Encoder + Simple Imputer + Standard Scaler'


def test_estimator_not_last(X_y):
    X, y = X_y

    class MockLogisticRegressionPipeline(PipelineBase):
        name = "Mock Logistic Regression Pipeline"

        def __init__(self, objective, penalty, C, impute_strategy,
                     number_features, n_jobs=-1, random_state=0):
            imputer = SimpleImputer(impute_strategy=impute_strategy)
            enc = OneHotEncoder()
            scaler = StandardScaler()
            estimator = LogisticRegressionClassifier(random_state=random_state,
                                                     penalty=penalty,
                                                     C=C,
                                                     n_jobs=-1)
            super().__init__(objective=objective, component_list=[enc, imputer, estimator, scaler], n_jobs=n_jobs, random_state=random_state)

    err_msg = "A pipeline must have an Estimator as the last component in component_list."
    with pytest.raises(ValueError, match=err_msg):
        MockLogisticRegressionPipeline(objective='recall', penalty='l2', C=1.0, impute_strategy='mean', number_features=len(X[0]), random_state=0)


def test_multi_format_creation(X_y):
    X, y = X_y
    clf = PipelineBase('precision', component_list=['Simple Imputer', 'categorical_encoder', StandardScaler(), ComponentTypes.CLASSIFIER], n_jobs=-1, random_state=0)
    correct_components = [SimpleImputer, OneHotEncoder, StandardScaler, LogisticRegressionClassifier]
    for component, correct_components in zip(clf.component_list, correct_components):
        assert isinstance(component, correct_components)
    assert clf.model_type == ModelTypes.LINEAR_MODEL
    assert clf.problem_types == [ProblemTypes.BINARY, ProblemTypes.MULTICLASS]

    clf.fit(X, y)
    clf.score(X, y)
    assert not clf.feature_importances.isnull().all().all()


def test_multiple_feature_selectors(X_y):
    X, y = X_y
    clf = PipelineBase('precision', component_list=['Simple Imputer', 'categorical_encoder', ComponentTypes.FEATURE_SELECTION_CLASSIFIER, StandardScaler(), ComponentTypes.FEATURE_SELECTION_CLASSIFIER, ComponentTypes.CLASSIFIER], n_jobs=-1, random_state=0)
    correct_components = [SimpleImputer, OneHotEncoder, RFClassifierSelectFromModel, StandardScaler, RFClassifierSelectFromModel, LogisticRegressionClassifier]
    for component, correct_components in zip(clf.component_list, correct_components):
        assert isinstance(component, correct_components)
    assert clf.model_type == ModelTypes.LINEAR_MODEL
    assert clf.problem_types == [ProblemTypes.BINARY, ProblemTypes.MULTICLASS]

    clf.fit(X, y)
    clf.score(X, y)
    assert not clf.feature_importances.isnull().all().all()


def test_component_list_init(X_y):
    X, y = X_y
    component_list = ['Simple Imputer', 'Logistic Regression Classifier']
    pipeline = PipelineBase(objective='recall', component_list=component_list, n_jobs=-1, random_state=0)
    assert pipeline.model_type == ModelTypes.LINEAR_MODEL
    assert pipeline.problem_types == [ProblemTypes.BINARY, ProblemTypes.MULTICLASS]

    class Mock_Component(Estimator):
        name = "Mock_Component"
        problem_types = [ProblemTypes.BINARY]
        model_type = ModelTypes.LINEAR_MODEL

        def __init__(self, param_1, param_2, param_3):
            self.parameters = {
                'param_1': param_1,
                'param_2': param_2,
                'param_3': param_3
            }
            pass

    params = {
        'param_1': 0.1,
        'param_2': 0.2,
        'param_3': 0.03,
    }

    pipeline = PipelineBase(objective='recall', component_list=[Mock_Component], n_jobs=-1, random_state=0, **params)
    assert pipeline.model_type == ModelTypes.LINEAR_MODEL
    assert pipeline.problem_types == [ProblemTypes.BINARY]

    params = {
        'param_1': 0.1,
        'param_2': 0.2,
    }
    with pytest.raises(Exception):
        pipeline = PipelineBase(objective='recall', component_list=[Mock_Component], n_jobs=-1, random_state=0, **params)


def test_register_pipelines():
    pipeline = ['Simple Imputer', 'Random Forest Classifier']
    register_pipeline(pipeline)
    assert pipeline in get_component_lists('binary')

    pipelines = [['Simple Imputer', 'XGboost Classifier'], ['Simple Imputer', 'Logistic Regression Classifier']]
    register_pipelines(pipelines)
    for pipeline in pipelines:
        assert pipeline in get_component_lists('binary')


def test_yaml_path():
    dir_path = os.path.dirname(os.path.realpath(__file__))
    path = os.path.join(dir_path, 'test_custom_pipelines.yaml')
    register_pipeline_yaml(path)
    pipelines = [['Simple Imputer', 'XGBoost Classifier'], ['Simple Imputer', 'Logistic Regression Classifier']]
    for pipeline in pipelines:
        assert pipeline in get_component_lists('binary')
