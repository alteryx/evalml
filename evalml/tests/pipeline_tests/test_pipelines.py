import os

import pytest

from evalml.model_types import ModelTypes
from evalml.objectives import FraudCost, Precision
from evalml.pipelines import LogisticRegressionBinaryPipeline, PipelineBase
from evalml.pipelines.components import (
    LogisticRegressionClassifier,
    OneHotEncoder,
    RFClassifierSelectFromModel,
    SimpleImputer,
    StandardScaler
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


def test_serialization(X_y, tmpdir):
    X, y = X_y
    path = os.path.join(str(tmpdir), 'pipe.pkl')
    objective = Precision()

    pipeline = LogisticRegressionBinaryPipeline(penalty='l2', C=1.0, impute_strategy='mean', number_features=len(X[0]))
    pipeline.fit(X, y, objective)
    save_pipeline(pipeline, path)
    assert pipeline.score(X, y, [objective]) == load_pipeline(path).score(X, y, [objective])


@pytest.fixture
def pickled_pipeline_path(X_y, tmpdir):
    X, y = X_y
    path = os.path.join(str(tmpdir), 'pickled_pipe.pkl')
    MockPrecision = type('MockPrecision', (Precision,), {})
    objective = MockPrecision()
    pipeline = LogisticRegressionBinaryPipeline(penalty='l2', C=1.0, impute_strategy='mean', number_features=len(X[0]))
    pipeline.fit(X, y, objective)
    save_pipeline(pipeline, path)
    return path


def test_load_pickled_pipeline_with_custom_objective(X_y, pickled_pipeline_path):
    X, y = X_y
    # checks that class is not defined before loading in pipeline
    with pytest.raises(NameError):
        MockPrecision()  # noqa: F821: ignore flake8's "undefined name" error
    objective = Precision()
    pipeline = LogisticRegressionBinaryPipeline(penalty='l2', C=1.0, impute_strategy='mean', number_features=len(X[0]))
    pipeline.fit(X, y, 'precision')
    assert load_pipeline(pickled_pipeline_path).score(X, y, [objective]) == pipeline.score(X, y, [objective])


def test_reproducibility(X_y):
    X, y = X_y
    objective = FraudCost(
        retry_percentage=.5,
        interchange_fee=.02,
        fraud_payout_percentage=.75,
        amount_col=10
    )

    clf = LogisticRegressionBinaryPipeline(penalty='l2', C=1.0, impute_strategy='mean', number_features=len(X[0]), random_state=0)
    clf.fit(X, y, objective)

    clf_1 = LogisticRegressionBinaryPipeline(penalty='l2', C=1.0, impute_strategy='mean', number_features=len(X[0]), random_state=0)
    clf_1.fit(X, y, objective)

    assert clf_1.score(X, y, [objective]) == clf.score(X, y, [objective])


def test_indexing(X_y):
    X, y = X_y
    clf = LogisticRegressionBinaryPipeline(penalty='l2', C=1.0, impute_strategy='mean', number_features=len(X[0]), random_state=0)
    clf.fit(X, y, 'recall')

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
    lrp = LogisticRegressionBinaryPipeline(penalty='l2', C=1.0, impute_strategy='mean', number_features=len(X[0]), random_state=0)
    assert lrp.describe(True) == {'C': 1.0, 'impute_strategy': 'mean', 'penalty': 'l2'}


def test_name(X_y):
    X, y = X_y
    clf = LogisticRegressionBinaryPipeline(penalty='l2', C=1.0, impute_strategy='mean', number_features=len(X[0]), random_state=0)
    assert clf.name == 'Logistic Regression Classifier w/ One Hot Encoder + Simple Imputer + Standard Scaler'


def test_estimator_not_last(X_y):
    X, y = X_y

    class MockLogisticRegressionPipeline(PipelineBase):
        name = "Mock Logistic Regression Pipeline"

        def __init__(self, penalty, C, impute_strategy,
                     number_features, n_jobs=-1, random_state=0):
            imputer = SimpleImputer(impute_strategy=impute_strategy)
            enc = OneHotEncoder()
            scaler = StandardScaler()
            estimator = LogisticRegressionClassifier(random_state=random_state,
                                                     penalty=penalty,
                                                     C=C,
                                                     n_jobs=-1)
            super().__init__(component_list=[enc, imputer, estimator, scaler], n_jobs=n_jobs, random_state=random_state)

    err_msg = "A pipeline must have an Estimator as the last component in component_list."
    with pytest.raises(ValueError, match=err_msg):
        MockLogisticRegressionPipeline(penalty='l2', C=1.0, impute_strategy='mean', number_features=len(X[0]), random_state=0)


def test_multi_format_creation(X_y):
    X, y = X_y
    clf = PipelineBase(component_list=['Simple Imputer', 'One Hot Encoder', StandardScaler(), 'Logistic Regression Classifier'], n_jobs=-1, random_state=0)
    correct_components = [SimpleImputer, OneHotEncoder, StandardScaler, LogisticRegressionClassifier]
    for component, correct_components in zip(clf.component_list, correct_components):
        assert isinstance(component, correct_components)
    assert clf.model_type == ModelTypes.LINEAR_MODEL
    assert clf.problem_types == [ProblemTypes.BINARY, ProblemTypes.MULTICLASS]

    clf.fit(X, y, 'precision')
    clf.score(X, y, ['precision'])
    assert not clf.feature_importances.isnull().all().all()


def test_multiple_feature_selectors(X_y):
    X, y = X_y
    clf = PipelineBase(component_list=['Simple Imputer', 'One Hot Encoder', 'RF Classifier Select From Model', StandardScaler(), 'RF Classifier Select From Model', 'Logistic Regression Classifier'], n_jobs=-1, random_state=0)
    correct_components = [SimpleImputer, OneHotEncoder, RFClassifierSelectFromModel, StandardScaler, RFClassifierSelectFromModel, LogisticRegressionClassifier]
    for component, correct_components in zip(clf.component_list, correct_components):
        assert isinstance(component, correct_components)
    assert clf.model_type == ModelTypes.LINEAR_MODEL
    assert clf.problem_types == [ProblemTypes.BINARY, ProblemTypes.MULTICLASS]

    clf.fit(X, y, 'precision')
    clf.score(X, y, ['precision'])
    assert not clf.feature_importances.isnull().all().all()
