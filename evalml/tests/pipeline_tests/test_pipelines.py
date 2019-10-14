import errno
import os
import shutil

import pandas as pd
import pytest

import evalml.tests as tests
from evalml.model_types import ModelTypes
from evalml.objectives import FraudCost, Precision
from evalml.pipelines import LogisticRegressionPipeline
from evalml.pipelines.utils import (
    get_pipelines,
    list_model_types,
    load_pipeline,
    save_pipeline
)
from evalml.problem_types import ProblemTypes

CACHE = os.path.join(os.path.dirname(tests.__file__), '.cache')


def test_list_model_types():
    assert set(list_model_types(ProblemTypes.BINARY)) == set([ModelTypes.RANDOM_FOREST, ModelTypes.XGBOOST, ModelTypes.LINEAR_MODEL])
    assert set(list_model_types(ProblemTypes.REGRESSION)) == set([ModelTypes.RANDOM_FOREST])


def test_get_pipelines():
    assert len(get_pipelines(problem_type=ProblemTypes.BINARY)) == 3
    assert len(get_pipelines(problem_type=ProblemTypes.BINARY, model_types=[ModelTypes.LINEAR_MODEL])) == 1
    assert len(get_pipelines(problem_type=ProblemTypes.REGRESSION)) == 1
    with pytest.raises(RuntimeError, match="Unrecognized model type for problem type"):
        get_pipelines(problem_type=ProblemTypes.REGRESSION, model_types=["random_forest", "xgboost"])


@pytest.fixture
def path_management():
    path = CACHE
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno != errno.EEXIST:  # EEXIST corresponds to FileExistsError
            raise e
    yield path
    shutil.rmtree(path)


def test_serialization(X_y, path_management):
    X, y = X_y
    path = os.path.join(path_management, 'pipe.pkl')
    objective = Precision()

    pipeline = LogisticRegressionPipeline(objective=objective, penalty='l2', C=1.0, impute_strategy='mean', number_features=len(X[0]))
    pipeline.fit(X, y)
    save_pipeline(pipeline, path)
    assert pipeline.score(X, y) == load_pipeline(path).score(X, y)


@pytest.fixture
def pickled_pipeline_path(X_y, path_management):
    X, y = X_y
    path = os.path.join(path_management, 'pickled_pipe.pkl')
    MockPrecision = type('MockPrecision', (Precision,), {})
    objective = MockPrecision()
    pipeline = LogisticRegressionPipeline(objective=objective, penalty='l2', C=1.0, impute_strategy='mean', number_features=len(X[0]))
    pipeline.fit(X, y)
    save_pipeline(pipeline, path)
    return path


def test_load_pickled_pipeline_with_custom_objective(X_y, pickled_pipeline_path):
    X, y = X_y
    objective = Precision()
    pipeline = LogisticRegressionPipeline(objective=objective, penalty='l2', C=1.0, impute_strategy='mean', number_features=len(X[0]))
    pipeline.fit(X, y)
    assert load_pipeline(pickled_pipeline_path).score(X, y) ==  pipeline.score(X,y)


def test_reproducibility(X_y):
    X, y = X_y
    X = pd.DataFrame(X)

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
