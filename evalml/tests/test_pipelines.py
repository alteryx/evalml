import errno
import os
import shutil

import pandas as pd
import pytest
from sklearn import datasets

import evalml.tests as tests
from evalml import load_pipeline, save_pipeline
from evalml.objectives import FraudCost
from evalml.pipelines import LogisticRegressionPipeline
from evalml.pipelines.utils import get_pipelines, list_model_types

CACHE = os.path.join(os.path.dirname(tests.__file__), '.cache')


@pytest.fixture
def data():
    X, y = datasets.make_classification(n_samples=100, n_features=20,
                                        n_informative=2, n_redundant=2, random_state=0)

    return X, y


def test_list_model_types():
    assert set(list_model_types("classification")) == set(["random_forest", "xgboost", "linear_model"])
    assert set(list_model_types("regression")) == set(["random_forest"])


def test_get_pipelines():
    assert len(get_pipelines(problem_type="classification")) == 3
    assert len(get_pipelines(problem_type="classification", model_types=["linear_model"])) == 1
    assert len(get_pipelines(problem_type="regression")) == 1


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


def test_serialization(X_y, trained_model, path_management):
    X, y = X_y
    path = os.path.join(path_management, 'pipe.pkl')
    pipeline = trained_model.best_pipeline
    save_pipeline(pipeline, path)
    assert pipeline.score(X, y) == load_pipeline(path).score(X, y)


def test_reproducibility(X_y):
    X, y = X_y
    X = pd.DataFrame(X)
    y = pd.Series(y)

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

    clf_2 = LogisticRegressionPipeline(objective=objective, penalty='l2', C=1.0, impute_strategy='mean', number_features=len(X[0]), random_state=30)
    clf_2.fit(X, y)

    assert clf_1.score(X, y) == clf.score(X, y)
    assert clf.score(X, y) != clf_2.score(X, y)
