import errno
import os
import shutil

import numpy as np
import pytest

import evalml.tests as tests
from evalml import AutoClassifier, load_pipeline, save_pipeline
from evalml.objectives import Precision
from evalml.pipelines import (
    LogisticRegressionPipeline,
    RFClassificationPipeline,
    XGBoostPipeline
)
from evalml.pipelines.utils import get_pipelines, list_model_types

CACHE = os.path.join(os.path.dirname(tests.__file__), '.cache')


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
    clf = trained_model
    pipeline = clf.best_pipeline
    save_pipeline(pipeline, path)
    assert pipeline.score(X, y) == load_pipeline(path).score(X, y)

    other_p = clf.get_pipeline(4)
    path = os.path.join(path_management, 'pipe1.pkl')
    save_pipeline(other_p, path)
    assert pipeline.score(X, y) != load_pipeline(path).score(X, y)


def test_multi(X_y_multi):
    X, y = X_y_multi
    clf = LogisticRegressionPipeline(objective=Precision(), penalty='l2', C=1.0, impute_strategy='mean', number_features=0)
    clf.fit(X, y)
    clf.score(X, y)
    y_pred = clf.predict(X)
    assert len(np.unique(y_pred)) == 3

    clf = XGBoostPipeline(objective=Precision(), eta=0.1, min_child_weight=1, max_depth=3, impute_strategy='mean', percent_features=1.0, number_features=0)
    clf.fit(X, y)
    clf.score(X, y)
    y_pred = clf.predict(X)
    assert len(np.unique(y_pred)) == 3

    clf = RFClassificationPipeline(objective=Precision(), n_estimators=10, max_depth=3, impute_strategy='mean', percent_features=1.0, number_features=0)
    clf.fit(X, y)
    clf.score(X, y)
    y_pred = clf.predict(X)
    assert len(np.unique(y_pred)) == 3


def test_multi_auto(X_y_multi):
    X, y = X_y_multi
    clf = AutoClassifier(objective="precision")
    clf.fit(X, y)
    y_pred = clf.best_pipeline.predict(X)
    assert len(np.unique(y_pred)) == 3
