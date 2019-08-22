import errno
import os
import shutil

import pytest
from sklearn import datasets

import evalml.tests as tests
from evalml import AutoClassifier
from evalml.pipelines.utils import get_pipelines, list_model_types, load_pipeline, save_pipeline

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
    clf = trained_model
    pipeline = clf.best_pipeline
    save_pipeline(pipeline, path)
    assert pipeline.score(X, y) == load_pipeline(path).score(X, y)

    other_p = clf.get_pipeline(1)
    path = os.path.join(path_management, 'pipe1.pkl')
    save_pipeline(other_p, path)
    assert pipeline.score(X, y) != load_pipeline(path).score(X, y)
