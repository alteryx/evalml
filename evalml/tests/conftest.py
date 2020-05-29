import os

import numpy as np
import pandas as pd
import pytest
from sklearn import datasets
from skopt.space import Integer, Real

from evalml.model_family import ModelFamily
from evalml.pipelines import BinaryClassificationPipeline, RegressionPipeline
from evalml.pipelines.components import Estimator
from evalml.problem_types import ProblemTypes


def pytest_addoption(parser):
    parser.addoption("--has-minimal-dependencies", action="store_true", default=False,
                     help="If true, tests will assume only the dependencies in"
                     "core-requirements.txt have been installed.")


@pytest.fixture
def has_minimal_dependencies(pytestconfig):
    return pytestconfig.getoption("--has-minimal-dependencies")


@pytest.fixture
def X_y():
    X, y = datasets.make_classification(n_samples=100, n_features=20,
                                        n_informative=2, n_redundant=2, random_state=0)

    return X, y


@pytest.fixture
def X_y_reg():
    X, y = datasets.make_regression(n_samples=100, n_features=20,
                                    n_informative=3, random_state=0)
    return X, y


@pytest.fixture
def X_y_multi():
    X, y = datasets.make_classification(n_samples=100, n_features=20, n_classes=3,
                                        n_informative=3, n_redundant=2, random_state=0)
    return X, y


@pytest.fixture
def X_y_categorical_regression():
    data_path = os.path.join(os.path.dirname(__file__), "data/tips.csv")
    flights = pd.read_csv(data_path)

    y = flights['tip']
    X = flights.drop('tip', axis=1)

    # add categorical dtype
    X['smoker'] = X['smoker'].astype('category')
    return X, y


@pytest.fixture
def X_y_categorical_classification():
    data_path = os.path.join(os.path.dirname(__file__), "data/titanic.csv")
    titanic = pd.read_csv(data_path)

    y = titanic['Survived']
    X = titanic.drop('Survived', axis=1)
    return X, y


@pytest.fixture
def dummy_pipeline_hyperparameters():
    return {'Mock Classifier': {
        'param a': Integer(0, 10),
        'param b': Real(0, 10),
        'param c': ['option a', 'option b', 'option c'],
        'param d': ['option a', 'option b', 100, np.inf]
    }}


@pytest.fixture
def dummy_pipeline_hyperparameters_unicode():
    return {'Mock Classifier': {
        'param a': Integer(0, 10),
        'param b': Real(0, 10),
        'param c': ['option a 💩', 'option b 💩', 'option c 💩'],
        'param d': ['option a', 'option b', 100, np.inf]
    }}


@pytest.fixture
def dummy_pipeline_hyperparameters_small():
    return {'Mock Classifier': {
        'param a': ['most_frequent', 'median', 'mean'],
        'param b': ['a', 'b', 'c']
    }}


@pytest.fixture
def dummy_classifier_estimator_class():
    class MockEstimator(Estimator):
        name = "Mock Classifier"
        model_family = ModelFamily.NONE
        supported_problem_types = [ProblemTypes.BINARY, ProblemTypes.MULTICLASS]
        hyperparameter_ranges = {}

        def __init__(self, random_state=0):
            super().__init__(parameters={}, component_obj=None, random_state=random_state)
    return MockEstimator


@pytest.fixture
def dummy_binary_pipeline_class(dummy_classifier_estimator_class):
    MockEstimator = dummy_classifier_estimator_class

    class MockBinaryClassificationPipeline(BinaryClassificationPipeline):
        estimator = MockEstimator
        component_graph = [MockEstimator()]

    return MockBinaryClassificationPipeline


@pytest.fixture
def dummy_regressor_estimator_class():
    class MockRegressor(Estimator):
        name = "Mock Regressor"
        model_family = ModelFamily.NONE
        supported_problem_types = [ProblemTypes.REGRESSION]
        hyperparameter_ranges = {}

        def __init__(self, random_state=0):
            super().__init__(parameters={}, component_obj=None, random_state=random_state)

    return MockRegressor


@pytest.fixture
def dummy_regression_pipeline(dummy_regressor_estimator_class):
    MockRegressor = dummy_regressor_estimator_class

    class MockRegressionPipeline(RegressionPipeline):
        component_graph = [MockRegressor()]
    return MockRegressionPipeline(parameters={})
