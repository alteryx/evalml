import importlib
import os

import pandas as pd
from sklearn import datasets
from skopt.space import Integer, Real

import pytest

import pytest


def has_minimal_deps():
    for module in ['xgboost', 'catboost', 'plotly', 'plotly.graph_objects']:
        try:
            importlib.import_module(module)
        except ImportError:
            return True
    return False


@pytest.fixture
def minimal_deps():
    return has_minimal_deps()


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
def test_space():
    return [Integer(0, 10), Real(0, 10), ['option_a', 'option_b', 'option_c']]


@pytest.fixture
def test_space_unicode():
    return [Integer(0, 10), Real(0, 10), ['option_a 💩', u'option_b 💩', 'option_c 💩']]


@pytest.fixture
def test_space_small():
    list_of_space = list()
    list_of_space.append(['most_frequent', 'median', 'mean'])
    list_of_space.append(['a', 'b', 'c'])
    return list_of_space
