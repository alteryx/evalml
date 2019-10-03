import pandas as pd
import pytest
import os
from sklearn import datasets

from evalml import AutoClassifier


@pytest.fixture
def X_y():
    X, y = datasets.make_classification(n_samples=100, n_features=20,
                                        n_informative=2, n_redundant=2, random_state=0)

    return X, y


@pytest.fixture
def X_y_multi():
    X, y = datasets.make_classification(n_samples=100, n_features=20, n_classes=3,
                                        n_informative=3, n_redundant=2, random_state=0)
    return X, y


@pytest.fixture
def X_y_categorical_regression():
    flights = pd.read_csv('https://raw.githubusercontent.com/mwaskom/seaborn-data/master/tips.csv')
    y = flights['tip']
    X = flights.drop('tip', axis=1)

    # add categorical dtype
    X['smoker'] = X['smoker'].astype('category')
    return X, y


@pytest.fixture
def X_y_categorical_classification():
    titanic = pd.read_csv('https://featuretools-static.s3.amazonaws.com/evalml/Titanic/train.csv')
    y = titanic['Survived']
    X = titanic.drop('Survived', axis=1)
    return X, y


@pytest.fixture
def trained_model(X_y):
    X, y = X_y

    clf = AutoClassifier()

    clf.fit(X, y)

    return clf


@pytest.fixture
def rootdir():
    return os.path.dirname(os.path.abspath(__file__))