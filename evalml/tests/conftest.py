import pandas as pd
import pytest
from sklearn import datasets

from evalml import AutoClassifier


@pytest.fixture
def X_y():
    X, y = datasets.make_classification(n_samples=100, n_features=20,
                                        n_informative=2, n_redundant=2, random_state=0)

    return pd.DataFrame(X), pd.DataFrame(y)


@pytest.fixture
def X_y_multi():
    X, y = datasets.make_classification(n_samples=100, n_features=20, n_classes=3,
                                        n_informative=3, n_redundant=2, random_state=0)

    return pd.DataFrame(X), pd.DataFrame(y)


@pytest.fixture
def trained_model(X_y):
    X, y = X_y

    clf = AutoClassifier()

    clf.fit(X, y)

    return clf
