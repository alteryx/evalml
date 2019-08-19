import pandas as pd
from sklearn.datasets import load_diabetes as load_diabetes_sk


def load_diabetes():
    """Load diabetes dataset. Regression problem"""
    X, y = load_diabetes_sk(return_X_y=True)
    return pd.DataFrame(X), pd.Series(y)
