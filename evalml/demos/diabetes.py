import pandas as pd
from sklearn.datasets import load_diabetes as load_diabetes_sk


def load_diabetes():
    """Load diabetes dataset. Regression problem"""
    data = load_diabetes_sk()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target)
    return X, y
