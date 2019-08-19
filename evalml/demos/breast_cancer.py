import pandas as pd
from sklearn.datasets import load_breast_cancer as load_breast_cancer_sk


def load_breast_cancer():
    """Load wine dataset. Multiclass problem"""
    X, y = load_breast_cancer_sk(return_X_y=True)
    return pd.DataFrame(X), pd.Series(y)
