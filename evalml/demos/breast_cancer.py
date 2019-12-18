import pandas as pd
from sklearn.datasets import load_breast_cancer as load_breast_cancer_sk


def load_breast_cancer():
    """Load breast cancer dataset. Binary problem

    Returns:
        pd.DataFrame, pd.Series: X, y
    """
    data = load_breast_cancer_sk()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target)
    return X, y
