import pandas as pd
from sklearn.datasets import load_wine as load_wine_sk


def load_wine():
    """Load wine dataset. Multiclass problem

    Returns:
        pd.DataFrame, pd.Series: X, y
    """
    data = load_wine_sk()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    labels = data.target_names[data.target]
    y = pd.Series(labels)
    return X, y
