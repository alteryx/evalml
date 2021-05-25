import pandas as pd
import woodwork as ww
from sklearn.datasets import load_wine as load_wine_sk


def load_wine():
    """Load wine dataset. Multiclass problem.

    Returns:
        (pd.Dataframe, pd.Series): X and y
    """
    data = load_wine_sk()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target)
    y = y.map(lambda x: data["target_names"][x])
    X.ww.init()
    y = ww.init_series(y)
    return X, y
