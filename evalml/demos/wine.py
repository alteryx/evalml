import pandas as pd
import woodwork as ww
from sklearn.datasets import load_wine as load_wine_sk

import evalml
from evalml.preprocessing import load_data


def load_wine(use_local=False):
    """Load wine dataset. Multiclass problem.

    Returns:
        (pd.Dataframe, pd.Series): X and y
    """
    if use_local:
        data = load_wine_sk()
        X = pd.DataFrame(data.data, columns=data.feature_names)
        y = pd.Series(data.target)
        y = y.map(lambda x: data["target_names"][x])
    else:
        filepath = (
            "https://api.featurelabs.com/datasets/wine.csv?library=evalml&version="
            + evalml.__version__
        )
        X, y = load_data(filepath, index=None, target="target")
        y.name = None

    X.ww.init()
    y = ww.init_series(y)
    return X, y
