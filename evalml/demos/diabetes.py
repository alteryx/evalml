import pandas as pd
import woodwork as ww
from sklearn.datasets import load_diabetes as load_diabetes_sk

import evalml
from evalml.preprocessing import load_data


def load_diabetes(use_local=False):
    """Load diabetes dataset. Regression problem

    Returns:
        (pd.Dataframe, pd.Series): X and y
    """
    if use_local:
        data = load_diabetes_sk()
        X = pd.DataFrame(data.data, columns=data.feature_names)
        y = pd.Series(data.target)
    else:
        filename = (
            "https://api.featurelabs.com/datasets/diabetes.csv?library=evalml&version="
            + evalml.__version__
        )
        X, y = load_data(filename, index=None, target="target")
        y.name = None

    X.ww.init()
    y = ww.init_series(y)

    return X, y
