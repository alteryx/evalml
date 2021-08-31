"""Load the diabetes dataset, which can be used for regression problems."""

import woodwork as ww

import evalml
from evalml.preprocessing import load_data


def load_diabetes():
    """Load diabetes dataset. Used for regression problem.

    Returns:
        (pd.Dataframe, pd.Series): X and y
    """
    filename = (
        "https://api.featurelabs.com/datasets/diabetes.csv?library=evalml&version="
        + evalml.__version__
    )
    X, y = load_data(filename, index=None, target="target")
    y.name = None

    X.ww.init()
    y = ww.init_series(y)

    return X, y
