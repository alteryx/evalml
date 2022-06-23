"""Load the diabetes dataset, which can be used for regression problems."""
import pandas as pd
import woodwork as ww
from sklearn.preprocessing import scale

import evalml
from evalml.preprocessing import load_data

# def load_diabetes():
#     """Load diabetes dataset. Used for regression problem.

#     Returns:
#         (pd.Dataframe, pd.Series): X and y
#     """
#     filename = (
#         "https://api.featurelabs.com/datasets/diabetes.csv?library=evalml&version="
#         + evalml.__version__
#     )
#     X, y = load_data(filename, index=None, target="target")
#     y.name = None

#     X.ww.init()
#     y = ww.init_series(y)

#     return X, y


def load_diabetes():
    """Load diabetes dataset. Used for regression problem.

    Returns:
        (pd.Dataframe, pd.Series): X and y
    """
    filename = "evalml/demos/data/diabetes.csv"
    X, y = load_data(filename, index=None, target="target")
    numpy_of_X = X.to_numpy()
    numpy_of_X = scale(numpy_of_X, copy=False)
    numpy_of_X /= numpy_of_X.shape[0] ** 0.5
    X = pd.DataFrame(numpy_of_X, columns=X.columns)
    y.name = None

    X.ww.init()
    y = ww.init_series(y)

    return X, y
