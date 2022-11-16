"""Load the diabetes dataset, which can be used for regression problems."""
import pandas as pd
import woodwork as ww
from sklearn.preprocessing import scale

import evalml
from evalml.preprocessing import load_data


def load_diabetes():
    """Load diabetes dataset. Used for regression problem.

    Returns:
        (pd.Dataframe, pd.Series): X and y
    """
    filename = (
        "https://oss.alteryx.com/datasets/diabetes-2022-06-27.csv?library=evalml&version="
        + evalml.__version__
    )
    X, y = load_data(filename, index=None, target="target")
    y.name = None

    # This scales the feature variables by the standard deviation times the square root of n_samples
    # This change is necessary due to https://github.com/scikit-learn/scikit-learn/pull/16605
    # In previous versions the diabetes.csv data was returned, but now scikit-learn scales the data then returns
    y = y.astype(float)
    X_np = scale(X.to_numpy(float), copy=False)
    X_np /= X_np.shape[0] ** 0.5
    X = pd.DataFrame(X_np, columns=X.columns)
    X.ww.init()
    y = ww.init_series(y)

    return X, y
