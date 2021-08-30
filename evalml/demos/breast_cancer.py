"""Load the breast cancer dataset, which can be used for binary classification problems."""
import woodwork as ww

import evalml
from evalml.preprocessing import load_data


def load_breast_cancer():
    """Load breast cancer dataset. Binary classification problem.

    Returns:
        (pd.Dataframe, pd.Series): X and y
    """
    filepath = (
        "https://api.featurelabs.com/datasets/breast_cancer.csv?library=evalml&version="
        + evalml.__version__
    )
    X, y = load_data(filepath, index=None, target="target")
    y.name = None

    X.ww.init()
    y = ww.init_series(y)
    return X, y
