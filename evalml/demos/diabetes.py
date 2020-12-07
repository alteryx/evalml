import pandas as pd
import woodwork as ww
from sklearn.datasets import load_diabetes as load_diabetes_sk


def load_diabetes(return_pandas=False):
    """Load diabetes dataset. Regression problem

    Returns:
        Union[(ww.DataTable, ww.DataColumn), (pd.Dataframe, pd.Series)]: X and y
    """
    data = load_diabetes_sk()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target)
    if return_pandas:
        return X, y
    return ww.DataTable(X), ww.DataColumn(y)
