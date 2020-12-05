import pandas as pd
import woodwork as ww
from sklearn.datasets import load_wine as load_wine_sk


def load_wine(return_pandas=False):
    """Load wine dataset. Multiclass problem

    Returns:
        Union[ww.DataTable, pd.Dataframe], Union[ww.DataColumn, pd.Series]: X and y
    """
    data = load_wine_sk()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target)
    y = y.map(lambda x: data["target_names"][x])
    if return_pandas:
        return X, y
    return ww.DataTable(X), ww.DataColumn(y)
