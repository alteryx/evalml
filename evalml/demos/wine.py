import pandas as pd
from sklearn.datasets import load_wine as load_wine_sk

import woodwork as ww

def load_wine():
    """Load wine dataset. Multiclass problem

    Returns:
        pd.DataFrame, pd.Series: X, y
    """
    data = load_wine_sk()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target)
    y = y.map(lambda x: data["target_names"][x])
    X = ww.DataTable(X)
    y = ww.DataColumn(y)
    return X, y
