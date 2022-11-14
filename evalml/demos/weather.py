"""The Australian daily-min-termperatures weather dataset."""
import pandas as pd

import evalml
from evalml.preprocessing import load_data
from evalml.utils import infer_feature_types


def load_weather():
    """Load the Australian daily-min-termperatures weather dataset.

    Returns:
        (pd.Dataframe, pd.Series): X and y

    """
    filename = (
        "https://oss.alteryx.com/datasets/daily-min-temperatures.csv?library=evalml&version="
        + evalml.__version__
    )
    X, y = load_data(filename, index=None, target="Temp")

    missing_date_1 = pd.DataFrame([pd.to_datetime("1984-12-31")], columns=["Date"])
    missing_date_2 = pd.DataFrame([pd.to_datetime("1988-12-31")], columns=["Date"])
    missing_y_1 = pd.Series([14.5], name="Temp")
    missing_y_2 = pd.Series([14.5], name="Temp")

    X = pd.concat([X.iloc[:1460], missing_date_1, X.iloc[1460:]]).reset_index(drop=True)
    X = pd.concat([X.iloc[:2921], missing_date_2, X.iloc[2921:]]).reset_index(drop=True)
    y = pd.concat([y.iloc[:1460], missing_y_1, y.iloc[1460:]]).reset_index(drop=True)
    y = pd.concat([y.iloc[:2921], missing_y_2, y.iloc[2921:]]).reset_index(drop=True)

    X = infer_feature_types(X)
    y = infer_feature_types(y)

    return X, y
