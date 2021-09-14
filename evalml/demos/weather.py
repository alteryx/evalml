"""The Australian daily-min-termperatures weather dataset."""
import os

import pandas as pd

from evalml.utils import infer_feature_types


def load_weather():
    """Load the Australian daily-min-termperatures weather dataset."""
    currdir_path = os.path.dirname(os.path.abspath(__file__))
    data_folder_path = os.path.join(currdir_path, "data")
    weather_data_path = os.path.join(data_folder_path, "daily-min-temperatures.csv")
    X = pd.read_csv(weather_data_path)
    X = infer_feature_types(X, feature_types={"Date": "Datetime", "Temp": "Double"})
    y = X.ww.pop("Temp")
    return X, y
