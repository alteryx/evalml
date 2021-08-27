import pandas as pd
from evalml.utils import infer_feature_types
import os


def load_weather():
    currdir_path = os.path.dirname(os.path.abspath(__file__))
    data_folder_path = os.path.join(currdir_path, "data")
    churn_data_path = os.path.join(data_folder_path, "daily-min-temperatures.csv")
    X = pd.read_csv(churn_data_path)
    X = infer_feature_types(X, feature_types={"Date": "Datetime", "Temp": "Double"})
    y = X.ww.pop("Temp")
    return X, y