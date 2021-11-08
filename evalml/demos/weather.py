"""The Australian daily-min-termperatures weather dataset."""
import evalml
from evalml.preprocessing import load_data


def load_weather():
    """Load the Australian daily-min-termperatures weather dataset.

    Returns:
        (pd.Dataframe, pd.Series): X and y

    """
    filename = (
        "https://api.featurelabs.com/datasets/daily-min-temperatures.csv?library=evalml&version="
        + evalml.__version__
    )
    X, y = load_data(filename, index=None, target="Temp")
    return X, y
