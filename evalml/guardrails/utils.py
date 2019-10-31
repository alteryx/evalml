import pandas as pd
from sklearn.ensemble import IsolationForest


def detect_label_leakage(X, y, threshold=.95):
    """Check if any of the features are highly correlated with the target.

    Currently only supports binary and numeric targets and features

    Args:
        X (pd.DataFrame): The input features to check
        y (pd.Series): the labels
        threshold (float): the correlation threshold to be considered leakage. Defaults to .95

    Returns:
        leakage, dictionary of features with leakage and corresponding threshold
    """

    # only select numeric
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64', 'bool']
    X = X.select_dtypes(include=numerics)

    if len(X.columns) == 0:
        return {}

    corrs = X.corrwith(y).abs()
    out = corrs[corrs >= threshold]
    return out.to_dict()


def detect_highly_null(X, percent_threshold=.95):
    """ Checks if there are any highly-null columns in a dataframe.

    Args:
        X (DataFrame) : features
        percent_threshold(float): Require that percentage of null values to be considered "highly-null", defaults to .95

    Returns:
        A dictionary of features with column name or index and their percentage of null values
    """
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X)

    percent_null = (X.isnull().mean()).to_dict()
    highly_null_cols = {key: value for key, value in percent_null.items() if value >= percent_threshold}
    return highly_null_cols


def detect_outliers(X, random_state=0):
    """ Checks if there are any outliers in a dataframe by using first Isolation Forest to obtain the anomaly score
    of each index and then using IQR to determine score anomalies. Indices with score anomalies are considered outliers.

    Args:
        X (DataFrame) : features

    Returns:
        A set of indices that may have outlier data.
    """
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X)

    # only select numeric
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    X = X.select_dtypes(include=numerics)

    if len(X.columns) == 0:
        return {}

    def get_IQR(df, k=2.0):
        q1 = df.quantile(0.25)
        q3 = df.quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - (k * iqr)
        upper_bound = q3 + (k * iqr)
        return (lower_bound, upper_bound)

    clf = IsolationForest(random_state=random_state, behaviour="new", contamination=0.1)
    clf.fit(X)
    scores = pd.Series(clf.decision_function(X))
    lower_bound, upper_bound = get_IQR(scores, k=2)
    outliers = (scores < lower_bound) | (scores > upper_bound)
    outliers_indices = outliers[outliers].index.values.tolist()
    return outliers_indices
