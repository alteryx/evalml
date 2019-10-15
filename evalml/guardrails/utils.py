import pandas as pd


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
        a set of features that are highly-null
    """
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X)

    threshold = len(X) * percent_threshold
    is_highly_null = (X.isnull().sum() >= threshold)
    highly_null_indices = set(is_highly_null[is_highly_null].index)
    return highly_null_indices
