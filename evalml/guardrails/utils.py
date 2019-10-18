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
        A dictionary of features with column name or index and their percentage of null values
    """
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X)

    percent_null = (X.isnull().mean()).to_dict()
    highly_null_cols = {key: value for key, value in percent_null.items() if value >= percent_threshold}
    return highly_null_cols


def detect_collinearity(X, threshold=.95):
    """Check if collinearity exists.

    Currently only supports numeric features.

    Args:
        X (pd.DataFrame): The input features to check
        threshold (float): the correlation threshold to be considered correlated  Defaults to .95

    Returns:
        dictionary
    """

    # only select numeric
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    X = X.select_dtypes(include=numerics)

    if len(X.columns) == 0:
        return {}

    corrs = X.corr().abs()
    out = corrs[corrs >= threshold]
    return out.to_dict()


def detect_multicollinearity(X, threshold=.95):
    """Check if multicollinearity exists.

    Currently only supports numeric features.

    Args:
        X (pd.DataFrame): The input features to check
        threshold (float): the correlation threshold to be considered correlated. Defaults to .95

    Returns:
        dictionary
    """

    # only select numeric
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    X = X.select_dtypes(include=numerics)

    if len(X.columns) == 0:
        return {}

    # vif > 5, 10
    vif = pd.Series([variance_inflation_factor(X.values, i) for i in range(X.shape[1])], index=X.columns).to_dict()
    X_arr = X.to_numpy()

    # condition index > 30
    condition_num = np.linalg.cond(X_arr)

    return
