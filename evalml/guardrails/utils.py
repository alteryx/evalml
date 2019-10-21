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
        dictionary of potentially collinear features and their percent chance of being collinear
    """

    # only select numeric
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    X = X.select_dtypes(include=numerics)

    if len(X.columns) == 0:
        return {}

    corrs = X.corr().abs()
    corrs = corrs.mask(np.tril(np.ones(corrs.shape)).astype(bool)).stack()
    out = {key: value for (key, value) in corrs.items() if value >= threshold}
    return out


def detect_multicollinearity(X, vif_threshold=10, index_threshold=30):
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

    multicollinear_cols = {}
    # vif > 5, 10
    vif = pd.Series([variance_inflation_factor(X.values, i) for i in range(X.shape[1])], index=X.columns)
    vif = vif[vif >= threshold]
    multicollinear_cols = vif.to_dict()

    corr = np.corrcoef(X, rowvar=0)
    eig_vals, eig_vecs = np.linalg.eig(corr)
    print np.linalg.cond(corr)

    return multicollinear_cols
