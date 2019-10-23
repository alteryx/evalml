import numpy as np
import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor


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
        threshold (float): the correlation threshold to be considered correlated. Defaults to .95.

    Returns:
        A dictionary mapping potentially collinear features and their corresponding correlation coefficient
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


def detect_multicollinearity(X, threshold=5):
    """Check if multicollinearity exists.

    Currently only supports numeric features.

    Args:
        X (pd.DataFrame): The input features to check
        threshold (float): the VIF threshold to use to determine multicollinearity. Defaults to 10

    Returns:
        A dictionary of features with VIF scores greater than threshold mapped to their corresponding VIF score
    """

    # only select numeric
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    X = X.select_dtypes(include=numerics)
    from statsmodels.tools.tools import add_constant

    if len(X.columns) == 0:
        return {}

    multicollinear_cols = {}
    X = add_constant(X)  # since variance_inflation_factor doesn't add intercept
    vif = pd.Series([variance_inflation_factor(X.values, i) for i in range(X.shape[1])], index=X.columns)
    vif = vif[vif >= threshold]
    multicollinear_cols = vif.to_dict()
    return multicollinear_cols
