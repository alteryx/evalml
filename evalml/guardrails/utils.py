import numpy as np
import pandas as pd
import scipy.stats as scipy_stats
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


def detect_correlation(X, threshold=.90):
    """Check if correlation exists between features.

    Args:
        X (pd.DataFrame): The input features to check
        threshold (float): the correlation threshold to be considered correlated. Defaults to .95.

    Currently only supports checking between numeric-numeric and categorical-categorical features

    Returns:
        A dictionary mapping potentially correlated features and their corresponding correlation coefficient
    """
    correlated = {}
    correlated.update(detect_categorical_correlation(X))
    correlated.update(detect_collinearity(X))
    return correlated


def detect_categorical_correlation(X, threshold=.95):
    """Check if correlation exists between categorical features.

    Args:
        X (pd.DataFrame): The input features to check
        threshold (float): the correlation threshold to be considered correlated. Defaults to .95.

    Returns:
        A dictionary mapping potentially collinear features and their corresponding correlation coefficient
    """
    def cramers_v_bias_corrected(confusion_matrix):
        """ Calculate Cramer's V statistic for categorial-categorial correlation with bias correction."""
        chi2 = scipy_stats.chi2_contingency(confusion_matrix)[0]
        n = confusion_matrix.sum().sum()  # grand total of observations
        phi2 = chi2 / n
        r, k = confusion_matrix.shape
        phi2corr = max(0, phi2 - ((k - 1) * (r - 1)) / (n - 1))
        rcorr = r - np.square(r - 1) / (n - 1)
        kcorr = k - np.square(k - 1) / (n - 1)
        return np.sqrt(phi2corr / min((kcorr - 1), (rcorr - 1)))

    # only select categorical features
    X = X.select_dtypes(include=['category'])

    cramers_corr = {}
    num_cols = X.shape[1]
    for i in range(num_cols):
        for j in range(i + 1, num_cols):
            # only calculate Cramer's V for upper triangle since Cramer's V produces symmetric scores
            confusion_matrix = pd.crosstab(X.iloc[:, i], X.iloc[:, j])
            col_names = (X.columns[i], X.columns[j])
            cramers_v = cramers_v_bias_corrected(confusion_matrix)
            cramers_corr.update({col_names: cramers_v})
    out = {key: value for (key, value) in cramers_corr.items() if value >= threshold}
    return out


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
    """Check if multicollinearity exists amongst numerical features.

    Args:
        X (pd.DataFrame): The input features to check
        threshold (float): the VIF threshold to use to determine multicollinearity. Defaults to 10

    Returns:
        A dictionary of features with VIF scores greater than threshold mapped to their corresponding VIF score
    """

    # only select numeric
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    X = X.select_dtypes(include=numerics)
    if len(X.columns) == 0:
        return {}

    multicollinear_cols = {}
    X = X.assign(const=1)  # since variance_inflation_factor doesn't add intercept
    vif = pd.Series([variance_inflation_factor(X.values, i) for i in range(X.shape[1])], index=X.columns)
    vif = vif[vif >= threshold]
    multicollinear_cols = vif.to_dict()
    return multicollinear_cols
